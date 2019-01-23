from __future__ import print_function

import numpy as np

import math

from scipy.misc import logsumexp


import torch
import torch.utils.data
import torch.nn as nn
from torch.nn import Linear
from torch.autograd import Variable

from utils.distributions import log_Bernoulli, log_Normal_diag, log_Normal_standard, log_Logistic_256
from utils.visual_evaluation import plot_histogram
from utils.nn import he_init, GatedDense, NonLinear, NonGatedDense

from Model import Model


class linIAF(nn.Module):
    def __init__(self,args):
        super(linIAF, self).__init__()
        self.args = args

    def forward(self, L, z):
        '''
        :param L: batch_size (B) x latent_size^2 (L^2)
        :param z: batch_size (B) x latent_size (L)
        :return: z_new = L*z
        '''
        # L->tril(L)
        self.args.hidden_size = 300
        L_matrix = L.view( -1, self.args.hidden_size, self.args.hidden_size ) # resize to get B x L x L
        LTmask = torch.tril( torch.ones(self.args.hidden_size, self.args.hidden_size), diagonal=-1 ) # lower-triangular mask matrix (1s in lower triangular part)
        I = Variable( torch.eye(self.args.hidden_size, self.args.hidden_size).expand(L_matrix.size(0), self.args.hidden_size, self.args.hidden_size) )
        if self.args.cuda:
            LTmask = LTmask.cuda()
            I = I.cuda()
        LTmask = Variable(LTmask)
        LTmask = LTmask.unsqueeze(0).expand( L_matrix.size(0), self.args.hidden_size, self.args.hidden_size ) # 1 x L x L -> B x L x L
        LT = torch.mul( L_matrix, LTmask ) + I # here we get a batch of lower-triangular matrices with ones on diagonal

        # z_new = L * z
        z_new = torch.bmm( LT , z.unsqueeze(2) ).squeeze(2) # B x L x L * B x L x 1 -> B x L

        return z_new

class combination_L(nn.Module):
    def __init__(self,args):
        super(combination_L, self).__init__()
        self.args = args

    def forward(self, L, y):
        '''
        :param L: batch_size (B) x latent_size^2 * number_combination (L^2 * C)
        :param y: batch_size (B) x number_combination (C)
        :return: L_combination = y * L
        '''
        # calculate combination of Ls
        self.args.hidden_size = 300
        L_tensor = L.view( -1, self.args.hidden_size**2, self.args.number_combination ) # resize to get B x L^2 x C
        y = y.unsqueeze(1).expand(y.size(0), self.args.hidden_size**2, y.size(1)) # expand to get B x L^2 x C
        L_combination = torch.sum( L_tensor * y, 2 ).squeeze()
        return L_combination


#=======================================================================================================================
class VAE(Model):
    def __init__(self, args):
        super(VAE, self).__init__(args)

        # encoder: q(z | x)
        self.q_z_layers = [GatedDense(np.prod(self.args.input_size), 300)]
        for i in range(args.number_hidden):
            self.q_z_layers.append(NonGatedDense(300, 300))

        self.q_z_layers = nn.ModuleList(self.q_z_layers)

        self.q_z_mean = Linear(300, self.args.z1_size)
        self.q_z_logvar = NonLinear(300, self.args.z1_size, activation=nn.Hardtanh(min_val=-6.,max_val=2.))

        # decoder: p(x | z)
        self.p_x_layers = [GatedDense(self.args.z1_size, 300)]
        for i in range(args.number_hidden/2):
            self.p_x_layers.append(NonGatedDense(300, 300))
        self.p_x_layers = nn.ModuleList(self.p_x_layers)

        if self.args.input_type == 'binary':
            self.p_x_mean = NonLinear(300, np.prod(self.args.input_size), activation=nn.Sigmoid())
        elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            self.p_x_mean = NonLinear(300, np.prod(self.args.input_size), activation=nn.Sigmoid())
            self.p_x_logvar = NonLinear(300, np.prod(self.args.input_size), activation=nn.Hardtanh(min_val=-4.5,max_val=0))

        # convex combination linear Inverse Autoregressive flow
        self.encoder_y = [nn.Linear( 300, self.args.number_combination ) for j in range(len(self.q_z_layers))]
        self.encoder_L = [nn.Linear( 300, (300**2) * self.args.number_combination ) for j in range(len(self.q_z_layers))]

        self.encoder_y = nn.ModuleList(self.encoder_y)
        self.encoder_L = nn.ModuleList(self.encoder_L)

        self.decoder_y = [nn.Linear( 300, self.args.number_combination ) for j in range(len(self.p_x_layers))]
        self.decoder_L = [nn.Linear( 300, (300**2) * self.args.number_combination ) for j in range(len(self.p_x_layers))]

        self.decoder_y = nn.ModuleList(self.decoder_y)
        self.decoder_L = nn.ModuleList(self.decoder_L)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.linIAF = linIAF(self.args)
        self.combination_L = combination_L(self.args)

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)

        # add pseudo-inputs if VampPrior
        if self.args.prior == 'vampprior':
            self.add_pseudoinputs()


    # AUXILIARY METHODS
    def calculate_loss(self, x, beta=1., average=False):
        '''
        :param x: input image(s)
        :param beta: a hyperparam for warmup
        :param average: whether to average loss or not
        :return: value of a loss function
        '''
        # pass through VAE
        x_mean, x_logvar, z_q, z_q_mean, z_q_logvar = self.forward(x)

        # RE
        if self.args.input_type == 'binary':
            RE = log_Bernoulli(x, x_mean, dim=1)
        elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            RE = -log_Logistic_256(x, x_mean, x_logvar, dim=1)
        else:
            raise Exception('Wrong input type!')

        # KL
        log_p_z = self.log_p_z(z_q)
        log_q_z = log_Normal_diag(z_q, z_q_mean, z_q_logvar, dim=1)
        KL = -(log_p_z - log_q_z)

        loss = - RE + beta * KL

        if average:
            loss = torch.mean(loss)
            RE = torch.mean(RE)
            KL = torch.mean(KL)

        return loss, RE, KL

    def calculate_likelihood(self, X, dir, mode='test', S=5000, MB = 100):
        # set auxiliary variables for number of training and test sets
        N_test = X.size(0)

        # init list
        likelihood_test = []

        if S <= MB:
            R = 1
        else:
            R = S / MB
            S = MB

        for j in range(N_test):
            if j % 100 == 0:
                print('{:.2f}%'.format(j / (1. * N_test) * 100))
            # Take x*
            x_single = X[j].unsqueeze(0)

            a = []
            for r in range(0, R):
                # Repeat it for all training points
                x = x_single.expand(S, x_single.size(1))

                # pass through VAE
                x_mean, x_logvar, z_q, z_q_mean, z_q_logvar = self.forward(x)

                # RE
                RE = log_Bernoulli(x, x_mean, dim=1)

                # KL
                log_p_z = log_Normal_standard(z_q, dim=1)
                log_q_z = log_Normal_diag(z_q, z_q_mean, z_q_logvar, dim=1)
                KL = -(log_p_z - log_q_z)

                a_tmp = (RE - KL)

                a.append( a_tmp.cpu().data.numpy() )

            # calculate max
            a = np.asarray(a)
            a = np.reshape(a, (a.shape[0] * a.shape[1], 1))
            likelihood_x = logsumexp( a )
            likelihood_test.append(likelihood_x - np.log(len(a)))

        likelihood_test = np.array(likelihood_test)

        plot_histogram(-likelihood_test, dir, mode)

        return -np.mean(likelihood_test)

    def calculate_lower_bound(self, X_full, MB=100):
        # CALCULATE LOWER BOUND:
        lower_bound = 0.
        RE_all = 0.
        KL_all = 0.

        I = int(math.ceil(X_full.size(0) / MB))

        for i in range(I):
            x = X_full[i * MB: (i + 1) * MB].view(-1, np.prod(self.args.input_size))

            loss, RE, KL = self.calculate_loss(x,average=True)

            RE_all += RE.cpu().data[0]
            KL_all += KL.cpu().data[0]
            lower_bound += loss.cpu().data[0]

        lower_bound /= I

        return lower_bound

    def generate_x(self, N=25):
        if self.args.prior == 'standard':
            z_sample_rand = Variable( torch.FloatTensor(N, self.args.z1_size).normal_() )
            if self.args.cuda:
                z_sample_rand = z_sample_rand.cuda()

        elif self.args.prior == 'vampprior':
            means = self.means(self.idle_input)[0:N]
            z_sample_gen_mean, z_sample_gen_logvar = self.q_z(means)
            z_sample_rand = self.reparameterize(z_sample_gen_mean, z_sample_gen_logvar)

        samples_rand, _ = self.p_x(z_sample_rand)
        return samples_rand

    def reconstruct_x(self, x):
        x_mean, _, _, _, _ = self.forward(x)
        return x_mean


    # THE MODEL: VARIATIONAL POSTERIOR
    def q_z(self, x):
        
        #print(len(self.encoder_L))
        for i in range(len(self.q_z_layers)):  
            if i > 0:
                ori = x
            x = self.q_z_layers[i](x)
            if i > 0:
                x += ori#self.invert_Flow(ori,self.encoder_L[i-1],self.encoder_y[i-1])
                                      

        z_q_mean = self.q_z_mean(x)
        z_q_logvar = self.q_z_logvar(x)
        return z_q_mean, z_q_logvar, x


    # THE MODEL: HOUSEHOLDER FLOW
    def q_z_Flow(self, z, h_last):
        # ccLinIAF
        L = self.encoder_L(h_last)
        y = self.softmax(self.encoder_y(h_last))
        L_combination = self.combination_L(L, y)
        z['1'] = self.linIAF(L_combination, z['0'])
        return z

    # THE MODEL: HOUSEHOLDER FLOW
    def invert_Flow(self, h, generator_L, generator_y):
        # ccLinIAF
        L = generator_L(h)
        y = self.softmax(generator_y(h))
        L_combination = self.combination_L(L, y)
        h = self.linIAF(L_combination, h)
        return h

    # THE MODEL: GENERATIVE DISTRIBUTION
    def p_x(self, z):
        for i in range(len(self.p_x_layers)):
            if i > 0:
                ori = z
            z = self.p_x_layers[i](z)
            if i > 0:
                z += ori#self.invert_Flow(ori,self.decoder_L[i-1],self.decoder_y[i-1])                        

        x_mean = self.p_x_mean(z)
        if self.args.input_type == 'binary':
            x_logvar = 0.
        else:
            x_mean = torch.clamp(x_mean, min=0.+1./512., max=1.-1./512.)
            x_logvar = self.p_x_logvar(z)
        return x_mean, x_logvar

    # the prior
    def log_p_z(self, z):
        if self.args.prior == 'standard':
            log_prior = log_Normal_standard(z, dim=1)

        elif self.args.prior == 'vampprior':
            # z - MB x M
            C = self.args.number_components

            # calculate params
            X = self.means(self.idle_input)

            # calculate params for given data
            z_p_mean, z_p_logvar = self.q_z(X)  # C x M

            # expand z
            z_expand = z.unsqueeze(1)
            means = z_p_mean.unsqueeze(0)
            logvars = z_p_logvar.unsqueeze(0)

            a = log_Normal_diag(z_expand, means, logvars, dim=2) - math.log(C)  # MB x C
            a_max, _ = torch.max(a, 1)  # MB x 1

            # calculte log-sum-exp
            log_prior = a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1))  # MB x 1

        else:
            raise Exception('Wrong name of the prior!')

        return log_prior

    # THE MODEL: FORWARD PASS
    def forward(self, x):
        
        # z ~ q(z | x)
        z_q_mean, z_q_logvar, h_last = self.q_z(x)
        z= self.reparameterize(z_q_mean, z_q_logvar)
        # Householder Flow:
        #z = self.q_z_Flow(z, h_last)

        # x_mean = p(x|z)
        x_mean, x_logvar = self.p_x(z)#(z['1'] )

        return x_mean, x_logvar, z, z_q_mean, z_q_logvar
