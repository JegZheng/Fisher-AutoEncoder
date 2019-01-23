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
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

#=======================================================================================================================
class VAE(Model):
    def __init__(self, args):
        super(VAE, self).__init__(args)

        # encoder: q(z | x)
        self.q_z_layers = [NonGatedDense(np.prod(self.args.input_size), 300, activation=nn.ReLU())]
        for i in range(args.number_hidden):
            self.q_z_layers.append(NonGatedDense(300, 300, activation=nn.ReLU()))

        self.q_z_layers = nn.ModuleList(self.q_z_layers)

        self.q_z_mean = Linear(300, self.args.z1_size)
        self.q_z_logvar = Linear(300, self.args.z1_size)#NonLinear(300, self.args.z1_size, activation=nn.Hardtanh(min_val=-6.,max_val=2.))

        # decoder: p(x | z)
        self.p_x_layers = [NonGatedDense(self.args.z1_size, 300, activation=nn.ReLU())]
        for i in range(args.number_hidden):
            self.p_x_layers.append(NonGatedDense(300, 300, activation=nn.ReLU()))
        self.p_x_layers = nn.ModuleList(self.p_x_layers)

        if self.args.input_type == 'binary':
            self.p_x_mean = NonLinear(300, np.prod(self.args.input_size), activation=nn.Sigmoid())
        elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            self.p_x_mean = NonLinear(300, np.prod(self.args.input_size), activation=nn.Sigmoid())
            self.p_x_logvar = NonLinear(300, np.prod(self.args.input_size), activation=nn.Hardtanh(min_val=-4.5,max_val=0))

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
        
        #FI
        if self.args.FI is True:
            FI, gamma = self.FI(x)
        else:
            FI, gamma = self.FI(x)
            FI *= 0
        #FI = (torch.mean((torch.log(2*torch.pow(torch.exp( z_q_logvar ),2) + 1) - 2 * z_q_logvar)) - self.args.M ).abs()
        #FI = (torch.mean((1/torch.exp( z_q_logvar ) + 1/(2*torch.pow( torch.exp( z_q_logvar ), 2 )))) - self.args.M ).abs()
        
        # MI
        if self.args.MI is True:
            MI = self.MI(x)
        else:
            MI = self.MI(x) * 0

        loss = - RE + beta * KL #-  self.args.gamma * FI - self.args.ksi * MI #- self.args.gamma * torch.log(FI)

        if self.args.FI is True:
            loss -= torch.mean(FI * gamma, dim = 1)

        #print(FI)

        if average:
            loss = torch.mean(loss)
            RE = torch.mean(RE)
            KL = torch.mean(KL)
            FI = torch.mean(torch.exp(torch.mean(FI)))
            MI = torch.mean(MI)

        return loss, RE, KL, FI, MI

    def calculate_likelihood(self, X, dir, mode='test', S=5000, MB=100):
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

                a_tmp, _, _ = self.calculate_loss(x)

                a.append( -a_tmp.cpu().data.numpy() )

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
        FI_all = 0.
        MI_all = 0.

        I = int(math.ceil(X_full.size(0) / MB))

        for i in range(I):
            x = X_full[i * MB: (i + 1) * MB].view(-1, np.prod(self.args.input_size))

            loss, RE, KL, FI, MI = self.calculate_loss(x,average=True)

            loss += torch.log(FI)

            RE_all += RE.cpu().data[0]
            KL_all += KL.cpu().data[0]
            FI_all += FI.cpu().data[0]
            MI_all += MI.cpu().data[0]
            lower_bound += loss.cpu().data[0]

        lower_bound /= I

        return lower_bound

    def isnan(self,x):
        return x!=x

    # MUTUAL INFORMATION
    def MI(self, x):
        x_mean, x_logvar, z_q, z_q_mean, z_q_logvar = self.forward(x)
        z_q_mean, z_q_logvar, _ = self.q_z(x_mean)
        z_q = self.reparameterize(z_q_mean, z_q_logvar)
        log_q_z = log_Normal_diag(z_q, z_q_mean, z_q_logvar, average=True, dim=1)
        epsilon = 1e-10
        
        mi_loss = (torch.mean(torch.log(torch.add(torch.exp(log_q_z),epsilon))) - self.args.M).abs()
        
        if self.isnan(mi_loss.data[0]):
            print(log_q_z)

        return mi_loss

    # FISHER INFORMATION
    def FI(self, x):
        z_q_mean, z_q_logvar, _ = self.q_z(x)
        #FI = torch.log(torch.mean((1/torch.exp( z_q_logvar ) + 1/(2*torch.pow( torch.exp( z_q_logvar ), 2 ))))) - self.args.F 
        FI =  - 3 * z_q_logvar
        gamma = nn.functional.relu((1-torch.exp(z_q_logvar))/3.)

        #print("===========FI==============")
        #print(FI.abs())
        return FI, gamma

    # ADDITIONAL METHODS
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
        for i in range(len(self.q_z_layers)):  
            x = self.q_z_layers[i](x)

        z_q_mean = self.q_z_mean(x)
        z_q_logvar = self.q_z_logvar(x)
        return z_q_mean, z_q_logvar, x

    # THE MODEL: GENERATIVE DISTRIBUTION
    def p_x(self, z):
        for i in range(len(self.p_x_layers)):
            z = self.p_x_layers[i](z)

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
        z_q_mean, z_q_logvar, _ = self.q_z(x)
        z_q = self.reparameterize(z_q_mean, z_q_logvar)

        # x_mean = p(x|z)
        x_mean, x_logvar = self.p_x(z_q)

        return x_mean, x_logvar, z_q, z_q_mean, z_q_logvar
