from __future__ import print_function

import numpy as np

import math

from scipy.misc import logsumexp


import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable

from utils.distributions import log_Bernoulli, log_Normal_diag, log_Normal_standard, log_Logistic_256, diff_Normal_diag
from utils.visual_evaluation import plot_histogram, plot_latent_histogram
from utils.nn import he_init, GatedDense, NonLinear, \
    Conv2d, GatedConv2d, GatedResUnit, ResizeGatedConv2d, MaskedConv2d, ResUnitBN, ResizeConv2d, GatedResUnit, GatedConvTranspose2d

from Model import Model
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


#=======================================================================================================================
class VAE(Model):
    def __init__(self, args):
        super(VAE, self).__init__(args)

        h_size = 512
        nc = self.args.input_size[0] # number of channels
        nz = self.args.z1_size # size of latent vector
        ngf = 64 # decoder (generator) filter factor
        ndf = 64 # encoder filter factor


        self.q_z_layers = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 8, 4, 1, 0, bias=False)
        )


        # linear layers
        self.q_z_mean = NonLinear(h_size, self.args.z1_size, activation=None)
        self.q_z_logvar = NonLinear(h_size, self.args.z1_size, activation=nn.Hardtanh(min_val=-6., max_val=2.))

        
        # decoder: p(x | z)
        self.p_x_layers_z = nn.Sequential(
            GatedDense(self.args.z1_size, h_size)
        )
        

        # decoder: p(x | z)
        act = nn.ReLU(True)
        # joint
        self.p_x_layers_joint = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     h_size, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
            # state size. (nc) x 64 x 64
        )

        if self.args.input_type == 'binary':
            self.p_x_mean = Conv2d(64, 1, 1, 1, 0, activation=nn.Sigmoid())
        elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            self.p_x_mean = Conv2d(64, self.args.input_size[0], 1, 1, 0, activation=nn.Sigmoid())
            self.p_x_logvar = Conv2d(64, self.args.input_size[0], 1, 1, 0, activation=nn.Hardtanh(min_val=-4.5, max_val=0.))

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
        x = x.view(-1, np.prod(self.args.input_size))
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
        if self.isnan(z_q_mean.data[0][0]):
            print("mean:")
            print(z_q_mean)
        if self.isnan(z_q_logvar.data[0][0]):
            print("var:")
            print(z_q_logvar)
        
        loss = - RE + beta * KL

        #FI
        if self.args.FI is True:
            FI, gamma = self.FI(x)
            #loss -= torch.mean(FI * gamma, dim = 1)
        else:
            FI, gamma = self.FI(x)
            FI *= 0.
        #FI = (torch.mean((torch.log(2*torch.pow(torch.exp( z_q_logvar ),2) + 1) - 2 * z_q_logvar)) - self.args.M ).abs()
        #FI = (torch.mean((1/torch.exp( z_q_logvar ) + 1/(2*torch.pow( torch.exp( z_q_logvar ), 2 )))) - self.args.M ).abs()
        
        # MI
        if self.args.MI is True:
            MI = self.MI(x)
            #loss += self.args.ksi * (MI -self.args.M).abs()
        else:
            MI = self.MI(x) * 0.

        if self.args.adv is True:
            loss += self.args.ksi * (torch.exp(MI) - FI).abs()

        if average:
            loss = torch.mean(loss)
            RE = torch.mean(RE)
            KL = torch.mean(KL)
            FI = torch.mean(FI)
            MI = torch.mean(torch.exp(MI))

        return loss, RE, KL, FI, MI

    def isnan(self,x):
        return x!=x

    # MUTUAL INFORMATION
    def MI(self, x):
        #x = x.view(-1, self.args.input_size[0], self.args.input_size[1], self.args.input_size[2])
        x_mean, x_logvar, z_q, z_q_mean, z_q_logvar = self.forward(x)
        x_mean = x_mean.view(-1, self.args.input_size[0], self.args.input_size[1], self.args.input_size[2])
        z_q_mean, z_q_logvar, _ = self.q_z(x_mean)
        z_q = self.reparameterize(z_q_mean, z_q_logvar)
        #log_q_z = log_Normal_diag(z_q, z_q_mean, z_q_logvar, average=True, dim=1)
        log_q_z = torch.distributions.Normal(z_q_mean,torch.sqrt(torch.exp(z_q_logvar))).log_prob(z_q)
        epsilon = 1e-10

        q_z_pdf = torch.exp(torch.add(log_q_z,epsilon))

        
        mi_loss = torch.log(q_z_pdf/torch.mean(q_z_pdf))
        mi_loss = torch.clamp(mi_loss, max=88.)
        mi_loss = torch.exp(torch.mean(mi_loss))

        return mi_loss

    # FISHER INFORMATION
    def FI(self, x):
        #x = x.view(-1, self.args.input_size[0], self.args.input_size[1], self.args.input_size[2])
        x_mean, x_logvar, z_q, z_q_mean, z_q_logvar = self.forward(x)
        FI = torch.exp(-0.5 * (torch.pow( z_q - z_q_mean, 2 ) / torch.exp( z_q_logvar ) ))
        #FI =  torch.exp(- 1 * z_q_logvar)
        FI = torch.pow((diff_Normal_diag(z_q, z_q_mean, z_q_logvar)/FI),2)
        #FI = torch.clamp(FI, max=88.)
        FI = torch.mean(FI)
        gamma = nn.functional.relu((1-torch.exp(z_q_logvar))/3.)

        #print("===========FI==============")
        #print(FI.abs())
        return FI, gamma

    def calculate_dist(self, X, dir, mode='test'):
        # set auxiliary variables for number of training and test sets
        x_mean, x_logvar, z_q, z_q_mean, z_q_logvar = self.forward(x)
        dist = torch.norm(z_q_mean,p=2,dim=1)
        plot_latent_histogram(dist, dir, mode)

    def calculate_likelihood(self, X, dir, mode='test', S=5000, MB=100):
        # set auxiliary variables for number of training and test sets
        X = X.view(-1, np.prod(self.args.input_size))
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
                x = x_single.expand(S, x_single.size(1)).contiguous()

                a_tmp, _, _, _, _ = self.calculate_loss(x)

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

            if self.args.FI is True:
                FI_, gamma = self.FI(x)
                loss += torch.mean(FI_ * gamma)

            if self.args.MI is True:
                loss -= self.args.ksi * torch.mean((MI -self.args.M).abs())

            RE_all += RE.cpu().data[0]
            KL_all += KL.cpu().data[0]
            FI_all += FI.cpu().data[0]
            MI_all += MI.cpu().data[0]
            lower_bound += loss.cpu().data[0]

        lower_bound /= I

        return lower_bound


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

    def traversal(self, x):
        x_mean, x_logvar, z_q, z_q_mean, z_q_logvar = self.forward(x)
        indices = np.argsort(z_q_logvar)[:10]
        var = np.linspace(-20,20,10)
        z_sample_rand = z_q_mean
        for i in range(99):
            z_sample_rand = torch.cat((z_sample_rand, z_q_mean),0) 

        for i in range(z_sample_rand.size()[0]):
            idx = i / 10
            var_idx = i % 10
            z_sample_rand[i,idx] = z_sample_rand[i,idx] + var[var_idx]

        if self.args.cuda:
            z_sample_rand = z_sample_rand.cuda()

        samples_rand, _ = self.p_x(z_sample_rand)
        return samples_rand

    def reconstruct_x(self, x):
        x_mean, _, _, _, _ = self.forward(x)
        return x_mean

    # THE MODEL: VARIATIONAL POSTERIOR
    def q_z(self, x):
        # processing x
        #x = x.view(-1, self.args.input_size[0], self.args.input_size[1], self.args.input_size[2])
        h = self.q_z_layers(x)
        h = h.view(x.size(0),-1)

        # predict mean and variance
        z2_q_mean = self.q_z_mean(h)
        z2_q_logvar = self.q_z_logvar(h)
        return z2_q_mean, z2_q_logvar, x


    # THE MODEL: GENERATIVE DISTRIBUTION
    def p_x(self, z):
        # processing z
        h = self.p_x_layers_z(z)
        # processing z to deconv
        h = h.view(-1, h.size()[1], 1, 1)

        # joint decoder part of the decoder
        h_decoder = self.p_x_layers_joint(h)

        x_mean = self.p_x_mean(h_decoder).view(-1,np.prod(self.args.input_size))
        if self.args.input_type == 'binary':
            x_logvar = 0.
        else:
            x_mean = torch.clamp(x_mean, min=0.+1./512., max=1.-1./512.)
            x_logvar = self.p_x_logvar(h_decoder).view(-1,np.prod(self.args.input_size))

        return x_mean, x_logvar

    # the prior
    def log_p_z(self, z2):
        if self.args.prior == 'standard':
            log_prior = log_Normal_standard(z2, dim=1)

        elif self.args.prior == 'vampprior':
            # z - MB x M
            C = self.args.number_components

            # calculate params
            X = self.means(self.idle_input).view(-1, self.args.input_size[0], self.args.input_size[1], self.args.input_size[2])

            # calculate params for given data
            z2_p_mean, z2_p_logvar = self.q_z(X)  # C x M)

            # expand z
            z_expand = z2.unsqueeze(1)
            means = z2_p_mean.unsqueeze(0)
            logvars = z2_p_logvar.unsqueeze(0)

            a = log_Normal_diag(z_expand, means, logvars, dim=2) - math.log(C)  # MB x C
            a_max, _ = torch.max(a, 1)  # MB
            # calculte log-sum-exp
            log_prior = (a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1)))  # MB

        else:
            raise Exception('Wrong name of the prior!')

        return log_prior

    # THE MODEL: FORWARD PASS
    def forward(self, x):
        # z ~ q(z | x)
        x = x.view(-1, self.args.input_size[0], self.args.input_size[1], self.args.input_size[2])
        z_q_mean, z_q_logvar, _ = self.q_z(x)
        z_q = self.reparameterize(z_q_mean, z_q_logvar)

        # x_mean = p(x|z)
        x_mean, x_logvar = self.p_x(z_q)

        return x_mean, x_logvar, z_q, z_q_mean, z_q_logvar