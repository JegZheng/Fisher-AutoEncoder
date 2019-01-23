from __future__ import print_function

import numpy as np

import math
from math import log10

from scipy.misc import logsumexp


import torch
import torch.utils.data
import torch.nn as nn
from torch.nn import Linear
from torch.autograd import Variable

from utils.distributions import log_Bernoulli, log_Normal_diag, log_Normal_standard, log_Logistic_256
from utils.visual_evaluation import plot_histogram, plot_latent_histogram
from utils.nn import he_init, GatedDense, NonLinear, NonGatedDense, \
    Conv2d, GatedConv2d, MaskedConv2d, ResUnitBN, MaskedGatedConv2d

from Model import Model
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

#=======================================================================================================================
class VAE(Model):
    def __init__(self, args):
        super(VAE, self).__init__(args)

        # encoder: q(z | x)
        if self.args.dataset_name == 'freyfaces':
            h_size = 210
        elif self.args.dataset_name == 'cifar10' :
            h_size = 384
        elif self.args.dataset_name ==  'svhn':
            h_size = 384
        else:
            h_size = 294

        # encoder: q(z2 | x)
        self.q_z_layers = nn.Sequential(
            Conv2d(self.args.input_size[0], 32, 7, 1, 3,activation=nn.ReLU()),
            nn.BatchNorm2d(32),
            Conv2d(32, 32, 3, 2, 1,activation=nn.ReLU()),
            nn.BatchNorm2d(32),
            Conv2d(32, 64, 5, 1, 2,activation=nn.ReLU()),
            nn.BatchNorm2d(64),
            Conv2d(64, 64, 3, 2, 1,activation=nn.ReLU()),
            nn.BatchNorm2d(64),
            Conv2d(64, 6, 3, 1, 1,activation=nn.ReLU())

        )
        '''
        self.q_z_layers = [NonGatedDense(np.prod(self.args.input_size), 300, activation=nn.ReLU())]
        for i in range(args.number_hidden):
            self.q_z_layers.append(NonGatedDense(300, 300, activation=nn.ReLU()))

        self.q_z_layers = nn.ModuleList(self.q_z_layers)
        '''
        self.q_z_mean = Linear(h_size, self.args.z1_size)
        self.q_z_logvar = Linear(h_size, self.args.z1_size)#NonLinear(300, self.args.z1_size, activation=nn.Hardtanh(min_val=-6.,max_val=2.))

        # decoder: p(x | z)
        self.p_x_layers = [NonGatedDense(self.args.z1_size, 300, activation=nn.ReLU())]
        for i in range(args.number_hidden):
            self.p_x_layers.append(NonGatedDense(300, 300, activation=nn.ReLU()))
        self.p_x_layers.append(NonGatedDense(300, np.prod(self.args.input_size), activation=nn.ReLU()))
        self.p_x_layers = nn.ModuleList(self.p_x_layers)

        # PixelCNN
        act = nn.ReLU(True)
        self.pixelcnn = nn.Sequential(
            MaskedConv2d('A', self.args.input_size[0] + self.args.input_size[0], 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
            MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
            MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
            MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
            MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
            MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
            MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act,
            MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), act
        )

        if self.args.input_type == 'binary':
            self.p_x_mean = Conv2d(64, 1, 1, 1, 0, activation=nn.Sigmoid())
        elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            self.p_x_mean = Conv2d(64, self.args.input_size[0], 1, 1, 0, activation=nn.Sigmoid(), bias=False )
            self.p_x_logvar = Conv2d(64, self.args.input_size[0], 1, 1, 0, activation=nn.Hardtanh(min_val=-4.5, max_val=0.), bias=False)
        
        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)
            else:
                if torch.is_tensor(m): 
                    torch.nn.init.kaiming_normal(m)

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
        if self.isnan(z_q_mean.data[0][0]):
            print("mean:")
            print(z_q_mean)
        if self.isnan(z_q_logvar.data[0][0]):
            print("var:")
            print(z_q_logvar)
        
        #print(z_q_logvar)

        #FI
        if self.args.FI is True:
            FI, gamma = self.FI(x)
        else:
            FI = Variable(torch.zeros(1),requires_grad=False)
            if self.args.cuda:
                FI = FI.cuda()
        #FI = (torch.mean((torch.log(2*torch.pow(torch.exp( z_q_logvar ),2) + 1) - 2 * z_q_logvar)) - self.args.M ).abs()
        #FI = (torch.mean((1/torch.exp( z_q_logvar ) + 1/(2*torch.pow( torch.exp( z_q_logvar ), 2 )))) - self.args.M ).abs()
        
        # MI
        if self.args.MI is True:
            MI = self.MI(x)
        else:
            MI = Variable(torch.zeros(1),requires_grad=False)
            if self.args.cuda:
                MI = MI.cuda()

        loss = - RE + beta * KL #+  self.args.gamma * FI + self.args.ksi * MI #- self.args.gamma * torch.log(FI)

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


    def psnr(self,x):
        criterionMSE = nn.MSELoss()
        if self.args.cuda:
            criterionMSE = criterionMSE.cuda()
        x_mean, x_logvar, z_q, z_q_mean, z_q_logvar = self.forward(x)
        mse = criterionMSE(x, x_mean)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr = torch.mean(psnr)
        return avg_psnr


    def isnan(self,x):
        return x!=x

    def calculate_dist(self, X, dir, mode='test'):
        # set auxiliary variables for number of training and test sets
        x_mean, x_logvar, z_q, z_q_mean, z_q_logvar = self.forward(x)
        dist = torch.norm(z_q_mean,p=2,dim=1)
        plot_latent_histogram(dist, dir, mode)    

    # MUTUAL INFORMATION
    def MI(self, x):
        x = x.view(-1, self.args.input_size[0], self.args.input_size[1], self.args.input_size[2])
        x_mean, x_logvar, z_q, z_q_mean, z_q_logvar = self.forward(x)
        x_mean = x_mean.view(-1, self.args.input_size[0], self.args.input_size[1], self.args.input_size[2])
        z_q_mean, z_q_logvar, _ = self.q_z(x_mean)
        z_q = self.reparameterize(z_q_mean, z_q_logvar)
        #log_q_z = log_Normal_diag(z_q, z_q_mean, z_q_logvar, average=True, dim=1)
        log_q_z = torch.distributions.Normal(z_q_mean,torch.sqrt(torch.exp(z_q_logvar))).log_prob(z_q)
        epsilon = 1e-10
        
        mi_loss = (torch.mean(torch.log(torch.add(torch.exp(log_q_z),epsilon))) - self.args.M).abs()
        
        #if self.isnan(mi_loss.data[0]):
            #print(log_q_z)


        return mi_loss

    # FISHER INFORMATION
    def FI(self, x):
        x = x.view(-1, self.args.input_size[0], self.args.input_size[1], self.args.input_size[2])
        z_q_mean, z_q_logvar, _ = self.q_z(x)
        #FI = torch.log(torch.mean((1/torch.exp( z_q_logvar ) + 1/(2*torch.pow( torch.exp( z_q_logvar ), 2 ))))) - self.args.F 
        FI =  - 3 * z_q_logvar
        FI = torch.clamp(FI, max=88.)
        gamma = nn.functional.relu((1-torch.exp(z_q_logvar))/3.)

        #print("===========FI==============")
        #print(FI.abs())
        return FI, gamma

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

    # ADDITIONAL METHODS
    def pixelcnn_generate(self, z):
        # Sampling from PixelCNN
        x_zeros = torch.zeros(
            (z.size(0), self.args.input_size[0], self.args.input_size[1], self.args.input_size[2]))
        if self.args.cuda:
            x_zeros = x_zeros.cuda()

        for i in range(self.args.input_size[1]):
            for j in range(self.args.input_size[2]):
                samples_mean, samples_logvar = self.p_x(Variable(x_zeros, volatile=True), z)
                samples_mean = samples_mean.view(samples_mean.size(0), self.args.input_size[0], self.args.input_size[1],
                                                 self.args.input_size[2])

                if self.args.input_type == 'binary':
                    probs = samples_mean[:, :, i, j].data
                    x_zeros[:, :, i, j] = torch.bernoulli(probs).float()
                    samples_gen = samples_mean

                elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
                    binsize = 1. / 256.
                    samples_logvar = samples_logvar.view(samples_mean.size(0), self.args.input_size[0],
                                                         self.args.input_size[1], self.args.input_size[2])
                    means = samples_mean[:, :, i, j].data
                    logvar = samples_logvar[:, :, i, j].data
                    # sample from logistic distribution
                    u = torch.rand(means.size()).cuda()
                    y = torch.log(u) - torch.log(1. - u)
                    sample = means + torch.exp(logvar) * y
                    x_zeros[:, :, i, j] = torch.floor(sample / binsize) * binsize
                    samples_gen = samples_mean

        return samples_gen

    def generate_x(self, N=25):
        if self.args.prior == 'standard':
            z_sample_rand = Variable( torch.FloatTensor(N, self.args.z1_size).normal_() )
            if self.args.cuda:
                z_sample_rand = z_sample_rand.cuda()

        elif self.args.prior == 'vampprior':
            means = self.means(self.idle_input)[0:N]
            z_sample_gen_mean, z_sample_gen_logvar = self.q_z(means)
            z_sample_rand = self.reparameterize(z_sample_gen_mean, z_sample_gen_logvar)

        # Sampling from PixelCNN
        samples_gen = self.pixelcnn_generate(z_sample_rand)
        return samples_gen

    def traversal(self, x):
        x_mean, x_logvar, z_q, z_q_mean, z_q_logvar = self.forward(x)
        indices = np.argsort(z_q_logvar.data[0])[:10]
        var = np.linspace(-20,20,10)
        z_sample_rand = z_q_mean
        for i in range(99):
            z_sample_rand = torch.cat((z_sample_rand, z_q_mean),0) 

        for i in range(z_sample_rand.size()[0]):
            idx = i / 10
            var_idx = i % 10
            z_sample_rand[i,indices[idx]] = z_sample_rand[i,indices[idx]] + var[var_idx]

        if self.args.cuda:
            z_sample_rand = z_sample_rand.cuda()

        samples_rand, _ = self.p_x(z_sample_rand)
        return samples_rand

    def reconstruct_x(self, x):
        _, _, z, _, _ = self.forward(x)
        x_reconstructed = self.pixelcnn_generate(z)
        return x_reconstructed

    # THE MODEL: VARIATIONAL POSTERIOR
    def q_z(self, x):
        #for i in range(len(self.q_z_layers)):  
        #    x = self.q_z_layers[i](x)
        x = x#.view(-1, self.args.input_size[0], self.args.input_size[1], self.args.input_size[2])
        h = self.q_z_layers(x)
        x = h.view(x.size(0),-1)
        z_q_mean = self.q_z_mean(x)
        z_q_logvar = self.q_z_logvar(x)
        return z_q_mean, z_q_logvar, x

    # THE MODEL: GENERATIVE DISTRIBUTION    
    def p_x(self, x, z):
        for i in range(len(self.p_x_layers)):
            z = self.p_x_layers[i](z)
        
        z = z.view(-1, self.args.input_size[0], self.args.input_size[1], self.args.input_size[2])

        # concatenate x and z1 and z2
        h = torch.cat((x,z), 1)

        # pixelcnn part of the decoder
        h_pixelcnn = self.pixelcnn(h)

        x_mean = self.p_x_mean(h_pixelcnn).view(-1,np.prod(self.args.input_size))
        if self.args.input_type == 'binary':
            x_logvar = 0.
        else:
            x_mean = torch.clamp(x_mean, min=0.+1./512., max=1.-1./512.)
            x_logvar = self.p_x_logvar(h_pixelcnn).view(-1,np.prod(self.args.input_size))

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
        x = x.view(-1, self.args.input_size[0], self.args.input_size[1], self.args.input_size[2])
        z_q_mean, z_q_logvar, _ = self.q_z(x)
        z_q = self.reparameterize(z_q_mean, z_q_logvar)

        # x_mean = p(x|z)
        x = x.view(-1, self.args.input_size[0], self.args.input_size[1], self.args.input_size[2])
        x_mean, x_logvar = self.p_x(x, z_q)

        return x_mean, x_logvar, z_q, z_q_mean, z_q_logvar
