from __future__ import print_function

import torch
from torch.autograd import Variable

from utils.visual_evaluation import plot_images, visualize_latent, plot_manifold, plot_scatter

import numpy as np

import time

import os
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# ======================================================================================================================
def evaluate_vae(args, model, train_loader, data_loader, epoch, dir, mode):
    # set loss to 0
    evaluate_loss = 0
    evaluate_re = 0
    evaluate_kl = 0
    evaluate_fi = 0
    evaluate_mi = 0
    evaluate_psnr = 0
    # set model to evaluation mode
    model.eval()

    # evaluate
    for batch_idx, (data, target) in enumerate(data_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        x = data

        # calculate loss function
        loss, RE, KL, FI, MI = model.calculate_loss(x, average=True)

        if args.FI is True:
            FI_, gamma = model.FI(x)
            loss += torch.mean(FI_ * gamma)

        if args.MI is True:
            loss -= args.ksi * torch.mean((MI - args.M).abs())


        if not args.MI is True:
            MI = model.MI(x)


        if not args.FI is True:
            FI, gamma = model.FI(x)
            FI = torch.mean(torch.exp(torch.mean(FI)))

        psnr = model.psnr(x)

        evaluate_loss += loss.data[0]
        evaluate_re += -RE.data[0]
        evaluate_kl += KL.data[0]
        evaluate_fi += FI.data[0]
        evaluate_mi += MI.abs().data[0]
        evaluate_psnr += psnr

        # print N digits
        if batch_idx == 1 and mode == 'validation':
            if epoch == 1:
                if not os.path.exists(dir + 'reconstruction/'):
                    os.makedirs(dir + 'reconstruction/')
                # VISUALIZATION: plot real images
                plot_images(args, data.data.cpu().numpy()[0:9], dir + 'reconstruction/', 'real', size_x=3, size_y=3)
            x_mean = model.reconstruct_x(x)
            plot_images(args, x_mean.data.cpu().numpy()[0:9], dir + 'reconstruction/', str(epoch), size_x=3, size_y=3)

    if mode == 'test':
        # load all data
        if args.dataset_name == 'celeba':
            test_data = []
            test_target = []
            full_data = []
            for d, l in data_loader.dataset:
                test_data.append(d)
                test_target.append(l)

            for d, l in train_loader.dataset:
                full_data.append(d)

            test_data = Variable(torch.stack(test_data),volatile=True)
            test_target = Variable(torch.from_numpy(np.array(test_target)),volatile=True)
            full_data = Variable(torch.stack(full_data[60000]),volatile=True)


        else:
            test_data = Variable(data_loader.dataset.data_tensor,volatile=True)
            test_target = Variable(data_loader.dataset.target_tensor,volatile=True)
            full_data = Variable(train_loader.dataset.data_tensor,volatile=True)

        if args.cuda:
            test_data, test_target, full_data = test_data.cuda(), test_target.cuda(), full_data.cuda()

        if args.dynamic_binarization:
            full_data = torch.bernoulli(full_data)

        # print(model.means(model.idle_input))

        # VISUALIZATION: plot real images
        #for k in range(200):
        #    plot_images(args, test_data.data.cpu().numpy()[k*25:(k+1)*25], dir, 'real'+str(k), size_x=5, size_y=5)

        # VISUALIZATION: plot reconstructions

        
        if not os.path.exists(dir + 'test_reconstruction/'):
            os.makedirs(dir + 'test_reconstruction/')
        for k in range(200):
            samples = model.reconstruct_x(test_data[k*25:(k+1)*25])
            plot_images(args, samples.data.cpu().numpy(), dir, 'test_reconstruction/'+str(k), size_x=5, size_y=5)
        
        '''    
        # VISUALIZATION: plot real images
        plot_images(args, test_data.data.cpu().numpy()[0:25], dir, 'real', size_x=5, size_y=5)

        # VISUALIZATION: plot reconstructions
        samples = model.reconstruct_x(test_data[0:25])

        plot_images(args, samples.data.cpu().numpy(), dir, 'reconstructions', size_x=5, size_y=5)
        '''
        # VISUALIZATION: plot generations
        samples_rand = model.generate_x(25)

        plot_images(args, samples_rand.data.cpu().numpy(), dir, 'generations', size_x=5, size_y=5)

        # VISUALIZATION: plot traversal
        if args.z1_size > 10: 
            if args.dataset_name == 'celeba':
                cnt = 0
                for j in range(len(test_target)):
                    if cnt == 10:
                        break
                    samples_rand = model.traversal(test_data[j])
                    plot_images(args, samples_rand.data.cpu().numpy(), dir, 'attributes'+str(cnt), size_x=10, size_y=10)
                    cnt += 1

            else:
                for i in range(10):
                    for j in range(len(test_target)):
                        if test_target[j].data.cpu().numpy()[0]%10 == i:
                            samples_rand = model.traversal(test_data[j])
                            plot_images(args, samples_rand.data.cpu().numpy(), dir, 'attributes'+str(i), size_x=10, size_y=10)
                            break
        else:
            pass


        # VISUALIZATION: latent space
        if args.latent is True and args.dataset_name != 'celeba':
            z_mean_recon, z_logvar_recon, _ = model.q_z(test_data.view(-1, args.input_size[0], args.input_size[1], args.input_size[2]))
            print("latent visualization")
            #plot_scatter(model, test_data.view(-1, args.input_size[0], args.input_size[1], args.input_size[2]), test_target, dir)
            visualize_latent(z_mean_recon,test_target, dir + '/latent_'+ args.model_name + '_' + args.model_signature)

        if args.z1_size == 2:
            # VISUALIZATION: plot low-dimensional manifold
            plot_manifold(model, args, dir)

            # VISUALIZATION: plot scatter-plot
            #plot_scatter(model, test_data.view(-1, args.input_size[0], args.input_size[1], args.input_size[2]), test_target, dir)

        if args.prior == 'vampprior':
            # VISUALIZE pseudoinputs
            pseudoinputs = model.means(model.idle_input).cpu().data.numpy()

            plot_images(args, pseudoinputs[0:25], dir, 'pseudoinputs', size_x=5, size_y=5)

        # CALCULATE lower-bound
        t_ll_s = time.time()
        elbo_test = model.calculate_lower_bound(test_data, MB=args.MB)
        t_ll_e = time.time()
        print('Test lower-bound value {:.2f} in time: {:.2f}s'.format(elbo_test, t_ll_e - t_ll_s))

        # CALCULATE log-likelihood
        t_ll_s = time.time()
        try:
            elbo_train = 0. #model.calculate_lower_bound(full_data, MB=args.MB)
        except:
            elbo_train = 0.
        t_ll_e = time.time()
        print('Train lower-bound value {:.2f} in time: {:.2f}s'.format(elbo_train, t_ll_e - t_ll_s))

        # CALCULATE log-likelihood
        t_ll_s = time.time()
        log_likelihood_test = 0.#model.calculate_likelihood(test_data, dir, mode='test', S=args.S, MB=args.MB)
        t_ll_e = time.time()
        print('Test log_likelihood value {:.2f} in time: {:.2f}s'.format(log_likelihood_test, t_ll_e - t_ll_s))

        # CALCULATE log-likelihood
        t_ll_s = time.time()
        log_likelihood_train = 0. #model.calculate_likelihood(full_data, dir, mode='train', S=args.S, MB=args.MB)) #commented because it takes too much time
        t_ll_e = time.time()
        print('Train log_likelihood value {:.2f} in time: {:.2f}s'.format(log_likelihood_train, t_ll_e - t_ll_s))


        t_ll_s = time.time()
        model.calculate_dist(test_data[:int(args.S)],dir, mode='test')
        t_ll_e = time.time()
        print('Test latent distribution in time: {:.2f}s'.format(t_ll_e - t_ll_s))

    # calculate final loss
    evaluate_loss /= len(data_loader)  # loss function already averages over batch size
    evaluate_re /= len(data_loader)  # re already averages over batch size
    evaluate_kl /= len(data_loader)  # kl already averages over batch size
    evaluate_fi /= len(data_loader)
    evaluate_mi /= len(data_loader)
    evaluate_psnr /= len(data_loader)
    if mode == 'test':
        #print(model.q_z_layers)
        return evaluate_loss, evaluate_re, evaluate_kl, evaluate_fi, evaluate_mi, log_likelihood_test, log_likelihood_train, elbo_test, elbo_train, evaluate_psnr
    else:
        return evaluate_loss, evaluate_re, evaluate_kl, evaluate_fi, evaluate_mi, evaluate_psnr