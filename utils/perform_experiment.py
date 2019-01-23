from __future__ import print_function

import torch

import math

import time

from utils.nn import  MaskedConv2d

from utils.visual_evaluation import plot_info_evolution

from torch.optim.lr_scheduler import StepLR
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# ======================================================================================================================
def experiment_vae(args, train_loader, val_loader, test_loader, model, optimizer, dir, model_name='vae'):
    from utils.training import train_vae as train
    from utils.evaluation import evaluate_vae as evaluate

    # SAVING
    torch.save(args, dir + args.model_name + '.config')

    # best_model = model
    best_loss = 100000.
    e = 0
    train_loss_history = []
    train_re_history = []
    train_kl_history = []
    train_fi_history = []
    train_mi_history = []
    train_psnr_history = []

    val_loss_history = []
    val_re_history = []
    val_kl_history = []
    val_fi_history = []
    val_mi_history = []
    val_psnr_history = []

    time_history = []

    scheduler = StepLR(optimizer, step_size=10000, gamma=0.1)

    for epoch in range(1, args.epochs + 1):
        scheduler.step()
        time_start = time.time()
        model, train_loss_epoch, train_re_epoch, train_kl_epoch, train_fi_epoch, train_mi_epoch, train_psnr = train(epoch, args, train_loader, model,
                                                                             optimizer)

        val_loss_epoch, val_re_epoch, val_kl_epoch, val_fi_epoch, val_mi_epoch, val_psnr = evaluate(args, model, train_loader, val_loader, epoch, dir, mode='validation')
        time_end = time.time()

        time_elapsed = time_end - time_start

        # appending history
        train_loss_history.append(train_loss_epoch), train_re_history.append(train_re_epoch), train_kl_history.append(
            train_kl_epoch), train_fi_history.append(train_fi_epoch), train_mi_history.append(train_mi_epoch), train_psnr_history.append(train_psnr)
        val_loss_history.append(val_loss_epoch), val_re_history.append(val_re_epoch), val_kl_history.append(
            val_kl_epoch), val_fi_history.append(val_fi_epoch), val_mi_history.append(val_mi_epoch), val_psnr_history.append(val_psnr)
        time_history.append(time_elapsed)



        # printing results
        print('Epoch: {}/{}, Time elapsed: {:.2f}s\n'
              '* Train loss: {:.2f}   (RE: {:.2f}, KL: {:.2f}, FI: {:.2f}, MI: {:.2f}, PSNR: {:.2f})\n'
              'o Val.  loss: {:.2f}   (RE: {:.2f}, KL: {:.2f}, FI: {:.2f}, MI: {:.2f}, PSNR: {:.2f})\n'
              '--> Early stopping: {}/{} (BEST: {:.2f})\n'.format(
            epoch, args.epochs, time_elapsed,
            train_loss_epoch, train_re_epoch, train_kl_epoch, train_fi_epoch, train_mi_epoch, train_psnr,
            val_loss_epoch, val_re_epoch, val_kl_epoch, val_fi_epoch, val_mi_epoch, val_psnr,
            e, args.early_stopping_epochs, best_loss
        ))

        # early-stopping
        if val_loss_epoch < best_loss:
            e = 0
            best_loss = val_loss_epoch
            # best_model = model
            print('->model saved<-')
            torch.save(model, dir + args.model_name + '.model')
        else:
            e += 1
            if epoch < args.warmup:
                e = 0
            if e > args.early_stopping_epochs:
                break

        # NaN
        if math.isnan(val_loss_epoch):
            break

    # FINAL EVALUATION
    best_model = torch.load(dir + '/' + args.model_name + '.model')
    test_loss, test_re, test_kl, test_fi, test_mi, test_log_likelihood, train_log_likelihood, test_elbo, train_elbo, test_psnr = evaluate(args, best_model, train_loader, test_loader, 9999, dir, mode='test')

    if args.fisher is True:
        fishier = []
        hid = list(model.named_children())
        for i in hid:
            if i[0] == 'p_x_layers':
                #print(i)
                temp = 0
                cnt_number_layers = 0
                for j in range(1,len(i[1])):
                    #print("----------------------")
                    #print(i[1][j]) 
                    temp += torch.mm(torch.mm(i[1][j].h.weight.grad,torch.t(i[1][j].h.weight)),torch.t(torch.mm(i[1][j].h.weight.grad,torch.t(i[1][j].h.weight)))).abs().sum().data[0]
                    cnt_number_layers += 1
                fishier.append((i[0],temp/cnt_number_layers))

            if i[0] == 'pixelcnn':
                temp = 0
                cnt_number_layers = 0
                for j in range(len(i[1])):
                    if isinstance(i[1][j],MaskedConv2d):
                        g = i[1][j].weight.grad.view(i[1][j].weight.grad.size()[0],-1)
                        h = i[1][j].weight.view(i[1][j].weight.size()[0],-1)
                        temp += torch.mm(torch.mm(g,torch.t(h)),torch.t(torch.mm(g,torch.t(h)))).abs().sum().data[0]
                        cnt_number_layers += 1
                fishier.append((i[0],temp/cnt_number_layers))


            if i[0] == 'q_z_layers':
                #print(i)
                if args.model_name[:3] == 'rev':
                    cnt_number_layers = 0
                    temp = 0
                    for j in range(1,len(i[1].stack)):
                        #print("----------------------")
                        #print(i[1][j]) 
                        for l in i[1].stack[j].bottleneck_block.children():
                            if isinstance(l, torch.nn.modules.conv.Conv2d):
                                g = l.weight.grad.view(l.weight.grad.size()[0],-1)
                                h = l.weight.view(l.weight.size()[0],-1)
                                temp += torch.mm(torch.mm(g,torch.t(h)),torch.t(torch.mm(g,torch.t(h)))).abs().sum().data[0]
                                cnt_number_layers += 1
                    FI_linear = torch.mm(torch.mm(i[1].linear.weight.grad,torch.t(i[1].linear.weight)),torch.t(torch.mm(i[1].linear.weight.grad,torch.t(i[1].linear.weight)))).abs().sum().data[0]
                    temp += FI_linear
                    fishier.append((i[0],temp/(cnt_number_layers+1)))
                else:
                    temp = 0
                    cnt_number_layers = 0
                    for j in range(1,len(i[1])):
                        #print("----------------------")
                        #print(i[1][j]) 
                        temp += torch.mm(torch.mm(i[1][j].h.weight.grad,torch.t(i[1][j].h.weight)),torch.t(torch.mm(i[1][j].h.weight.grad,torch.t(i[1][j].h.weight)))).abs().sum().data[0]
                        cnt_number_layers += 1
                    fishier.append((i[0],temp/cnt_number_layers))


        with open(dir + args.model_name + '_' + args.model_signature + '_fishier.txt','a') as o :
            o.write("Fishier Information Ratio in Hidden Layers\n")
            o.write(str(args)+'\n')
            for i in fishier:
                o.write(i[0] + ": " + str(i[1]) + "\n")

    print('FINAL EVALUATION ON TEST SET\n'
          'LogL (TEST): {:.2f}\n'
          'LogL (TRAIN): {:.2f}\n'
          'ELBO (TEST): {:.2f}\n'
          'ELBO (TRAIN): {:.2f}\n'
          'Loss: {:.2f}\n'
          'RE: {:.2f}\n'
          'KL: {:.2f}\n'
          'FI: {:.2f}\n'
          'MI: {:.2f}\n'
          'PSNR: {:.2f}'.format(
        test_log_likelihood,
        train_log_likelihood,
        test_elbo,
        train_elbo,
        test_loss,
        test_re,
        test_kl,
        test_fi,
        test_mi,
        test_psnr
    ))

    with open(dir + 'vae_experiment_log.txt', 'a') as f:
        print('FINAL EVALUATION ON TEST SET\n'
          'LogL (TEST): {:.2f}\n'
          'LogL (TRAIN): {:.2f}\n'
          'ELBO (TEST): {:.2f}\n'
          'ELBO (TRAIN): {:.2f}\n'
          'Loss: {:.2f}\n'
          'RE: {:.2f}\n'
          'KL: {:.2f}\n'
          'FI: {:.2f}\n'
          'MI: {:.2f}\n'
          'PSNR: {:.2f}'.format(
        test_log_likelihood,
        train_log_likelihood,
        test_elbo,
        train_elbo,
        test_loss,
        test_re,
        test_kl,
        test_fi,
        test_mi,
        test_psnr
        ), file=f)

    # Plot MI and FI
    #if args.MI is True:
    plot_info_evolution(val_mi_history, dir, 'MI')

    #if args.FI is True:
    plot_info_evolution(val_fi_history, dir, 'FI')

    # SAVING
    torch.save(train_loss_history, dir + args.model_name + '.train_loss')
    torch.save(train_re_history, dir + args.model_name + '.train_re')
    torch.save(train_kl_history, dir + args.model_name + '.train_kl')
    torch.save(train_mi_history, dir + args.model_name + '.train_mi')
    torch.save(train_fi_history, dir + args.model_name + '.train_fi')
    torch.save(train_psnr_history, dir + args.model_name + '.train_psnr')
    torch.save(val_loss_history, dir + args.model_name + '.val_loss')
    torch.save(val_re_history, dir + args.model_name + '.val_re')
    torch.save(val_kl_history, dir + args.model_name + '.val_kl')
    torch.save(val_mi_history, dir + args.model_name + '.val_mi')
    torch.save(val_fi_history, dir + args.model_name + '.val_fi')
    torch.save(val_psnr_history, dir + args.model_name + '.val_psnr')
    torch.save(test_log_likelihood, dir + args.model_name + '.test_log_likelihood')
    torch.save(test_loss, dir + args.model_name + '.test_loss')
    torch.save(test_re, dir + args.model_name + '.test_re')
    torch.save(test_kl, dir + args.model_name + '.test_kl')
    torch.save(test_fi, dir + args.model_name + '.test_fi')
    torch.save(test_mi, dir + args.model_name + '.test_mi')
    torch.save(test_psnr, dir + args.model_name + '.test_psnr')
