import time
import numpy as np
import math
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import sys
from options.train_options import TrainOptions
# from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.metrics import *
import random
from loss.entropy import Entropy
ent = Entropy()

from dataloaders import fundus_dataloader as DL
from dataloaders import custom_transforms as tr


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
    # 1. dataset
    composed_transforms_tr = transforms.Compose([
        #tr.Resize(512),###
        tr.RandomScaleCrop(512),
        tr.RandomRotate(),
        tr.RandomFlip(),
        tr.elastic_transform(),
        tr.add_salt_pepper_noise(),
        tr.adjust_light(),
        tr.eraser(),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    composed_transforms_ts = transforms.Compose([
        # tr.RandomCrop(512),
        tr.Resize(512),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    domain = DL.FundusSegmentation(base_dir=opt.data_dir, dataset=opt.datasetS, split='train/ROIs', transform=composed_transforms_ts)
    domain_loaderS = DataLoader(domain, batch_size=opt.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    domain_iterS = iter(domain_loaderS)

    domain_T = DL.FundusSegmentation(base_dir=opt.data_dir, dataset=opt.datasetT, split='train/ROIs', transform=composed_transforms_ts)
    domain_loaderT = DataLoader(domain_T, batch_size=opt.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    domain_iterT = iter(domain_loaderT)

    domain_val = DL.FundusSegmentation(base_dir=opt.data_dir, dataset=opt.datasetT, split='test/ROIs', transform=composed_transforms_ts)
    domain_loader_val = DataLoader(domain_val, batch_size=opt.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    refine_npdata = np.load(opt.npfilename_new, allow_pickle=True)
    refine_pseudo_label_dic = refine_npdata['arr_0'].item()
    refine_prob_dic = refine_npdata['arr_1'].item()
    refine_prob_low_ent = []
    refine_prob_high_ent = []
    entropy_sorted_imgs = []
    for i in refine_prob_dic:
        img = torch.from_numpy(refine_prob_dic[i])
        img_ent = ent(img.unsqueeze(0)).item()
        entropy_sorted_imgs.append((img, img_ent))
        
    # After collecting all images, sort them by entropy
    entropy_sorted_imgs.sort(key=lambda x: x[1])  # Sort by entropy
    
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    max_dice = 0

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.

        
        # Calculate threshold index based on current epoch
        # Start with 20% in low entropy, gradually increase to 100%
        low_ent_ratio = 0.2 + (0.8 * (epoch - opt.epoch_count) / (opt.n_epochs + opt.n_epochs_decay))
        low_ent_ratio = min(1, max(0.2, low_ent_ratio))  # Keep between 20% and 100%
        threshold_idx = int(len(entropy_sorted_imgs) * low_ent_ratio)
        
        # Split images into low and high entropy groups
        refine_prob_low_ent = [img.unsqueeze(0) for img, _ in entropy_sorted_imgs[:threshold_idx]]
        refine_prob_high_ent = [img.unsqueeze(0) for img, _ in entropy_sorted_imgs[threshold_idx:]]
        
        random.shuffle(refine_prob_low_ent)
        random.shuffle(refine_prob_high_ent)
        prob_iterl = iter(refine_prob_low_ent)
        prob_iterh = iter(refine_prob_high_ent)

        for i in range(max(len(refine_prob_low_ent), len(refine_prob_high_ent))):
            try:
                A = next(prob_iterl)
            except Exception as err:
                prob_iterl = iter(refine_prob_low_ent)
                A = next(prob_iterl)
            try:
                B = next(prob_iterh)
            except Exception as err:
                prob_iterh = iter(refine_prob_high_ent)
                B = next(prob_iterh)

            # Resize target_pl to 1x2x256x256
            # target_pl = F.interpolate(target_pl, size=(256, 256), mode='bilinear', align_corners=False)
            data = {"A": A.float(), "B": B}
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals() , epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                # if opt.display_id > 0:
                visualizer.plot_current_losses(epoch, epoch_iter, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
#                 model.save_networks(save_suffix)
                model.save_networks(epoch)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            refine_dice_cup, refine_dice_disc = 0, 0
            for sample in domain_loaderT:
                target_image, target_label, target_img_name = sample['image'], sample['map'], sample['img_name']
                target_prob_pl = torch.stack([torch.from_numpy(refine_prob_dic.get(i)) for i in target_img_name])
                # target_prob_pl = F.interpolate(target_prob_pl, size=(256, 256), mode='bilinear', align_corners=False)
                target_label = target_label.to(device)
                target_prob_pl = target_prob_pl.to(device)
                refine_target_new_pl, _, _ = model.get_output_B(target_prob_pl)
                refine_target_new_pl[refine_target_new_pl > 0.75] = 1
                refine_target_new_pl[refine_target_new_pl <= 0.75] = 0
                refine_dice_prob_cup, refine_dice_prob_disc = dice_coeff_2label(refine_target_new_pl, target_label)
                refine_dice_cup += refine_dice_prob_cup
                refine_dice_disc += refine_dice_prob_disc
            refine_dice_cup /= len(domain_loaderT)
            refine_dice_disc /= len(domain_loaderT)
            visualizer.plot_current_metrics({"refine_dice_cup":refine_dice_cup,"refine_dice_disc":refine_dice_disc})
            print('refine_dice_cup: %.4f, refine_dice_disc: %.4f' % (refine_dice_cup, refine_dice_disc))
            model.save_networks('latest')
            if (refine_dice_cup + refine_dice_disc) / 2 > max_dice:
                max_dice = (refine_dice_cup + refine_dice_disc) / 2
                print("______________________________best____________________________")
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
