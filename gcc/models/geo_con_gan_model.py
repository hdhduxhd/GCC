import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import itertools
import sys
from util.image_pool import ImagePool
from models.base_model import BaseModel
from models import networks
from loss.entropy import Entropy
from ddpm.new_diffusion import *


class GeoConGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True) 
        parser.add_argument('--max_timestep', type=int, default=500, help='output dim of linear')
        parser.add_argument('--shift', type=int, default=0, help='start index of timestep')
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_N', type=float, default=20.0, help='weight for diff loss (denoise)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A']
        visual_names_B = ['real_B', 'fake_A', 'real_B_noise']
        # if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
        #     visual_names_A.append('idt_B')
        #     visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_B', 'D_B', 'G_N']
        else:  # during test time, only load Gs
            self.model_names = ['G_B', 'G_N']

        # define networks (both Generators and discriminators)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_N = networks.define_N(opt.input_nc, opt.ngf, opt.n_layers_N, opt.max_timestep, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.diffusion = GaussianDiffusion()

        if self.isTrain:  # define discriminators
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionCycle_pixel = torch.nn.L1Loss(reduction='none')
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_N = torch.optim.Adam(self.netG_N.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_N)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
    
    def forward(self):
        batch_size = self.real_A.shape[0]
        self.t = torch.randint(1, 500, (batch_size,), device=self.device).long()
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.noise_real_A = torch.randn_like(self.real_A)
        self.real_A_noise = self.diffusion.q_sample(self.real_A, self.t, noise=self.noise_real_A)

        self.noise_real_B = torch.randn_like(self.real_B)
        self.real_B_noise = self.diffusion.q_sample(self.real_B, self.t, noise=self.noise_real_B)
        self.fake_A = self.netG_B(self.real_B_noise)  # G_B(B)

    def forward_with_N(self):
        logits1, logits2, logits3 = self.netG_N(self.real_B)
        # y = get_rep_outputs(logits, 0.5, True)
        y1 = get_rep_outputs(logits1, 0.5, True)
        y2 = get_rep_outputs(logits2, 0.5, True)
        y3 = get_rep_outputs(logits3, 0.5, True)
        # column_vector = torch.arange(self.opt.shift+1, self.opt.shift+self.opt.max_timestep+1).view(self.opt.max_timestep, 1).cuda()
        column_vector1 = torch.arange(0, self.opt.max_timestep//100).view(self.opt.max_timestep//100, 1).cuda()
        column_vector2 = torch.arange(0, 10).view(10, 1).cuda()
        column_vector3 = torch.arange(1, 10).view(9, 1).cuda()
        self.t = (y1 @ column_vector1.float()) * 100 + (y2 @ column_vector2.float()) * 10 + (y3 @ column_vector3.float())
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        self.noise_real_B = torch.randn_like(self.real_B)
        self.real_B_noise = self.diffusion.q_sample(self.real_B, self.t, noise=self.noise_real_B)
        self.fake_A = self.netG_B(self.real_B_noise)  # G_B(B)
        

    def get_output_B(self, input, t=None, get_loss=False):
        batch_size = input.shape[0]
        if t is None:
#             t = torch.randint(1, 500, (batch_size,), device=self.device).long()
#             y = get_rep_outputs(logits, 0.5, True)
#             column_vector = torch.arange(self.opt.shift+1, self.opt.shift+self.opt.max_timestep+1).view(self.opt.max_timestep, 1).cuda()
#             t = y @ column_vector.float()
            logits1, logits2, logits3 = self.netG_N(input)
            y1 = get_rep_outputs(logits1, 0.5, True)
            y2 = get_rep_outputs(logits2, 0.5, True)
            y3 = get_rep_outputs(logits3, 0.5, True)
            column_vector1 = torch.arange(0, self.opt.max_timestep//100).view(self.opt.max_timestep//100, 1).cuda()
            column_vector2 = torch.arange(0, 10).view(10, 1).cuda()
            column_vector3 = torch.arange(1, 10).view(9, 1).cuda()
            # t = (y1.permute(0,2,3,1) @ column_vector1.float()) * 100 + (y2.permute(0,2,3,1) @ column_vector2.float()) * 10 + (y3.permute(0,2,3,1) @ column_vector3.float())
            t = (y1 @ column_vector1.float()) * 100 + (y2 @ column_vector2.float()) * 10 + (y3 @ column_vector3.float())

        noise_input = torch.randn_like(input)
        input_noise = self.diffusion.q_sample(input, t, noise=noise_input)

        output1 = self.netG_B(input_noise)  # denoise one step

        return output1, input_noise, t

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_N = self.opt.lambda_N
        # Identity loss
        if lambda_idt > 0:
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A_noise)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_B = 0

        # GAN loss D_B(G_B(B)) L_str
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # L_sem
        weight_ent_pixel = torch.exp(-1 * (self.real_B * (self.real_B + 1e-16).log()))
        self.loss_cycle_B = self.criterionCycle_pixel(self.fake_A, self.real_B) * weight_ent_pixel
        self.loss_cycle_B = self.loss_cycle_B.mean() * lambda_B

        self.ent_loss = Entropy()
        #L_nem
        self.penalty = self.ent_loss(self.real_B).detach()*10*torch.exp((self.fake_A - self.real_B).mean())

        self.loss_G = self.loss_G_B + self.loss_cycle_B + self.loss_idt_B + self.ent_loss(self.fake_A) + self.penalty
        self.loss_G.backward()
        
    def backward_N(self):
        self.ent_loss = Entropy()
        self.penalty = self.ent_loss(self.real_B).detach()*10*torch.exp((self.fake_A - self.real_B).mean())
        self.loss_N = self.penalty
        self.loss_N.backward()
        
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_B
        self.set_requires_grad([self.netD_B, self.netG_N], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
        
        # G_N
        self.forward_with_N()
        self.set_requires_grad([self.netG_B], False)
        self.set_requires_grad([self.netG_N], True)
        self.optimizer_N.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_N()      # calculate gradients for D_A
        self.optimizer_N.step()  # update D_A and D_B's weights
        self.set_requires_grad([self.netG_B], True)
