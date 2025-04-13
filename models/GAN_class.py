from . import networks
from util.util import *
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from perceive_loss.pytorch_loss_function import ResNet18_3D_Loss, VGGLoss_3D, L1_Charbonnier_loss

class GANclass(nn.Module):
    def __init__(self, opt):
        super(GANclass, self).__init__()
        self.isTrain = opt.isTrain
        self.max_epoch = opt.max_epochs
        self.resolution = [opt.depthSize, opt.ImageSize, opt.ImageSize]
        self.VGG_loss = opt.VGG_loss
        self.gpu_ids = opt.gpu_ids
        self.device = opt.device
        self.lambda_L1 = opt.lambda_L1
        self.G_model = opt.G_model
        if self.VGG_loss:
            self.loss_names = ['G_GAN', 'G_L1', 'G_loss', 'D_real', 'D_fake', 'D_loss', 'G_perceive','classification']
        else:
            self.loss_names = ['G_GAN', 'G_L1', 'G_loss', 'D_real', 'D_fake', 'D_loss', 'classification']
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, self.resolution, opt.G_model, opt.G_norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.D_model,
                                          opt.n_layers_D, opt.D_norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            self.BCELoss = torch.nn.BCELoss().to(self.device)
            if self.VGG_loss:
                self.criterionPreLoss = VGGLoss_3D(opt.pretrain_model_path).to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr_max, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr_max, betas=(opt.beta1, opt.beta2))

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.mask = input['mask'].to(self.device)
        self.label = input['label'].to(self.device)

    def train_once(self):
        # epoch参数没用，直接去掉更好
        set_requires_grad(self.netG, True)
        self.fake_B, self.pre_label = self.netG(self.real_A)

        set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()  # set D's gradients to zero

        # calculate gradients for D
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach(), isDetach=True)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB, isDetach=True)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D_loss = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D_loss.backward()
        self.optimizer_D.step()  # update D's weights

        # update G
        set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        # self.netG.moudle.train()
        # self.netD.moudle.eval()

        # backward_G
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB, isDetach=False)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.lambda_L1

        one_hot_key = torch.FloatTensor(self.label.size(0), 2).zero_().to(self.device)
        idx = self.label.view(-1, 1)
        one_hot_key = one_hot_key.scatter_(1, idx, 1)

        self.loss_classification = self.BCELoss(self.pre_label, one_hot_key) + 0.01
        if self.VGG_loss:
            self.loss_G_perceive = self.criterionPreLoss(self.fake_B, self.real_B)
            self.loss_G_loss = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_perceive + self.loss_classification
        else:
            self.loss_G_loss = self.loss_G_GAN + self.loss_G_L1 + self.loss_classification

        # combine loss and calculate gradients
        self.loss_G_loss.backward()
        self.optimizer_G.step()  # udpate G's weights

    # 兼容性代码，epoch原本就没用
    # 不知道为什么命名为 forward()， 实际上就是训练一次
    # 我将原来的forward改名为train_once了，将forward()改为调用train_once()
    def forward(self, epoch=None): 
        self.train_once(self)
        
    def eval_once(self):
        with torch.no_grad():
            # set G&D to eval mode
            set_requires_grad(self.netG, False)
            set_requires_grad(self.netD, False)
            # generate fake_B
            self.fake_B, self.pre_label = self.netG(self.real_A)
            
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
            # loss D
            pred_real = self.netD(fake_AB, isDetach=True)
            self.loss_D_real = self.criterionGAN(pred_real, True)
            self.loss_D_loss = (self.loss_D_fake + self.loss_D_real) * 0.5
            # loss G
            pred_fake = self.netD(fake_AB, isDetach=False)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.lambda_L1
            
            one_hot_key = torch.FloatTensor(self.label.size(0), 2).zero_().to(self.device)
            idx = self.label.view(-1, 1)
            one_hot_key = one_hot_key.scatter_(1, idx, 1)

            self.loss_classification = self.BCELoss(self.pre_label, one_hot_key) + 0.01
            if self.VGG_loss:
                self.loss_G_perceive = self.criterionPreLoss(self.fake_B, self.real_B)
                self.loss_G_loss = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_perceive + self.loss_classification
            else:
                self.loss_G_loss = self.loss_G_GAN + self.loss_G_L1 + self.loss_classification
        



