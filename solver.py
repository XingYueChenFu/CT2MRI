import os
import random
import time

import argparse
import pathlib
import yaml
import json
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR

from tensorboardX import SummaryWriter

from data.dataset import create_datasets
from util.util import *
from util.Nii_utils import NiiDataRead
from models.GAN_class import *





class Solver:
    def __init__(self, config):
        # ===== 加载配置 ===== 
        tqdm.write(f'\033[1;34m[INFO]\033[0m\t Loading config...')
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config['device'] = self.device
        # 兼容性处理
        self.config['max_epochs'] = self.config['num_epochs']
        self.config['depthSize'] = self.config['target_depth_size']
        self.config['ImageSize'] = self.config['target_image_size']
        # self.config['pretrain_model_path'] = self.config['pretrain_model']
        self.opt = argparse.Namespace(**config)
        # 设置目录
        self.output_path = os.path.join(config['output_root'], config['name'])
        self.log_root = os.path.join(self.output_path, 'log')
        self.checkpoint_root = os.path.join(self.output_path, 'checkpoint')
        self.sample_root = os.path.join(self.output_path, 'sample')
        self.tb_root = os.path.join(self.output_path, 'tensorboard')
        # 创建目录
        for path in [self.output_path, self.log_root, self.checkpoint_root, self.sample_root, self.tb_root]:
            os.makedirs(path, exist_ok=True)
        # 设置随机数种子
        if config['use_seed']:
            self.seed = config['seed']
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # ===== 加载预训练模型 =====
        if config['VGG_loss']:
            tqdm.write(f'\033[1;34m[INFO]\033[0m\t Checking pre-trained model...')
            if not os.path.exists(config['pretrain_model_path']):
                tqdm.write(f'\033[1;34m[INFO]\033[0m\t Downloading VGG model...')
                os.makedirs(os.path.dirname(config['pretrain_model_path']), exist_ok=True)
                os.system(f'wget https://download.pytorch.org/models/vgg19-dcbb9e9d.pth -O {config["pretrain_model_path"]}')
            
        
        # ===== 构造模型等 =====
        tqdm.write(f'\033[1;34m[INFO]\033[0m\t Building model...')
        self.model = GANclass(self.opt) # GANclass使用的是argparse.Namespace
        
        self.ssim = SSIM(data_range=self.config['data_tange']).to(self.device)
        self.psnr = PSNR(data_range=self.config['data_tange']).to(self.device)
        
        # ===== 加载checkpoint模型 =====
        tqdm.write(f'\033[1;34m[INFO]\033[0m\t Loading model...')
        self._load_model()
        
        # ===== 加载数据集 ===== 
        tqdm.write(f'\033[1;34m[INFO]\033[0m\t Loading datasets...')
        self._load_data()
    
    def _load_data(self):
        self.trainset, self.valset, self.testset = create_datasets(config)
        # DataLoader
        self.train_loader = DataLoader(self.trainset, batch_size=config['batch_size'], shuffle=True, num_workers=4) if self.trainset else None
        tqdm.write(f'\t Trainset: \033[34m{len(self.train_loader.dataset)}\033[0m \tTrainLoader: \033[34m{len(self.train_loader)}\033[0m' if self.trainset else f'\tTrainset: \033[34m{None}\033[0m \tTrainLoader: \033[34m{None}\033[0m')
        self.val_loader = DataLoader(self.valset, batch_size=config['batch_size'], shuffle=False, num_workers=4) if self.valset else None
        tqdm.write(f'\t Valset: \033[34m{len(self.val_loader.dataset)}\033[0m \tValLoader: \033[34m{len(self.val_loader)}\033[0m' if self.valset else f'\tValset: \033[34m{None}\033[0m \tValLoader: \033[34m{None}\033[0m')
        self.test_loader = DataLoader(self.testset, batch_size=config['batch_size'], shuffle=False, num_workers=4) if self.testset else None
        tqdm.write(f'\t Testset: \033[34m{len(self.test_loader.dataset)}\033[0m \tTestLoader: \033[34m{len(self.test_loader)}\033[0m' if self.testset else f'\tTestset: \033[34m{None}\033[0m \tTestLoader: \033[34m{None}\033[0m')
    
    # 加载模型，没有模型时，从epoch为0开始
    def _load_model(self, checkpoint_path=None):
        checkpoint_path = checkpoint_path if checkpoint_path else os.path.join(self.checkpoint_root, f"{self.config['start_epoch']}.pth")
        
        # 不存在模型时，直接从epoch为0开始
        if not os.path.exists(checkpoint_path):
            tqdm.write(f'\033[1;33m[WARNING]\033[0m\t No checkpoint provided, while loading model. Starting from scratch...')
            self.config['start_epoch'] = 0
            return
        
        # 加载模型
        state = torch.load(checkpoint_path, map_location='cpu')
        pretrained_netG_dict = state['netG_state_dict']
        model_netG_dict = self.model.netG.state_dict()
        pretrained_netG_dict = {k: v for k, v in pretrained_netG_dict.items() if k in model_netG_dict}
        model_netG_dict.update(pretrained_netG_dict)
        self.model.netG.load_state_dict(model_netG_dict)  # torch.load: 加载训练好的模型 load_state_dict: 将torch.load加载出来的数据加载到net中

        if opt.isTrain:
            pretrained_netD_dict = state['netD_state_dict']
            model_netD_dict = self.model.netD.state_dict()
            pretrained_netD_dict = {k: v for k, v in pretrained_netD_dict.items() if k in model_netD_dict}
            model_netD_dict.update(pretrained_netD_dict)
            self.model.netD.load_state_dict(model_netD_dict)  # torch.load: 加载训练好的模型 load_state_dict: 将torch.load加载出来的数据加载到net中

            self.model.optimizer_G.load_state_dict(state['optimizer_G'])
            self.model.optimizer_D.load_state_dict(state['optimizer_D'])
        self.config['start_epoch'] = state['epoch']
        self.opt.epoch_count = state['epoch']
        tqdm.write(f'\033[1;32m[Success]\033[0m\t Successfully loaded model from "{checkpoint_path}", starting from epoch {self.config["start_epoch"]}...')
    
    # 保存模型
    def _save_model(self, checkpoint_path=None, epoch=None):
        epoch = epoch if epoch else self.epoch
        
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.checkpoint_root, f'default_{epoch}.pth')
            tqdm.write(f'\033[1;33m[WARNING]\033[0m\t No checkpoint path provided, using default path "{checkpoint_path}"')
        
        state = {
            'epoch': epoch + 1,
            'netG_state_dict': self.model.netG.state_dict(),
            'netD_state_dict': self.model.netD.state_dict(),
            'optimizer_G': self.model.optimizer_G.state_dict(),
            'optimizer_D': self.model.optimizer_D.state_dict()
        }
        torch.save(state, checkpoint_path)
        tqdm.write(f'\033[1;32m[Success]\033[0m\t Successfully saved model to "{checkpoint_path}", epoch {epoch}')
        
    
    def train(self):
        tqdm.write(f'\n\033[1;34m[INFO]\033[0m\t Training...')
        # 变量
        self.best_SSIM = 0
        self.train_losses, self.train_metrics = [], [] # 目前没在训练时加上评估
        self.eval_losses, self.eval_metrics = [], []
        tb_writer = SummaryWriter(self.tb_root)
        
        epoch_bar = tqdm(range(self.config['start_epoch']+1, self.config['max_epochs']+1), desc='\033[34mTraining Progress\033[0m', unit='epoch', position=0, leave=True, dynamic_ncols=True)
        
        self.start_time = time.time()
        
        for self.epoch in epoch_bar:
            # ///// 训练 /////
            train_batch_bar = tqdm(range(len(self.train_loader)), desc='\033[34mTraining\033[0m Batch Progress', unit='batch', position=1, leave=False, dynamic_ncols=True)
            for batch_idx, batch_data in enumerate(self.train_loader):
                batch_losses, batch_metrics = [], []
                # ===== 训练逻辑 =====
                # 加载数据
                self.model.set_input(batch_data)
                # 训练一次
                self.model.train_once()
                # 更新学习率
                update_learning_rate(self.model, self.config['max_epochs'], self.epoch, self.config['lr_max'])

                # ===== 日志&展示逻辑 =====
                # 计算损失
                losses = get_current_losses(self.model)
                batch_losses.append(losses)
                
                # 计算指标
                ssim = self.ssim(self.model.fake_B, self.model.real_B)
                psnr = self.psnr(self.model.fake_B, self.model.real_B)
                batch_metrics.append({'SSIM': ssim, 'PSNR': psnr})
                
                # 进度条
                train_batch_bar.update(1)
                train_batch_bar.set_postfix({'G_loss': losses['G_loss'], 'D_loss': losses['D_loss'], 'SSIM': ssim.mean().item(), 'PSNR': psnr.mean().item()})
            train_batch_bar.close()

            # 存放损失&指标
            mean_batch_losses = {
                k: torch.mean(torch.tensor([losses[k].item() if torch.is_tensor(losses[k]) else losses[k] 
                    for losses in batch_losses])).item()
                for k in batch_losses[0].keys()
            }
            mean_batch_metrics = {
                k: torch.mean(torch.tensor([metrics[k].item() if torch.is_tensor(metrics[k]) else metrics[k] 
                    for metrics in batch_metrics])).item()
                for k in batch_metrics[0].keys()
            }
            
            self.train_losses.append(mean_batch_losses)
            self.train_metrics.append(mean_batch_metrics)
            # 保存tensorboard
            tb_writer.add_scalar('train_SSIM', mean_batch_metrics['SSIM'], self.epoch)
            tb_writer.add_scalar('train_PSNR', mean_batch_metrics['PSNR'], self.epoch)
            tb_writer.add_scalar('train_G_loss', mean_batch_losses['G_loss'], self.epoch)
            tb_writer.add_scalar('train_D_loss', mean_batch_losses['D_loss'], self.epoch)
            
            # ///// 评估 /////
            if self.epoch % self.config['eval_interval'] == 1: # 本来应该是== 0（想在1st epoch后就看看效果，这里改为== 1）
                mean_eval_losses, mean_eval_metrics, samples = self.evaluate(self.val_loader, need_sample = True)
                
                self.eval_losses.append(mean_eval_losses)
                self.eval_metrics.append(mean_eval_metrics)
                
                # 保存评估结果
                # 保存log: {self.epoch}.json
                log_path = os.path.join(self.log_root, f'{self.epoch}.json')
                with open(log_path, 'w') as f:
                    # json.dump({'epoch': self.epoch, 'eval_losses': mean_eval_losses, 'eval_metrics': mean_eval_metrics}, f, indent=4)
                    json.dump({'epoch': self.epoch, 'train_losses': mean_batch_losses, 'train_metrics': mean_batch_metrics, 'eval_losses': mean_eval_losses, 'eval_metrics': mean_eval_metrics}, f, indent=4)
                # 保存sample: {self.epoch}_{类型}_npy
                for key, value in samples.items():
                    sample_path = os.path.join(self.sample_root, f'{self.epoch}_{key}.npy')
                    np.save(sample_path, value.cpu().numpy()) # (2, 1, 32, 256, 256)
                # 保存tensorboard
                tb_writer.add_scalar('val_SSIM', mean_eval_metrics['SSIM'], self.epoch)
                tb_writer.add_scalar('val_PSNR', mean_eval_metrics['PSNR'], self.epoch)
                tb_writer.add_scalar('val_G_loss', mean_eval_losses['G_loss'], self.epoch)
                tb_writer.add_scalar('val_D_loss', mean_eval_losses['D_loss'], self.epoch)
                
                # 保存最优模型
                if mean_eval_metrics['SSIM'] > self.best_SSIM:
                    self.best_SSIM = mean_eval_metrics['SSIM']
                    # 保存checkpoint
                    checkpoint_path = os.path.join(self.checkpoint_root, f'{self.epoch}.pth')
                    self._save_model(checkpoint_path)
                
            # 进度条
            epoch_bar.update(1)
        epoch_bar.close()
        tqdm.write(f'\033[1;32m[Success]\033[0m\t Training finished, total time: {time.time() - self.start_time:.2f}s')
    
    def evaluate(self, dataloader=None, need_sample=False):
        # 如果need_sample，返回最后的评估样例
        # 初始化变量
        # 优先级: 传入数据集 > test_loader > val_loader > train_loader > None
        dataloader = dataloader if dataloader else (self.test_loader if self.test_loader else (self.val_loader if self.val_loader else (self.train_loader if self.train_loader else None)))
        if dataloader is None:
            tqdm.write(f'\033[1;31m[ERROR]\033[0m\t No dataloader provided')
            return None, None, None
        
        eval_batch_bar = tqdm(range(len(dataloader)), desc='\033[34mEvaluation\033[0m Progress', unit='batch', position=1, leave=True, dynamic_ncols=True)
        batch_losses, batch_metrics = [], []
        
        for batch_idx, batch_data in enumerate(dataloader):
            # 读取数据
            self.model.set_input(batch_data)
            # 评估一次
            self.model.eval_once()
            
            # 计算指标
            losses = get_current_losses(self.model)
            batch_losses.append(losses)
            
            # 计算指标
            ssim = self.ssim(self.model.fake_B, self.model.real_B)
            psnr = self.psnr(self.model.fake_B, self.model.real_B)
            batch_metrics.append({'SSIM': ssim, 'PSNR': psnr})
            
            # 进度条
            eval_batch_bar.update(1)
            eval_batch_bar.set_postfix({'G_loss': losses['G_loss'], 'D_loss': losses['D_loss'], 'SSIM': ssim.mean().item(), 'PSNR': psnr.mean().item()})
        eval_batch_bar.close()
        # 处理数据并返回
        # mean_batch_losses = {k: np.mean([loss[k] for loss in batch_losses]) for k in batch_losses[0].keys()}
        # mean_batch_metrics = {k: np.mean([metric[k] for metric in batch_metrics]) for k in batch_metrics[0].keys()}
        mean_batch_losses = {
            k: torch.mean(torch.tensor([losses[k].item() if torch.is_tensor(losses[k]) else losses[k] 
                for losses in batch_losses])).item()
            for k in batch_losses[0].keys()
        }
        mean_batch_metrics = {
            k: torch.mean(torch.tensor([metrics[k].item() if torch.is_tensor(metrics[k]) else metrics[k] 
                for metrics in batch_metrics])).item()
            for k in batch_metrics[0].keys()
        }
        
        if not need_sample:
            return mean_batch_losses, mean_batch_metrics
        else:
            samples = {'input': self.model.real_A, 'output': self.model.fake_B, 'target': self.model.real_B}
            return mean_batch_losses, mean_batch_metrics, samples
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    opt = parser.parse_args()

    with open(opt.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if config['use_seed']:
        os.environ['PYTHONHASHSEED'] = str(config['seed']) # python内置随机数种子
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # GPU与CPU同步
    
    # 测试数据集加载
    # from data.dataset import MedicalDataset3D
    # md3d = MedicalDataset3D(config)
    # tqdm.write(md3d[0]['A'].shape) # (2,1,32,256,256)

    # 测试模型训练
    solver = Solver(config)
    solver.train()
    

        