import os
import random
import time

import argparse
import pathlib
import yaml
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics import PeakSignalNoiseRatio as PSNR

from tensorboardX import SummaryWriter

from data.dataset import create_datasets
from util.util import *
from util.Nii_utils import NiiDataRead
from models.GAN_class import *





class Solver:
    def __init__(self, config):
        # ===== 加载配置 ===== 
        tqdm.write(f'\033[1;34m[INFO]\033[0m Loading config...')
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
        self.log_path = os.path.join(self.output_path, 'log')
        self.checkpoint_path = os.path.join(self.output_path, 'checkpoint')
        self.sample_path = os.path.join(self.output_path, 'sample')
        self.tb_path = os.path.join(self.output_path, 'tensorboard')
        # 创建目录
        for path in [self.output_path, self.log_path, self.checkpoint_path, self.sample_path, self.tb_path]:
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
            
        # ===== 加载数据集 ===== 
        tqdm.write(f'\033[1;34m[INFO]\033[0m Loading datasets...')
        self._load_data()
        
        # ===== 加载预训练模型 =====
        if config['VGG_loss']:
            tqdm.write(f'\033[1;34m[INFO]\033[0m Checking pre-trained model...')
            if not os.path.exists(config['pretrain_model_path']):
                tqdm.write(f'\033[1;34m[INFO]\033[0m Downloading VGG model...')
                os.makedirs(os.path.dirname(config['pretrain_model_path']), exist_ok=True)
                os.system(f'wget https://download.pytorch.org/models/vgg19-dcbb9e9d.pth -O {config["pretrain_model_path"]}')
            
        
        # ===== 构造模型等 =====
        tqdm.write(f'\033[1;34m[INFO]\033[0m Building model...')
        self.model = GANclass(self.opt) # GANclass使用的是argparse.Namespace
        
        self.ssim = SSIM(data_range=self.config['data_tange']).to(self.device)
        self.psnr = PSNR(data_range=self.config['data_tange']).to(self.device)
        
        # ===== 加载模型 =====
        tqdm.write(f'\033[1;34m[INFO]\033[0m Loading model...')
        self._load_model()
    
    def _load_data(self):
        self.trainset, self.valset, self.testset = create_datasets(config)
        # DataLoader
        self.train_loader = DataLoader(self.trainset, batch_size=config['batch_size'], shuffle=True, num_workers=4) if self.trainset else None
        tqdm.write(f'\033[1;34m[INFO]\033[0m ' + (f'\tTrainset: \033[34m{len(self.train_loader.dataset)}\033[0m \tTrainLoader: \033[34m{len(self.train_loader)}\033[0m' if self.trainset else f'\tTrainset: \033[34m{None}\033[0m \tTrainLoader: \033[34m{None}\033[0m'))
        self.val_loader = DataLoader(self.valset, batch_size=config['batch_size'], shuffle=False, num_workers=4) if self.valset else None
        tqdm.write(f'\033[1;34m[INFO]\033[0m ' + (f'\tValset: \033[34m{len(self.val_loader.dataset)}\033[0m \tValLoader: \033[34m{len(self.val_loader)}\033[0m' if self.valset else f'\tValset: \033[34m{None}\033[0m \tValLoader: \033[34m{None}\033[0m'))
        self.test_loader = DataLoader(self.testset, batch_size=config['batch_size'], shuffle=False, num_workers=4) if self.testset else None
        tqdm.write(f'\033[1;34m[INFO]\033[0m ' + (f'\tTestset: \033[34m{len(self.test_loader.dataset)}\033[0m \tTestLoader: \033[34m{len(self.test_loader)}\033[0m' if self.testset else f'\tTestset: \033[34m{None}\033[0m \tTestLoader: \033[34m{None}\033[0m'))
    
    # TODO 加载模型的逻辑，没有模型时，从0开始训练
    def _load_model(self):
        pass
    
    # TODO
    def _save_model(self):
        pass
    
    def train(self):
        tqdm.write(f'\n\033[1;34m[INFO]\033[0m Training...')
        # 变量
        self.best_SSIM = 0
        self.train_losses, self.train_metics = [], [] # 目前没在训练时加上评估
        self.eval_losses, self.eval_metics = [], []
        
        epoch_bar = tqdm(range(self.config['start_epoch'], self.config['max_epochs']), desc='\033[34mTraining Progress\033[0m', unit='epoch', position=0, leave=True)
        
        self.start_time = time.time()
        
        for self.epoch in epoch_bar:
            # ///// 训练 /////
            train_batch_bar = tqdm(range(len(self.train_loader)), desc='\033[34mTraining\033[0m Batch Progress', unit='batch', position=1, leave=False)
            for batch_idx, batch_data in enumerate(self.train_loader):
                batch_losses, batch_metics = [], []
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
                batch_metics.append({'SSIM': ssim, 'PSNR': psnr})
                
                # 进度条
                train_batch_bar.update(1)
                train_batch_bar.set_postfix({'G_loss': losses['G_loss'], 'D_loss': losses['D_loss'], 'SSIM': ssim, 'PSNR': psnr})
            train_batch_bar.close()

            # 存放损失&指标
            mean_batch_losses = {k: np.mean([loss[k] for loss in batch_losses]) for k in batch_losses[0].keys()}
            self.train_losses.append(mean_batch_losses)
            mean_batch_metics = {k: np.mean([metric[k] for metric in batch_metics]) for k in batch_metics[0].keys()}
            self.train_metics.append(mean_batch_metics)
            
            # ///// 评估 /////
            if self.epoch % self.config['eval_interval'] == 0:
                this_eval_losses, this_eval_metrics, samples = self.evaluate(self.val_loader, need_sample = True)
                
                self.eval_losses.append(this_eval_losses)
                self.eval_metics.append(this_eval_metrics)
                
                tqdm.write(f'\033[1;34m[INFO]\033[0m Evaluation at epoch {self.epoch}: SSIM: {this_eval_metrics["SSIM"]:.4f}, PSNR: {this_eval_metrics["PSNR"]:.4f}')
                # 保存评估结果，保存最优模型，保存样本
                if this_eval_metrics['SSIM'] > self.best_SSIM:
                    self.best_SSIM = this_eval_metrics['SSIM']
                    # 保存模型
                    # self.model.save(self.checkpoint_path, self.epoch)

                
                
            # 进度条
            epoch_bar.update(1)
            break # DEBUG
        epoch_bar.close()
        pass
    
    def evaluate(self, dataloader=None, need_sample=False):
        # 如果need_sample，返回最后的评估样例
        # 初始化变量
        # 优先级: 传入数据集 > test_loader > val_loader > train_loader > None
        dataloader = dataloader if dataloader else (self.test_loader if self.test_loader else (self.val_loader if self.val_loader else (self.train_loader if self.train_loader else None)))
        if dataloader is None:
            tqdm.write(f'\033[1;32m[ERROR]\033[0m No dataloader provided')
            return None, None, None
        
        eval_batch_bar = tqdm(range(len(dataloader)), desc='\033[34mEvaluation\033[0m Progress', unit='batch', position=1, leave=True)
        batch_losses, batch_metics = [], []
        
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
            batch_metics.append({'SSIM': ssim, 'PSNR': psnr})
            
            # 进度条
            eval_batch_bar.update(1)
            eval_batch_bar.set_postfix({'G_loss': losses['G_loss'], 'D_loss': losses['D_loss'], 'SSIM': ssim, 'PSNR': psnr})
        eval_batch_bar.close()
        # 处理数据并返回
        mean_batch_losses = {k: np.mean([loss[k] for loss in batch_losses]) for k in batch_losses[0].keys()}
        mean_batch_metics = {k: np.mean([metric[k] for metric in batch_metics]) for k in batch_metics[0].keys()}
        
        if not need_sample:
            return mean_batch_losses, mean_batch_metics
        else:
            samples = {'input': self.model.real_A, 'output': self.model.fake_B, 'target': self.model.real_B}
            return mean_batch_losses, mean_batch_metics, samples
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    opt = parser.parse_args()

    with open(opt.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if config['use_seed']:
        os.environ['PYTHONHASHSEED'] = str(config['seed']) # python内置随机数种子
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # GPU与CPU同步

    solver = Solver(config)
    solver.train()

        