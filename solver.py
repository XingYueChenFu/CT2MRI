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
        self.trainset, self.valset, self.testset = create_datasets(config)
        # DataLoader
        self.train_loader = DataLoader(self.trainset, batch_size=config['batch_size'], shuffle=True, num_workers=4) if self.trainset else None
        tqdm.write(f'\033[1;34m[INFO]\033[0m ' + (f'\tTrainset: \033[34m{len(self.train_loader.dataset)}\033[0m \tTrainLoader: \033[34m{len(self.train_loader)}\033[0m' if self.trainset else f'\tTrainset: \033[34m{None}\033[0m \tTrainLoader: \033[34m{None}\033[0m'))
        self.val_loader = DataLoader(self.valset, batch_size=config['batch_size'], shuffle=False, num_workers=4) if self.valset else None
        tqdm.write(f'\033[1;34m[INFO]\033[0m ' + (f'\tValset: \033[34m{len(self.val_loader.dataset)}\033[0m \tValLoader: \033[34m{len(self.val_loader)}\033[0m' if self.valset else f'\tValset: \033[34m{None}\033[0m \tValLoader: \033[34m{None}\033[0m'))
        self.test_loader = DataLoader(self.testset, batch_size=config['batch_size'], shuffle=False, num_workers=4) if self.testset else None
        tqdm.write(f'\033[1;34m[INFO]\033[0m ' + (f'\tTestset: \033[34m{len(self.test_loader.dataset)}\033[0m \tTestLoader: \033[34m{len(self.test_loader)}\033[0m' if self.testset else f'\tTestset: \033[34m{None}\033[0m \tTestLoader: \033[34m{None}\033[0m'))
        
        # ===== 加载预训练模型 =====
        
        if config['VGG_loss']:
            tqdm.write(f'\033[1;34m[INFO]\033[0m Checking pre-trained model...')
            if not os.path.exists(config['pretrain_model_path']):
                tqdm.write(f'\033[1;34m[INFO]\033[0m Downloading VGG model...')
                os.makedirs(os.path.dirname(config['pretrain_model_path']), exist_ok=True)
                os.system(f'wget https://download.pytorch.org/models/vgg19-dcbb9e9d.pth -O {config["pretrain_model_path"]}')
            
        
        # ===== 构造模型 =====
        tqdm.write(f'\033[1;34m[INFO]\033[0m Building model...')
        self.model = GANclass(self.opt) # GANclass使用的是argparse.Namespace
        
        # ===== 加载模型 =====
        tqdm.write(f'\033[1;34m[INFO]\033[0m Loading model...')
        self._load_model()
    
    # TODO 加载模型的逻辑，没有模型时，从0开始训练
    def _load_model(self):
        pass
        
        


if __name__ == '__main__':
    # os.environ['PYTHONHASHSEED'] = '8' # python内置随机数种子
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # GPU与CPU同步
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    opt = parser.parse_args()

    with open(opt.config, 'r') as f:
        config = yaml.safe_load(f)

    solver = Solver(config)
    # for batch_idx, batch_data in enumerate(solver.train_loader):
    #     print(f"Batch {batch_idx} A: {batch_data['A'].shape}") # ([2, 1, 256, 256, 32])
    #     print(f"Batch {batch_idx} B: {batch_data['B'].shape}") # ([2, 1, 256, 256, 32])
    #     print(f"Batch {batch_idx} mask: {batch_data['mask'].shape}")
    #     print(f"Batch {batch_idx} label: {batch_data['label']}")
    #     print()
    #     # break
        