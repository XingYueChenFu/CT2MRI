import os
from os import listdir
from os.path import join
import random

import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

from skimage import transform

from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from util.Nii_utils import NiiDataRead
from util.util import *

        

# ===== 已做修改 =====
# 新的数据集类MedicalDataset3D不需要
def randomcrop_Npatch(crop_size, crop_Npatch, mri1, ct, ct_mask):
    this_frame = crop_size
    img = mri1

    non_zero_z, non_zero_x, non_zero_y = np.where(ct_mask == 1)
    non_zero_num = non_zero_x.shape[0]

    patch_index = random.sample(range(0, non_zero_num), crop_Npatch)
    patch_mri1 = np.zeros([crop_Npatch, this_frame[0], this_frame[1], this_frame[2]]).astype(np.float32)
    patch_ct = np.zeros([crop_Npatch, this_frame[0], this_frame[1], this_frame[2]]).astype(np.float32)
    patch_mask = np.zeros([crop_Npatch, this_frame[0], this_frame[1], this_frame[2]]).astype(np.int32)

    for idx in range(crop_Npatch):
        z_med = non_zero_z[patch_index[idx]]
        x_med = non_zero_x[patch_index[idx]]
        y_med = non_zero_y[patch_index[idx]]
        z_frame_size = int(this_frame[0] / 2)
        x_frame_size = int(this_frame[1] / 2)
        y_frame_size = int(this_frame[2] / 2)

        # 计算裁剪区域
        z_this_min = max(0, z_med - z_frame_size)
        z_this_max = min(img.shape[0], z_med + z_frame_size)
        x_this_min = max(0, x_med - x_frame_size)
        x_this_max = min(img.shape[1], x_med + x_frame_size)
        y_this_min = max(0, y_med - y_frame_size)
        y_this_max = min(img.shape[2], y_med + y_frame_size)
        
        # 获取实际裁剪的区域
        crop_mri = mri1[z_this_min:z_this_max, x_this_min:x_this_max, y_this_min:y_this_max]
        crop_ct = ct[z_this_min:z_this_max, x_this_min:x_this_max, y_this_min:y_this_max]
        crop_mask = ct_mask[z_this_min:z_this_max, x_this_min:x_this_max, y_this_min:y_this_max]
        
        # 计算需要填充的大小
        pad_width = [
            (0, this_frame[0] - crop_mri.shape[0]),
            (0, this_frame[1] - crop_mri.shape[1]),
            (0, this_frame[2] - crop_mri.shape[2])
        ]
        
        # 进行填充
        padded_mri = np.pad(crop_mri, pad_width, mode='constant', constant_values=0)
        padded_ct = np.pad(crop_ct, pad_width, mode='constant', constant_values=0)
        padded_mask = np.pad(crop_mask, pad_width, mode='constant', constant_values=0)
        
        # 确保填充后的尺寸正确
        padded_mri = padded_mri[:this_frame[0], :this_frame[1], :this_frame[2]]
        padded_ct = padded_ct[:this_frame[0], :this_frame[1], :this_frame[2]]
        padded_mask = padded_mask[:this_frame[0], :this_frame[1], :this_frame[2]]
        
        patch_mri1[idx] = padded_mri
        patch_ct[idx] = padded_ct
        patch_mask[idx] = padded_mask

    return np.ascontiguousarray(patch_mri1), np.ascontiguousarray(patch_ct), np.ascontiguousarray(patch_mask)


# ===== 原文（没法用） =====
class DatasetFromFolder_train(data.Dataset):
    def __init__(self, opt, region):
        self.image_dir = opt.image_dir
        self.Max_CT = opt.Max_CT

        if region=='Headandneck':
            self.train_txt = os.path.join(opt.code_dir, 'data', 'headneck_train.txt')
        elif region=='All':
            self.train_txt = os.path.join(opt.code_dir, 'data', 'all_train.txt')
        with open(self.train_txt, 'r') as f:
            name_list = f.readlines()
        self.image_filenames = [n.strip('\n') for n in name_list]
        self.crop_size = [opt.depthSize, opt.ImageSize, opt.ImageSize]
        self.crop_Npatch = opt.Npatch
        self.all_patch_num = self.crop_Npatch * len(self.image_filenames)

    def __getitem__(self, index):
        this_index = int(index // self.crop_Npatch)
        self.ran_num = 1
        patient_name = self.image_filenames[this_index]

        if 'Abdomen' in patient_name:
            a_mri1, spacing, origin, direction = NiiDataRead(
                join(self.image_dir, patient_name, 'MR.nii.gz'))
            b_ct, spacing1, origin1, direction1 = NiiDataRead(
                join(self.image_dir, patient_name, 'CT.nii.gz'))
            b_mask, spacing, origin, direction = NiiDataRead(
                join(self.image_dir, patient_name, 'mask.nii.gz'))
            ct_max = self.Max_CT
            label = 1
        else:
            a_mri1, spacing, origin, direction = NiiDataRead(
                join(self.image_dir, patient_name, 'MR.nii.gz'))
            b_ct, spacing1, origin1, direction1 = NiiDataRead(
                join(self.image_dir, patient_name, 'CT.nii.gz'))
            b_mask, spacing, origin, direction = NiiDataRead(
                join(self.image_dir, patient_name, 'mask.nii.gz'))
            ct_max = self.Max_CT
            label = 0

        a_mri1 = normalization(a_mri1, 0, 255)
        ct_min = -1000

        b_ct[b_ct < ct_min] = ct_min
        b_ct[b_ct > ct_max] = ct_max
        b_ct[b_mask == 0] = ct_min
        b_ct = normalization(b_ct, ct_min, ct_max)

        a_patch_mri1, b_patch_ct, b_patch_mask = randomcrop_Npatch(self.crop_size, self.ran_num, a_mri1, b_ct, b_mask)

        a = torch.tensor(a_patch_mri1).float()
        b = torch.tensor(b_patch_ct).float()
        mask = torch.tensor(b_patch_mask).float()

        p1 = np.random.choice([0, 1])
        p2 = np.random.choice([0, 1])
        self.trans = transforms.Compose([
                                  transforms.RandomHorizontalFlip(p1),
                                  transforms.RandomVerticalFlip(p2),
                                       ])
        a = self.trans(a)
        b = self.trans(b)
        mask = self.trans(mask)
        label = torch.tensor(int(label)).long()


        return {
            'A': a,
            'B': b,
            'mask': mask,
            'label':label
        }

    def __len__(self):
        return self.all_patch_num


# ===== 兼容（屎山） =====

class CustomDatasetFromFolder_train(data.Dataset):
    def __init__(self, opt, region):
        
        self.data_root = '/home/featurize/data'  # 您的数据根目录
        self.Max_CT = opt.Max_CT
        
        # 根据区域确定要包含哪些数据
        self.regions = []
        if region == 'brain':
            self.regions = ['brain']
        elif region == 'pelvis':
            self.regions = ['pelvis']
        elif region == 'All':
            self.regions = ['brain', 'pelvis']
        
        # 收集所有符合条件的病例路径
        self.case_paths = []
        for task in ['Task1']:
            for region in self.regions:
                region_path = join(self.data_root, task, region)
                for case in os.listdir(region_path):
                    case_path = join(region_path, case)
                    if os.path.isdir(case_path) and case != 'overview':
                        # 检查必要的文件是否存在
                        if all(os.path.exists(join(case_path, f)) for f in ['ct.nii.gz', 'mr.nii.gz', 'mask.nii.gz']):
                            self.case_paths.append(case_path)
        
        self.crop_size = [opt.depthSize, opt.ImageSize, opt.ImageSize]
        self.crop_Npatch = opt.Npatch
        self.all_patch_num = self.crop_Npatch * len(self.case_paths)
        
        # 定义转换
        self.trans = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])
        

    def __getitem__(self, index):
        this_index = int(index // self.crop_Npatch)
        case_path = self.case_paths[this_index]
        
        # 读取数据
        a_mri, spacing, origin, direction = NiiDataRead(join(case_path, 'mr.nii.gz'))
        b_ct, spacing1, origin1, direction1 = NiiDataRead(join(case_path, 'ct.nii.gz'))
        b_mask, spacing, origin, direction = NiiDataRead(join(case_path, 'mask.nii.gz'))
        
        # 设置标签：brain为0，pelvis为1
        label = 0 if 'brain' in case_path else 1
        
        # 数据预处理
        a_mri = normalization(a_mri, 0, 255)
        ct_min = -1000
        ct_max = self.Max_CT
        
        b_ct[b_ct < ct_min] = ct_min
        b_ct[b_ct > ct_max] = ct_max
        b_ct[b_mask == 0] = ct_min
        b_ct = normalization(b_ct, ct_min, ct_max)
        #
        
        # 随机裁剪
        # self.ran_num = 1
        # a_patch_mri, b_patch_ct, b_patch_mask = randomcrop_Npatch(self.crop_size, self.ran_num, a_mri, b_ct, b_mask)
        a_patch_mri, b_patch_ct, b_patch_mask = randomcrop_Npatch(self.crop_size, 1, a_mri, b_ct, b_mask)
        
        # 转换为tensor
        a = torch.tensor(a_patch_mri).float()
        b = torch.tensor(b_patch_ct).float()
        mask = torch.tensor(b_patch_mask).float()
        
        # 随机翻转
        if random.random() > 0.5:
            a = self.trans(a)
            b = self.trans(b)
            mask = self.trans(mask)
        
        label = torch.tensor(int(label)).long()

        return {
            'A': a,
            'B': b,
            'mask': mask,
            'label': label
        }

    def __len__(self):
        return self.all_patch_num



# ===== NEW =====
import os
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm
import nibabel as nib

import torch
from torch.utils.data import Dataset, random_split




class MedicalDataset3D(Dataset):
    def __init__(self, config: Dict, mode: str = 'train'):
        """
        Initialize the MedicalDataset3D.
        
        Args:
            config (Dict): Loaded config dictionary
            mode (str): One of 'train', 'val', or 'test'
        """
        self.config = config
        self.preload = config['preload']
        self.mode = mode
        self.ignore_ids = set(self.config['ignore_ids'])
        self.target_depth = self.config['target_depth_size']
        self.start_slice = self.config['start_slice']
        self.target_size = self.config['target_image_size']
        self.ct_window_level = self.config['ct_window_level']
        self.ct_window_width = self.config['ct_window_width']
        
        # Collect all valid data paths
        self.data_paths = []
        self.labels = []
        
        if mode in ['train', 'val']:
            for root, label in zip(self.config['train_root'], self.config['train_label']):
                self._collect_data_from_root(root, label)
        elif mode == 'test':
            for root, label in zip(self.config['test_root'], self.config['test_label']):
                self._collect_data_from_root(root, label)

        if self.preload:
            tqdm.write(f'\t\033[1;34m[INFO]\033[0m Preloading data into memory...')
            # 加载数据到内存
            self.data = []
            path_bar = tqdm(total=len(self.data_paths), desc='\tPreloading data', unit='sample')
            for idx, path in enumerate(self.data_paths):
                label = self.labels[idx]
                
                ct = nib.load(os.path.join(path, 'ct.nii.gz')).get_fdata()
                mr = nib.load(os.path.join(path, 'mr.nii.gz')).get_fdata()
                mask = nib.load(os.path.join(path, 'mask.nii.gz')).get_fdata()
                self.data.append(self._process_data(ct, mr, mask, label))
                
                path_bar.update(1)
            path_bar.close()
            
            # 统计占用
            for items in self.data: 
                self.data_memory += sys.getsizeof(items)
                for ele in items[:3]: # tensor管理的张量需要额外获取 
                    self.data_memory += ele.element_size() * ele.nelement()
            
            tqdm.write(f'\t\033[1;34m[INFO]\033[0m Preloaded \033[34m{len(self.data_paths)}\033[0m samples. Total memory used: \033[34m{self.data_memory / (1024 * 1024):.2f} MB\033[0m')
        
    def _collect_data_from_root(self, root: str, label: int):
        """Collect data paths from a root directory."""
        for folder in os.listdir(root):
            if folder in self.ignore_ids:
                continue
                
            folder_path = os.path.join(root, folder)
            if os.path.isdir(folder_path):
                # Check if all required files exist
                required_files = {'ct.nii.gz', 'mask.nii.gz', 'mr.nii.gz'}
                existing_files = set(os.listdir(folder_path))
                
                if required_files.issubset(existing_files):
                    self.data_paths.append(folder_path)
                    self.labels.append(label)
    
    def __len__(self) -> int:
        return len(self.data_paths)
    
    
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.preload:
            ct, mr, mask, label = self.data[idx]
        else:
            folder_path = self.data_paths[idx]
            label = self.labels[idx]
            
            ct = nib.load(os.path.join(folder_path, 'ct.nii.gz')).get_fdata()
            mr = nib.load(os.path.join(folder_path, 'mr.nii.gz')).get_fdata()
            mask = nib.load(os.path.join(folder_path, 'mask.nii.gz')).get_fdata()
            
            ct, mr, mask, label = self._process_data(ct, mr, mask, label)
        
        return {
            'A': ct,    # CT
            'B': mr,    # MRI
            'mask': mask,
            'label': label
        }
    
    def _process_data(self, ct, mr, mask, label):
        ct =self._normalize_ct(ct)
        mr = self._normalize_mr(mr)
        mask = (mask > 0).astype(np.float32)
        
        ct = self._select_and_pad(ct)
        mr = self._select_and_pad(mr)
        mask = self._select_and_pad(mask)
        
        ct = np.expand_dims(ct, axis=0)
        mr = np.expand_dims(mr, axis=0)
        mask = np.expand_dims(mask, axis=0)
        
        ct = np.transpose(ct, (0, 3, 1, 2))
        mr = np.transpose(mr, (0, 3, 1, 2))
        mask = np.transpose(mask, (0, 3, 1, 2))
        
        ct = torch.from_numpy(ct).float()
        mr = torch.from_numpy(mr).float()
        mask = torch.from_numpy(mask).float()
        
        # label不做处理
        return ct, mr, mask, label
    
    def _normalize_ct(self, ct: np.ndarray) -> np.ndarray:
        """Apply CT windowing (window level/width)."""
        window_min = self.ct_window_level - self.ct_window_width / 2
        window_max = self.ct_window_level + self.ct_window_width / 2
        
        ct = np.clip(ct, window_min, window_max)
        ct = (ct - window_min) / (window_max - window_min) * 255.0
        return ct.astype(np.float32)
    
    def _normalize_mr(self, mr: np.ndarray) -> np.ndarray:
        """Normalize MRI to 0-255 range."""
        mr_min = mr.min()
        mr_max = mr.max()
        
        if mr_max != mr_min:
            mr = (mr - mr_min) / (mr_max - mr_min) * 255.0
        return mr.astype(np.float32)
    
    def _select_and_pad(self, volume: np.ndarray) -> np.ndarray:
        """Select slices and pad to target size."""
        depth = volume.shape[-1]
        
        # Select slices
        if self.start_slice == -1:
            # Start from middle
            start = max(0, depth // 2 - self.target_depth // 2)
        else:
            start = self.start_slice
        
        end = start + self.target_depth
        if end > depth:
            raise ValueError(f"Not enough slices (depth={depth}) for target depth {self.target_depth} starting at {start}")
        
        volume = volume[..., start:end]
        
        # Pad height and width to target size
        h, w = volume.shape[0], volume.shape[1]
        pad_h = max(0, self.target_size - h)
        pad_w = max(0, self.target_size - w)
        
        if len(volume.shape) == 3:  # D, H, W
            pad_width = ((0, pad_h), (0, pad_w), (0, 0))
        else:  # H, W
            pad_width = ((0, pad_h), (0, pad_w))
        
        # Use minimum value for padding
        pad_value = volume.min()
        volume = np.pad(volume, pad_width, mode='constant', constant_values=pad_value)
        
        # If still not big enough (unlikely), crop
        if volume.shape[0] > self.target_size or volume.shape[1] > self.target_size:
            volume = volume[:self.target_size, :self.target_size, ...]
        
        return volume

def create_datasets(config: Dict) -> Tuple[Dataset, Optional[Dataset], Dataset]:
    """
    Create train, validation, and test datasets based on config.
    
    Args:
        config (Dict): Loaded config dictionary
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
        If validation is disabled in config, returns None for val_dataset
    """
    # 创建完整训练集
    full_train = MedicalDataset3D(config, mode='train')
    
    # 创建测试集
    test_dataset = MedicalDataset3D(config, mode='test') if config['need_test'] and 'test_root' in config and config['test_root'] else None
    
    # 划分训练集和验证集
    if config['need_validation']:
        val_size = int(len(full_train) * config['validation_ratio'])
        train_size = len(full_train) - val_size
        
        # Use fixed random seed for reproducibility
        if config['use_seed']:
            generator = torch.Generator().manual_seed(config['seed'])
        else:
            generator = torch.Generator()
        train_dataset, val_dataset = random_split(
            full_train, 
            [train_size, val_size],
            generator=generator
        )
    else:
        train_dataset = full_train
        val_dataset = None
    
    return train_dataset, val_dataset, test_dataset