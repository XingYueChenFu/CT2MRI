#-*- coding:utf-8 -*-
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Lambda
from glob import glob
from utils.dtypes import LabelEnum
import matplotlib.pyplot as plt
import nibabel as nib
import torchio as tio
import numpy as np
import torch
import re
import os


class NiftiImageGenerator(Dataset):
    def __init__(self, imagefolder, input_size, depth_size, transform=None):
        self.imagefolder = imagefolder
        self.input_size = input_size
        self.depth_size = depth_size
        self.inputfiles = glob(os.path.join(imagefolder, '*.nii.gz'))
        self.scaler = MinMaxScaler()
        self.transform = transform

    def read_image(self, file_path):
        img = nib.load(file_path).get_fdata()
        img = self.scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape) # 0 -> 1 scale
        return img

    def plot_samples(self, n_slice=15, n_row=4):
        samples = [self[index] for index in np.random.randint(0, len(self), n_row*n_row)]
        for i in range(n_row):
            for j in range(n_row):
                sample = samples[n_row*i+j]
                sample = sample[0]
                plt.subplot(n_row, n_row, n_row*i+j+1)
                plt.imshow(sample[:, :, n_slice])
        plt.show()

    def __len__(self):
        return len(self.inputfiles)

    def __getitem__(self, index):
        inputfile = self.inputfiles[index]
        img = self.read_image(inputfile)
        h, w, d= img.shape
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            img = tio.ScalarImage(inputfile)
            cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
            img = np.asarray(cop(img))[0]

        if self.transform is not None:
            img = self.transform(img)
        return img

class NiftiPairImageGenerator(Dataset):
    def __init__(self,
            input_folder: str,
            target_folder: str,
            input_size: int,
            depth_size: int,
            input_channel: int = 3,
            transform=None,
            target_transform=None,
            full_channel_mask=False,
            combine_output=False
        ):
        self.input_folder = input_folder
        self.target_folder = target_folder
        self.pair_files = self.pair_file()
        self.input_size = input_size
        self.depth_size = depth_size
        self.input_channel = input_channel
        self.scaler = MinMaxScaler()
        self.transform = transform
        self.target_transform = target_transform
        self.full_channel_mask = full_channel_mask
        self.combine_output = combine_output

    def pair_file(self):
        input_files = sorted(glob(os.path.join(self.input_folder, '*')))
        target_files = sorted(glob(os.path.join(self.target_folder, '*')))
        pairs = []
        for input_file, target_file in zip(input_files, target_files):
            assert int("".join(re.findall("\d", input_file))) == int("".join(re.findall("\d", target_file)))
            pairs.append((input_file, target_file))
        return pairs

    def label2masks(self, masked_img):
        result_img = np.zeros(masked_img.shape + ( self.input_channel - 1,))
        result_img[masked_img==LabelEnum.BRAINAREA.value, 0] = 1
        result_img[masked_img==LabelEnum.TUMORAREA.value, 1] = 1
        return result_img

    def read_image(self, file_path, pass_scaler=False):
        img = nib.load(file_path).get_fdata()
        if not pass_scaler:
            img = self.scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape) # 0 -> 1 scale
        return img

    def plot(self, index, n_slice=30):
        data = self[index]
        input_img = data['input']
        target_img = data['target']
        plt.subplot(1, 2, 1)
        plt.imshow(input_img[:, :, n_slice])
        plt.subplot(1, 2, 2)
        plt.imshow(target_img[:, :, n_slice])
        plt.show()

    def resize_img(self, img):
        h, w, d = img.shape
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            img = tio.ScalarImage(tensor=img[np.newaxis, ...])
            cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
            img = np.asarray(cop(img))[0]
        return img

    def resize_img_4d(self, input_img):
        h, w, d, c = input_img.shape
        result_img = np.zeros((self.input_size, self.input_size, self.depth_size, 2))
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            for ch in range(c):
                buff = input_img.copy()[..., ch]
                img = tio.ScalarImage(tensor=buff[np.newaxis, ...])
                cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
                img = np.asarray(cop(img))[0]
                result_img[..., ch] += img
            return result_img
        else:
            return input_img

    def sample_conditions(self, batch_size: int):
        indexes = np.random.randint(0, len(self), batch_size)
        input_files = [self.pair_files[index][0] for index in indexes]
        input_tensors = []
        for input_file in input_files:
            input_img = self.read_image(input_file, pass_scaler=self.full_channel_mask)
            input_img = self.label2masks(input_img) if self.full_channel_mask else input_img
            input_img = self.resize_img(input_img) if not self.full_channel_mask else self.resize_img_4d(input_img)
            if self.transform is not None:
                input_img = self.transform(input_img).unsqueeze(0)
                input_tensors.append(input_img)
        return torch.cat(input_tensors, 0).cuda()

    def __len__(self):
        return len(self.pair_files)

    def __getitem__(self, index):
        input_file, target_file = self.pair_files[index]
        input_img = self.read_image(input_file, pass_scaler=self.full_channel_mask)
        input_img = self.label2masks(input_img) if self.full_channel_mask else input_img
        input_img = self.resize_img(input_img) if not self.full_channel_mask else self.resize_img_4d(input_img)

        target_img = self.read_image(target_file)
        target_img = self.resize_img(target_img)

        if self.transform is not None:
            input_img = self.transform(input_img)
        if self.target_transform is not None:
            target_img = self.target_transform(target_img)

        if self.combine_output:
            return torch.cat([target_img, input_img], 0)

        return {'input':input_img, 'target':target_img}

from tqdm import tqdm

class NpyPairImageGenerator(Dataset):
    def __init__(
        self,
        data_root: str,  # 根目录，如 "/home/featurize/data/Task1_Liu_split/train"
        input_modality: str = "CT",  # 输入模态（CT）
        target_modality: str = "MR",  # 目标模态（MR）
        input_size: int = 256,  # 调整大小（如 256x256）
        depth_size: int = 64,  # 深度（切片数）
        input_channel: int = 1,  # 输入通道数（CT通常是单通道）
        transform=None,
        target_transform=None,
        full_channel_mask=False,
        combine_output=False,
        preload=False,  # 是否预加载数据
    ):
        self.data_root = data_root
        self.input_folder = os.path.join(data_root, input_modality)  # e.g., ".../train/CT"
        self.target_folder = os.path.join(data_root, target_modality)  # e.g., ".../train/MR"
        self.pair_files = self.pair_file()  # 获取配对的CT-MR文件列表
        self.input_size = input_size
        self.depth_size = depth_size
        self.input_channel = input_channel
        self.scaler = MinMaxScaler()
        self.transform = transform
        self.target_transform = target_transform
        self.full_channel_mask = full_channel_mask
        self.combine_output = combine_output
        
        self.preload = preload  # 是否预加载数据
        if self.preload:
            self.preloaded_data = []
            # for input_file, target_file in self.pair_files:
            for index in tqdm(range(len(self.pair_files)), desc="Preloading data"):
                # 获取配对的CT和MR文件
                input_file, target_file = self.pair_files[index]
                input_img = self.read_npy(input_file, pass_scaler=self.full_channel_mask)
                target_img = self.read_npy(target_file)
                # 调整大小
                input_img = self.resize_img(input_img)
                target_img = self.resize_img(target_img)
                
                # 增加通道维度
                if input_img.ndim == 3:
                    input_img = np.expand_dims(input_img, axis=0)
                if target_img.ndim == 3:
                    target_img = np.expand_dims(target_img, axis=0)
                    
                # 转换为Tensor
                if self.transform is not None:
                    input_img = self.transform(input_img)
                else:
                    input_img = torch.from_numpy(input_img).float()
                if self.target_transform is not None:
                    target_img = self.target_transform(target_img)
                else:
                    target_img = torch.from_numpy(target_img).float()
                    
                if self.combine_output:
                    self.preloaded_data.append(torch.cat([input_img, target_img], 0))
                else:
                    self.preloaded_data.append({"input": input_img, "target": target_img})
                

    def pair_file(self):
        # 获取CT和MR文件夹下的所有.npy文件，并按文件名配对
        input_files = sorted(glob(os.path.join(self.input_folder, "*.npy")))
        target_files = sorted(glob(os.path.join(self.target_folder, "*.npy")))
        
        # 确保文件名一致（如 "1BA001.npy" 在CT和MR中都存在）
        pairs = []
        for input_file in input_files:
            filename = os.path.basename(input_file)  # e.g., "1BA001.npy"
            target_file = os.path.join(self.target_folder, filename)
            if os.path.exists(target_file):
                pairs.append((input_file, target_file))
            else:
                print(f"Warning: {filename} not found in target folder, skipping.")
        return pairs

    def read_npy(self, file_path, pass_scaler=False):
        # 读取.npy文件（如果是NIfTI格式，可以用nibabel.load）
        img = np.load(file_path)
        if not pass_scaler:
            img = self.scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
        return img

    def resize_img(self, img):
        # 调整3D图像大小（H x W x D）
        h, w, d = img.shape
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            img = tio.ScalarImage(tensor=img[np.newaxis, ...])
            cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
            img = np.asarray(cop(img))[0]
        return img

    def __len__(self):
        return len(self.pair_files)

    def __getitem__(self, index):
        # 如果预加载数据，则直接返回预加载的数据
        if self.preload:
            return self.preloaded_data[index]
        
        
        input_file, target_file = self.pair_files[index]
        input_img = self.read_npy(input_file, pass_scaler=self.full_channel_mask)
        target_img = self.read_npy(target_file)

        # 调整大小
        input_img = self.resize_img(input_img)
        target_img = self.resize_img(target_img)
        
        # 增加通道维度
        if input_img.ndim == 3:
            input_img = np.expand_dims(input_img, axis=0)
        if target_img.ndim == 3:
            target_img = np.expand_dims(target_img, axis=0)

        # 转换为Tensor（如果transform=None，默认只转Tensor）
        if self.transform is not None:
            input_img = self.transform(input_img)
        else:
            input_img = torch.from_numpy(input_img).float()

        if self.target_transform is not None:
            target_img = self.target_transform(target_img)
        else:
            target_img = torch.from_numpy(target_img).float()

        if self.combine_output:
            return torch.cat([input_img, target_img], 0)  # 合并输入和目标（可选）
        return {"input": input_img, "target": target_img}
    
if __name__ == "__main__":
    # Example usage
    
    transform = Compose([
        Lambda(lambda t: torch.tensor(t).float()),
        Lambda(lambda t: (t * 2) - 1),
        # Lambda(lambda t: t.unsqueeze(0)),
        # Lambda(lambda t: t.transpose(3, 1)),
        Lambda(lambda t: t.permute(0, 3, 1, 2)),
    ])

    input_transform = Compose([
        Lambda(lambda t: torch.tensor(t).float()),
        Lambda(lambda t: (t * 2) - 1),
        # Lambda(lambda t: t.permute(3, 0, 1, 2)),
        # Lambda(lambda t: t.transpose(3, 1)),
        Lambda(lambda t: t.permute(0, 3, 1, 2)),
    ])
    
    
    train_dataset = NpyPairImageGenerator(
        data_root="/home/featurize/data/Task1_Liu_split_128/train",
        input_modality="CT",
        target_modality="MR",
        input_size=128,
        depth_size=32,
        transform=input_transform,
        target_transform=transform,
        full_channel_mask=True,
        preload=True
    )
    
    # 检查数据
    sample = train_dataset[0]
    print(sample["input"].shape, sample["target"].shape)  # e.g., torch.Size([1, 32, 128, 128])
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    