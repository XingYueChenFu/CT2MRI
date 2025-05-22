import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F

class CT2MRIDataset(Dataset):
    def __init__(self, data_root, image_size=None, use_rgb=False, transform=None):
        """
        Args:
            data_root (str): 数据根目录，包含 'CT' 和 'MR' 子目录
            image_size (int): 输出图像大小
            use_rgb (bool): 是否使用伪RGB图像（True: [3, H, W], False: [1, H, W]）
            transform (callable): 可选的数据增强操作
        """
        self.data_root = data_root
        self.image_size = image_size
        self.use_rgb = use_rgb
        self.transform = transform
        
        self.nch = 3 if use_rgb else 1
        
        self._load_data_paths()
        self._build_index_map()

    def _load_data_paths(self):
        """加载CT和MRI的npy文件路径"""
        ct_files = glob.glob(os.path.join(self.data_root, "CT", "*.npy"))
        mri_files = glob.glob(os.path.join(self.data_root, "MR", "*.npy"))

        self.data_dict = {}

        for ct_file in ct_files:
            prefix = os.path.splitext(os.path.basename(ct_file))[0]
            self.data_dict[prefix] = {"CT": ct_file}

        for mri_file in mri_files:
            prefix = os.path.splitext(os.path.basename(mri_file))[0]
            if prefix in self.data_dict:
                self.data_dict[prefix]["MR"] = mri_file
            else:
                self.data_dict[prefix] = {"MR": mri_file}

        # 只保留同时有CT和MRI的样本
        self.data_dict = {k: v for k, v in self.data_dict.items() if "CT" in v and "MR" in v}
        self.keys = list(self.data_dict.keys())

    def _build_index_map(self):
        """构建索引到切片的映射"""
        self.index_map = []
        for key in self.keys:
            ct_path = self.data_dict[key]["CT"]
            depth = np.load(ct_path).shape[0]
            for i in range(depth):
                self.index_map.append((key, i))

    def _minmax_normalize(self, img):
        """Min-Max 归一化到 [0, 1]"""
        img_min, img_max = img.min(), img.max()
        return (img - img_min) / (img_max - img_min + 1e-8)

    def _to_nch(self, img, nch):
        """将灰度图扩展为3通道"""
        return np.stack([img] * nch, axis=-1)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        key, slice_idx = self.index_map[idx]
        ct_path = self.data_dict[key]["CT"]
        mri_path = self.data_dict[key]["MR"]

        # 加载切片
        ct_img = np.load(ct_path)[slice_idx]
        mri_img = np.load(mri_path)[slice_idx]

        # Min-Max 归一化到 [0, 1]
        ct_img = self._minmax_normalize(ct_img)
        mri_img = self._minmax_normalize(mri_img)

        # 转换为张量 [H, W] -> [C, H, W] # use_rgb=True时为[3, H, W], 否则为[1, H, W]
        ct_img = self._to_nch(ct_img, self.nch)
        mri_img = self._to_nch(mri_img, self.nch)

        ct_tensor = torch.from_numpy(ct_img).float().permute(2, 0, 1)
        mri_tensor = torch.from_numpy(mri_img).float().permute(2, 0, 1)

        # Resize # 如果指定了图像大小
        if self.image_size and ct_tensor.shape[-1] != self.image_size:
            ct_tensor = F.interpolate(ct_tensor.unsqueeze(0), size=(self.image_size, self.image_size),
                                      mode='bilinear', align_corners=False).squeeze(0)
            mri_tensor = F.interpolate(mri_tensor.unsqueeze(0), size=(self.image_size, self.image_size),
                                       mode='bilinear', align_corners=False).squeeze(0)

        # 应用变换
        if self.transform:
            ct_tensor = self.transform(ct_tensor)
            mri_tensor = self.transform(mri_tensor)

        return {"image": ct_tensor, "condition": mri_tensor}

if __name__ == "__main__":
    use_rgb = True
    data_root = "/home/featurize/data/Task1_Liu/"
    traindata_root = "/home/featurize/data/Task1_Liu_split/train"
    valdata_root = "/home/featurize/data/Task1_Liu_split/val"
    
    dataset = CT2MRIDataset(data_root=data_root, image_size=None, use_rgb=use_rgb)
    print(len(dataset)) # 180*64=11520
    print(dataset[0]['image'].shape) # torch.Size([3, 256, 256])
    print(dataset[0]['condition'].shape) # torch.Size([3, 256, 256])
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    
    for batch in dataloader:
        print(batch['image'].shape)
        print(batch['condition'].shape)
        break
    
    
    
    
    traindataset = CT2MRIDataset(data_root=traindata_root, image_size=None, use_rgb=None)
    print(len(traindataset)) # 144*64=9216
    print(traindataset[0]['image'].shape) # torch.Size([1, 256, 256])
    print(traindataset[0]['condition'].shape) # torch.Size([1, 256, 256])
    
    trainloader = DataLoader(traindataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    for batch in trainloader:
        print(batch['image'].shape)
        print(batch['condition'].shape)
        break
    