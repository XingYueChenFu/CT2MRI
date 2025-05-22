import os
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf

from data.dataset import CT2MRIDataset
from ldm.models.diffusion.ddpm import LatentDiffusion

def main():
    # 加载配置
    config = OmegaConf.load("configs/ct2mri.yaml")
    
    # 创建数据集和数据加载器
    dataset = CT2MRIDataset(data_root="/home/featurize/data/Task1_Liu_split/train")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    
    # 创建模型
    model = LatentDiffusion(**config.model.params)
    model.learning_rate = config.model.base_learning_rate
    
    # 设置训练器
    trainer = Trainer(
        max_epochs=100,
        accelerator='gpu',
        devices=1,
        callbacks=[
            ModelCheckpoint(
                dirpath='checkpoints',
                filename='ct2mri-{epoch:02d}',
                save_top_k=3,
                monitor='val/loss_simple_ema'
            )
        ]
    )
    
    # 开始训练
    trainer.fit(model, dataloader)

if __name__ == "__main__":
    main()