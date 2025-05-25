"""
预训练LDM，

使用VQ-VAE-CT提取输入的CT图像的特征作为嵌入

使用VQ-VAE-MRI提取输入的MRI图像(加噪)，作为让模型降噪的输入

将CT的潜在表示与MRI(加噪)的潜在表示concat，作为模型的输入

"""

# 0. 导包
import os, sys, argparse, datetime
from pathlib import Path

import json
import numpy as np
from tqdm import tqdm

from omegaconf import OmegaConf

import torch
from torch import nn
from torch.utils.data import DataLoader

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddpm import LDM
from data.dataset import CT2MRIDataset
from solver import get_parser, print_device_info

# SSIM 与 PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR

import lpips  # pip install lpips

def main():
    # ===== ===== =====
    # 1. 读取参数
    # ===== ===== =====
    debug = 1
    print_device_info()
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())
    
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    # Load configs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'\033[1;34m[DEBUG]\033[0m Using device: {device}')
    cli = OmegaConf.from_dotlist(unknown)
    configs = [OmegaConf.load(cfg) for cfg in list(opt.base)]
    config = OmegaConf.merge(*configs, cli)
    
    # model
    if debug:
        print('\033[1;34m[DEBUG]\033[0m Loading Model...')
    # model = instantiate_from_config(config.model)
    # model = model.to(device)
    if debug:
        print('\033[1;34m[DEBUG]\033[0m Model Loaded')
    # ===== ===== =====
    # 2. 读取数据集
    # ===== ===== =====
    # dataset
    trainset = CT2MRIDataset(config.data.params.train.folder, use_rgb=config.data.params.train.use_rgb)
    valset = CT2MRIDataset(config.data.params.validation.folder, use_rgb=config.data.params.validation.use_rgb)
    
    # dataloader
    trainloader = DataLoader(
        trainset,
        batch_size=config.data.params.train.batch_size,
        shuffle=True,
        num_workers=config.data.params.train.num_workers,
        pin_memory=True,
    )
    valloader = DataLoader(
        valset,
        batch_size=config.data.params.validation.batch_size,
        shuffle=False,
        num_workers=config.data.params.validation.num_workers,
        pin_memory=True,
    )
    
    # ===== ===== =====
    # 3. 定义模型（加载模型）
    # ===== ===== =====
    tqdm.write(f'\033[1;34m[DEBUG]\033[0m Loading Model...')
    # mri_vae_model
    mri_root = '/home/featurize/output/VQ-VAE-MRI/checkpoints/best/'
    mri_vae_model = load_model(config.model.params.first_stage_config, mri_root, device)
    tqdm.write(f'\033[1;34m[DEBUG]\033[0m MRI VAE Model Loaded')
    # ct_vae_model
    ct_root = '/home/featurize/output/VQ-VAE-CT/checkpoints/best/'
    ct_vae_model = load_model(config.model.params.cond_stage_config, ct_root, device)
    tqdm.write(f'\033[1;34m[DEBUG]\033[0m CT VAE Model Loaded')
    # unet_model
    ldm_model = LDM(config.model.params,
                    # config.model.params.unet_config, 
                    # config.model.params.conditioning_key, 
                    mri_vae_model,
                    ct_vae_model)
    # ldm_model.mri_vae = mri_vae_model
    # ldm_model.ct_vae = ct_vae_model
    
    # 创建模型
    ldm_model.model.to(device)
    
    
    # ===== ===== =====
    # 4. 其它定义
    # ===== ===== =====
    # 启用 EMA（可选）
    # if model.use_ema:
    #     model.model_ema.to(device)
    # 优化器
    optimizer = torch.optim.Adam(ldm_model.parameters(), lr=1.0e-06, betas=(0.9, 0.999))
    
    # LPIPS损失？
    
    # ===== ===== =====
    # 5. 训练模型 & 验证模型
    # ===== ===== =====
    
    # from diffusers.schedulers import DDIMScheduler  # 或 DDPMScheduler
    # noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
    # from torch.optim.lr_scheduler import LambdaLR
    # scheduler = [
    #             {
    #                 'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
    #                 'interval': 'step',
    #                 'frequency': 1
    #             }]
    # ldm_model.scheduler = scheduler
    # ldm_model.use_scheduler = True  # 是否使用调度器
    
    optimizer = torch.optim.Adam(ldm_model.parameters(), lr=1e-4)
    ldm_model.optimizer = optimizer
    
    train_ldm(ldm_model, trainloader, valloader, optimizer, epochs=200, device=device)

def train_ldm(model, trainloader, valloader=None, optimizer=None, epochs=100, device='cpu', save_path="./checkpoints"):
    model.train()
    best_ssim = 0.0
    os.makedirs(save_path, exist_ok=True)

    # optional: EMA 移动平均参数
    if getattr(model, "use_ema", False):
        model.model_ema.to(device)

    epoch_bar = tqdm(total=epochs, desc="\033[1;34m[Training Progress]\033[0m", position=0, leave=False)
    for epoch in range(epochs):
        # eval_ldm(model, valloader, device) # DEBUG 看看是否正常工作
        # ---------- 训练 ----------
        batch_bar = tqdm(total=len(trainloader), desc=f"[Training]", position=1, leave=False)
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(trainloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            loss = model.training_step(batch, batch_idx)
            loss.backward()
            optimizer.step()

            # 更新 EMA
            if getattr(model, "use_ema", False):
                model.model_ema.update(model.model)

            batch_bar.set_postfix(loss=loss.item())
            batch_bar.update()
            epoch_loss += loss.item()
        
        batch_bar.close()
        epoch_bar.update()
        
        tqdm.write(f"\033[1;34m[LOG]\033[0m Epoch [{epoch+1}/{epochs}] Avg Loss: {epoch_loss / len(trainloader):.4f}")

        # ---------- 验证（仅简单验证） ----------
        # 全部2304张验证集全部跑32步，需要2:47:33（4.38s/it） 这里只跑15个batch
        eval_log = eval_ldm(model, valloader, device, num_steps=32, max_batchs=15) 
        
        # ---------- log, 保存模型， 保存样本 ----------
        # 保存数据的位置
        output_root = '/home/featurize/output/LDM-CT2MRI'
        log_path = os.path.join(output_root, 'logs')
        checkpoint_path = os.path.join(output_root, 'checkpoints')
        sample_path = os.path.join(output_root, 'samples')
        best_model_root = os.path.join(checkpoint_path, f"best")
        last_model_root = os.path.join(checkpoint_path, f"last")
        # 创建文件夹
        os.makedirs(log_path, exist_ok=True)
        os.makedirs(checkpoint_path, exist_ok=True)
        os.makedirs(sample_path, exist_ok=True)
        os.makedirs(best_model_root, exist_ok=True)
        os.makedirs(last_model_root, exist_ok=True)
        
        # 1. 日志
        log_dict = {
            'epoch': epoch + 1,
            'train_loss': epoch_loss / len(trainloader),
            'val_loss': eval_log['loss'],
            'val_ssim': eval_log['ssim'],
            'val_psnr': eval_log['psnr'],
        }
        epoch_log_path = os.path.join(log_path, f"epoch_{epoch + 1}.json")
        with open(epoch_log_path, 'w') as f:
            json.dump(log_dict, f, indent=4)
        
        # 2. 保存模型
        if eval_log['ssim'] > best_ssim:
            best_ssim = eval_log['ssim']
            best_model_path = os.path.join(best_model_root, f"epoch_{epoch + 1}_ssim_{best_ssim:.4f}.pth")
            torch.save(model.state_dict(), best_model_path)
            tqdm.write(f"\033[1;32m[LOG]\033[0m Best model saved at {best_model_path} with SSIM: {best_ssim:.4f}")
        
        
    epoch_bar.close()

@torch.no_grad()
def eval_ldm(model, valloader, device='cpu', num_steps=32, max_batchs=None):
    model.eval()
    ssim = SSIM(data_range=1.0).to(device)
    psnr = PSNR(data_range=1.0).to(device)
    val_loss = 0.0
    val_ssim = []
    val_psnr = []
    
    total_len = len(valloader) if max_batchs is None else min(len(valloader), max_batchs)
    eval_bar = tqdm(total=total_len, desc=f"[Validation]", position=1, leave=False)
    
    for val_batch_idx, val_batch in enumerate(valloader):
        val_batch = {k: v.to(device) for k, v in val_batch.items()}
        val_loss += model.validation_step(val_batch, val_batch_idx).item() # 使用了@torch.no_grad()
    
        # 计算 SSIM 和 PSNR
        output = model.generate(ct=val_batch['condition'], num_steps=num_steps) 
        ssim_value = ssim(output, val_batch['image']).item()
        psnr_value = psnr(output, val_batch['image']).item()
        
        val_ssim.append(ssim_value)
        val_psnr.append(psnr_value)

        eval_bar.set_postfix(val_loss=val_loss / (val_batch_idx + 1), ssim=val_ssim[-1], psnr=val_psnr[-1])
        eval_bar.update()
        
        if max_batchs is not None and val_batch_idx >= max_batchs - 1:
            break
        
    eval_bar.close()
    
    avg_val_loss = val_loss / len(valloader)
    avg_val_ssim = np.mean(val_ssim)
    avg_val_psnr = np.mean(val_psnr)
    tqdm.write(f"\033[1;34m[LOG]\033[0m Validation Loss: {avg_val_loss:.4f}, SSIM: {avg_val_ssim:.4f}, PSNR: {avg_val_psnr:.4f}")
    
    model.train()
    
    return {'loss': avg_val_loss, 'ssim': avg_val_ssim, 'psnr': avg_val_psnr}
            


def load_model(config, state_dict_root, device='cpu'):
    # 加载模型
    model = instantiate_from_config(config)
    # 如果传入文件夹，找到该文件夹中最新的.pth文件
    if os.path.isdir(state_dict_root):
        state_dict_root = sorted(
            [os.path.join(state_dict_root, f) for f in os.listdir(state_dict_root) if f.endswith(".pth")],
            key=lambda x: int(x.split("/")[-1].split(".")[0]),
        )[-1]
    # 加载模型参数
    state_dict = torch.load(state_dict_root)
    model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    return model

if __name__ == "__main__":
    main()