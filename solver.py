"""
预训练VQ-VAE模型，提取MRI图像的潜在表示

"""
import os, sys, argparse, datetime

import json
import numpy as np
from tqdm import tqdm

from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader

from ldm.util import instantiate_from_config
from data.dataset import CT2MRIDataset
# SSIM 与 PSNR
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics import PeakSignalNoiseRatio as PSNR

def print_device_info():
    import psutil
    import GPUtil
    print("\n=== 设备使用情况 ===")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.current_device()}")
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 打印CPU和内存使用情况
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    print(f"\nCPU使用率: {cpu_percent}%")
    print(f"内存使用: {memory.used/1024/1024/1024:.2f}GB / {memory.total/1024/1024/1024:.2f}GB")
    
    # 打印GPU使用情况
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"\nGPU {gpu.id} {gpu.name}:")
            print(f"  内存使用: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
            print(f"  GPU利用率: {gpu.load*100}%")
    except Exception as e:
        print(f"无法获取GPU使用情况: {e}")

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    return parser

def main():
    debug = 1
    
    print_device_info()
    
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())
    
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    # Load configs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cli = OmegaConf.from_dotlist(unknown)
    configs = [OmegaConf.load(cfg) for cfg in list(opt.base)]
    config = OmegaConf.merge(*configs, cli)
    
    # model
    if debug:
        print('\033[1;34m[DEBUG]\033[0m Loading Model...')
    model = instantiate_from_config(config.model)
    model = model.to(device)
    if debug:
        print('\033[1;34m[DEBUG]\033[0m Model Loaded')
    
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
    
    # 优化器
    opt_ae = torch.optim.AdamW(model.parameters(), lr=4.5e-4, weight_decay=1e-4)
    opt_disc = torch.optim.AdamW(model.loss.discriminator.parameters(), lr=4.5e-4, weight_decay=1e-4)
    
    # 启用 EMA（可选）
    if model.use_ema:
        model.model_ema.to(device)
        
    # 评价指标
    ssim = SSIM(data_range=1.0).to(device)
    psnr = PSNR(data_range=1.0).to(device)
    best_ssim = 0.0
    
    # # 尝试输入一组数据，看看模型是否能正常运行
    # if debug:
    #     print('\033[1;34m[DEBUG]\033[0m Testing Model with Dummy Data...')
    # for batch in trainloader:
    #     ct, mri = batch["image"], batch["condition"]
    #     mri = mri.to(model.device)
        
        
    #     break
    # if debug:
    #     print('\033[1;34m[DEBUG]\033[0m Model Test Completed')
    
    # 保存数据的位置
    output_root = '/home/featurize/output/VQ-VAE-MRI'
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
    
    # 训练
    global_step = 0
    max_steps = 10000
    
    realmri_sample = None
    recomri_sample = None
    
    step_bar = tqdm(range(global_step, max_steps), desc="step", leave=True, position=0)
    for epoch in range(global_step, (max_steps-1) // len(trainloader) + 1):
        train_epoch_ssims = []
        train_epoch_psnrs = []
        eval_epoch_ssims = []
        eval_epoch_psnrs = []
        
        # ===== 训练逻辑 ===== 
        for i, batch in enumerate(trainloader):
            # ct = batch["image"]
            mri = batch["condition"]
            x = mri.to(model.device)
            
            
            # ================================
            # Step 1: Train Autoencoder
            # ================================
            opt_ae.zero_grad()

            # 前向传播
            xrec, qloss = model.forward(x)  # 返回 xrec: [4, 3, 256, 256] 与 qloss: 标量
            # 计算总损失
            aeloss, log_dict_ae = model.loss(
                codebook_loss=qloss,       # 新增参数
                inputs=x,
                reconstructions=xrec,
                optimizer_idx=0,           # 生成器模式
                global_step=global_step,
                last_layer=model.get_last_layer(),
                split="train"
            )

            # 反向传播 + 优化
            aeloss.backward()
            opt_ae.step()

            # ================================
            # Step 2: Train Discriminator
            # ================================
            opt_disc.zero_grad()

            # 再次前向传播（需要时重新计算）
            with torch.no_grad():
                xrec, qloss = model(x)

            # 计算判别器损失
            discloss, log_dict_disc = model.loss(
                codebook_loss=qloss.detach(),  # 判别器不需要梯度
                inputs=x,
                reconstructions=xrec,
                optimizer_idx=1,           # 判别器模式
                global_step=global_step,
                last_layer=None,           # 判别器不涉及生成器的最后一层
                split="train"
            )
            # 反向传播 + 优化
            discloss.backward()
            opt_disc.step()

            # ================================
            # Step 3: EMA update and ...
            # ================================
            
            # 更新 EMA（如果启用）
            if model.use_ema:
                model.model_ema(model)

            train_epoch_ssims.append(ssim(xrec, x).item())
            train_epoch_psnrs.append(psnr(xrec, x).item())
            step_bar.set_postfix(
                aeloss=aeloss.item(),
                discloss=discloss.item(),
                ssim=train_epoch_ssims[-1],
                psnr=train_epoch_psnrs[-1],
            )
            global_step += 1
            step_bar.update(1)
            
            # 控制最大训练步数
            if global_step >= max_steps:
                break
        train_ssim = torch.mean(torch.tensor(train_epoch_ssims)).item()
        train_psnr = torch.mean(torch.tensor(train_epoch_psnrs)).item()
        
        # ===== 测试逻辑 ===== 
        eval_bar = tqdm(range(len(valloader)), desc="eval", leave=True, position=1)
        for i, batch in enumerate(valloader):
            # ct = batch["image"]
            mri = batch["condition"]
            x = mri.to(model.device)
            
            with torch.no_grad():
                xrec, qloss = model(x)
                ssim_score = ssim(xrec, x).item()
                psnr_score = psnr(xrec, x).item()
                
            eval_epoch_ssims.append(ssim_score)
            eval_epoch_psnrs.append(psnr_score)
            eval_bar.set_postfix(
                ssim=eval_epoch_ssims[-1],
                psnr=eval_epoch_psnrs[-1],
            )
            eval_bar.update(1)
            
            if i == 0: # Log用
                realmri_sample = x.cpu().numpy()
                recomri_sample = xrec.cpu().numpy()
            
        eval_bar.close()
        
        eval_ssim = torch.mean(torch.tensor(eval_epoch_ssims)).item()
        eval_psnr = torch.mean(torch.tensor(eval_epoch_psnrs)).item()
        
        # ===== Log逻辑 =====
        # 1. 保存日志
        log_dict = {
            "train/ssim": train_ssim,
            "train/psnr": train_psnr,
            "eval/ssim": eval_ssim,
            "eval/psnr": eval_psnr,
        }
        # 以{global_step}.json保存
        step_log_path = os.path.join(log_path, f"{global_step}.json")
        with open(step_log_path, "w") as f:
            json.dump(log_dict, f)
        
        # 2. 保存模型 & 样本
        # 保存eval_ssim最高的model与最近一次model，分别保存到best和last文件夹
        # 注意需要删除之前的模型        
        best_path = os.path.join(best_model_root, f"{global_step}.pth")
        last_path = os.path.join(last_model_root, f"{global_step}.pth")
        if eval_ssim > best_ssim:
            best_ssim = eval_ssim
            # 删除之前的模型
            for file in os.listdir(best_model_root):
                os.remove(os.path.join(best_model_root, file))
            # 保存当前模型
            torch.save(model.state_dict(), best_path)
            
            # 保存原图与重建图{global_step}_realMRI与{}global_step}_reconMRI
            np.save(os.path.join(sample_path, f"{global_step}_realMRI.npy"), realmri_sample)
            np.save(os.path.join(sample_path, f"{global_step}_reconMRI.npy"), recomri_sample)
            
            
        # 删除之前的模型
        for file in os.listdir(last_model_root):
            os.remove(os.path.join(last_model_root, file))
        # 保存当前模型
        torch.save(model.state_dict(), last_path)
                
        if global_step >= max_steps:
            break
    step_bar.close()
    
if __name__ == "__main__":
    main()