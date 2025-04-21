#-*- coding:utf-8 -*-
# +
from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, ToTensor, Lambda
from diffusion_model.trainer import GaussianDiffusion, Trainer
from diffusion_model.unet import create_model
from dataset import NiftiImageGenerator, NiftiPairImageGenerator, NpyPairImageGenerator
import argparse
import torch

import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# -

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputfolder', type=str, default="dataset/mask/")
parser.add_argument('-t', '--targetfolder', type=str, default="dataset/image/")
parser.add_argument('--input_size', type=int, default=128) # 128
parser.add_argument('--depth_size', type=int, default=32) # 128
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--num_res_blocks', type=int, default=1)
parser.add_argument('--num_class_labels', type=int, default=2) # 3
parser.add_argument('--train_lr', type=float, default=1e-5)
parser.add_argument('--batchsize', type=int, default=1)
parser.add_argument('--epochs', type=int, default=50000) # epochs parameter specifies the number of training iterations
parser.add_argument('--timesteps', type=int, default=250)
parser.add_argument('--save_and_sample_every', type=int, default=1000) # 每多少步测试并一次模型
# parser.add_argument('--with_condition', action='store_true') 
parser.add_argument('--with_condition', default=True) 
# 换用数据集
parser.add_argument('--preload', default=False) # 是否预加载全部数据到内存
parser.add_argument('--data_root', type=str, default='/home/featurize/data/Task1_Liu_split_128/train')
parser.add_argument('--input_modality', type=str, default='CT')
parser.add_argument('--target_modality', type=str, default='MR')

parser.add_argument('-r', '--resume_weight', type=str, default="") # "model/model_128.pt"
args = parser.parse_args()

inputfolder = args.inputfolder
targetfolder = args.targetfolder
input_size = args.input_size
depth_size = args.depth_size
num_channels = args.num_channels
num_res_blocks = args.num_res_blocks
num_class_labels = args.num_class_labels
save_and_sample_every = args.save_and_sample_every
with_condition = args.with_condition
resume_weight = args.resume_weight
train_lr = args.train_lr

# input tensor: (B, 1, H, W, D)  value range: [-1, 1]
transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    Lambda(lambda t: (t * 2) - 1),
    # Lambda(lambda t: t.unsqueeze(0)),
    # Lambda(lambda t: t.transpose(3, 1)),
    Lambda(lambda t: t.permute(0, 3, 1, 2)), # (c, h, w, d) -> (c, d, h, w)
])

input_transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    Lambda(lambda t: (t * 2) - 1),
    # Lambda(lambda t: t.permute(3, 0, 1, 2)),
    # Lambda(lambda t: t.transpose(3, 1)),
    Lambda(lambda t: t.permute(0, 3, 1, 2)),
])

if with_condition:
    # dataset = NiftiPairImageGenerator(
    #     inputfolder,
    #     targetfolder,
    #     input_size=input_size,
    #     depth_size=depth_size,
    #     transform=input_transform if with_condition else transform,
    #     target_transform=transform,
    #     full_channel_mask=True
    # )
    dataset = NpyPairImageGenerator(
        data_root=args.data_root,
        input_modality=args.input_modality,
        target_modality=args.target_modality,
        input_size=input_size,
        depth_size=depth_size,
        transform=input_transform if with_condition else transform,
        target_transform=transform,
        full_channel_mask=True,
        preload=args.preload
    )
    # print(f"Len(dataset): {len(dataset)}")
    print(f"\033[33m[DEBUG] dataset length: {len(dataset)}\033[0m")
    # dataset[0].shape
    print(f"\033[33mdataset[0]['input'].shape: {dataset[0]['input'].shape}\033[0m")
    print(f"\033[33mdataset[0]['target'].shape: {dataset[0]['target'].shape}\033[0m")
    
        
else:
    dataset = NiftiImageGenerator(
        inputfolder,
        input_size=input_size,
        depth_size=depth_size,
        transform=transform
    )

print(len(dataset))

in_channels = num_class_labels if with_condition else 1
out_channels = 1


model = create_model(input_size, num_channels, num_res_blocks, in_channels=in_channels, out_channels=out_channels).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = input_size,
    depth_size = depth_size,
    timesteps = args.timesteps,   # number of steps
    loss_type = 'l1',    # L1 or L2
    with_condition=with_condition,
    channels=out_channels
).cuda()

if len(resume_weight) > 0:
    weight = torch.load(resume_weight, map_location='cuda')
    diffusion.load_state_dict(weight['ema'])
    print("Model Loaded!")

trainer = Trainer(
    diffusion,
    dataset,
    image_size = input_size,
    depth_size = depth_size,
    train_batch_size = args.batchsize,
    train_lr = train_lr,
    train_num_steps = args.epochs,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = False,#True,                       # turn on mixed precision training with apex
    with_condition=with_condition,
    save_and_sample_every = save_and_sample_every,
)

trainer.train()
