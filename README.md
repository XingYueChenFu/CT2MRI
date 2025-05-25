# CT2MRI-LDM

从`https://github.com/CompVis/latent-diffusion`修改而来。

基本照搬的

## 训练

### 下载&处理数据

#### 汇总
```bash
# ===== 下载数据 =====
mkdir data
cd data
featurize dataset download 5df2ea90-3abe-41e8-9a78-9af7bc77feed # 下载数据
7z x nii.zip -mmt # 7z多线程解压
rm nii.zip # 删除压缩包

# ===== 处理数据 =====
cd ../work/CT2MRI-LDM
conda activate
python ./data/Liu_preprocess.py
rm ../../data/nii -r # 删除原始数据

# ===== 划分数据集 =====
python ./data/Liu_split.py
# rm ../../data/Task1_Liu -r # 删除未划分的数据

# ===== 复制并解压预训练模型 =====
cd ../..
cp ./work/output.7z ./
7z x output.7z
rm output.7z

# 进入工作目录
cd work/CT2MRI-LDM
```

#### 分步

> 执行了**汇总**后，不要再分步执行

下载
```bash
mkdir data
cd data
featurize dataset download 5df2ea90-3abe-41e8-9a78-9af7bc77feed # 下载数据
7z x nii.zip -mmt # 7z多线程解压
rm nii.zip # 删除压缩包
```
处理1：预处理数据
```bash
cd ../work/CT2MRI-LDM
conda activate
python ./data/Liu_preprocess.py
rm ../../data/nii -r # 删除原始数据
```
处理2：划分数据集
```bash
python ./data/Liu_split.py
# rm ../../data/Task1_Liu -r # 删除未划分的数据
```

解压：预训练的VAE
```bash
cd ../..
cp ./work/output.7z ./
7z x output.7z
rm output.7z
```

### 创建虚拟环境
配置参考：
[CSDN中博客](https://blog.csdn.net/qq_42940160/article/details/131284998)

```bash
conda env create -f environment.yaml # 比较耗时 尤其是pip安装各个包时
conda activate ldm

pip install transformers==4.19.2 scann kornia==0.6.4 torchmetrics==0.6.0
pip install git+https://github.com/arogozhnikov/einops.git # 这步可能报错 我使用的：pip install einops

```

### 训练
配置过程。略

弃用：
```bash
python main.py --base configs/ct2mri.yaml --train --gpus 0 # 训练autoencoder

python main.py --base configs/ct2mri_ldm.yaml --train
```

#### 1. 预训练 VQ-VAE （提取MRI的特征）
```bash
python solver.py --base configs/ct2mri_pretrain_vae.yaml --train
```

#### 2. 预训练 VQ-VAE （提取CT的特征）
```bash
python solver2.py --base configs/ct2mri_pretrain_vae.yaml --train
```

#### 3. 训练LDM
```bash
python solver3.py --base configs/ct2mri_ldm.yaml --train
```
TODO



快出现吧【天选 の 随机种子】🎯