
# MTT-Net
用MTT-Net实现CT to MRI


## 方法来源：
[MTT-Net: Multi-scale Tokens-Aware Transformer Network for Multi-region and Multi-sequence MR-to-CT Synthesis in A Single Model](https://github.com/SMU-MedicalVision/MTT-Net)

## 训练

### 加载数据集

下载SynthRAD2023比赛的Task1数据集

解压到`config.yaml`中train_root中的路径下，或自己配置

  注意，我使用的数据的mri经过了N4偏置场校正（可以不用，有小幅度提升）

  你可以参考使用项目中`preprocess/data_N4BFC.py`进行配准，非常耗时

  注意对应改`config.yaml`中的路径

### 下载预训练模型

你可以参考使用项目中`preprocess/pretrainedmodel_downloader.ipynb`下载使用的预训练模型

  注意对应改`config.yaml`中的路径

### 运行

```bash
# 默认
python solver.py

# 自定义配置文件路径
python ./solver.py --config ./config.yaml
```

### 检测
```bash
# 终端运行
tensorboard --logdir /home/featurize/data/output/MTT-Net/tensorboard --bind-all

# 开新终端，暴露端口
featurize port export 6006
# 会给你服务器转发的网址

# 结束时，记得取消暴露
featurize port unexport 6006  
```
