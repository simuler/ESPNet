English | [简体中文](README_CN.md)

# ESPNet

## 1 简介
本项目基于paddlepaddle框架复现了ESPNet语义分割模型，该论文作者利用卷积因子分解原理设计了非常精巧的EESP模块，并基于次提出了一个轻量级、效率高的通用卷积神经网络模型ESPNet，能够大大减少模型的参数并且保持模型的性能。

### 论文:
[1] Mehta S ,  Rastegari M ,  Caspi A , et al. ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation

### 项目参考：
https://github.com/sacmehta/ESPNet

## 2 复现精度
>在CityScapes val数据集的测试效果如下表。


| |steps|opt|image_size|batch_size|dataset|memory|card|mIou|config|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|ESPNet|120k|adam|1024x512|8|CityScapes|32G|4|0.6417|[espnet_cityscapes_1024_512_120k.yml](configs/espnet_cityscapes_1024_512_120k.yml)|

## 3 数据集
[CityScapes dataset](https://www.cityscapes-dataset.com/)

- 数据集大小:
    - 训练集: 2975
    - 验证集: 500

## 4 环境依赖
- 硬件: Tesla V100 * 4

- 框架:
    - PaddlePaddle == develop


