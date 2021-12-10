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
|ESPNet|120k|adam|1024x512|4|CityScapes|32G|4|0.6365|[espnet_cityscapes_1024x512_120k.yml](configs/espnetv1/espnetv1_cityscapes_1024x512_120k.yml)|

## 3 数据集
[CityScapes dataset](https://www.cityscapes-dataset.com/)

- 数据集大小:
    - 训练集: 2975
    - 验证集: 500

## 4 环境依赖
- 硬件: Tesla V100 * 4

- 框架:
    - PaddlePaddle == develop


## 5 快速开始

### 第一步：克隆本项目
```bash
# clone this repo
git clone https://github.com/simuler/ESPNet.git
cd ESPNet
```

**安装第三方库**
```bash
pip install -r requirements.txt
```

### 第二步：计算交叉熵损失的权重
运行compute_classweight.py文件，注意修改文件内的数据路径，将运行打印的输出结果作为配置文件的损失函数权重。
配置文件中已经放置了计算过的损失函数权重，无需再次计算

### 第三步：训练模型
单卡训练：
```bash
python train.py --config configs/espnetv1/espnetv1_cityscapes_1024x512_120k.yml  --do_eval --use_vdl --log_iter 100 --save_interval 1000 --save_dir output
```
多卡训练：
```bash
python -m paddle.distributed.launch train.py --config configs/espnetv1/espnetv1_cityscapes_1024x512_120k.yml  --do_eval --use_vdl --log_iter 100 --save_interval 1000 --save_dir output
```

### 第四步：测试
output目录下包含已经训练好的模型参数以及对应的日志文件。
```bash
python val.py --config configs/espnetv1/espnetv1_cityscapes_1024x512_120k.yml --model_path output/best_model/model.pdparams
```

## 6 代码结构与说明
**代码结构**
```
├─configs                          
├─log                         
├─output                           
├─paddleseg
├─tools                                               
│  export.py                     
│  predict.py                        
│  README.md                        
│  compute_classweight.py                    
│  requirements.txt                      
│  setup.py                   
│  train.py                
│  val.py                       
```
**说明**
1、本项目在Aistudio平台，使用Tesla V100 * 4 脚本任务训练120K miou达到63.65%。
2、本项目基于PaddleSeg开发。

## 7 模型信息

相关信息:

| 信息 | 描述 |
| --- | --- |
| 作者 | 宁文彬、郎督|
| 日期 | 2021年11月 |
| 框架版本 | PaddlePaddle==2.2.0 |
| 应用场景 | 语义分割 |
| 硬件支持 | GPU、CPU |
| 在线体验 | [notebook](https://aistudio.baidu.com/aistudio/projectdetail/3193232?contributionType=1), [Script](https://aistudio.baidu.com/aistudio/clusterprojectdetail/3193362)|
