# PRLNet: A Two-Stage Lightweight Framework for Accurate Human Pose Estimation

The official code repository for the paper "PRLNet: A Two-Stage Lightweight Framework for Accurate Human Pose Estimation".
![PRLNet.png](PRLNet.jpg)

## Environment：
* Python3.7
* Pytorch1.10.1
* It is best to use GPU training
* For detailed environment configuration, see 'requirements.txt'(pip install -r requirements.txt)
 
## Data preparation：
* Official website of COCO dataset：https://cocodataset.org/
* You need to download three files of the coco2017 dataset：
    * 2017 Train images [118K/18GB]
    * 2017 Val images [5K/1GB]
    * 2017 Train/Val annotations [241MB]
* Unzip them all to the 'coco2017' folder, and you will get the following folder structure：

├── coco2017
     ├── train2017
     ├── val2017
     └── annotations
              ├── instances_train2017.json
              ├── instances_val2017.json
              ├── captions_train2017.json
              ├── captions_val2017.json
              ├── person_keypoints_train2017.json
              └── person_keypoints_val2017.json

## Training Methods：
* 确保提前准备好数据集
* 确保提前下载好对应预训练模型权重
* 若要使用单GPU训练直接使用train.py训练脚本
* 若要使用多GPU训练，使用`torchrun --nproc_per_node=8 train_multi_GPU.py`指令,`nproc_per_node`参数为使用GPU数量


## 注意事项
1. 在使用训练脚本时，注意要将`--data-path`设置为自己存放数据集的**根目录**：
假设要使用COCO数据集，启用自定义数据集读取CocoDetection并将数据集解压到成/data/coco2017目录下
```
python train.py --data-path /data/coco2017
```
2. 训练过程中保存的`results.txt`是每个epoch在验证集上的COCO指标，前10个值是COCO指标，后面两个值是训练平均损失以及学习率
3. 在使用预测脚本时，如果要读取自己训练好的权重要将`weights_path`设置为你自己生成的权重路径。

## This project mainly refers to the following code base：
* https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_keypoint/HRNet. Thank you very much for his contribution！
