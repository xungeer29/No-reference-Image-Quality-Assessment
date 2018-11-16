# 无参考图像质量评价

## RankIQA
### 简介

参考[RankIQA](https://github.com/xialeiliu/RankIQA)中的方法，将回归问题转化为分类+回归的
问题，对高质量图像进行distortion，生成大量不同等级的图像，先训练一个图像质量rank的网络，
网络结构为Siamese，Siamese内为VGG；在训练完成后，在其基础上进行fine-tuning，使用公开集
数据(大约为2000多张)微调该网络，损失函数为L2，回归图像的具体质量分数。

该方法可以突破无参考图像质量评价的小数据集限制，可以使用大网络来训练而不产生过拟合。

![Models](./figs/models.png)

### 文件说明
* ./data: 

