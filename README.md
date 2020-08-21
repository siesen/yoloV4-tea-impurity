# YOLOV4：基于yoloV4的茶叶杂质检测
---



### 目录
1. [所需环境 Environment](#所需环境)
2. [注意事项 Attention](#注意事项)
3. [小技巧的设置 TricksSet](#小技巧的设置)
4. [文件下载 Download](#文件下载)
5. [训练步骤 How2train](#训练步骤)
6. [参考资料 Reference](#Reference)

### 概述
检测茶叶中的杂质，并利用混淆矩阵计算出精确率和误判率

目标检测YOLOV4框架源码源自https://github.com/bubbliiiing/yolov4-tf2
yoloV4源码详解UP主B站有视频

视频地址：https://www.bilibili.com/video/BV1yK411J7Zc

### 所需环境
python==3.7
tensorflow-gpu==2.2.0
opencv==4.2.0

### 注意事项

**代码有改进和提高的地方请告诉我哈，感谢赐教!**  

### 小技巧的设置
在train.py和train_eager.py文件下：   
1、mosaic参数可用于控制是否实现Mosaic数据增强。   
2、Cosine_scheduler可用于控制是否使用学习率余弦退火衰减。   
3、label_smoothing可用于控制是否Label Smoothing平滑。  

在train_eager.py文件下：   
1、regularization参数可用于控制是否实现正则化损失。  

### 文件下载
相关文件链接在 文件链接.txt中
init-loss21.h5是本案例检测茶叶杂质的权重
yolo4_weights.h5是coco数据集的权重。  
yolo4_voc_weights.h5是voc数据集的权重。
原始图片文件夹存放待训练的原始图片
Annotations文件夹存放labelimg生成的xml文件
img2，img3文件夹存放压缩后的图片
img_detected_with2,3maxpooling文件夹存放预测后的图片

### 训练步骤
1、本文使用VOC格式进行训练。  
2、训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。  
3、训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。  
4、在训练前利用voc2yolo3.py文件生成对应的txt。  
5、再运行根目录下的voc_annotation.py，运行前需要将classes改成你自己的classes。  
```python
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
```
6、就会生成对应的2007_train.txt，每一行对应其图片位置及其真实框的位置。  
7、在训练前需要修改model_data里面的voc_classes.txt文件，需要将classes改成你自己的classes。  
8、运行train.py即可开始训练。
训练过程可以参考原UP主

### Reference
https://github.com/bubbliiiing/yolov4-tf2
https://github.com/qqwweee/keras-yolo3/  
https://github.com/Cartucho/mAP  
https://github.com/Ma-Dan/keras-yolo4  
# -yoloV4-
