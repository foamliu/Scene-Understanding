# 语义分割

用SegNet进行室内语义分割。

## 依赖
- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
- [Keras](https://keras.io/#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)

## 数据集

![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/dataset.png)

按照 [说明](http://3dvision.princeton.edu/projects/2015/SUNrgbd/) 下载 SUN RGB-D 数据集，放在 data 目录内。

```bash
$ wget http://3dvision.princeton.edu/projects/2015/SUNrgbd/data/SUNRGBD.zip
$ wget http://3dvision.princeton.edu/projects/2015/SUNrgbd/data/SUNRGBDtoolbox.zip
```

## 架构

![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/segnet.png)


## ImageNet 预训练模型
下载 [VGG16](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5) 放在 models 目录内。

## 用法
### 数据预处理
该数据集包含SUNRGBD V1的10335个RGBD图像，执行下述命令提取训练图像：
```bash
$ python pre-process.py
```

### 训练
```bash
$ python train.py
```

如果想可视化训练过程，可执行：
```bash
$ tensorboard --logdir path_to_current_dir/logs
```

### 演示

```bash
$ python demo.py
```

图例
![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/legend.png)

输入 | 真实 | 输出 |
|---|---|---|
|![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/0_image.png)  | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/0_gt.png) | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/0_out.png)|
|![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/1_image.png)  | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/1_gt.png) | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/1_out.png)|
|![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/2_image.png)  | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/2_gt.png) | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/2_out.png)|
|![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/3_image.png)  | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/3_gt.png) | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/3_out.png)|
|![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/4_image.png)  | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/4_gt.png) | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/4_out.png)|
|![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/5_image.png)  | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/5_gt.png) | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/5_out.png)|
|![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/6_image.png)  | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/6_gt.png) | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/6_out.png)|
|![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/7_image.png)  | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/7_gt.png) | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/7_out.png)|
|![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/8_image.png)  | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/8_gt.png) | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/8_out.png)|
|![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/9_image.png)  | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/9_gt.png) | ![image](https://github.com/foamliu/Semantic-Segmentation/raw/master/images/9_out.png)|

