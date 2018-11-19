# 人力 Tensorflow

* 请运行`main.py`
* 请在`main.py`中编写网络，并优化
* layers 内含网络层
* model 内含模型broker
* optimizer 内含优化器


# 简介
本框架实现了如下内容

1. 网络层
    * - [x] Layer基类
    * - [x] Input层
    * - [x] Dense层
    * - [x] Softmax层
    * - [x] CrossEntropy Loss层
    * - [x] Flatten层
    * - [x] ExponentialLinearUnit层
    * - [x] Conv2D层
1. Sequential模型通用容器
    * - [x] Model类
1. Optimizer
    * - [x] Optimizer基类
    * - [x] SGDOptimizer
    * - [x] MomentumOptimizer
    * - [x] BatchMomentumOptimizer
1. 数据处理工具



#命令格式

### 在训练集和测试集上计算loss和精度

    python main.py --phase evaluate

### 将某个文件夹中的图片进行inference

    python main.py --phase inference --dataset dataset/test
结果会被写到`out.txt`文件中。当输入图片少于20张时，会把结果打到屏幕上

### 对mnist 的测试集进行inference

    python main.py --phase inference --dataset mnist
或

    python main.py --phase inference 

结果会被写到`out.txt`文件中。当输入图片少于20张时，会把结果打到屏幕上

### 展示模型结构

    python main.py --show 