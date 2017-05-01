# Train
基于OpenCV 2.4训练SVM模型。

运行前准备
准备训练集置于根目录下，名字为train.txt。
格式为，第一个数字指明该样本为正/负样本，之后为特征向量，每一维度以空格隔开。

1/-1 <feature vector, delim space>

在train.cpp中修改训练样本个数及维度。

编译及运行：

```
cmake .
make
./Train
