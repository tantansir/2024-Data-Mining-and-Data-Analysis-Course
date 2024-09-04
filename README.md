report是整个项目的报告
reference documents是参考文献
members是小组成员

下面是对experiment文件夹的解释说明：

experiment是实验文件夹，但是删除了预训练模型、数据集和训练记录（训练记录有30个g）

项目使用模型库为：https://github.com/Fafa-DL/Awesome-Backbones?tab=readme-ov-file
详细操作请参考链接
项目使用数据集为CNFOOD241：https://data.mendeley.com/datasets/fspyss5zbb/1

环境配置：
按照requirements.txt安装

简易实验复现流程：
1.下载数据集，编写annotations.txt。编辑tools/split_data.py中的数据集路径，输入python tools/split_data.py运行。
在datasets中得到训练验证比为8：2的数据集。
2.数据集信息文件制作，输入python tools/get_annotation.py运行得到数据集信息文件train.txt与test.txt。
3.修改对应模型参数，在models中选择对应模型进行修改
预训练模型链接：
Resnet：https://download.openmmlab.com/mmclassification/v0/resnet/resnet152_8xb32_in1k_20210901-4d7582fa.pth
Densenet：https://download.openmmlab.com/mmclassification/v0/densenet/densenet201_4xb256_in1k_20220426-05cae4ef.pth
Vision Transformer：https://download.openmmlab.com/mmclassification/v0/vit/finetune/vit-base-p32_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-9cea8599.pth
4.训练，以resnet为例，输入python tools/train.py models/resnet/resnet152.py进行训练