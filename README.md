# Seed_Cup_TextCNN
### 文档说明
`data/`:存放比赛的所有数据

`model/`:存放神经网络的模型文件

`config.py`:配置文件

`class_idx.py`:本次比赛的类别是离散程度很大的值，class_idx中class_idx类可以实现由把目标类映射到输出空间或者把输出还原为目标类

`word_idx.py`:给所有的词生成onehot形式的编号

`accurancy.py`:其中的F1类通过save_data保存batch中生成的数据，caculate_f1计算f1值

`Dataloader.py`:加载文件

`main.py`:包含train()和val()

`test.py`:加载模型并生成提交文件

### 效果说明
测试集输出取得了0.8604的分数

f1_cate1: 0.9577

f1_cate2: 0.8884

f1_cate3: 0.8302
### 使用说明
在本目录下：`python main.py`即可开始训练

生成模型之后，用`python test.py`即可生成对应的提交文件
### 编写环境
Ubuntu 18.04

python:3.7.6

cuda:9.0.176

torch 0.4.1
