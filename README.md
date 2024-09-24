# Synapse
一个深度学习框架。以Restormer实现低光图像增强为例。

## 1. 框架规范
### 1.1 编码
所有文本文件需使用无BOM的UTF-8编码。

### 1.2 文件组织
项目目录下有若干目录和文件，功能如下：

#### 1.2.1 arch
arch目录中存放实现神经网络的代码。例如，实现 Restormer 网络的代码应被放在arch/Restormer 目录下。

#### 1.2.2 config
config目录中存放训练参数json文件。

#### 1.2.3 data
data目录中存放实现DataSet、DataLoader等训练时数据集操作相关的代码，不包含训练前数据集预处理代码。

#### 1.2.4 loss
loss目录中存放实现损失函数的代码，所有loss类需继承```LossWithWeights```类。

#### 1.2.5 metric
metric目录中存放实现评估指标的代码。

#### 1.2.6 trainer
trainer目录中存放实现训练循环相关的代码，所有trainer类需继承```BaseTrainer```类，并且其类名和文件名需以“Trainer”结尾。

#### 1.2.7 util
util目录中存放在data目录、loss目录、trainer目录和script目录中可能会使用的公共函数和公共类。

#### 1.2.8 script
script目录中存放与训练无关的代码，例如数据集预处理、邮件自动化、原型实验代码等。

#### 1.2.9 experiments
experiments目录中存放训练记录目录和script目录中代码可能的输出文件，会在训练开始时尝试建立。

#### 1.2.10 train.py
train.py为训练代码，用以启动训练，运行下列命令以启动训练：
```
python3 train.py --config config_path --phase train
```
```phase```参数可省略，默认为train，也可设置为其他模式。

#### 1.2.11 test.py
test.py为利用预训练权重计算并输出的代码，用以推理，按任务需求设置输入参数和实现。

### 1.3 神经网络实现
所有神经网络的实现类必须实现```train_forward```函数和```test_forward```函数，前者用以训练，后者用以推理。

### 1.4 script目录组织
script目录下至少应有如下目录，也可按需新增目录：
1. util：存放script目录代码中可能用到的公共函数和公共类，公开；
2. mail：存放邮件自动化相关代码，不公开；
3. data_preprocess：存放数据集预处理代码，公开；
4. prototype：存放原型实验代码，不公开；

### 1.5 train.py phase参数
```phase```参数可省略，默认为train，不同参数选择效果如下：
1. train：正常训练；
2. debug：将训练参数json文件中```train.val_iter```重载为20；
3. profile：启用PyTorch的性能分析模式；
4. detect：启用PyTorch的异常检测模式；

### 1.6 训练记录目录
训练记录目录存放在experiments目录下，以```phase```参数、json文件中指定的```name```字段与训练开始时间（yyyyMMdd_HHmmss格式）以下划线“_”连接构成的字符串命名（例如“train_NaturalSRx4_ReMambaWithCAB_DF2K_Ubuntu_Charbonnier_240806_165843”）。该目录下有多个子目录和文件：
1. checkpoint：目录，存放训练时保存的权重，包括目前最优权重和一定迭代次数后保存的权重；
2. code：目录，存放训练开始时 arch、config、data、loss、trainer、utils目录和train.py、test.py的备份副本；
3. tf-logs：目录，存放TensorBoard记录文件；
4. config.json：文件，训练使用的json训练参数文件；
5. metric.csv：文件，每次验证时的指标值；
6. run.log：文件，日志文件；
