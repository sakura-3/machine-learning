# machine-learning

# usage

## create virtual environment(recommend)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## install dependencies
```bash
pip3 install -r requirements.txt
```

## update dependencies
```bash
pip3 freeze > requirements.txt
```

## run
```bash
python3 -u main.py
```

# weights

`resnet18.npy` 来自[pytorch](https://download.pytorch.org/models/resnet18-f37072fd.pth),为了避免引入`pytorch`依赖,我们将其转为`.npy` 格式

# 分界

## resnet_np.py
np实现的resnet18模型，并包含了测试
## resnet.py
pytorch实现的resnet18模型
## 数据集
Cifar100，如需运行测试加载数据集时设置download=True
## resnet18-cifar100.pth与resnet18_cifar100.npy
分别对应pytorch与np实现的模型的权重文件(同为训练200个epoch情况下在验证集上取得最好效果的权重，只是格式不一样，仅此而已)
## test.py与train.py
pytorch实现模型对应的测试文件与训练文件
## weight_trans.py
即resnet18-cifar100.pth -> resnet18_cifar100.npy

## 测试结果对比
             acc-1     acc-5        参数量
pytorch：    0.7643     0.9349      11220132
np：         0.7875     0.9375      11229752

二者本地测试推理运行时间相差大几十甚至上百倍。（pytorch框架推理速度远快于np框架推理速度）