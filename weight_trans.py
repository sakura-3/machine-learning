import torch
import numpy as np

pth_path = '/Users/liangwei13/BlockChain_MAS_POS/machine-learning/resnet18-cifar100.pth'
output_path = './resnet18_cifar100.npy'

state_dict = torch.load(pth_path, map_location='cpu')
# 转为numpy字典
numpy_dict = {k: v.cpu().numpy() for k, v in state_dict.items()}

np.save(output_path, numpy_dict)
print(f"Saved dict to {output_path}")