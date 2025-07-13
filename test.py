import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
from resnet import resnet18
import time
import numpy as np
from sklearn.metrics import confusion_matrix,precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
def main():
    # ===== 1. 命令行参数 =====
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--data_dir', type=str, default='./data', help='CIFAR-100 dataset directory')
    args = parser.parse_args()

    DEVICE = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'

    # ===== 2. 数据加载 =====
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])
    testset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # ===== 3. 网络定义与加载权重 =====
    net = resnet18().to(DEVICE)
    net.load_state_dict(torch.load(args.weights, map_location=DEVICE))
    net.eval()

    # ===== 4. 测试 =====
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        start_time = time.time()
        for inputs, targets in testloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = net(inputs)
            _, pred = outputs.topk(5, 1, largest=True, sorted=True)  # (batch, 5)
            total += targets.size(0)
            correct = pred.eq(targets.view(-1, 1).expand_as(pred))
            correct_top1 += correct[:, :1].sum().item()
            correct_top5 += correct.sum().item()
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(pred[:, :1].cpu().numpy())
        end_time = time.time()
    top1_acc = correct_top1 / total
    top5_acc = correct_top5 / total

    print('------Weighted------')
    print('Weighted precision:{:.4f}'.format(precision_score(all_targets, all_preds, average='weighted')))
    print('Weighted recall:{:.4f}'.format(recall_score(all_targets, all_preds, average='weighted')))
    print('Weighted f1-score:{:.4f}'.format(f1_score(all_targets, all_preds, average='weighted')))
    print('------Macro------')
    print('Macro precision:{:.4f}'.format(precision_score(all_targets, all_preds, average='macro')))
    print('Macro recall:{:.4f}'.format(recall_score(all_targets, all_preds, average='macro')))
    print('Macro f1-score:{:.4f}'.format(f1_score(all_targets, all_preds, average='macro')))
    print('------Micro------')
    print('Micro precision:{:.4f}'.format(precision_score(all_targets, all_preds, average='micro')))
    print('Micro recall:{:.4f}'.format(recall_score(all_targets, all_preds, average='micro')))
    print('Micro f1-score:{:.4f}'.format(f1_score(all_targets, all_preds, average='micro')))

    cm = confusion_matrix(np.array(all_targets),np.array(all_preds))

    class_names = testloader.dataset.classes

    # 绘制归一化混淆矩阵
    plot_confusion_matrix(cm, class_names, normalize=True, title='Normalized Confusion Matrix')
    plt.savefig('normalized_confusion_matrix_np.png', dpi=300, bbox_inches='tight')
    plt.show()

    print('Test set: Top-1 Acc: {:.4f}, Top-5 Acc: {:.4f}, Top-1 Error: {:.4f}, Top-5 Error: {:.4f}'.format(
        top1_acc, top5_acc, 1-top1_acc, 1-top5_acc))
    print('Total Params: {}'.format(sum(p.numel() for p in net.parameters())))
    print(end_time - start_time)

def plot_confusion_matrix(cm, class_names=None, figsize=(10, 8), 
                          normalize=False, title='Confusion Matrix', 
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized ' + title
    else:
        fmt = 'd'
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    
    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax)
    
    # 设置坐标轴
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names if class_names else np.arange(cm.shape[1]),
           yticklabels=class_names if class_names else np.arange(cm.shape[0]),
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    
    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    fig.tight_layout()
    return fig, ax

if __name__ == '__main__':
    main()