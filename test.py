import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
from resnet import resnet18
import time
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

    with torch.no_grad():
        start_time = time.time()
        i=0
        for inputs, targets in testloader:
            if i > 10:
                break
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = net(inputs)
            _, pred = outputs.topk(5, 1, largest=True, sorted=True)  # (batch, 5)
            total += targets.size(0)
            correct = pred.eq(targets.view(-1, 1).expand_as(pred))
            correct_top1 += correct[:, :1].sum().item()
            correct_top5 += correct.sum().item()
            i += 1
        end_time = time.time()
    top1_acc = correct_top1 / total
    top5_acc = correct_top5 / total

    print('Test set: Top-1 Acc: {:.4f}, Top-5 Acc: {:.4f}, Top-1 Error: {:.4f}, Top-5 Error: {:.4f}'.format(
        top1_acc, top5_acc, 1-top1_acc, 1-top5_acc))
    print('Total Params: {}'.format(sum(p.numel() for p in net.parameters())))
    print(end_time - start_time)

if __name__ == '__main__':
    main()