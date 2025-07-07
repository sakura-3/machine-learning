import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

def main():
    # ===== 1. 配置参数 =====
    BATCH_SIZE = 128
    EPOCHS = 200
    LEARNING_RATE = 0.1
    MILESTONES = [60, 120, 160]
    GAMMA = 0.2
    CHECKPOINT_DIR = './'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ===== 2. 定义数据加载 =====
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],  # CIFAR-100 官方均值
            std=[0.2675, 0.2565, 0.2761])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761])
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # ===== 3. 定义网络=====
    from resnet import resnet18  

    net = resnet18().to(DEVICE)

    # ===== 4. 损失函数和优化器 =====
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=GAMMA)

    # ===== 5. 训练和评估函数 =====
    def train(epoch):
        net.train()
        total_loss = 0
        correct = 0
        total = 0
        start_time = time.time()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Epoch {:3d} | Train Loss: {:.4f} | Train Acc: {:.4f} | Time: {:.2f}s'.format(
            epoch, total_loss/total, correct/total, time.time()-start_time))

    def test(epoch):
        net.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * targets.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        acc = correct/total
        print('Epoch {:3d} | Test  Loss: {:.4f} | Test  Acc: {:.4f}'.format(
            epoch, total_loss/total, acc))
        return acc

    # ===== 6. 主训练循环 =====
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    if __name__ == '__main__':
        best_acc = 0.0
        for epoch in range(1, EPOCHS+1):
            train(epoch)
            acc = test(epoch)
            scheduler.step()

            # 保存最佳模型
            if acc > best_acc:
                print('Saving best model ...')
                torch.save(net.state_dict(), os.path.join(CHECKPOINT_DIR, 'resnet18-cifar100.pth'))
                best_acc = acc
if __name__ == '__main__':
    main()