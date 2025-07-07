import numpy as np
import os
import argparse
from PIL import Image
from typing import List
from tqdm import tqdm
import time
params_path = "resnet18_cifar100.npy"

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def global_avg_pool2d(x):
    return np.mean(x, axis=(2, 3))

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def conv2d(x, w, b=None, stride=1, padding=0):
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels, _, filter_height, filter_width = w.shape

    out_height = (in_height + 2 * padding - filter_height) // stride + 1
    out_width = (in_width + 2 * padding - filter_width) // stride + 1

    x_padded = np.pad(
        x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant"
    )

    out = np.zeros((batch_size, out_channels, out_height, out_width))

    for batch in range(batch_size):
        for chan in range(out_channels):
            for i in range(out_height):
                for j in range(out_width):
                    h_start = i * stride
                    h_end = h_start + filter_height
                    w_start = j * stride
                    w_end = w_start + filter_width

                    x_region = x_padded[batch, :, h_start:h_end, w_start:w_end]
                    out[batch, chan, i, j] = np.sum(x_region * w[chan])
            if b is not None:
                out[batch, chan] += b[chan]
    return out

def batch_norm(x, gamma, beta, mean, var, eps=1e-5):
    gamma = gamma.reshape(1, -1, 1, 1)
    beta = beta.reshape(1, -1, 1, 1)
    mean = mean.reshape(1, -1, 1, 1)
    var = var.reshape(1, -1, 1, 1)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

class BasicBlock:
    expansion = 1
    def __init__(self):
        pass

    def load_weights(self, params, prefix, downsample=False):
        self.conv1 = params[f"{prefix}.residual_function.0.weight"]
        self.bn1_weight = params[f"{prefix}.residual_function.1.weight"]
        self.bn1_bias = params[f"{prefix}.residual_function.1.bias"]
        self.bn1_mean = params[f"{prefix}.residual_function.1.running_mean"]
        self.bn1_var = params[f"{prefix}.residual_function.1.running_var"]

        self.conv2 = params[f"{prefix}.residual_function.3.weight"]
        self.bn2_weight = params[f"{prefix}.residual_function.4.weight"]
        self.bn2_bias = params[f"{prefix}.residual_function.4.bias"]
        self.bn2_mean = params[f"{prefix}.residual_function.4.running_mean"]
        self.bn2_var = params[f"{prefix}.residual_function.4.running_var"]

        self.downsample = downsample
        if downsample:
            self.downsample_conv = params[f"{prefix}.shortcut.0.weight"]
            self.downsample_bn_weight = params[f"{prefix}.shortcut.1.weight"]
            self.downsample_bn_bias = params[f"{prefix}.shortcut.1.bias"]
            self.downsample_bn_mean = params[f"{prefix}.shortcut.1.running_mean"]
            self.downsample_bn_var = params[f"{prefix}.shortcut.1.running_var"]

    def forward(self, x, stride=1):
        identity = x

        out = conv2d(x, self.conv1, None, stride=stride, padding=1)
        out = batch_norm(out, self.bn1_weight, self.bn1_bias, self.bn1_mean, self.bn1_var)
        out = relu(out)

        out = conv2d(out, self.conv2, None, stride=1, padding=1)
        out = batch_norm(out, self.bn2_weight, self.bn2_bias, self.bn2_mean, self.bn2_var)

        if self.downsample:
            identity = conv2d(identity, self.downsample_conv, None, stride=stride)
            identity = batch_norm(identity, self.downsample_bn_weight, self.downsample_bn_bias, self.downsample_bn_mean, self.downsample_bn_var)

        out += identity
        out = relu(out)
        return out

class ResNet18:
    def __init__(self):
        self.conv1 = None
        self.bn1_weight = None
        self.bn1_bias = None
        self.bn1_mean = None
        self.bn1_var = None
        self.layer1 = []
        self.layer2 = []
        self.layer3 = []
        self.layer4 = []
        self.fc_weight = None
        self.fc_bias = None

    def load_weights(self, params):
        self.conv1 = params["conv1.0.weight"]
        self.bn1_weight = params["conv1.1.weight"]
        self.bn1_bias = params["conv1.1.bias"]
        self.bn1_mean = params["conv1.1.running_mean"]
        self.bn1_var = params["conv1.1.running_var"]

        self.layer1 = [BasicBlock(), BasicBlock()]
        self.layer1[0].load_weights(params, "conv2_x.0")
        self.layer1[1].load_weights(params, "conv2_x.1")

        self.layer2 = [BasicBlock(), BasicBlock()]
        self.layer2[0].load_weights(params, "conv3_x.0", downsample=True)
        self.layer2[1].load_weights(params, "conv3_x.1")

        self.layer3 = [BasicBlock(), BasicBlock()]
        self.layer3[0].load_weights(params, "conv4_x.0", downsample=True)
        self.layer3[1].load_weights(params, "conv4_x.1")

        self.layer4 = [BasicBlock(), BasicBlock()]
        self.layer4[0].load_weights(params, "conv5_x.0", downsample=True)
        self.layer4[1].load_weights(params, "conv5_x.1")

        self.fc_weight = params["fc.weight"].T
        self.fc_bias = params["fc.bias"]

    def forward(self, x):
        x = conv2d(x, self.conv1, None, stride=1, padding=1)
        x = batch_norm(x, self.bn1_weight, self.bn1_bias, self.bn1_mean, self.bn1_var)
        x = relu(x)

        for i, block in enumerate(self.layer1):
            x = block.forward(x)
        for i, block in enumerate(self.layer2):
            x = block.forward(x, stride=2 if i == 0 else 1)
        for i, block in enumerate(self.layer3):
            x = block.forward(x, stride=2 if i == 0 else 1)
        for i, block in enumerate(self.layer4):
            x = block.forward(x, stride=2 if i == 0 else 1)

        x = global_avg_pool2d(x)
        x = x.reshape(x.shape[0], -1)
        x = np.dot(x, self.fc_weight) + self.fc_bias
        return x

def preprocess_image(img32: np.ndarray) -> np.ndarray:
    mean = np.array([0.5071, 0.4867, 0.4408]).reshape(3, 1, 1)
    std = np.array([0.2675, 0.2565, 0.2761]).reshape(3, 1, 1)
    img = img32.transpose(2, 0, 1) / 255.0
    img = (img - mean) / std
    img = img[np.newaxis, ...]
    return img

def test_all(params_path: str, data_dir: str):
    import torch
    import torchvision
    import torchvision.transforms as transforms

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

    model = ResNet18()
    params = np.load(params_path, allow_pickle=True).item()
    model.load_weights(params)

    correct_top1 = 0
    correct_top5 = 0
    total = 0
    start_time = time.time()
    for imgs, labels in tqdm(testloader):
        print("Processing batch...")
        imgs = imgs.numpy()
        batch = []
        for img in imgs:
            img = (img * np.array([0.2675, 0.2565, 0.2761]).reshape(3,1,1)) + np.array([0.5071, 0.4867, 0.4408]).reshape(3,1,1)
            img = np.clip(img * 255, 0, 255).astype(np.float32).transpose(1,2,0)
            batch.append(preprocess_image(img)[0])
        batch = np.stack(batch)
        logits = model.forward(batch)
        probs = softmax(logits)
        top5_preds = np.argsort(-probs, axis=1)[:, :5]

        for i, label in enumerate(labels):
            total += 1
            if label in top5_preds[i, :1]:
                correct_top1 += 1
            if label in top5_preds[i]:
                correct_top5 += 1
    end_time = time.time()
    top1_acc = correct_top1 / total
    top5_acc = correct_top5 / total

    print("Test set: Top-1 Acc: {:.4f}, Top-5 Acc: {:.4f}, Top-1 Error: {:.4f}, Top-5 Error: {:.4f}".format(
        top1_acc, top5_acc, 1 - top1_acc, 1 - top5_acc))

    total_params = sum(v.size for v in params.values())
    print("Total Params: {}".format(total_params))
    print(end_time - start_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=params_path)
    parser.add_argument('--data_dir', type=str, default='./data')
    args = parser.parse_args()

    test_all(args.weights, args.data_dir)
