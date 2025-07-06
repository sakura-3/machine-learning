import numpy as np
from typing import Dict, List

# 权重文件
params_path = "resnet18.npy"
# 测试图片
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
# 类别文件
class_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"


def relu(x: np.ndarray) -> np.ndarray:
    """激活函数"""
    return np.maximum(0, x)


def global_avg_pool2d(x):
    return np.mean(x, axis=(2, 3))


def softmax(x):
    """分类输出"""
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def conv2d(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
    stride: int = 1,
    padding: int = 0,
    debug: bool = False,
) -> np.ndarray:
    """
    二维卷积

    params:
        x: 输入数据
        w: 卷积核, 从权重文件中读取
        b: bias, 从权重文件中读取
        stride: 步长，默认为 1
        padding: 填充，默认为 0
        debug: 打印调试信息
    """
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels, _, filter_height, filter_width = w.shape

    out_height = (in_height + 2 * padding - filter_height) // stride + 1
    out_width = (in_width + 2 * padding - filter_width) // stride + 1

    # 只在 height 和 width 进行 padding
    x_padded = np.pad(
        x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant"
    )

    out = np.zeros((batch_size, out_channels, out_height, out_width))
    if debug:
        print(x.shape, w.shape, out.shape)

    for batch in range(batch_size):
        for chan in range(out_channels):
            for i in range(out_height):
                for j in range(out_width):
                    h_start = i * stride
                    h_end = h_start + filter_height
                    w_start = j * stride
                    w_end = w_start + filter_width

                    x_region = x_padded[batch, :, h_start:h_end, w_start:w_end]

                    out[batch, chan, i, j] = np.sum(x_region * w[chan]) + b[chan]

    return out


def batch_norm(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    debug: bool = False,
    eps=1e-5,
) -> np.ndarray:
    """
    归一化

    params:
        x: 输入张量
        gamma: 缩放参数
        beta: 平移参数
        mean: 均值
        var: 方差
        debug: 是否打印调试信息
    """

    # TODO 为什么 mean 和 var 来自权重文件
    gamma = gamma.reshape(1, -1, 1, 1)
    beta = beta.reshape(1, -1, 1, 1)
    mean = mean.reshape(1, -1, 1, 1)
    var = var.reshape(1, -1, 1, 1)

    if debug:
        print(x.shape, gamma.shape, beta.shape, mean.shape, var.shape)

    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def max_pool2d(
    x: np.ndarray, pool_size: int = 2, stride: int = 2, debug: bool = False
) -> np.ndarray:
    """二维最大池化操作"""
    batch_size, in_channels, in_height, in_width = x.shape

    out_height = (in_height - pool_size) // stride + 1
    out_width = (in_width - pool_size) // stride + 1

    out = np.zeros((batch_size, in_channels, out_height, out_width))
    if debug:
        print(out.shape)

    for batch in range(batch_size):
        for chan in range(in_channels):
            for i in range(out_height):
                for j in range(out_width):
                    h_start = i * stride
                    h_end = h_start + pool_size
                    w_start = j * stride
                    w_end = w_start + pool_size

                    # 取区域内最大值输出
                    out[batch, chan, i, j] = np.max(
                        x[batch, chan, h_start:h_end, w_start:w_end]
                    )

    return out


class BasicBlock:
    """ResNet18的基本残差块"""

    def __init__(self):
        self.conv1: np.ndarray
        self.bn1_weight: np.ndarray
        self.bn1_bias: np.ndarray
        self.bn1_running_mean: np.ndarray
        self.bn1_running_var: np.ndarray

        self.conv2: np.ndarray
        self.bn2_weight: np.ndarray
        self.bn2_bias: np.ndarray
        self.bn2_running_mean: np.ndarray
        self.bn2_running_var: np.ndarray

        self.downsample: bool
        self.downsample_conv: np.ndarray
        self.downsample_bn_weight: np.ndarray
        self.downsample_bn_bias: np.ndarray
        self.downsample_bn_running_mean: np.ndarray
        self.downsample_bn_running_var: np.ndarray

    def load_weights(self, params, prefix, downsample=False):
        self.conv1 = params[f"{prefix}.conv1.weight"]
        self.bn1_weight = params[f"{prefix}.bn1.weight"]
        self.bn1_bias = params[f"{prefix}.bn1.bias"]
        self.bn1_running_mean = params[f"{prefix}.bn1.running_mean"]
        self.bn1_running_var = params[f"{prefix}.bn1.running_var"]

        self.conv2 = params[f"{prefix}.conv2.weight"]
        self.bn2_weight = params[f"{prefix}.bn2.weight"]
        self.bn2_bias = params[f"{prefix}.bn2.bias"]
        self.bn2_running_mean = params[f"{prefix}.bn2.running_mean"]
        self.bn2_running_var = params[f"{prefix}.bn2.running_var"]

        self.downsample = downsample
        if downsample:
            self.downsample_conv = params[f"{prefix}.downsample.0.weight"]
            self.downsample_bn_weight = params[f"{prefix}.downsample.1.weight"]
            self.downsample_bn_bias = params[f"{prefix}.downsample.1.bias"]
            self.downsample_bn_running_mean = params[
                f"{prefix}.downsample.1.running_mean"
            ]
            self.downsample_bn_running_var = params[
                f"{prefix}.downsample.1.running_var"
            ]

    def forward(self, x):
        identity = x

        out = conv2d(
            x,
            self.conv1,
            np.zeros_like(self.bn1_bias),
            stride=1 if not self.downsample else 2,
            padding=1,
        )
        out = batch_norm(
            out,
            self.bn1_weight,
            self.bn1_bias,
            self.bn1_running_mean,
            self.bn1_running_var,
        )
        out = relu(out)

        out = conv2d(out, self.conv2, np.zeros_like(self.bn2_bias), stride=1, padding=1)
        out = batch_norm(
            out,
            self.bn2_weight,
            self.bn2_bias,
            self.bn2_running_mean,
            self.bn2_running_var,
        )

        # 残差连接
        if self.downsample:
            identity = conv2d(
                identity,
                self.downsample_conv,
                np.zeros_like(self.downsample_bn_bias),
                stride=2,
            )
            identity = batch_norm(
                identity,
                self.downsample_bn_weight,
                self.downsample_bn_bias,
                self.downsample_bn_running_mean,
                self.downsample_bn_running_var,
            )

        out += identity
        out = relu(out)

        return out


class ResNet18:
    """ResNet18模型"""

    def __init__(self):
        self.conv1: np.ndarray
        self.bn1_weight: np.ndarray
        self.bn1_bias: np.ndarray
        self.bn1_running_mean: np.ndarray
        self.bn1_running_var: np.ndarray

        self.layer1 = [BasicBlock(), BasicBlock()]
        self.layer2 = [BasicBlock(), BasicBlock()]
        self.layer3 = [BasicBlock(), BasicBlock()]
        self.layer4 = [BasicBlock(), BasicBlock()]

        self.fc_weight: np.ndarray
        self.fc_bias: np.ndarray

    def load_weights(self, params):
        # 初始卷积层
        self.conv1 = params["conv1.weight"]
        self.bn1_weight = params["bn1.weight"]
        self.bn1_bias = params["bn1.bias"]
        self.bn1_running_mean = params["bn1.running_mean"]
        self.bn1_running_var = params["bn1.running_var"]

        # 残差层
        self.layer1[0].load_weights(params, "layer1.0")
        self.layer1[1].load_weights(params, "layer1.1")

        self.layer2[0].load_weights(params, "layer2.0", downsample=True)
        self.layer2[1].load_weights(params, "layer2.1")

        self.layer3[0].load_weights(params, "layer3.0", downsample=True)
        self.layer3[1].load_weights(params, "layer3.1")

        self.layer4[0].load_weights(params, "layer4.0", downsample=True)
        self.layer4[1].load_weights(params, "layer4.1")

        # 全连接层
        self.fc_weight = params["fc.weight"].T
        self.fc_bias = params["fc.bias"]

    def forward(self, x):
        # 初始卷积层
        x = conv2d(x, self.conv1, np.zeros_like(self.bn1_bias), stride=2, padding=3)
        x = batch_norm(
            x,
            self.bn1_weight,
            self.bn1_bias,
            self.bn1_running_mean,
            self.bn1_running_var,
        )
        x = relu(x)
        x = max_pool2d(x, pool_size=3, stride=2)

        # 残差层
        for block in self.layer1:
            x = block.forward(x)
        for block in self.layer2:
            x = block.forward(x)
        for block in self.layer3:
            x = block.forward(x)
        for block in self.layer4:
            x = block.forward(x)

        # 全连接层
        x = global_avg_pool2d(x)
        x = x.reshape(x.shape[0], -1)
        x = np.dot(x, self.fc_weight) + self.fc_bias

        return x


def preprocess_image(image_url):
    from PIL import Image
    import requests
    from io import BytesIO

    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).convert("RGB")

    img = img.resize((224, 224))
    img_numpy = np.array(img).astype(np.float32)
    img_numpy = img_numpy.transpose(2, 0, 1)

    # 归一化
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_numpy = (img_numpy / 255.0 - mean) / std

    # 添加批次维度 [C, H, W] → [1, C, H, W]
    img_numpy = img_numpy[np.newaxis, ...]

    return img_numpy


def get_classes(class_url: str) -> List[str]:
    import requests

    response = requests.get(class_url)
    classes = [line.strip() for line in response.text.split("\n")]
    return classes


def classify_image(model, image_url: str, class_url: str) -> Dict:
    image = preprocess_image(image_url)

    output = model.forward(image)
    output = softmax(output)
    predicted_idx = np.argmax(output, axis=1)[0]
    confidence = output[0, predicted_idx]

    classes = get_classes(class_url)

    return {
        "class_id": predicted_idx,
        "class_name": classes[predicted_idx],
        "confidence": confidence,
    }


if __name__ == "__main__":
    params = np.load(params_path, allow_pickle=True).item()
    # print(params.keys())
    
    model = ResNet18()
    model.load_weights(params)

    result = classify_image(model, image_url, class_url)
    print(result)
