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
