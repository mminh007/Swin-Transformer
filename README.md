The script `train.py` performs model training

___________
## Installing dependencies
```
> git clone https://github.com/mminh007/swin-transformer.git
> cd swin-transformer
```
_________
## Config
the *config.yaml* file is located *./configs* 

example: ./configs/tiny_patch4_window7_224_cifar10.yaml
_________
## Run script
```
python ./train.py --config_file ./configs/tiny_patch4_window7_224_cifar10.yaml
```
________
# NOTE
The model is set up with two configurations. 

Dataset: **CIFAR10**

```
tiny model:
    - image size: 224
    - embed_dim: 96
    - patch size: 4
    - num heads: [3, 6, 12, 24]
    - depths: [2, 2, 6, 2]
    - window_size: 7
```

```
base model:
    - image: 224
    - embed_dim: 96
    - patch size: 4
    - num heads: [4, 8, 16, 32]
    - depths: [2, 2, 18, 2]
    - window size: 7
```

