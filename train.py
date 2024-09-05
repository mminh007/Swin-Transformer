import torch
import torch.nn as nn
import os
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import torchvision.transforms as transforms
import torchvision
import tqdm
import argparse
import gc
from config import update_config

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cfg", type=int, required= True, help="path to config file")

    parser.add_argument("--model", 
                        type=str, help="Type of ViT model")
    
    parser.add_argument("--labels",  
                        type=int, help="Number of classes")
    
    parser.add_argument("--patch-size", 
                        type=int, help="Size of image patch")
    
    parser.add_argument("--num-heads", 
                        type=list, help="Number of attention heads")
    
    parser.add_argument("--embed-dim", 
                        type=int, help="Size of each attention head for value")
    
    parser.add_argument("--depths", 
                        type=list, help="number of attention layers")
    
    parser.add_argument("--mlp-ratio", 
                        type=int, help="Demension of hidden layer in MLP Block")
    
    parser.add_argument("--ape", type=bool)

    parser.add_argument("use-pos-rel", type=bool)
    
    parser.add_argument("--lr", type=float, help="Learning rate")

    parser.add_argument("--optimizer", type=int)
    
    parser.add_argument("--weight-decay", type=float)
    
    parser.add_argument("--batch-size", type=int)
    
    parser.add_argument("--epochs", type=int)
    
    parser.add_argument("--imgsz",  type=int)
    
    parser.add_argument("--in_chans", type=int)
    
    parser.add_argument("--beta1", type=float)
    
    parser.add_argument("--beta2", type=float)
    
    parser.add_argument("--eps", type=float)
    
    parser.add_argument("--dropout", type=float)
    
    parser.add_argument("--norm-eps", type=float)

    parser.add_argument("--dataset-dir", type=str)
    
    parser.add_argument("--annotated-file", type=str)
    
    parser.add_argument("--outputs-dir", type=str)
    
    parser.add_argument("--logger", type=str)

    parser.add_argument("--devices", type=str)
    
    parser.add_argument("--name", type=int)
    
    args = parser.parse_args()

    config = update_config(args)

    return args, config


def main(config):
    pass 
