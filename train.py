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
from config import setup_parse, update_config



def main(args):
    pass













if __name__ == "__main__":
    parser = setup_parse()

    args = parser.parse_args()
    args = update_config(args, parser)

    main(args)
