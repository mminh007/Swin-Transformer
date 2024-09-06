import yaml
import os
import argparse

def setup_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cfg", type=int, required= True, help="path to config file")

    parser.add_argument("--model", 
                        type=str, help="Type of Swin model")
    
    parser.add_argument("--mode", type=str, default="train")
    
    parser.add_argument("--name", type=int)

    # config datasets
    parser.add_argument("--data", type=str, default="cifar10")

    parser.add_argument("--imgsz",  type=int)
    
    parser.add_argument("--in_chans", type=int)
    
    parser.add_argument("--labels",  
                        type=int, help="Number of classes")
    
    parser.add_argument("--patch_size", 
                        type=int, help="Size of image patch")
    
    # parser.add_argument("--dataset-dir", type=str)
    
    # parser.add_argument("--annotated-file", type=str)

    
    # config model
    parser.add_argument("--num-heads", 
                        type=list, help="Number of attention heads")
    
    parser.add_argument("--embed-dim", 
                        type=int, help="Size of each attention head for value")
    
    parser.add_argument("--depths", 
                        type=list, help="number of transformer layers")
    
    parser.add_argument("--window_size", type=int)
    
    parser.add_argument("--mlp-ratio", 
                        type=int)
    
    parser.add_argument("--qkv_bias", type=bool)
    
    parser.add_argument("--ape", type=bool)

    parser.add_argument("use-pos-rel", type=bool)

    parser.add_argument("--dropout", type=float)
    
    parser.add_argument("--norm-eps", type=float)


    # config training
    parser.add_argument("--epochs", type=int)

    parser.add_argument("--batch", type=int)

    parser.add_argument("--devices", type=str)

    parser.add_argument("--optimizer", type=int)

    parser.add_argument("--weight-decay", type=float)

    parser.add_argument("--lr", type=float, help="Learning rate")

    parser.add_argument("--beta1", type=float)
    
    parser.add_argument("--beta2", type=float)
    
    parser.add_argument("--eps", type=float)

    parser.add_argument("--num_workers", type=int, default=1)
    
    parser.add_argument("--outputs-dir", type=str)
    
    parser.add_argument("--logger", type=str)
    
    return parser


def update_config(args: argparse.Namespace, parser: argparse.ArgumentParser):
    if not args.config_file:
        return args
    
    cfg_path = args.config_file + "yaml" if not args.config_file.endwiths(".yaml") else args.config_file

    with open(cfg_path, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    

    config_args = argparse.Namespace(**data)
    args = parser.parse_args(namespace=config_args)
    args.config_file = os.path.splitext(args.config_file)[0]

    return args

