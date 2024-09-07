import yaml
import os
import argparse

def setup_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file", type=str, required= True, help="path to config file")
    return parser


def update_config(args: argparse.Namespace, parser: argparse.ArgumentParser):
    if not args.config_file:
        return args
    
    cfg_path = args.config_file + ".yaml" if not args.config_file.endswith(".yaml") else args.config_file

    with open(cfg_path, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    
    config_args = argparse.Namespace(**data)
    args = parser.parse_args(namespace=config_args)
    args.config_file = os.path.splitext(args.config_file)[0]

    return args

