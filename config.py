import yaml


def update_config(args):
    with open(args.cfg, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    
    # update config file
    data["model"] = args.model
    data["name"] = args.name
    data["in_chans"] = args.in_chans
    data["imgsz"] = args.imgsz
    data["labels"] = args.labels
    data["patches"] = args.patch_size
    data["nheads"] = args.num_heads
    data["embed_dim"] = args.embed_dim
    data["depths"] = args.depths
    data["window_size"] = args.window_size
    data["mlp_ratio"] = args.mlp_ratio
    data["qkv_bias"] = args.qkv_bias
    data["ape"] = args.ape
    data["use_pos_rel"] = args.use_pos_rel
    data["drop_out"] = args.drop_out
    data["norm_eps"] = args.norm_eps

    data["epochs"] = args.epochs
    data["batch"] = args.batch_size
    data["devices"] = args.devices
    data["optimizer"] = args.optimizer
    data["weight_decay"] = args.weight_decay
    data["lr"] = args.lr
    data["beta1"] = args.beta1
    data["beta2"] = args.beta2
    data["eps"] = args.eps
    data["outputs_dir"] = args.output_dirs 
    data["logger"] = args.logger

    with open(args.cfg, "w") as f:
        yaml.dump(data, f)
    
    return data