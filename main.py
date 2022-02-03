import os
import argparse
import yaml
import warnings

warnings.filterwarnings("ignore")

import torch


from exp.exp_GraphTransformer import ExpGraphTransformer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrajGAT")
    parser.add_argument("-C", "--config", type=str)
    parser.add_argument("-G", "--gpu", type=str, default="0")
    parser.add_argument("-L", "--load-model", type=str, default=None)
    parser.add_argument("-J", "--just_embedding", action="store_true")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f)

    print("Args in experiment:")
    print(config)
    print("GPU:", args.gpu)
    print("Load model:", args.load_model)
    print("Store embeddings:", args.just_embedding, "\n")

    if args.just_embedding:
        ExpGraphTransformer(config=config, gpu_id=args.gpu, load_model=args.load_model, just_embeddings=args.just_embedding).embedding()
    else:
        ExpGraphTransformer(config=config, gpu_id=args.gpu, load_model=args.load_model, just_embeddings=args.just_embedding).train()

    torch.cuda.empty_cache()
