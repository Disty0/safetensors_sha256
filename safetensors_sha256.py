import argparse
import torch
from safetensors import safe_open
import hashlib

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default=None,
    help="path to checkpoint of model",
)
opt = parser.parse_args()

if opt.model is not None:
    sd_model = {}
    with safe_open(opt.model, framework="pt", device="cpu") as f:
        for key in f.keys():
            hashed_line = hashlib.sha256(str(f.get_tensor(key)).encode('utf-8')).hexdigest()
            print(f"{key}: {hashed_line}")
