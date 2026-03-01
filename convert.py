"""Convert a Kokoro .pt voice file to .bin format for use with kokoro-onnx."""
import argparse
import numpy as np
import torch

parser = argparse.ArgumentParser(description='Convert Kokoro .pt voice to .bin')
parser.add_argument('input', help='Path to input .pt file')
parser.add_argument('output', nargs='?', help='Path to output .bin file (default: same name as input)')
args = parser.parse_args()

out = args.output or args.input.replace('.pt', '.bin')

pack = torch.load(args.input, map_location='cpu', weights_only=True)
assert pack.ndim == 3 and pack.shape[1:] == (1, 256), f'Unexpected shape: {pack.shape}'

pack.numpy().astype(np.float32).tofile(out)

check = np.fromfile(out, dtype=np.float32).reshape(-1, 1, 256)
assert check.shape == tuple(pack.shape), f'Verification failed: {check.shape}'

print(f'{args.input} -> {out}  {tuple(pack.shape)}')
