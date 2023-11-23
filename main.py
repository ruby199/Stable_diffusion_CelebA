# Image-to-Image generation example code using CelebA dataset
# output: new_image
import sys
import os
import model_loader
import pipeline
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch
import torchvision.transforms as transforms
import random
import numpy as np


# Add the parent directory of the 'helper' directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

# Import CelebA dataloader
from helper.celeba_data_loader import load_celeba_data

DEVICE = "cpu"

ALLOW_CUDA = True
ALLOW_MPS = False

# print(torch.cuda.is_available())

# Use GPU if available
if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

# Current file's directory
current_dir = os.path.dirname(__file__)

# Modify file paths
tokenizer_vocab_path = os.path.join(current_dir, '../data/tokenizer_vocab.json')
tokenizer_merges_path = os.path.join(current_dir, '../data/tokenizer_merges.txt')
output_dir = os.path.join(current_dir, 'output')
tokenizer = CLIPTokenizer(tokenizer_vocab_path, merges_file=tokenizer_merges_path)


model_file = "../data/v1-5-pruned-emaonly.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

# Move each model in the dictionary to the specified device
for model_name, model in models.items():
    model.to(DEVICE)

# Load CelebA Data
celeba_data_path = '../data/img_align_celeba/img_align_celeba'  # Set your CelebA images path
celeba_label_path = '../data/list_attr_celeba.txt'  # Set your CelebA labels path
celeba_loader, (cond, neg_cond) = load_celeba_data(celeba_data_path, celeba_label_path, batch_size=1)
celeba_iter = iter(celeba_loader)

# Fetch one sample from CelebA
input_image, (label_1_val, label_2_val) = next(celeba_iter)
# print("Type of input_image:", type(input_image))
# print("Shape of input_image:", input_image.shape)


###Convert the tensor to a PIL image###

# Ensure the tensor is on CPU and in the correct data type (float32)
input_image = input_image.cpu().squeeze(0).permute(1, 2, 0).numpy()
input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())

# Convert the NumPy array to a PIL image
input_image = Image.fromarray((input_image * 255).astype('uint8'))

###############

# Define your labels based on the values
label_1 = cond if label_1_val == 1 else "Not" + str(cond)
label_2 = neg_cond if label_2_val == 1 else "Not" + str(neg_cond)


do_cfg = True
cfg_scale = 8  # min: 1, max: 14


## IMAGE TO IMAGE

# Higher values means more noise will be added to the input image, so the result will further from the input image.
# Lower values means less noise is added to the input image, so output will be closer to the input image.
strength = 0.9

## SAMPLER

sampler = "ddpm"
num_inference_steps = 50
seed = 42
# seed = random.randint(0, 99999)

output_image = pipeline.generate_celeba(
    prompt=label_1,  # Replaced with label_1
    uncond_prompt=label_2,  # Replaced with label_2
    input_image=input_image,
    strength=strength,
    do_cfg=do_cfg,
    cfg_scale=cfg_scale,
    sampler_name=sampler,
    n_inference_steps=num_inference_steps,
    seed=seed,
    models=models,
    device=DEVICE,
    idle_device="cuda",
    tokenizer=tokenizer,
)

# Combine the input image and the output image into a single image.
# Image.fromarray(output_image)
output_dir = os.path.join(current_dir, 'output_celeb')

# Make sure output directory exists
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Saving the output image...
output_path = os.path.join(output_dir, f'output_image_{seed}_{label_1}_{label_2}.png')
Image.fromarray(output_image).save(output_path)