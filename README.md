# Diffusion CelebA: A Stable Diffusion Framework for CelebA Image Generation


## Project Overview:
This project, Stable Diffusion CelebA is basically a model that laverages the Stable Diffusion framework to create a  model for generating and transforming images from the CelebA dataset. 
If anyone interested in topics such as image-to-image generation, generative ai, etc, this is the model just right for you! 


## Features:

* **Model Architecture:** Utilizes the Stable Diffusion framework and architecture, offering flexibility and efficiency in image generation.
* **Customizable Output:** Incorporates Configuration Guidance (CFG) for personalized control over the generation process.
* **CelebA Dataset Integration:** Processes images from the popular CelebA dataset, known for its vast collection of celebrity faces.


## Download the pretrained model (Stable-diffusion-v1-5)

Download the pretrained model from this link: https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.ckpt

## Download Data Files

1. CelebA Dataset
The CelebA dataset consists of two main components:
  
Image Files: 
  * `img_align_celeba.zip`
  
Attribute List: 
  * `list_attr_celeba.txt`
  
  Source: [CelebA Official Website](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
  Direct Download Link: [Google Drive](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg&usp=sharing)


2. Stable Diffusion - Tokenizer & Model
This part includes the tokenizer and the model files necessary for Stable Diffusion:
  
Tokenizer Files:
  * `tokenizer_merges.txt`
  * `tokenizer_vocab.json`
  
  Download Source: [Hugging Face - Stable Diffusion v1.5 Tokenizer](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main/tokenizer)


## Usage:
The main script (main.py) performs the entire process, from data loading to image generation and saving. It allows for flexibility in terms of device usage and offers options for different sampling methods and CFG scales. Simply run this python file and the user-defined text conditioned image would be generated! Super Cool!

## Output:
Generated images are saved in a specified output directory!
