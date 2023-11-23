# Table Diffusion CelebA: A Stable Diffusion Framework for CelebA Image Generation

Build a model using stable diffusion framework &amp;  architecture &amp; cfg 


## Project Overview:
Table Diffusion CelebA leverages the Stable Diffusion framework to create a robust model for generating and transforming images from the CelebA dataset. 
This project integrates cutting-edge techniques in image-to-image generation, focusing on high-quality, configurable outputs.


## Features:

* **Advanced Model Architecture:** Utilizes the Stable Diffusion framework and architecture, offering flexibility and efficiency in image generation.
* **Customizable Output:** Incorporates Configuration Guidance (CFG) for personalized control over the generation process.
* **CelebA Dataset Integration:** Seamlessly processes images from the popular CelebA dataset, known for its vast collection of celebrity faces.



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
The main script (main.py) orchestrates the entire process, from data loading to image generation and saving. It allows for flexibility in terms of device usage and offers options for different sampling methods and CFG scales.

## Output:
Generated images are saved in a specified output directory, showcasing the transformation from the original CelebA images to the new, modified versions.
