# utils/model_loader.py
import torch
from transformers import CLIPProcessor, CLIPModel
from diffusers import AutoencoderKL

def load_vae_and_clip_models(device):
    print("Loading VAE and CLIP models...")

    # Keep in fp32 for stable grads through decode (we're not training VAE)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float32).to(device)
    vae.eval()

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model.eval()
    # Freeze params (but keep graph through ops)
    for p in clip_model.parameters():
        p.requires_grad = False

    print("Models loaded successfully.")
    return vae, clip_model, clip_processor