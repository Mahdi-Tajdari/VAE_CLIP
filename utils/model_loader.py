import torch
from transformers import CLIPProcessor, CLIPModel
from diffusers import AutoencoderKL

def load_vae_and_clip_models(device):
    """
    Loads pre-trained VAE and CLIP models.
    """
    print("Loading VAE and CLIP models...")
    
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to(device)
    vae.eval()
    
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model.eval()
    
    print("Models loaded successfully.")
    
    return vae, clip_model, clip_processor

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae, clip_model, clip_processor = load_vae_and_clip_models(device)
    print(f"VAE model type: {type(vae)}")
    print(f"CLIP model type: {type(clip_model)}")