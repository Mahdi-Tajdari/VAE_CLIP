import torch
from transformers import CLIPProcessor, CLIPModel
from diffusers import AutoencoderKL

def load_vae_and_clip_models(device):
    """
    Loads pre-trained VAE and CLIP models.
    """
    print("Loading VAE and CLIP models...")
    
    # Load pre-trained VAE model
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to(device)
    vae.eval()
    
    # Load pre-trained CLIP model
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model.eval()
    
    print("Models loaded successfully.")
    
    return vae, clip_model, clip_processor

if __name__ == '__main__':
    # This block is for testing the module
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae, clip_model, clip_processor = load_vae_and_clip_models(device)
    print(f"VAE model type: {type(vae)}")
    print(f"CLIP model type: {type(clip_model)}")