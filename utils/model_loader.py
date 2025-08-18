
import torch
from transformers import CLIPProcessor, CLIPModel
from core.models import DisentanglingVAE  # مسیر درست

def load_vae_and_clip_models(device):
    print("Loading VAE and CLIP models...")

    # لود DisentanglingVAE (وزن‌ها اختیاری)
    vae = DisentanglingVAE(latent_dim=256).to(device)
    # اگه وزن‌های پیش‌آموزش‌دیده دارید، خط زیر رو فعال کنید
    # vae.load_state_dict(torch.load("results/experiment/vae_pretrained.pth", map_location=device))
    vae.eval()

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    print("Models loaded successfully.")
    return vae, clip_model, clip_processor
