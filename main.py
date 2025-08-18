import torch
from utils.model_loader import load_vae_and_clip_models
from trainer import train_auxiliary_network

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    vae, clip_model, clip_processor = load_vae_and_clip_models(device)

    aux_net = train_auxiliary_network(
        vae=vae,
        clip_model=clip_model,
        clip_processor=clip_processor,
        image_dir="dataset/celeba/img_align_celeba/img_align_celeba/",
        num_epochs=10,          # Reduced for debugging
        steps_per_epoch=10,     # Reduced for debugging
        lr=1e-4,
        log_every=1,
        ckpt_every=5,
        device=device
    )