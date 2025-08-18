import torch
from utils.model_loader import load_vae_and_clip_models
from trainer import train_auxiliary_network

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # لود مدل‌ها
    vae, clip_model, clip_processor = load_vae_and_clip_models(device)

    # شروع آموزش
    aux_net = train_auxiliary_network(
        vae=vae,
        clip_model=clip_model,
        clip_processor=clip_processor,
        image_dir="dataset/celeba/img_align_celeba/img_align_celeba/",
        num_epochs=100,          # کل ایپاک‌ها
        steps_per_epoch=50,     # هر ایپاک چند آپدیت بزنه
        lr=1e-4,
        log_every=1,
        ckpt_every=5,
        device=device
    )