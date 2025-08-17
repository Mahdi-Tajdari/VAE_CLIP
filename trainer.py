import os
import glob
import random
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.io import read_image
from torch.optim import Adam

from core.auxiliary_network import AuxiliaryNetwork
from core.loss_function import HeuristicLoss

# ------------------------
# ثابت‌ها
# ------------------------
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)
SD_SCALE  = 0.18215

def preprocess_for_clip(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B,3,H,W) در بازه [-1,1]
    خروجی: (B,3,224,224) نرمال‌شده برای CLIP
    """
    x = (x + 1.0) / 2.0  # به بازه [0,1]
    x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
    mean = torch.tensor(CLIP_MEAN, device=x.device).view(1, 3, 1, 1)
    std  = torch.tensor(CLIP_STD,  device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x

def load_random_image(image_paths, size=512, device="cuda"):
    """یک عکس رندوم می‌خونه و به [-1,1] تبدیل می‌کنه"""
    path = random.choice(image_paths)
    img = read_image(path)  # uint8 [0,255]
    if img.shape[0] == 1:   # grayscale → RGB
        img = img.repeat(3, 1, 1)
    if img.shape[0] == 4:   # RGBA → RGB
        img = img[:3]
    tx = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ConvertImageDtype(torch.float32),
    ])
    img = tx(img)           # [0,1]
    img = img * 2.0 - 1.0   # [-1,1]
    return img.unsqueeze(0).to(device)  # (1,3,H,W)

# ------------------------
# حلقه آموزش
# ------------------------
def train_auxiliary_network(
    vae, clip_model, clip_processor,
    image_dir="dataset/celeba/img_align_celeba/img_align_celeba/",
    num_epochs=500, steps_per_epoch=500,
    lr=1e-4, log_every=20, ckpt_every=50,
    device=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # مسیر تصاویر
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    print(f"Found {len(image_paths)} images. Using random sampling each step.")

    # مدل‌ها
    aux_net = AuxiliaryNetwork().to(device)
    optimizer = Adam(aux_net.parameters(), lr=lr)
    criterion = HeuristicLoss(clip_model).to(device)

    # embedding متن هدف فقط یک بار ساخته میشه
    with torch.no_grad():
        txt_inputs = clip_processor(
            text=["a photo of a person wearing glasses"], return_tensors="pt", padding=True
        ).to(device)
        text_embeds_target = clip_model.get_text_features(**txt_inputs).float()

    # فولدر خروجی
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("inference_samples", exist_ok=True)

    print(f"Starting training: epochs={num_epochs}, steps_per_epoch={steps_per_epoch}")
    global_step = 0

    for epoch in range(1, num_epochs + 1):
        for step in range(steps_per_epoch):
            imgs = load_random_image(image_paths, size=512, device=device)

            # encode → latent space
            with torch.no_grad():
                posterior = vae.encode(imgs)
                z = posterior.latent_dist.sample() * SD_SCALE

            # auxnet → delta z
            dz = aux_net(z)
            z_prime = z + dz

            # decode
            img_init = vae.decode(z / SD_SCALE).sample.float()
            img_edit = vae.decode(z_prime / SD_SCALE).sample.float()

            # برای CLIP آماده کن
            im_init_clip = preprocess_for_clip(img_init)
            im_edit_clip = preprocess_for_clip(img_edit)

            # لا‌س
            total_loss, feature_loss, identity_loss = criterion(
                im_init_clip, im_edit_clip, text_embeds_target, alpha=0.9
            )

            # آپدیت
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            global_step += 1

            if step % log_every == 0:
                print(f"Epoch {epoch} step {step} | "
                      f"Loss: {total_loss.item():.4f} "
                      f"Feat: {feature_loss.item():.4f} "
                      f"Id: {identity_loss.item():.6f}")

        # هر 50 ایپاک یکبار → چک‌پوینت و نمونه خروجی
        if epoch % ckpt_every == 0:
            ckpt_path = f"checkpoints/auxnet_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': aux_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss.item(),
            }, ckpt_path)
            print(f"Checkpoint saved at {ckpt_path}")

            # اینفرنس
            with torch.no_grad():
                test_img = load_random_image(image_paths, size=512, device=device)
                posterior = vae.encode(test_img)
                z = posterior.latent_dist.sample() * SD_SCALE
                dz = aux_net(z)
                z_prime = z + dz
                img_init = vae.decode(z / SD_SCALE).sample.float()
                img_edit = vae.decode(z_prime / SD_SCALE).sample.float()

                from torchvision.utils import save_image
                save_image((img_init + 1) / 2, f"inference_samples/original_epoch_{epoch}.png")
                save_image((img_edit + 1) / 2, f"inference_samples/edited_epoch_{epoch}.png")
                print(f"Inference images saved for epoch {epoch}")

    print("Training finished.")
    torch.save(aux_net.state_dict(), "auxiliary_network_final.pth")
    return aux_net
