import torch
from torch.optim import Adam
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

from core.auxiliary_network import AuxiliaryNetwork
from core.loss_function import HeuristicLoss
from utils.model_loader import load_vae_and_clip_models

def train_auxiliary_network(vae, clip_model, clip_processor, num_epochs=1000, image_dir='dataset/celeba/img_align_celeba/', batch_size=1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    print("Preparing training batch from dataset...")
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = ImageFolder(root=image_dir, transform=transform)
    
    indices = torch.randperm(len(dataset))[:batch_size]
    fixed_batch_images = torch.stack([dataset[i][0] for i in indices]).to(device, dtype=torch.float16)

    with torch.no_grad():
        z_initial = vae.encode(fixed_batch_images).latent_dist.sample() * 0.18215
    
    aux_net = AuxiliaryNetwork().to(device).to(torch.float16)
    optimizer = Adam(aux_net.parameters(), lr=1e-4)
    
    prompt_with_glasses = "a photo of a person wearing glasses"
    
    with torch.no_grad():
        text_embeds_target = clip_model.get_text_features(
            **clip_processor(text=[prompt_with_glasses], return_tensors="pt", padding=True).to(device)
        )
    
    print("Starting training process on the fixed batch... 🚀")
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        delta_z = aux_net(z_initial)
        z_modified = z_initial + delta_z
        
        image_initial_tensor = vae.decode(z_initial / 0.18215).sample
        image_modified_tensor = vae.decode(z_modified / 0.18215).sample
        
        image_initial_norm = (image_initial_tensor / 2 + 0.5).clamp(0, 1)
        image_modified_norm = (image_modified_tensor / 2 + 0.5).clamp(0, 1)
        
        image_initial_pil = [Image.fromarray((i.permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)) for i in image_initial_norm]
        image_modified_pil = [Image.fromarray((i.permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)) for i in image_modified_norm]

        image_initial_processed = clip_processor(images=image_initial_pil, return_tensors="pt").pixel_values.to(device)
        image_modified_processed = clip_processor(images=image_modified_pil, return_tensors="pt").pixel_values.to(device)
        
        total_loss, feature_loss, identity_loss = HeuristicLoss(clip_model).to(device)(image_initial_processed, image_modified_processed, text_embeds_target)
        
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] | Total Loss: {total_loss.item():.4f} | Feature Loss: {feature_loss.item():.4f} | Identity Loss: {identity_loss.item():.4f}")
        
        # Save model checkpoint after each epoch
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': aux_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss.item(),
                }, f"checkpoint_epoch_{epoch+1}.pth")
            print(f"Checkpoint saved at epoch {epoch+1}")
    
    print("آموزش به پایان رسید.")
    
    torch.save(aux_net.state_dict(), "auxiliary_network_weights.pth")