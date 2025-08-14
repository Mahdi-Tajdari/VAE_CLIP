import torch
from torch.optim import Adam
from PIL import Image
import numpy as np

from core.auxiliary_network import AuxiliaryNetwork
from core.loss_function import HeuristicLoss
from utils.model_loader import load_vae_and_clip_models

def train_auxiliary_network(vae, clip_model, clip_processor, num_epochs=1000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Define auxiliary network and optimizer
    aux_net = AuxiliaryNetwork().to(device).to(torch.float16)  # --- این خط اصلاح شده است ---
    optimizer = Adam(aux_net.parameters(), lr=1e-3)
    
    # Define text prompt for the heuristic loss
    prompt_with_glasses = "a photo of a person wearing glasses"
    
    # Encode the text prompt with CLIP
    with torch.no_grad():
        text_embeds_target = clip_model.get_text_features(
            **clip_processor(text=[prompt_with_glasses], return_tensors="pt", padding=True).to(device)
        )
    
    print("شروع فرآیند آموزش... 🚀")
    
    for epoch in range(num_epochs):
        # 1. Generate a random latent vector Z
        z_initial = torch.randn((1, 4, 64, 64), device=device, dtype=torch.float16)
        
        # 2. Forward pass
        optimizer.zero_grad()
        
        # Auxiliary network accepts and returns the 4D tensor
        delta_z = aux_net(z_initial)
        z_modified = z_initial + delta_z
        
        # 3. Decode latent vectors to images using the VAE
        image_initial_tensor = vae.decode(z_initial / 0.18215).sample
        image_modified_tensor = vae.decode(z_modified / 0.18215).sample
        
        # Normalization to [0, 1] range
        image_initial_norm = (image_initial_tensor / 2 + 0.5).clamp(0, 1)
        image_modified_norm = (image_modified_tensor / 2 + 0.5).clamp(0, 1)
        
        # Convert tensors to PIL Image format by detaching from the graph
        image_initial_pil = [Image.fromarray((i.permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)) for i in image_initial_norm]
        image_modified_pil = [Image.fromarray((i.permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)) for i in image_modified_norm]

        # Pre-process images for CLIP
        image_initial_processed = clip_processor(images=image_initial_pil, return_tensors="pt").pixel_values.to(device)
        image_modified_processed = clip_processor(images=image_modified_pil, return_tensors="pt").pixel_values.to(device)
        
        # 4. Calculate loss
        loss = HeuristicLoss(clip_model).to(device)(image_initial_processed, image_modified_processed, text_embeds_target)
        
        # 5. Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    print("آموزش به پایان رسید.")
    
    # Save the trained network weights for later use
    torch.save(aux_net.state_dict(), "auxiliary_network_weights.pth")