import torch
from utils.model_loader import load_vae_and_clip_models
from trainer import train_auxiliary_network
from core.auxiliary_network import AuxiliaryNetwork
from diffusers import AutoencoderKL
from PIL import Image

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load models
    vae, clip_model, clip_processor = load_vae_and_clip_models(device)
    
    # 2. Start the training process
    print("\n--- Starting the training loop ---")
    # train_auxiliary_network(vae, clip_model, clip_processor) # Commented out after first run
    print("\n--- Training complete ---")
    
    # 3. Here you can load the trained network and test it
    print("\n--- Loading trained network and generating example image ---")
    
    # Load the trained auxiliary network
    aux_net = AuxiliaryNetwork().to(device)
    aux_net.load_state_dict(torch.load("auxiliary_network_weights.pth"))
    aux_net = aux_net.to(torch.float16)  # --- این خط اصلاح شده است ---
    aux_net.eval()
    
    # Generate an initial image
    z_original = torch.randn((1, 4, 64, 64), device=device, dtype=torch.float16)
    with torch.no_grad():
        image_original_tensor = vae.decode(z_original / 0.18215).sample
        
        # Apply the learned disentangled vector
        delta_z = aux_net(z_original)
        z_edited = z_original + delta_z
        image_edited_tensor = vae.decode(z_edited / 0.18215).sample
        
        # Normalize and convert to PIL Image format
        image_original_norm = (image_original_tensor / 2 + 0.5).clamp(0, 1)
        image_edited_norm = (image_edited_tensor / 2 + 0.5).clamp(0, 1)
        
        image_original_pil = [Image.fromarray((i.permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)) for i in image_original_norm]
        image_edited_pil = [Image.fromarray((i.permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)) for i in image_edited_norm]

        # Save the images
        image_original_pil[0].save("original_image.png")
        image_edited_pil[0].save("edited_image_with_glasses.png")
    
    print("Example images generated and saved. You can now implement your interactive control panel!")