import torch
from utils.model_loader import load_vae_and_clip_models
from trainer import train_auxiliary_network
from core.auxiliary_network import AuxiliaryNetwork
from diffusers import AutoencoderKL
from PIL import Image
from torchvision.transforms import transforms
import os

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    vae, clip_model, clip_processor = load_vae_and_clip_models(device)
    
    print("\n--- Starting the training loop ---")
    
    # Check for existing checkpoint
    if os.path.exists("auxiliary_network_weights.pth"):
        print("Found existing model weights. Loading model...")
        aux_net = AuxiliaryNetwork().to(device)
        aux_net.load_state_dict(torch.load("auxiliary_network_weights.pth"))
        aux_net = aux_net.to(torch.float16)
        aux_net.eval()
        
        print("\n--- Model loaded. You can now use it for inference or continue training. ---")
        
    else:
        train_auxiliary_network(vae, clip_model, clip_processor, num_epochs=1000)
        print("\n--- Training complete ---")
    
    print("\n--- Loading trained network and generating example image ---")
    
    aux_net = AuxiliaryNetwork().to(device)
    aux_net.load_state_dict(torch.load("auxiliary_network_weights.pth"))
    aux_net = aux_net.to(torch.float16)
    aux_net.eval()
    
    image_path_test = "dataset/celeba/img_align_celeba/000001.jpg"
    image_test_pil = Image.open(image_path_test).convert("RGB").resize((512, 512))
    image_test_tensor = transforms.ToTensor()(image_test_pil).unsqueeze(0).to(device, dtype=torch.float16) * 2 - 1
    
    with torch.no_grad():
        z_original = vae.encode(image_test_tensor).latent_dist.sample() * 0.18215
        
        delta_z = aux_net(z_original)
        z_edited = z_original + delta_z
        
        image_original_tensor = vae.decode(z_original / 0.18215).sample
        image_edited_tensor = vae.decode(z_edited / 0.18215).sample
        
        image_original_pil = [Image.fromarray((i.permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)) for i in (image_original_tensor / 2 + 0.5).clamp(0, 1)]
        image_edited_pil = [Image.fromarray((i.permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)) for i in (image_edited_tensor / 2 + 0.5).clamp(0, 1)]
        
        image_original_pil[0].save("original_image_test.png")
        image_edited_pil[0].save("edited_image_with_glasses_test.png")
        
    print("Example images generated and saved. You can now implement your interactive control panel!")