import torch
import torch.nn as nn
import torch.nn.functional as F

class HeuristicLoss(nn.Module):
    """
    Intelligent loss function for the auxiliary network.
    """
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.cos_sim = nn.CosineSimilarity(dim=1)
    
    def forward(self, image_initial, image_modified, text_embeds_target):
        """
        Inputs:
        image_initial: Original image from VAE
        image_modified: Modified image from VAE
        text_embeds_target: CLIP embedding of the target text ("with glasses")
        
        Output: a single loss value
        """
        image_initial_embeds = self.clip_model.get_image_features(pixel_values=image_initial)
        image_modified_embeds = self.clip_model.get_image_features(pixel_values=image_modified)
        
        feature_loss = 1 - self.cos_sim(image_modified_embeds, text_embeds_target).mean()
        
        identity_loss = 1 - self.cos_sim(image_modified_embeds, image_initial_embeds).mean()
        
        # Increased alpha to prioritize adding the glasses feature
        alpha = 0.9 
        total_loss = alpha * feature_loss + (1 - alpha) * identity_loss
        
        return total_loss, feature_loss, identity_loss