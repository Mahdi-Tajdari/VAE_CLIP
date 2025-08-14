import torch
import torch.nn as nn
import torch.nn.functional as F

class HeuristicLoss(nn.Module):
    """
    تابع هزینه هوشمندانه برای بهینه‌سازی شبکه کمکی.
    """
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.cos_sim = nn.CosineSimilarity(dim=1)
    
    def forward(self, image_initial, image_modified, text_embeds_target):
        """
        ورودی‌ها:
        image_initial: تصویر اصلی از VAE
        image_modified: تصویر تغییریافته از VAE
        text_embeds_target: امبدینگ متن "با عینک"
        
        خروجی: مقدار عددی هزینه (loss value)
        """
        # 1. گرفتن امبدینگ‌های تصویر با CLIP
        image_initial_embeds = self.clip_model.get_image_features(pixel_values=image_initial)
        image_modified_embeds = self.clip_model.get_image_features(pixel_values=image_modified)
        
        # 2. محاسبه هزینه ویژگی (Feature Loss)
        feature_loss = 1 - self.cos_sim(image_modified_embeds, text_embeds_target).mean()
        
        # 3. محاسبه هزینه هویت (Identity Loss)
        identity_loss = 1 - self.cos_sim(image_modified_embeds, image_initial_embeds).mean()
        
        # 4. ترکیب دو هزینه با یک ضریب برای تعادل
        alpha = 0.5
        total_loss = alpha * feature_loss + (1 - alpha) * identity_loss
        
        return total_loss