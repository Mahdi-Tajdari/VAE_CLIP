# core/loss_function.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class HeuristicLoss(nn.Module):
    """
    CLIP-driven feature loss (+ identity consistency).
    """
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.cos_sim = nn.CosineSimilarity(dim=1)

    @staticmethod
    def _l2n(x, dim=1, eps=1e-6):
        return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)

    def forward(self, image_initial, image_modified, text_embeds_target, alpha=0.9):
        """
        image_initial / image_modified: (B,3,224,224) preprocessed for CLIP (differentiable)
        text_embeds_target: (1,D) or (B,D) CLIP text features (no grad needed)
        """
        # forward through CLIP (keep graph!)
        img_init_emb = self.clip_model.get_image_features(pixel_values=image_initial)
        img_mod_emb  = self.clip_model.get_image_features(pixel_values=image_modified)

        # L2 normalize
        img_init_emb = self._l2n(img_init_emb, dim=1)
        img_mod_emb  = self._l2n(img_mod_emb,  dim=1)
        txt_tgt_emb  = self._l2n(text_embeds_target, dim=1)

        # expand text embedding to batch if needed
        if txt_tgt_emb.shape[0] == 1 and img_mod_emb.shape[0] > 1:
            txt_tgt_emb = txt_tgt_emb.expand(img_mod_emb.shape[0], -1)

        feature_loss  = 1.0 - self.cos_sim(img_mod_emb, txt_tgt_emb).mean()
        identity_loss = 1.0 - self.cos_sim(img_mod_emb, img_init_emb).mean()

        total_loss = alpha * feature_loss + (1.0 - alpha) * identity_loss
        return total_loss, feature_loss, identity_loss
