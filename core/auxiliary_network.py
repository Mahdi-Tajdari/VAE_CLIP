import torch.nn as nn

class AuxiliaryNetwork(nn.Module):
    """
    شبکه عصبی سبک برای پیدا کردن جهت تفکیک‌شده در فضای پنهان.
    """
    def __init__(self, latent_dim=4 * 64 * 64, hidden_dim=256):
        super().__init__()
        # MLP با ابعاد ورودی و خروجی صحیح
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh()  
        )
        
    def forward(self, latent_vector):
        """
        ورودی: بردار فضای پنهان (latent_vector) چهاربعدی
        خروجی: بردار تغییر کوچک (delta_vector) چهاربعدی
        """
        # بردار را صاف می کند تا به لایه خطی داده شود
        batch_size = latent_vector.shape[0]
        flattened_vector = latent_vector.view(batch_size, -1)
        
        # پیش بینی بردار تغییر در فضای صاف
        delta_flattened = self.net(flattened_vector)
        
        # بردار را به ابعاد اولیه برمی گرداند
        delta_vector = delta_flattened.view(latent_vector.shape)
        
        return delta_vector