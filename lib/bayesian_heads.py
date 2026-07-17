import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianHead(nn.Module):
    def __init__(self, hid_dim, num_classes, rel_type=None, sample_n=5):
        super(BayesianHead, self).__init__()
        self.num_classes = num_classes
        self.rel_type = rel_type
        self.sample_n = sample_n  # number of MC samples during training

        self.weight_mu = nn.Linear(hid_dim, num_classes)
        self.weight_logvar = nn.Linear(hid_dim, num_classes)

        if rel_type == 'attention' or rel_type is None:
            self.activation = nn.Softmax(dim=-1)
        else:
            self.activation = nn.Sigmoid()

    def forward(self, x, phase='train', unc=False):
        """
        Args:
            x: (B, D)
            phase: 'train' or 'test'
            unc: whether to output uncertainty (epistemic + aleatoric)
        Returns:
            (B, C) prediction distribution or (aleatoric, epistemic) uncertainty
        """
        mu = self.weight_mu(x)             # (B, C)
        logvar = self.weight_logvar(x)     # (B, C)
        var = torch.exp(logvar)

        if unc:
            # 预测不确定性（可视化时使用）
            return var, torch.zeros_like(var)

        if phase == 'train':
            # 多次采样估计分布
            preds = []
            for _ in range(self.sample_n):
                eps = torch.randn_like(var)
                sampled_logits = mu + eps * torch.sqrt(var)
                preds.append(self.activation(sampled_logits))
            preds = torch.stack(preds, dim=0)  # (S, B, C)
            return preds.mean(dim=0)  # (B, C)
        else:
            return self.activation(mu)