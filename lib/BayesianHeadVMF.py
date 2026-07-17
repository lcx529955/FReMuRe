import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianHeadVMF(nn.Module):
    """
    vMF-Mixture 关系分类头（替换版）
    - __init__ 参数与原版完全一致：hid_dim, num_classes, rel_type=None, sample_n=5
    - 多了 K（混合分量数）、use_shared_comp（共享分量开关），有默认值
    - forward 与原版兼容，保留 unc 和 sample_n 参数
    """
    def __init__(
            self,
            hid_dim,
            num_classes,
            rel_type=None,
            sample_n= 5,
            K=4,
            use_shared_comp=True,
            activation='sigmoid',
            temperature_init=1.0
    ):
        super().__init__()
        self.out_dim = num_classes
        self.K = K
        self.use_shared = use_shared_comp
        self.activation = activation
        self.sample_n = sample_n
        self.rel_type = rel_type  # 保留接口一致性

        # 投影到球面并归一化
        self.proj = nn.Linear(hid_dim, hid_dim, bias=False)  # 投影维度与 hid_dim 一致

        # 分量中心 μ
        if self.use_shared:
            self.mu_shared = nn.Parameter(F.normalize(torch.randn(K, hid_dim), dim=-1))
            self.mu_delta  = nn.Parameter(torch.zeros(num_classes, K, hid_dim))
        else:
            self.mu_cls = nn.Parameter(F.normalize(torch.randn(num_classes, K, hid_dim), dim=-1))

        # 由 z 预测 κ 和 π
        self.kappa_head = nn.Linear(hid_dim, num_classes * K)
        self.pi_head    = nn.Linear(hid_dim, num_classes * K)

        # 温度
        self.tau = nn.Parameter(torch.tensor(float(temperature_init)))
        self._eps = 1e-9

    @torch.no_grad()
    def shrink_with_memory_prototypes(self, omega: torch.Tensor, alpha: float = 0.2):
        if omega is None:
            return
        assert omega.shape[0] == self.out_dim, "omega shape mismatch"
        omega = F.normalize(omega, dim=-1)
        if self.use_shared:
            mu = self.mu_shared.unsqueeze(0) + self.mu_delta
            mu = F.normalize((1 - alpha) * mu + alpha * omega.unsqueeze(1), dim=-1)
            self.mu_delta.data.copy_(mu - self.mu_shared.unsqueeze(0))
        else:
            mu = F.normalize((1 - alpha) * self.mu_cls + alpha * omega.unsqueeze(1), dim=-1)
            self.mu_cls.data.copy_(mu)

    def _get_mu(self):
        if self.use_shared:
            mu = self.mu_shared.unsqueeze(0) + self.mu_delta
        else:
            mu = self.mu_cls
        return F.normalize(mu, dim=-1)

    def forward(self, x, training=True, unc=False, n_samples=None):
        # 保留 sample_n 接口，但 vMF 不使用采样
        z = F.normalize(self.proj(x), dim=-1)   # [B, d]
        mu = self._get_mu()                     # [C, K, d]

        kappa = F.softplus(self.kappa_head(z)).view(z.size(0), self.out_dim, self.K) + 1e-6
        logits_pi = self.pi_head(z).view(z.size(0), self.out_dim, self.K)
        pi = F.softmax(logits_pi, dim=-1)

        dot = torch.einsum('bd,ckd->bck', z, mu)
        comp_score = kappa * dot
        s = torch.logsumexp(torch.log(pi + self._eps) + comp_score, dim=-1)
        s = s / self.tau.clamp_min(1e-3)

        if self.activation is None:
            pred = s
        elif self.activation.lower() == 'sigmoid':
            pred = torch.sigmoid(s)
        elif self.activation.lower() == 'softmax':
            pred = F.softmax(s, dim=-1)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

        if not unc:
            return pred

        # aleatoric 不确定性估计
        pi_entropy = (-pi * (pi.clamp_min(self._eps).log())).sum(-1)
        kappa_term = (pi / (1.0 + kappa)).sum(-1)
        U_al = (kappa_term + pi_entropy).detach()
        U_ep = torch.zeros_like(U_al)
        return pred, {"aleatoric": U_al, "epistemic": U_ep}
