import torch
import torch.nn as nn
import torch.nn.functional as F

class GMMHeadPlus(nn.Module):
    """
    TEMPURA GMM head — upgraded (same interface as original):
    - __init__(hid_dim, num_classes, k=4, activation='sigmoid')
    - forward(x, phase='train', unc=False, sample_n=1)

    Internal upgrades:
    - Class-wise mixture weights π per class (softmax over K)
    - Variance via softplus + clamp (floor/ceiling)
    - Optional temperature τ for calibration
    - Uncertainty hooks (aleatoric/epistemic proxy) & regularizers
    """
    def __init__(
            self,
            hid_dim,
            num_classes,
            k = 4,
            activation = 'sigmoid',
            # Extra knobs with defaults to stay backward-compatible:
            var_floor = 1e-3,
            var_ceiling = 5.0,
            classwise_pi = True,
            learnable_tau = True,
            init_tau = 1.0,
            *args, **kwargs,
    ):
        super().__init__()
        self.hid_dim = hid_dim
        self.num_classes = num_classes
        self.k = k
        self.activation_type = activation
        self.var_floor = var_floor
        self.var_ceiling = var_ceiling
        self.classwise_pi = classwise_pi
        self.learnable_tau = learnable_tau

        # Per-component projection heads (match original naming schema mu_i, pi_i, var_i)
        self.heads = nn.ModuleDict()
        for i in range(1, k + 1):
            self.heads[f"mu_{i}"]  = nn.Linear(hid_dim, num_classes)
            pi_out = num_classes if classwise_pi else 1
            self.heads[f"pi_{i}"]  = nn.Linear(hid_dim, pi_out)
            self.heads[f"var_{i}"] = nn.Linear(hid_dim, num_classes)

        # Optional temperature parameter for calibration (train/test consistency)
        if learnable_tau:
            self.tau = nn.Parameter(torch.tensor(float(init_tau)))
        else:
            self.register_buffer('tau', torch.tensor(float(init_tau)), persistent=False)

        # Activation to mirror original semantics
        if activation is None:
            self.activation = lambda x: x
        elif activation.lower() == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation.lower() == 'softmax':
            self.activation = lambda x: F.softmax(x, dim=-1)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self._eps = 1e-9

    @torch.no_grad()
    def set_temperature(self, tau: float):
        if self.learnable_tau:
            self.tau.data.fill_(float(tau))
        else:
            self.tau = torch.tensor(float(tau), device=self.tau.device)

    def _compute_params(self, x):
        """
        Compute μ, π, var for all components.
        Returns:
            conf_mu (list of tensors): K items of [B, C]
            conf_pi_norm (tensor): [B, K, C] (class-wise over K)
            conf_var (list of tensors): K items of [B, C]
        """
        mu_list, var_list, pi_list = [], [], []
        for i in range(1, self.k + 1):
            mu_i = self.heads[f"mu_{i}"](x)                     # [B, C]
            raw_var_i = self.heads[f"var_{i}"](x)               # [B, C]
            var_i = F.softplus(raw_var_i) + 1e-6
            var_i = var_i.clamp(min=self.var_floor, max=self.var_ceiling)

            pi_i = self.heads[f"pi_{i}"](x)                     # [B, C] or [B,1]
            if not self.classwise_pi:
                pi_i = pi_i.expand(-1, self.num_classes)

            mu_list.append(mu_i)
            var_list.append(var_i)
            pi_list.append(pi_i)

        pi_stack = torch.stack(pi_list, dim=1)  # [B, K, C]
        pi_norm  = torch.softmax(pi_stack, dim=1)
        return mu_list, pi_norm, var_list

    def forward(self, x, phase: str = 'train', unc: bool = False, sample_n: int = 1):
        """Same signature as original forward.
        Args:
            x: [B, hid_dim]
            phase: 'train' or 'test' (any non-'train' treated as eval)
            unc: if True, also return uncertainty dict
            sample_n: kept for API compatibility; we use one sample per forward
        Returns:
            pred  (and optionally uncertainty dict)
        """
        mu_list, pi_norm, var_list = self._compute_params(x)

        comp_logits = []
        if phase == 'train':
            for k in range(self.k):
                eps = torch.randn_like(var_list[k])
                logits_k = mu_list[k] + eps * torch.sqrt(var_list[k])
                comp_logits.append(logits_k)
        else:
            for k in range(self.k):
                comp_logits.append(mu_list[k])

        tau = self.tau.clamp_min(1e-3)
        comp_logits = [l / tau for l in comp_logits]

        comp_probs = [self.activation(l) for l in comp_logits]   # each [B, C]
        probs_stack = torch.stack(comp_probs, dim=1)             # [B, K, C]
        weighted = (pi_norm * probs_stack).sum(dim=1)            # [B, C]
        pred = weighted

        if not unc:
            return pred

        with torch.no_grad():
            var_stack = torch.stack(var_list, dim=1)             # [B, K, C]
            kappa_term = (pi_norm / (1.0 + var_stack)).sum(dim=1)
            pi_entropy = (-(pi_norm * (pi_norm.clamp_min(self._eps).log()))).sum(dim=1)
            U_al = kappa_term + pi_entropy
            U_ep = torch.zeros_like(U_al)
        return pred, {"aleatoric": U_al, "epistemic": U_ep}

    def uncertainty(self, x):
        """Match original-style hook to fetch uncertainties given features x."""
        with torch.no_grad():
            _, pi_norm, var_list = self._compute_params(x)
            var_stack = torch.stack(var_list, dim=1)
            kappa_term = (pi_norm / (1.0 + var_stack)).sum(dim=1)
            pi_entropy = (-(pi_norm * (pi_norm.clamp_min(self._eps).log()))).sum(dim=1)
            U_al = kappa_term + pi_entropy
            U_ep = torch.zeros_like(U_al)
        return U_al, U_ep

    def regularizers(self, pi_weight: float = 0.0, var_weight: float = 0.0):
        """Optional regularization terms; call and add to total loss externally."""
        reg = 0.0
        with torch.no_grad():
            device = next(self.parameters()).device
            dummy = torch.zeros(1, self.hid_dim, device=device)
            _, pi_norm, var_list = self._compute_params(dummy)
        if pi_weight > 0.0:
            pi_entropy = (-(pi_norm * (pi_norm.clamp_min(self._eps).log()))).sum(dim=1).mean()
            reg = reg - pi_weight * pi_entropy
        if var_weight > 0.0:
            var_stack = torch.stack(var_list, dim=0)
            reg = reg + var_weight * var_stack.pow(2).mean()
        return reg
