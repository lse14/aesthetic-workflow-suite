import torch
import torch.nn as nn


class FusionMultiTaskHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | tuple[int, ...] = (1024, 256),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend(
                [
                    nn.LayerNorm(prev),
                    nn.Linear(prev, h),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.reg_heads = nn.ModuleDict(
            {
                "aesthetic": nn.Linear(prev, 1),
                "composition": nn.Linear(prev, 1),
                "color": nn.Linear(prev, 1),
                "sexual": nn.Linear(prev, 1),
            }
        )
        self.cls_head = nn.Linear(prev, 1)

    def forward(self, fused_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.trunk(fused_features)
        out = [
            self.reg_heads[name](x)
            for name in ("aesthetic", "composition", "color", "sexual")
        ]
        reg_raw = torch.cat(out, dim=-1)
        reg_pred = torch.sigmoid(reg_raw) * 4.0 + 1.0
        cls_logit = self.cls_head(x).squeeze(-1)
        return reg_pred, cls_logit


class FusionRegressorHead(FusionMultiTaskHead):
    # Backward-compatible wrapper for old imports.
    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        reg_pred, _ = super().forward(fused_features)
        return reg_pred
