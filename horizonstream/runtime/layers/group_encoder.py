from functools import partial
from typing import Callable, List, Optional, Type

import torch
import torch.nn as nn


# ------------------------- Residual Block -------------------------


class ResidualBlock(nn.Module):
    """Residual block used in DenseRepresentationEncoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act_layer: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.act = act_layer()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = out + identity
        return self.act(out)


# --------------------- Dense Representation Encoder ---------------------


class DenseRepresentationEncoder(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 1024,
        patch_size: int = 14,
        intermediate_dims: Optional[List[int]] = None,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Optional[Callable[..., nn.Module]] = partial(
            nn.LayerNorm, eps=1e-6
        ),
        pretrained_checkpoint_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        if intermediate_dims is None:
            intermediate_dims = [588, 768, 1024]
        self.intermediate_dims = intermediate_dims

        # (B, C, H, W) -> (B, C * P^2, H/P, W/P)
        self.unshuffle = nn.PixelUnshuffle(self.patch_size)
        self.conv_in = nn.Conv2d(
            in_channels=self.in_chans * (self.patch_size**2),
            out_channels=self.intermediate_dims[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Residual blocks
        layers: List[nn.Module] = []
        for i in range(len(self.intermediate_dims) - 1):
            layers.append(
                ResidualBlock(
                    in_channels=self.intermediate_dims[i],
                    out_channels=self.intermediate_dims[i + 1],
                    act_layer=act_layer,
                )
            )

        layers.append(
            nn.Conv2d(
                in_channels=self.intermediate_dims[-1],
                out_channels=self.embed_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )
        self.encoder = nn.Sequential(*layers)

        self.norm_layer = norm_layer(embed_dim) if norm_layer else nn.Identity()
        if isinstance(self.norm_layer, nn.LayerNorm):
            nn.init.constant_(self.norm_layer.bias, 0.0)
            nn.init.constant_(self.norm_layer.weight, 1.0)

        self._init_weights()

        # Load pretrained weights if provided
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        if self.pretrained_checkpoint_path is not None:
            print(
                f"Loading custom pretrained Dense Representation Encoder "
                f"checkpoint from {self.pretrained_checkpoint_path} ..."
            )
            ckpt = torch.load(self.pretrained_checkpoint_path, weights_only=False)
            print(self.load_state_dict(ckpt["model"], strict=False))

    def _init_weights(self) -> None:
        # 1) conv: Kaiming, bias=0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # 2) residual 的最后一层 conv2: zero-init（关键稳定技巧）
        for m in self.modules():
            if isinstance(m, ResidualBlock):
                nn.init.zeros_(m.conv2.weight)
                if m.conv2.bias is not None:
                    nn.init.zeros_(m.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: Tensor of shape (B, C, H, W)

        Returns:
            Tensor of shape (B, embed_dim, H // patch_size, W // patch_size)
        """
        assert isinstance(x, torch.Tensor), "Input must be a torch.Tensor"
        assert x.ndim == 4, "Input must be of shape (B, C, H, W)"
        assert x.shape[1] == self.in_chans, f"Input channels must be {self.in_chans}"
        B, _, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, (
            f"Input shape must be divisible by patch size={self.patch_size}, "
            f"got H={H}, W={W}"
        )

        # patchify + conv + residual blocks
        feats = self.unshuffle(x)  # (B, C * P^2, H/P, W/P)
        feats = self.conv_in(feats)
        feats = self.encoder(feats)  # (B, C, H/P, W/P)

        # (B, C, H', W') -> (B, N, C)
        feats = feats.flatten(2).transpose(1, 2)
        return self.norm_layer(feats)


# --------------------- Global Representation Encoder ---------------------


class GlobalRepresentationEncoder(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 1024,
        intermediate_dims: Optional[List[int]] = None,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Optional[Callable[..., nn.Module]] = partial(
            nn.LayerNorm, eps=1e-6
        ),
        pretrained_checkpoint_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        if intermediate_dims is None:
            intermediate_dims = [128, 256, 512]
        self.intermediate_dims = intermediate_dims

        # simple MLP
        layers: List[nn.Module] = []
        in_dim = self.in_chans
        for hidden_dim in self.intermediate_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_layer())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.embed_dim))
        self.encoder = nn.Sequential(*layers)

        # final norm
        self.norm_layer = norm_layer(embed_dim) if norm_layer else nn.Identity()
        if isinstance(self.norm_layer, nn.LayerNorm):
            nn.init.constant_(self.norm_layer.bias, 0.0)
            nn.init.constant_(self.norm_layer.weight, 1.0)

        # Load pretrained weights if provided
        if self.pretrained_checkpoint_path is not None:
            print(
                f"Loading pretrained Global Representation Encoder checkpoint "
                f"from {self.pretrained_checkpoint_path} ..."
            )
            ckpt = torch.load(self.pretrained_checkpoint_path, weights_only=False)
            print(self.load_state_dict(ckpt["model"], strict=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: Tensor of shape (B, C)

        Returns:
            Tensor of shape (B, embed_dim)
        """
        assert x.ndim == 2, "Input data must have shape (B, C)"
        assert (
            x.shape[1] == self.in_chans
        ), f"Input data must have {self.in_chans} channels"

        feats = self.encoder(x)
        return self.norm_layer(feats)


# --------------------------- Camera Encoder ---------------------------


class GroupEncoder(nn.Module):
    """Encode per-pixel raymap of intrisics rotations and centers into a dense feature map."""

    def __init__(
        self,
        embed_dim: int = 1024,
        patch_size: int = 14,
        dense_intermediate_dims: Optional[List[int]] = None,
        act_layer: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # output (B, H'*W', embed_dim)
        self.group_enc = DenseRepresentationEncoder(
            in_chans=3,
            embed_dim=embed_dim,
            patch_size=patch_size,
            intermediate_dims=dense_intermediate_dims,
            act_layer=act_layer,
            pretrained_checkpoint_path=None,
            norm_layer=None,
        )

        self.norm_layer = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(
        self,
        group_maps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            group_maps: (B, V, 9, H, W)

        Returns:
            features: (B, V*H'*W', embed_dim)
        """
        # Basic shape checks
        assert group_maps.ndim == 5, "Inputs must be 5D tensors"

        B, V_, C_in, H, W = group_maps.shape
        assert C_in == 3, f"Expected channel=3, got {C_in}"
        assert V_ % 3 == 0, "Expected V is divisible by 3."

        group_maps = group_maps.reshape(-1, 3, H, W)  # [B*3*V, 3, H, W]
        features = self.group_enc(group_maps)  # [B*3*V, H'*W', C]
        features = features.reshape(B, 3, -1, self.embed_dim).sum(1)  # [B, V*H'*W', C]
        return self.norm_layer(features) # [B, V*H'*W', C]
