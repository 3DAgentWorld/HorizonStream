# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple, Optional

import math
from typing import List, Literal
import numpy as np

import torch
import torch.nn as nn

from ..layers import Mlp
from ..layers.block import Block
from .head_act import activate_camera_pose
from ..pose_utils import create_attn_mask


class CustomSequential(nn.Sequential):
    def forward(self, x, *inputs, kv_cache_list=None, **kwargs):
        """
        Forward pass through sequential modules.
        
        Args:
            x: Input tensor
            *inputs: Additional positional arguments
            kv_cache_list: Optional list of kv_cache for each layer. If provided, will be updated inplace.
            **kwargs: Additional keyword arguments
        """
        if kv_cache_list is not None:
            assert len(kv_cache_list) == len(self), "kv_cache_list must have the same length as the number of layers"
        for i, module in enumerate(self):
            kv_cache = kv_cache_list[i] if kv_cache_list is not None else None
            x = module(x, *inputs, kv_cache=kv_cache, **kwargs)
        return x


class CameraHead(nn.Module):
    """
    CameraHead predicts camera parameters from token representations using iterative refinement.

    It applies a series of transformer blocks (the "trunk") to dedicated camera tokens.
    """

    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
        scale_act: str = "relu",
        shift_act: str = "linear",
        conf_act: str = "expp1",
        covis_act: str = "linear",
        use_checkpoint=False,
        use_reentrant=True,
    ):
        super().__init__()

        if pose_encoding_type == "absT_quaR_FoV":
            self.nonlinear_def = [("trans", 3, trans_act), ("quat", 4, quat_act), ("fl", 2, fl_act)]
        elif pose_encoding_type == "absT_quaR":
            self.nonlinear_def = [("trans", 3, trans_act), ("quat", 4, quat_act)]
        elif pose_encoding_type == "absT_quaR_FoV_confTR":
            self.nonlinear_def = [("trans", 3, trans_act), ("quat", 4, quat_act), ("fl", 2, fl_act), ("conf", 2, conf_act)]
        elif pose_encoding_type == "FoV":
            self.nonlinear_def = [("fl", 2, fl_act)]
        elif pose_encoding_type == "absT_quaR_FoV_Scale":
            self.nonlinear_def = [("trans", 3, trans_act), ("quat", 4, quat_act), ("fl", 2, fl_act), ("dpt_scale", 1, scale_act)]
        elif pose_encoding_type == "absT_quaR_FoV_Affine":
            self.nonlinear_def = [("trans", 3, trans_act), ("quat", 4, quat_act), ("fl", 2, fl_act), ("dpt_scale", 1, scale_act), ("dpt_shift", 1, shift_act)]
        else:
            raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")
        self.pose_encoding_type = pose_encoding_type
    
        self.target_dim = sum(dim for _, dim, _ in self.nonlinear_def)
    
        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.conf_act = conf_act
        self.covis_act = covis_act
        self.trunk_depth = trunk_depth
        self.pose_encoding_type = pose_encoding_type

        # Gradient checkpointing
        self.use_checkpoint = use_checkpoint
        self.use_reentrant = use_reentrant

        # Build the trunk using a sequence of transformer blocks.
        self.trunk = CustomSequential(
            *[
                Block(
                    dim=dim_in,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                )
                for _ in range(trunk_depth)
            ]
        )

        # Normalizations for camera token and trunk output.
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)

        # Learnable empty camera pose token.
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in)

        # Module for producing modulation parameters: shift, scale, and a gate.
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))

        # Adaptive layer normalization without affine parameters.
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
        self.pose_branch = Mlp(
            in_features=dim_in,
            hidden_features=dim_in // 2,
            out_features=self.target_dim,
            drop=0,
        )

    def forward(self, aggregated_tokens_list: list, num_iterations: int = 4,
        attn_mode: str = "full",
        kv_cache_list: Optional[List[List[List[torch.Tensor]]]] = None) -> list:
        """
        Forward pass to predict camera parameters.

        Args:
            aggregated_tokens_list (list): List of token tensors from the network;
                the last tensor is used for prediction.
            num_iterations (int, optional): Number of iterative refinement steps. Defaults to 4.
            attn_mode (str): Global attention mode, could be either "causal", "window" or "full"
            kv_cache_list (List[List[List[torch.Tensor]]]): List of cached key-value pairs for 
                each iterations and each attention layer of the camera head 

        Returns:
            list: A list of predicted camera encodings (post-activation) from each iteration.
        """
        # Use tokens from the last block for camera prediction.
        tokens = aggregated_tokens_list[-1]

        # Extract the camera tokens
        pose_tokens = tokens[:, :, 0].cuda()
        pose_tokens = self.token_norm(pose_tokens)

        B, S, C = pose_tokens.shape
        attn_mask = None
        if kv_cache_list is None:
            attn_mask = create_attn_mask(S, P=1, mode=attn_mode, device=pose_tokens.device)

        pred_pose_enc_list = self.trunk_fn(pose_tokens, num_iterations, attn_mask, kv_cache_list)
        return pred_pose_enc_list

    def trunk_fn(self, pose_tokens: torch.Tensor, num_iterations: int,
        attn_mask: torch.Tensor, kv_cache_list: List[Tuple[torch.Tensor, torch.Tensor]] = None) -> list:
        """
        Iteratively refine camera pose predictions.

        Args:
            pose_tokens (torch.Tensor): Normalized camera tokens with shape [B, 1, C].
            num_iterations (int): Number of refinement iterations.
            attn_mask (torch.Tensor): attention mask
            kv_cache_list (List[List[List[torch.Tensor]]]): List of cached key-value pairs for 
                each iterations and each attention layer of the camera head 

        Returns:
            list: List of activated camera encodings from each iteration.
        """
        B, S, C = pose_tokens.shape
        pred_pose_enc = None
        pred_pose_enc_list = []

        if kv_cache_list is not None:
            assert len(kv_cache_list) == num_iterations, "kv_cache_list must have the same length as the number of iterations"

        for index in range(num_iterations):
            # Use a learned empty pose for the first iteration.
            if pred_pose_enc is None:
                module_input = self.embed_pose(self.empty_pose_tokens.expand(B, S, -1))
            else:
                # Detach the previous prediction to avoid backprop through time.
                pred_pose_enc = pred_pose_enc.detach()
                module_input = self.embed_pose(pred_pose_enc)

            # Generate modulation parameters and split them into shift, scale, and gate components.
            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)

            # Adaptive layer normalization and modulation.
            pose_tokens_modulated = gate_msa * modulate(self.adaln_norm(pose_tokens), shift_msa, scale_msa)
            pose_tokens_modulated = pose_tokens_modulated + pose_tokens

            if self.use_checkpoint:
                # Use gradient checkpointing to save memory.
                pose_tokens_modulated = torch.utils.checkpoint.checkpoint(
                    self.trunk,
                    pose_tokens_modulated,
                    attn_mask=attn_mask,
                    kv_cache_list=kv_cache_list[index] if kv_cache_list is not None else None,
                    use_reentrant=self.use_reentrant,
                )
            else:
                # Without checkpoint, we can use CustomSequential directly
                pose_tokens_modulated = self.trunk(
                    pose_tokens_modulated,
                    attn_mask=attn_mask,
                    kv_cache_list=kv_cache_list[index] if kv_cache_list is not None else None
                )

            # Compute the delta update for the pose encoding.
            pred_pose_enc_delta = self.pose_branch(self.trunk_norm(pose_tokens_modulated))

            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta

            # Apply final activation functions for translation, quaternion, and field-of-view.
            activated_pose = self.predict_pose(pred_pose_enc)
            pred_pose_enc_list.append(activated_pose)

        return pred_pose_enc_list
    
    def predict_pose(self, pred_pose_enc: torch.Tensor) -> torch.Tensor:
        activated_pose = activate_camera_pose(
            pred_pose_enc,
            self.nonlinear_def,
        )
        return activated_pose
        


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate the input tensor using scaling and shifting parameters.
    """
    # modified from https://github.com/facebookresearch/DiT/blob/796c29e532f47bba17c5b9c5eb39b9354b8b7c64/models.py#L19
    return x * (1 + scale) + shift

class BatchCameraHead(CameraHead):
    """
    CameraHead predicts camera parameters from token representations using iterative refinement.

    It applies a series of transformer blocks (the "trunk") to dedicated camera tokens.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # def forward(self, win_pose_tokens: torch.Tensor, chunk_idx: int, num_iterations: int = 4) -> list:
    #     def create_head_mask(Win: int, device: torch.device, dtype: torch.dtype = torch.float32,) -> torch.Tensor:
    #         row_head_num = Win * (Win - 1) // 2
    #         neg = torch.finfo(dtype).min
    #         mask_head = torch.full((row_head_num, Win, Win), neg, dtype=dtype, device=device)
    #         j_vals = torch.arange(1, Win, device=device, dtype=torch.long)
    #         row_j = j_vals.repeat_interleave(j_vals)[:, None]  # (row_head_num, 1)

    #         q = torch.arange(Win, device=device, dtype=torch.long)[None, :]  # (1, Win)
    #         valid_q = (q < row_j)  # (row_head_num, Win)
    #         valid = valid_q[:, :, None] & valid_q[:, None, :]
            
    #         mask_head.masked_fill_(valid, 0.0)
    #         mask_head[..., 0].masked_fill_(~valid_q, 0.0)
    #         return mask_head.unsqueeze(1) # explicit head dim
        
    #     B, row_num, Win, C = win_pose_tokens.shape
        
    #     win_pose_tokens = self.token_norm(win_pose_tokens)
    #     row_head_num = Win * (Win - 1) // 2
            
    #     if chunk_idx != 0:
    #         win_pose_tokens = win_pose_tokens.reshape(B * row_num, Win, C)
    #         pred_pose_enc_list = self.trunk_fn(win_pose_tokens, num_iterations, attn_mask=None)
    #     else:
    #         head_pose_tokens = win_pose_tokens[:, :row_head_num].flatten(0, 1) # (B * row_head_num, Win, C)
    #         head_attn_mask = create_head_mask(Win, win_pose_tokens.device, win_pose_tokens.dtype)
    #         head_attn_mask = head_attn_mask.unsqueeze(0).expand(B, -1, -1, -1, -1).flatten(0, 1) # (B * row_head_num, 1, Win, Win)
    #         head_enc_list = self.trunk_fn(head_pose_tokens, num_iterations, attn_mask=head_attn_mask)
    #         tail_pose_tokens = win_pose_tokens[:, row_head_num:].flatten(0, 1)
    #         tail_enc_list = self.trunk_fn(tail_pose_tokens, num_iterations, attn_mask=None)
            
    #         pred_pose_enc_list = []
    #         for i in range(num_iterations):
    #             h = head_enc_list[i].view(B, -1, Win, self.target_dim)
    #             t = tail_enc_list[i].view(B, -1, Win, self.target_dim)
    #             pred_pose_enc_list.append(torch.cat([h, t], dim=1).flatten(0, 1))  # (B*row_num, Win, self.target_dim)
    #     return pred_pose_enc_list
    
    def forward(self, win_pose_tokens: torch.Tensor, chunk_idx: int, num_iterations: int = 4) -> list:        
        _, Fr, Win, _ = win_pose_tokens.shape
        
        win_pose_tokens = win_pose_tokens.flatten(0, 1) # (B * F, Win, 2C)
        win_pose_tokens = self.token_norm(win_pose_tokens)
        
        attn_mask = None
        if chunk_idx == 0:
            assert Fr == Win, f"first chunk F: {Fr}, Win: {Win}"
            device = win_pose_tokens.device
            steps = torch.arange(Win, device=device)
            limit = (steps + 1).view(Win, 1, 1) # Win==F
            grid_idx = steps.view(1, -1) # (1, Win)
            attn_mask = (grid_idx.unsqueeze(-1) < limit) & (grid_idx.unsqueeze(0) < limit)
            attn_mask[:, :, 0] = True # avoid no valid attn for tail query
            attn_mask = attn_mask.unsqueeze(1) # (Win==F, 1, Win, Win) broadcast for head and batch dim
        pred_pose_enc_list = self.trunk_fn(win_pose_tokens, num_iterations, attn_mask=attn_mask)
        
        return pred_pose_enc_list
