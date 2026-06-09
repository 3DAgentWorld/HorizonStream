# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn.functional as F


def activate_camera_pose(pred_pose_enc, nonlinear_act):
    """
    Activate pose parameters with specified activation functions.

    Args:
        pred_pose_enc: Tensor containing encoded pose parameters [translation, quaternion, focal length]
        nonlinear_act: Tuple of nonlinear activation functions

    Returns:
        Activated pose parameters tensor
    """
    total_dim = sum(dim for _, dim, _ in nonlinear_act)
    assert total_dim == pred_pose_enc.shape[-1], "The number of dimensions of the predicted pose encoding does not match the number of dimensions of the nonlinear activation functions"
    
    start_idx = 0
    out = []
    for _, dim, act in nonlinear_act:
        end_idx = start_idx + dim
        out.append(base_pose_act(pred_pose_enc[..., start_idx:end_idx], act))
        start_idx = end_idx
    return torch.cat(out, dim=-1)


def activate_pose(pred_pose_enc, pose_encoding_type="absT_quaR_FoV", trans_act="linear", quat_act="linear", fl_act="relu", covis_act="linear", conf_act="expp1"):
    """
    Activate pose parameters with specified activation functions.

    Args:
        pred_pose_enc: Tensor containing encoded pose parameters [translation, quaternion, focal length]
        trans_act: Activation type for translation component
        quat_act: Activation type for quaternion component
        fl_act: Activation type for focal length component

    Returns:
        Activated pose parameters tensor
    """

    if pose_encoding_type == "absT_quaR_FoV":
        T = pred_pose_enc[..., :3]
        quat = pred_pose_enc[..., 3:7]
        fl = pred_pose_enc[..., 7:]  # or fov

        T = base_pose_act(T, trans_act)
        quat = base_pose_act(quat, quat_act)
        fl = base_pose_act(fl, fl_act)  # or fov
        quat = torch.nn.functional.normalize(quat, p=2, dim=-1, eps=1e-8)

        pred_pose_enc = torch.cat([T, quat, fl], dim=-1)
    else:
        raise ValueError(f"Unknown pose encoding type: {pose_encoding_type}")

    return pred_pose_enc


def base_pose_act(pose_enc, act_type="linear"):
    """
    Apply basic activation function to pose parameters.

    Args:
        pose_enc: Tensor containing encoded pose parameters
        act_type: Activation type ("linear", "inv_log", "exp", "relu")

    Returns:
        Activated pose parameters
    """
    if act_type == "linear":
        return pose_enc
    elif act_type == "inv_log":
        return inverse_log_transform(pose_enc)
    elif act_type == "exp":
        return torch.exp(pose_enc)
    elif act_type == "relu":
        return F.relu(pose_enc)
    elif act_type == "expp1":
        return torch.exp(pose_enc) + 1        
    else:
        raise ValueError(f"Unknown act_type: {act_type}")


def activate_head(
    out,
    activation="norm_exp",
    exps_max=6.9,  # math.log(1000.)
    conf_activation="expp1",
    conf_min=1.0,
    conf_max=100.,
    use_conf: bool = True,
):
    """
    Process network output to extract 3D points and confidence values.

    Args:
        out: Network output tensor (B, C, H, W)
        activation: Activation type for 3D points
        conf_activation: Activation type for confidence values

    Returns:
        Tuple of (3D points tensor, confidence tensor)
    """
    # Move channels from last dim to the 4th dimension => (B, H, W, C)
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,C expected

    # Split into xyz (first C-1 channels) and confidence (last channel)
    xyz = fmap[:, :, :, :-1] if use_conf else fmap
    conf = fmap[:, :, :, -1] if use_conf else None

    if activation in [
        'norm_exp', 'exp', 'inv_log', 'xy_inv_log'
    ]:
        # Clamp xyz to avoid overflow
        xyz = torch.clamp(xyz, max=exps_max)

    if activation == "norm_exp":
        d = xyz.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        xyz_normed = xyz / d
        pts3d = xyz_normed * torch.expm1(d)
    elif activation == "norm":
        pts3d = xyz / xyz.norm(dim=-1, keepdim=True)
    elif activation == "exp":
        pts3d = torch.exp(xyz)
    elif activation == "relu":
        pts3d = F.relu(xyz)
    elif activation == 'leaky_relu':
        pts3d = F.leaky_relu(xyz)
    elif activation == "inv_log":
        pts3d = inverse_log_transform(xyz)
    elif activation == "xy_inv_log":
        xy, z = xyz.split([2, 1], dim=-1)
        z = inverse_log_transform(z)
        pts3d = torch.cat([xy * z, z], dim=-1)
    elif activation == "sigmoid":
        pts3d = torch.sigmoid(xyz)
    elif activation == "linear":
        pts3d = xyz
    else:
        raise ValueError(f"Unknown activation: {activation}")

    if not use_conf: return pts3d, None
    
    if conf_activation == "expp1":
        conf_out = 1 + conf.exp()
    elif conf_activation == "expp0":
        conf_out = conf.exp()
    elif conf_activation == "sigmoid":
        conf_out = torch.sigmoid(conf)
    elif conf_activation == "elup2":
        conf_out = F.elu(conf) + 2
    elif conf_activation == "symelup2":
        conf_max = 32.
        conf_out = F.elu(conf)
        mask = conf > conf_max
        conf_out[mask] = conf_max - F.elu(conf_max - conf[mask])
        conf_out = conf_out + 2
    else:
        raise ValueError(f"Unknown conf_activation: {conf_activation}")

    return pts3d, conf_out


def inverse_log_transform(y):
    """
    Apply inverse log transform: sign(y) * (exp(|y|) - 1)

    Args:
        y: Input tensor

    Returns:
        Transformed tensor
    """
    return torch.sign(y) * (torch.expm1(torch.abs(y)))
