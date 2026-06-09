import torch
import torch.nn.functional as F


class SafeCat(torch.autograd.Function):
    """A safe version of torch.cat that prevents NaN/Inf gradients from propagating.

    This function acts as a 'fuse' - it performs normal concatenation in forward pass,
    but in backward pass it replaces any NaN/Inf gradients with zeros before passing
    them back to the input tensors.
    """

    @staticmethod
    def forward(ctx, *args):

        *xs, dim = args
        ctx.dim = dim
        ctx.sizes = [x.size(dim) for x in xs]
        return torch.cat(xs, dim=dim)

    @staticmethod
    def backward(ctx, g_out):

        dxs = []
        s = 0
        for k in ctx.sizes:
            sl = [slice(None)] * g_out.ndim
            sl[ctx.dim] = slice(s, s + k)
            g = g_out[tuple(sl)]

            g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            dxs.append(g)
            s += k

        return (*dxs, None)


def activate_pose(
    pred_pose_enc, trans_act="linear", quat_act="linear", fl_act="linear"
):
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

    T = base_pose_act(pred_pose_enc[..., :3], trans_act)
    quat = base_pose_act(pred_pose_enc[..., 3:7], quat_act)
    fl = base_pose_act(pred_pose_enc[..., 7:], fl_act)

    quat = torch.nn.functional.normalize(quat, p=2, dim=-1, eps=1e-8)

    pred_pose_enc = SafeCat.apply(T, quat, fl, -1)

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
    else:
        raise ValueError(f"Unknown act_type: {act_type}")


def activate_head(out, activation="norm_exp", conf_activation="expp1"):
    """
    Process network output to extract 3D points and confidence values.

    Args:
        out: Network output tensor (B, C, H, W)
        activation: Activation type for 3D points
        conf_activation: Activation type for confidence values

    Returns:
        Tuple of (3D points tensor, confidence tensor)
    """

    fmap = out.permute(0, 2, 3, 1)

    xyz = fmap[:, :, :, :-1]
    conf = fmap[:, :, :, -1]

    if activation == "norm_exp":
        d = xyz.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        pts3d = (xyz / d) * torch.expm1(d)
    elif activation == "norm":
        pts3d = xyz / xyz.norm(dim=-1, keepdim=True)
    elif activation == "exp":
        pts3d = torch.exp(xyz)
    elif activation == "relu":
        pts3d = F.relu(xyz)
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

    if conf_activation == "expp1":
        conf_out = 1 + conf.exp()
    elif conf_activation == "expp0":
        conf_out = conf.exp()
    elif conf_activation == "sigmoid":
        conf_out = torch.sigmoid(conf)
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
