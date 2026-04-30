from __future__ import annotations

import torch


def make_transform(position: torch.Tensor, quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    """Build homogeneous transforms from positions and quaternions.

    Args:
        position: Translation tensor with shape ``(..., 3)``.
        quaternion_wxyz: Orientation tensor with shape ``(..., 4)`` in
            ``(w, x, y, z)`` order. The orientation maps local-frame vectors to
            the parent/world frame.

    Returns:
        Homogeneous transform tensor with shape ``(..., 4, 4)``.
    """
    from .rotations import quaternion_wxyz_to_matrix

    rotation = quaternion_wxyz_to_matrix(quaternion_wxyz)
    batch_shape = position.shape[:-1]
    transform = torch.zeros(*batch_shape, 4, 4, dtype=position.dtype, device=position.device)
    transform[..., :3, :3] = rotation
    transform[..., :3, 3] = position
    transform[..., 3, 3] = 1.0
    return transform


def ensure_batch(tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Ensure a tensor has a leading batch dimension.

    Args:
        tensor: Tensor with either unbatched shape ``(D,)`` or batched shape
            ``(B, D)``. Higher-rank tensors are treated as already batched.

    Returns:
        Tuple ``(batched_tensor, was_single)``. ``batched_tensor`` has shape
        ``(1, D)`` when the input was unbatched, otherwise it is returned
        unchanged. ``was_single`` indicates whether a singleton batch dimension
        was added.
    """
    if tensor.ndim == 1:
        return tensor.unsqueeze(0), True
    return tensor, False
