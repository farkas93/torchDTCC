import numpy as np
import torch
from typing import Optional, Callable

def check_shape(x):
    assert x.ndim == 3, f"Input must be [batch, seq_len, features], got {x.shape}"

def ensure_float32(x):
    if x.dtype != np.float32:
        return x.astype(np.float32)
    return x

def torch_augmentation_wrapper(
    aug_fn: Callable, 
    x: torch.Tensor, 
    labels: Optional[torch.Tensor] = None,
    **kwargs
) -> torch.Tensor:
    """
    Wrapper to apply numpy augmentations on torch tensors.
    """
    x_np = x.detach().cpu().numpy()
    if labels is not None:
        labels_np = labels.detach().cpu().numpy()
        aug_np = aug_fn(x_np, labels_np, **kwargs)
    else:
        aug_np = aug_fn(x_np, **kwargs)
    return torch.from_numpy(aug_np).to(x.device).type_as(x)

# def random_time_series_augmentation(
#     x: np.ndarray,
#     labels: Optional[np.ndarray] = None,
#     random_state: Optional[int] = None,
#     allow_label_dependent: bool = False
# ) -> np.ndarray:
#     """
#     Randomly applies a time-series augmentation.
#     If labels are provided and allow_label_dependent=True, may select label-dependent augmentations.
#     """
#     _check_shape(x)
#     rng = np.random.RandomState(random_state)
#     # List of (func, needs_labels)
#     aug_funcs: List[Tuple[Callable, bool]] = [
#         (jitter, False),
#         (scaling, False),
#         (rotation, False),
#         (permutation, False),
#         (magnitude_warp, False),
#         (time_warp, False),
#         (window_slice, False),
#         (window_warp, False),
#     ]
#     if allow_label_dependent and labels is not None:
#         # These require labels and external DTW code, so only include if possible
#         try:
#             from utils import dtw  # Will raise ImportError if not available
#             aug_funcs.extend([
#                 # (spawner, True),  # Uncomment if dtw is available and tested
#                 # (wdba, True),
#                 # (random_guided_warp, True),
#                 # (random_guided_warp_shape, True),
#                 # (discriminative_guided_warp, True),
#                 # (discriminative_guided_warp_shape, True),
#             ])
#         except ImportError:
#             pass

#     idx = rng.randint(0, len(aug_funcs))
#     func, needs_labels = aug_funcs[idx]
#     if needs_labels and labels is not None:
#         return func(x, labels, random_state=random_state)
#     return func(x, random_state=random_state)
