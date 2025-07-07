import numpy as np
from tqdm import tqdm
from typing import Optional, Callable, List, Tuple, Union
from timeseries.dtw import DTW

def _check_shape(x):
    assert x.ndim == 3, f"Input must be [batch, seq_len, features], got {x.shape}"

def _ensure_float32(x):
    if x.dtype != np.float32:
        return x.astype(np.float32)
    return x

def jitter(x, sigma=0.03, random_state=None):
    """
    Adds Gaussian noise to the input.
    Reference: https://arxiv.org/pdf/1706.00527.pdf
    """
    _check_shape(x)
    rng = np.random.RandomState(random_state)
    return _ensure_float32(x + rng.normal(loc=0., scale=sigma, size=x.shape))

def scaling(x, sigma=0.1, random_state=None):
    """
    Multiplies input by a random scaling factor for each feature.
    Reference: https://arxiv.org/pdf/1706.00527.pdf
    """
    _check_shape(x)
    rng = np.random.RandomState(random_state)
    factor = rng.normal(loc=1., scale=sigma, size=(x.shape[0], x.shape[2]))
    return _ensure_float32(np.multiply(x, factor[:, np.newaxis, :]))

def rotation(x, random_state=None):
    """
    Randomly flips and permutes feature axes.
    """
    _check_shape(x)
    rng = np.random.RandomState(random_state)
    flip = rng.choice([-1, 1], size=(x.shape[0], x.shape[2]))
    rotate_axis = np.arange(x.shape[2])
    rng.shuffle(rotate_axis)
    return _ensure_float32(flip[:, np.newaxis, :] * x[:, :, rotate_axis])

def permutation(x, max_segments=5, seg_mode="equal", random_state=None):
    """
    Randomly permutes segments of the time series.
    """
    _check_shape(x)
    rng = np.random.RandomState(random_state)
    orig_steps = np.arange(x.shape[1])
    num_segs = rng.randint(1, max_segments, size=(x.shape[0]))
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = rng.choice(x.shape[1]-2, num_segs[i]-1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(rng.permutation(splits)).ravel()
            ret[i] = pat[warp]
        else:
            ret[i] = pat
    return _ensure_float32(ret)

def magnitude_warp(x, sigma=0.2, knot=4, random_state=None):
    """
    Warps the magnitude of the series using random smooth curves.
    """
    from scipy.interpolate import CubicSpline
    _check_shape(x)
    rng = np.random.RandomState(random_state)
    orig_steps = np.arange(x.shape[1])
    random_warps = rng.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        warper = np.array([CubicSpline(warp_steps[:,dim], random_warps[i,:,dim])(orig_steps) for dim in range(x.shape[2])]).T
        ret[i] = pat * warper
    return _ensure_float32(ret)

def time_warp(x, sigma=0.2, knot=4, random_state=None):
    """
    Warps the time axis using smooth curves.
    """
    from scipy.interpolate import CubicSpline
    _check_shape(x)
    rng = np.random.RandomState(random_state)
    orig_steps = np.arange(x.shape[1])
    random_warps = rng.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp_curve = CubicSpline(warp_steps[:,dim], warp_steps[:,dim] * random_warps[i,:,dim])(orig_steps)
            scale = (x.shape[1]-1)/time_warp_curve[-1] if time_warp_curve[-1] != 0 else 1.0
            ret[i,:,dim] = np.interp(orig_steps, np.clip(scale*time_warp_curve, 0, x.shape[1]-1), pat[:,dim]).T
    return _ensure_float32(ret)

def window_slice(x, reduce_ratio=0.9, random_state=None):
    """
    Randomly slices a window from the time series and rescales back to original size.
    Reference: https://halshs.archives-ouvertes.fr/halshs-01357973/document
    """
    _check_shape(x)
    rng = np.random.RandomState(random_state)
    target_len = np.ceil(reduce_ratio*x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return _ensure_float32(x)
    starts = rng.randint(low=0, high=x.shape[1]-target_len, size=(x.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            ret[i,:,dim] = np.interp(np.linspace(0, target_len-1, num=x.shape[1]), np.arange(target_len), pat[starts[i]:ends[i],dim]).T
    return _ensure_float32(ret)

def window_warp(x, window_ratio=0.1, scales=[0.5, 2.], random_state=None):
    """
    Warps a random window in the series by a random scale and resamples to original length.
    Reference: https://halshs.archives-ouvertes.fr/halshs-01357973/document
    """
    _check_shape(x)
    rng = np.random.RandomState(random_state)
    warp_scales = rng.choice(scales, x.shape[0])
    warp_size = np.ceil(window_ratio*x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)
    window_starts = rng.randint(low=1, high=x.shape[1]-warp_size-1, size=(x.shape[0])).astype(int)
    window_ends = (window_starts + warp_size).astype(int)
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            start_seg = pat[:window_starts[i],dim]
            window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales[i])), window_steps, pat[window_starts[i]:window_ends[i],dim])
            end_seg = pat[window_ends[i]:,dim]
            warped = np.concatenate((start_seg, window_seg, end_seg))
            ret[i,:,dim] = np.interp(np.arange(x.shape[1]), np.linspace(0, warped.size-1, num=x.shape[1]), warped).T
    return _ensure_float32(ret)

def random_time_series_augmentation(
    x: np.ndarray,
    labels: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
    allow_label_dependent: bool = False
) -> np.ndarray:
    """
    Randomly applies a time-series augmentation.
    If labels are provided and allow_label_dependent=True, may select label-dependent augmentations.
    """
    _check_shape(x)
    rng = np.random.RandomState(random_state)
    # List of (func, needs_labels)
    aug_funcs: List[Tuple[Callable, bool]] = [
        (jitter, False),
        (scaling, False),
        (rotation, False),
        (permutation, False),
        (magnitude_warp, False),
        (time_warp, False),
        (window_slice, False),
        (window_warp, False),
    ]
    if allow_label_dependent and labels is not None:
        # These require labels and external DTW code, so only include if possible
        try:
            from utils import dtw  # Will raise ImportError if not available
            aug_funcs.extend([
                # (spawner, True),  # Uncomment if dtw is available and tested
                # (wdba, True),
                # (random_guided_warp, True),
                # (random_guided_warp_shape, True),
                # (discriminative_guided_warp, True),
                # (discriminative_guided_warp_shape, True),
            ])
        except ImportError:
            pass

    idx = rng.randint(0, len(aug_funcs))
    func, needs_labels = aug_funcs[idx]
    if needs_labels and labels is not None:
        return func(x, labels, random_state=random_state)
    return func(x, random_state=random_state)

def torch_augmentation_wrapper(
    aug_fn: Callable, 
    x: 'torch.Tensor', 
    labels: Optional['torch.Tensor'] = None,
    **kwargs
) -> 'torch.Tensor':
    """
    Wrapper to apply numpy augmentations on torch tensors.
    """
    import torch
    x_np = x.detach().cpu().numpy()
    if labels is not None:
        labels_np = labels.detach().cpu().numpy()
        aug_np = aug_fn(x_np, labels_np, **kwargs)
    else:
        aug_np = aug_fn(x_np, **kwargs)
    return torch.from_numpy(aug_np).to(x.device).type_as(x)

# Example docstring for a label-dependent augmentation:
def spawner(x, labels, sigma=0.05, verbose=0, random_state=None) -> Tuple[np.ndarray, List[int]]:
    """
    SPAWNER augmentation: averages two intra-class patterns using DTW path.
    Returns:
        - Augmented data (np.ndarray)
        - List of indices where no augmentation was possible (skipped indices)
    Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6983028/
    """
    _check_shape(x)
    rng = np.random.RandomState(random_state)
    random_points = rng.randint(low=1, high=x.shape[1]-1, size=x.shape[0])
    window = np.ceil(x.shape[1] / 10.).astype(int)
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
    ret = np.zeros_like(x)
    skipped = []
    for i, pat in enumerate(tqdm(x)):
        choices = np.delete(np.arange(x.shape[0]), i)
        choices = np.where(l[choices] == l[i])[0]
        if choices.size > 0:
            random_sample = x[rng.choice(choices)]
            path1 = DTW.dtw(pat[:random_points[i]], random_sample[:random_points[i]], DTW.RETURN_PATH, slope_constraint="symmetric", window=window)
            path2 = DTW.dtw(pat[random_points[i]:], random_sample[random_points[i]:], DTW.RETURN_PATH, slope_constraint="symmetric", window=window)
            combined = np.concatenate((np.vstack(path1), np.vstack(path2+random_points[i])), axis=1)
            mean = np.mean([pat[combined[0]], random_sample[combined[1]]], axis=0)
            for dim in range(x.shape[2]):
                ret[i,:,dim] = np.interp(orig_steps, np.linspace(0, x.shape[1]-1., num=mean.shape[0]), mean[:,dim]).T
        else:
            if verbose > -1:
                print(f"Only one pattern of class {l[i]}, skipping pattern average.")
            ret[i,:] = pat
            skipped.append(i)
    return _ensure_float32(jitter(ret, sigma=sigma, random_state=random_state)), skipped

def wdba(
    x: np.ndarray, 
    labels: np.ndarray, 
    batch_size: int = 6, 
    slope_constraint: str = "symmetric", 
    use_window: bool = True, 
    verbose: int = 0, 
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, List[int]]:
    """
    Weighted DBA (DTW Barycenter Averaging) augmentation.
    Returns augmented data and list of skipped indices.
    Reference: https://ieeexplore.ieee.org/document/8215569
    """
    _check_shape(x)
    rng = np.random.RandomState(random_state)
    window = np.ceil(x.shape[1] / 10.).astype(int) if use_window else None
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
    ret = np.zeros_like(x)
    skipped = []
    for i in tqdm(range(ret.shape[0])):
        choices = np.where(l == l[i])[0]
        if choices.size > 0:
            k = min(choices.size, batch_size)
            random_prototypes = x[rng.choice(choices, k, replace=False)]
            # DTW matrix
            dtw_matrix = np.zeros((k, k))
            for p, prototype in enumerate(random_prototypes):
                for s, sample in enumerate(random_prototypes):
                    if p == s:
                        dtw_matrix[p, s] = 0.
                    else:
                        dtw_matrix[p, s] = DTW.dtw(
                            prototype, sample, DTW.RETURN_VALUE, 
                            slope_constraint=slope_constraint, window=window
                        )
            medoid_id = np.argsort(np.sum(dtw_matrix, axis=1))[0]
            nearest_order = np.argsort(dtw_matrix[medoid_id])
            medoid_pattern = random_prototypes[medoid_id]
            average_pattern = np.zeros_like(medoid_pattern)
            weighted_sums = np.zeros((medoid_pattern.shape[0]))
            for nid in nearest_order:
                if nid == medoid_id or dtw_matrix[medoid_id, nearest_order[1]] == 0.:
                    average_pattern += medoid_pattern
                    weighted_sums += np.ones_like(weighted_sums)
                else:
                    idx1, idx2 = DTW.dtw(
                        medoid_pattern, random_prototypes[nid], DTW.RETURN_PATH,
                        slope_constraint=slope_constraint, window=window
                    )
                    dtw_value = dtw_matrix[medoid_id, nid]
                    warped = random_prototypes[nid][idx2]
                    weight = np.exp(np.log(0.5) * dtw_value / (dtw_matrix[medoid_id, nearest_order[1]] + 1e-8))
                    average_pattern[idx1] += weight * warped
                    weighted_sums[idx1] += weight
            ret[i, :] = average_pattern / (weighted_sums[:, np.newaxis] + 1e-8)
        else:
            if verbose > -1:
                print(f"Only one pattern of class {l[i]}, skipping pattern average.")
            ret[i, :] = x[i]
            skipped.append(i)
    return _ensure_float32(ret), skipped

def random_guided_warp(
    x: np.ndarray, 
    labels: np.ndarray, 
    slope_constraint: str = "symmetric", 
    use_window: bool = True, 
    dtw_type: str = "normal", 
    verbose: int = 0, 
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, List[int]]:
    """
    Random guided DTW warping using intra-class prototypes.
    Returns augmented data and list of skipped indices.
    """
    _check_shape(x)
    rng = np.random.RandomState(random_state)
    window = np.ceil(x.shape[1] / 10.).astype(int) if use_window else None
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
    ret = np.zeros_like(x)
    skipped = []
    for i, pat in enumerate(tqdm(x)):
        choices = np.delete(np.arange(x.shape[0]), i)
        choices = np.where(l[choices] == l[i])[0]
        if choices.size > 0:
            random_prototype = x[rng.choice(choices)]
            if dtw_type == "shape":
                raise NotImplementedError("shapeDTW is not implemented.")
            else:
                idx1, idx2 = DTW.dtw(random_prototype, pat, DTW.RETURN_PATH, slope_constraint=slope_constraint, window=window)
            warped = pat[idx2]
            for dim in range(x.shape[2]):
                ret[i, :, dim] = np.interp(orig_steps, np.linspace(0, x.shape[1]-1., num=warped.shape[0]), warped[:, dim]).T
        else:
            if verbose > -1:
                print(f"Only one pattern of class {l[i]}, skipping timewarping.")
            ret[i, :] = pat
            skipped.append(i)
    return _ensure_float32(ret), skipped

def discriminative_guided_warp(
    x: np.ndarray, 
    labels: np.ndarray, 
    batch_size: int = 6, 
    slope_constraint: str = "symmetric", 
    use_window: bool = True, 
    dtw_type: str = "normal", 
    use_variable_slice: bool = True, 
    verbose: int = 0, 
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, List[int]]:
    """
    Discriminative guided warping based on intra- and inter-class prototypes.
    Returns augmented data and list of skipped indices.
    """
    _check_shape(x)
    rng = np.random.RandomState(random_state)
    window = np.ceil(x.shape[1] / 10.).astype(int) if use_window else None
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
    positive_batch = int(np.ceil(batch_size / 2))
    negative_batch = int(np.floor(batch_size / 2))
    ret = np.zeros_like(x)
    warp_amount = np.zeros(x.shape[0])
    skipped = []
    for i, pat in enumerate(tqdm(x)):
        choices = np.delete(np.arange(x.shape[0]), i)
        positive = np.where(l[choices] == l[i])[0]
        negative = np.where(l[choices] != l[i])[0]
        if positive.size > 0 and negative.size > 0:
            pos_k = min(positive.size, positive_batch)
            neg_k = min(negative.size, negative_batch)
            positive_prototypes = x[rng.choice(positive, pos_k, replace=False)]
            negative_prototypes = x[rng.choice(negative, neg_k, replace=False)]
            pos_aves = np.zeros((pos_k))
            neg_aves = np.zeros((pos_k))
            if dtw_type == "shape":
                raise NotImplementedError("shapeDTW is not implemented.")
            else:
                for p, pos_prot in enumerate(positive_prototypes):
                    for ps, pos_samp in enumerate(positive_prototypes):
                        if p != ps:
                            pos_aves[p] += (1./(pos_k-1.))*DTW.dtw(pos_prot, pos_samp, DTW.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                    for ns, neg_samp in enumerate(negative_prototypes):
                        neg_aves[p] += (1./neg_k)*DTW.dtw(pos_prot, neg_samp, DTW.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                selected_id = np.argmax(neg_aves - pos_aves)
                idx1, idx2 = DTW.dtw(positive_prototypes[selected_id], pat, DTW.RETURN_PATH, slope_constraint=slope_constraint, window=window)
            warped = pat[idx2]
            warp_path_interp = np.interp(orig_steps, np.linspace(0, x.shape[1]-1., num=warped.shape[0]), idx2)
            warp_amount[i] = np.sum(np.abs(orig_steps-warp_path_interp))
            for dim in range(x.shape[2]):
                ret[i, :, dim] = np.interp(orig_steps, np.linspace(0, x.shape[1]-1., num=warped.shape[0]), warped[:, dim]).T
        else:
            if verbose > -1:
                print(f"Only one pattern of class {l[i]}, skipping discriminative warping.")
            ret[i, :] = pat
            warp_amount[i] = 0.
            skipped.append(i)
    if use_variable_slice:
        max_warp = np.max(warp_amount)
        if max_warp == 0:
            ret = window_slice(ret, reduce_ratio=0.9)
        else:
            for i, pat in enumerate(ret):
                ret[i] = window_slice(pat[np.newaxis, :, :], reduce_ratio=0.9+0.1*warp_amount[i]/max_warp)[0]
    return _ensure_float32(ret), skipped