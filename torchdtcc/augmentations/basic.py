import numpy as np
from .helper import check_shape, ensure_float32

def jitter(x, sigma=0.03, random_state=None):
    """
    Adds Gaussian noise to the input.
    Reference: https://arxiv.org/pdf/1706.00527.pdf
    """
    check_shape(x)
    rng = np.random.RandomState(random_state)
    return ensure_float32(x + rng.normal(loc=0., scale=sigma, size=x.shape))

def scaling(x, sigma=0.1, random_state=None):
    """
    Multiplies input by a random scaling factor for each feature.
    Reference: https://arxiv.org/pdf/1706.00527.pdf
    """
    check_shape(x)
    rng = np.random.RandomState(random_state)
    factor = rng.normal(loc=1., scale=sigma, size=(x.shape[0], x.shape[2]))
    return ensure_float32(np.multiply(x, factor[:, np.newaxis, :]))

def rotation(x, random_state=None):
    """
    Randomly flips and permutes feature axes.
    """
    check_shape(x)
    rng = np.random.RandomState(random_state)
    flip = rng.choice([-1, 1], size=(x.shape[0], x.shape[2]))
    rotate_axis = np.arange(x.shape[2])
    rng.shuffle(rotate_axis)
    return ensure_float32(flip[:, np.newaxis, :] * x[:, :, rotate_axis])

def permutation(x, max_segments=5, seg_mode="equal", random_state=None):
    """
    Randomly permutes segments of the time series.
    """
    check_shape(x)
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
    return ensure_float32(ret)

def magnitude_warp(x, sigma=0.2, knot=4, random_state=None):
    """
    Warps the magnitude of the series using random smooth curves.
    """
    from scipy.interpolate import CubicSpline
    check_shape(x)
    rng = np.random.RandomState(random_state)
    orig_steps = np.arange(x.shape[1])
    random_warps = rng.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        warper = np.array([CubicSpline(warp_steps[:,dim], random_warps[i,:,dim])(orig_steps) for dim in range(x.shape[2])]).T
        ret[i] = pat * warper
    return ensure_float32(ret)

def time_warp(x, sigma=0.2, knot=4, random_state=None):
    """
    Warps the time axis using smooth curves.
    """
    from scipy.interpolate import CubicSpline
    check_shape(x)
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
    return ensure_float32(ret)

def window_slice(x, reduce_ratio=0.9, random_state=None):
    """
    Randomly slices a window from the time series and rescales back to original size.
    Reference: https://halshs.archives-ouvertes.fr/halshs-01357973/document
    """
    check_shape(x)
    rng = np.random.RandomState(random_state)
    target_len = np.ceil(reduce_ratio*x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return ensure_float32(x)
    starts = rng.randint(low=0, high=x.shape[1]-target_len, size=(x.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            ret[i,:,dim] = np.interp(np.linspace(0, target_len-1, num=x.shape[1]), np.arange(target_len), pat[starts[i]:ends[i],dim]).T
    return ensure_float32(ret)

def window_warp(x, window_ratio=0.1, scales=[0.5, 2.], random_state=None):
    """
    Warps a random window in the series by a random scale and resamples to original length.
    Reference: https://halshs.archives-ouvertes.fr/halshs-01357973/document
    """
    check_shape(x)
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
    return ensure_float32(ret)