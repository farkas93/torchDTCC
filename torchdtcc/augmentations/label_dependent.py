import numpy as np
from tqdm import tqdm
from typing import Optional, List, Tuple
from .helper import check_shape, ensure_float32
from .basic import window_slice, jitter
from .dtw import DTW

# Example docstring for a label-dependent augmentation:
def spawner(x, labels, sigma=0.05, verbose=0, random_state=None) -> Tuple[np.ndarray, List[int]]:
    """
    SPAWNER augmentation: averages two intra-class patterns using DTW path.
    Returns:
        - Augmented data (np.ndarray)
        - List of indices where no augmentation was possible (skipped indices)
    Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6983028/
    """
    check_shape(x)
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
    return ensure_float32(jitter(ret, sigma=sigma, random_state=random_state)), skipped

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
    check_shape(x)
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
    return ensure_float32(ret), skipped

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
    check_shape(x)
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
    return ensure_float32(ret), skipped

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
    check_shape(x)
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
    return ensure_float32(ret), skipped