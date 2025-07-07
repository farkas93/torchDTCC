from tslearn.metrics import dtw_path, dtw
import numpy as np

class DTW:
    RETURN_PATH = "path"
    RETURN_VALUE = "value"
    RETURN_ALL = "all"
    
    @staticmethod
    def dtw(ts1, ts2, return_flag="path", slope_constraint="symmetric", window=None):
        """
        ts1, ts2: np.ndarray, shape [seq_len, n_features]
        return_flag: 'path', 'value', or 'all'
        slope_constraint: unused for tslearn (tslearn always symmetric)
        window: ignored for now (can use Sakoe-Chiba band if needed)
        """
        # tslearn expects 2D arrays
        if ts1.ndim == 1:
            ts1 = ts1[:, None]
        if ts2.ndim == 1:
            ts2 = ts2[:, None]
        # Optionally handle windowing (Sakoe-Chiba band)
        if window is not None:
            # tslearn uses sakoe_chiba_radius, which is half the window size
            sakoe_chiba_radius = int(window // 2)
        else:
            sakoe_chiba_radius = None
        path, dist = dtw_path(ts1, ts2, global_constraint="sakoe_chiba" if sakoe_chiba_radius else None, sakoe_chiba_radius=sakoe_chiba_radius)
        if return_flag == DTW.RETURN_PATH:
            # Unzip path to index arrays
            idx1, idx2 = zip(*path)
            return np.array(idx1), np.array(idx2)
        elif return_flag == DTW.RETURN_VALUE:
            return dist
        elif return_flag == DTW.RETURN_ALL:
            idx1, idx2 = zip(*path)
            return dist, None, None, (np.array(idx1), np.array(idx2))
        else:
            raise ValueError(f"Unknown return_flag: {return_flag}")
        

    @staticmethod
    def shapeDTW(ts1, ts2, window_size=5, dist_func=None, return_flag="path"):
        """
        shapeDTW: Dynamic Time Warping with local shape descriptors.
        
        Args:
            ts1, ts2: np.ndarray, shape (seq_len, n_features)
            window_size: int, size of window for local descriptors (must be odd, default 5)
            dist_func: callable, function to compute distance between two windows (default: Euclidean)
            return_flag: "path" (indices), "value" (total cost), or "all"
        Returns:
            Depending on return_flag:
            - "path": idx1, idx2 (warping path indices)
            - "value": total cost (float)
            - "all": total cost, None, None, (idx1, idx2)
        """
        assert ts1.ndim == 2 and ts2.ndim == 2, "shapeDTW: ts1 and ts2 must be 2D arrays"
        assert window_size % 2 == 1, "shapeDTW: window_size should be odd"
        if dist_func is None:
            def dist_func(w1, w2):
                return np.linalg.norm(w1 - w2)

        half = window_size // 2
        # Pad the series to handle borders
        ts1_padded = np.pad(ts1, ((half, half), (0, 0)), mode='reflect')
        ts2_padded = np.pad(ts2, ((half, half), (0, 0)), mode='reflect')
        # Extract all windows
        descs1 = np.array([ts1_padded[i:i+window_size] for i in range(ts1.shape[0])])
        descs2 = np.array([ts2_padded[i:i+window_size] for i in range(ts2.shape[0])])
        # Compute cost matrix
        cost_matrix = np.zeros((ts1.shape[0], ts2.shape[0]))
        for i in range(ts1.shape[0]):
            for j in range(ts2.shape[0]):
                cost_matrix[i, j] = dist_func(descs1[i], descs2[j])
        # Use tslearn DTW on cost matrix (use 1D cost matrix as multivariate)
        # We'll use dtw_path on cost matrix directly
        path, cost = dtw_path(cost_matrix.reshape(-1, 1), np.zeros((cost_matrix.shape[1], 1)))
        idx1, idx2 = zip(*path)
        if return_flag == "path":
            return np.array(idx1), np.array(idx2)
        elif return_flag == "value":
            return cost
        elif return_flag == "all":
            return cost, None, None, (np.array(idx1), np.array(idx2))
        else:
            raise ValueError(f"Unknown return_flag: {return_flag}")