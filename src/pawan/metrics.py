import numpy as np


def match_kpoints(k_small, k_large, atol=1e-6):
    """
    For each point in k_small, find a matching point in k_large (within atol).
    Returns (idx_small, idx_large, fully_included):
      - idx_small, idx_large: index arrays such that k_small[idx_small] == k_large[idx_large] pairwise
      - fully_included: True if every point in k_small has a match in k_large
    """
    k_small = np.asarray(k_small)
    k_large = np.asarray(k_large)
    close = np.all(np.abs(k_small[:, None, :] - k_large[None, :, :]) < atol, axis=-1)  # (n_small, n_large)
    matched = close.any(axis=1)
    idx_small = np.where(matched)[0]
    idx_large = close[idx_small].argmax(axis=1)
    return idx_small, idx_large, bool(matched.all())


def least_square_deviation(dft_eig_energies, k_dft, wannier_interpolated_energies, k_wannier, outer_win,
                           per_nk=False, root=False, scale=100):
    """
    Compute the Sigma metric from Zhang's paper, which is the least square deviation between DFT eigenvalues and Wannier interpolated eigenvalues.
    """
    y_true = np.asarray(dft_eig_energies)  # (spin,npoints, nbands)
    y_true = y_true[0]  # (npoints, nbands)
    y_pred = np.asarray(wannier_interpolated_energies)  # (npoints, nwann), nwann < nbands
    # restrict to outer window
    band_mask = np.any((y_true >= outer_win[0]) & (y_true <= outer_win[1]), axis=0)
    y_true = y_true[:, band_mask]

    # Compare only the common number of bands
    nbands = min(y_true.shape[1], y_pred.shape[1])
    y_true = y_true[:, :nbands]
    y_pred = y_pred[:, :nbands]

    k_dft = np.asarray(k_dft)
    k_wannier = np.asarray(k_wannier)

    if k_dft.shape[0] <= k_wannier.shape[0]:
        idx_dft, idx_wann, fully_included = match_kpoints(k_dft, k_wannier)
    else:
        idx_wann, idx_dft, fully_included = match_kpoints(k_wannier, k_dft)

    if fully_included:
        # one k-grid is a subset of (or equal to) the other -> use the shared points directly
        y_true = y_true[idx_dft]
        y_pred_interp = y_pred[idx_wann]
        mask = np.ones(y_pred_interp.shape[0], dtype=bool)
    else:
        print("The DFT and Wannier k-paths do not overlap. Interpolating the Wannier bands onto the DFT k-path.",k_dft[:5],k_wannier[:5])
        def path_coordinate(k):
            """Convert a k-path into a curvilinear coordinate."""
            k = np.asarray(k)
            if k.ndim == 1:
                return k
            ds = np.linalg.norm(np.diff(k, axis=0), axis=1)
            return np.concatenate([[0.0], np.cumsum(ds)])

        s_dft = path_coordinate(k_dft)
        s_wann = path_coordinate(k_wannier)

        y_pred_interp = np.empty_like(y_true)
        for ib in range(nbands):
            y_pred_interp[:, ib] = np.interp(
                s_dft,
                s_wann,
                y_pred[:, ib],
                left=np.nan,
                right=np.nan,
            )

        mask = np.isfinite(y_pred_interp).all(axis=1)

        if not np.any(mask):
            raise ValueError("The DFT and Wannier k-paths do not overlap.")

    residuals = y_true[mask] - y_pred_interp[mask]
    ytk = y_true[mask]
    if per_nk:
        fmask = (ytk >= outer_win[0]) & (ytk <= outer_win[1])
    else:
        fmask = np.broadcast_to(
            np.any((ytk >= outer_win[0]) & (ytk <= outer_win[1]), axis=0), ytk.shape)
    n_eigenvalues = fmask.sum()
    if n_eigenvalues == 0:
        raise ValueError("no (band,k) states fall inside the window.")
    ms = np.sum((residuals ** 2) * fmask) / n_eigenvalues
    return scale * (np.sqrt(ms) if root else ms)

def least_square_deviation_within_frozen(dft_eig_energies,k_dft, wannier_interpolated_energies,k_wannier,outer_win,frozen_win,
                                         per_nk=True, root=True, scale=1000):
    """
    Compute the Sigma metric from Zhang's paper within the frozen energy window, which is the least square deviation between DFT eigenvalues and Wannier interpolated eigenvalues.
    """
    y_true = np.asarray(dft_eig_energies)[0]
    band_mask = np.any((y_true >= outer_win[0]) & (y_true <= outer_win[1]), axis=0)
    dft_aligned = dft_eig_energies[:, :, band_mask]
    return least_square_deviation(dft_aligned, k_dft, wannier_interpolated_energies, k_wannier, frozen_win,
                                  per_nk=per_nk, root=root, scale=scale)
