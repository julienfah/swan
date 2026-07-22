import numpy as np


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
    same_kpoints = k_dft.shape == k_wannier.shape and np.allclose(k_dft, k_wannier)
    if same_kpoints:
        # k-grids identical (e.g. built from the same bandpath) -> no interpolation needed
        y_pred_interp = y_pred
        mask = np.ones(y_pred_interp.shape[0], dtype=bool)
    else:
        def path_coordinate(k):
            """Convert a k-path into a curvilinear coordinate."""
            k = np.asarray(k)
            if k.ndim == 1:
                return k
            ds = np.linalg.norm(np.diff(k, axis=0), axis=1)
            return np.concatenate([[0.0], np.cumsum(ds)])

        s_dft = path_coordinate(k_dft)
        s_wann = path_coordinate(k_wannier)

        # Interpolate Wannier energies onto the DFT k-grid
        y_pred_interp = np.empty_like(y_true)
        for ib in range(nbands):
            y_pred_interp[:, ib] = np.interp(
                s_dft,
                s_wann,
                y_pred[:, ib],
                left=np.nan,
                right=np.nan,
            )

        # Keep only points inside the interpolation interval
        mask = np.isfinite(y_pred_interp).all(axis=1)

        if not np.any(mask):
            raise ValueError("The DFT and Wannier k-paths do not overlap.")

    residuals = y_true[mask] - y_pred_interp[mask]  # shape (n_kpoints_kept, nbands)
    ytk = y_true[mask]
    if per_nk:
        fmask = (ytk >= outer_win[0]) & (ytk <= outer_win[1])
    else:
        fmask = np.broadcast_to(
            np.any((ytk >= outer_win[0]) & (ytk <= outer_win[1]), axis=0), ytk.shape)
    n_eigenvalues = fmask.sum()
    if n_eigenvalues == 0:
        raise ValueError("no (band,k) states fall inside the window.")
    # Sigma = scale * [sqrt] ( sum f_nk (E - E_wann)^2 / sum f_nk )
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
