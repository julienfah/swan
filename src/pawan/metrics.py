import numpy as np


def least_square_deviation(dft_eig_energies,k_dft, wannier_interpolated_energies,k_wannier,outer_win):
    """
    Compute the Sigma metric from Zhang's paper, which is the least square deviation between DFT eigenvalues and Wannier interpolated eigenvalues.
    """
    y_true = np.asarray(dft_eig_energies) #(spin,npoints, nbands)
    y_true = y_true[0]#(npoints, nbands),
    y_pred = np.asarray(wannier_interpolated_energies) #(npoints, nwann), nwann < nbands
    #restrict to outer window
    print(f"y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
    mask = np.any((y_true >= outer_win[0]) & (y_true <= outer_win[1]), axis=0)
    y_true = y_true[:, mask]

    print(f"y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
    # Compare only the common number of bands
    nbands = min(y_true.shape[1], y_pred.shape[1])
    y_true = y_true[:, :nbands]
    y_pred = y_pred[:, :nbands]

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

    return np.sum((y_true[mask] - y_pred_interp[mask]) ** 2) * 100 / np.sum(mask)

def least_square_deviation_within_frozen(dft_eig_energies,k_dft, wannier_interpolated_energies,k_wannier,outer_win,frozen_win):
    """
    Compute the Sigma metric from Zhang's paper within the frozen energy window, which is the least square deviation between DFT eigenvalues and Wannier interpolated eigenvalues.
    """
    mask = np.any((dft_eig_energies >= frozen_win[0]) & (dft_eig_energies <= frozen_win[1]), axis=(0, 1))
    print(mask.shape)
    print(dft_eig_energies.shape)
    dft_eig_frozen = dft_eig_energies[:,:,mask]
    return least_square_deviation(dft_eig_frozen, k_dft, wannier_interpolated_energies, k_wannier, outer_win)
