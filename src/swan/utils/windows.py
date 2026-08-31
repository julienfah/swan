"""Energy windows from the band structure alone.

Extracted from the old sphere driver, which is gone: everything else in it
-- the sphere selection, the atomicity gate, the fixed-point refinement --
was superseded by the amn route. These three are pure band arithmetic and
are used by candidate_scan.amn_projection_method.
"""

from __future__ import annotations

import numpy as np

__all__ = ["band_blocks", "emax_from_band_count", "cap_frozen_window"]

def band_blocks(eps_kn, gap_thres=0.1):
    """Groups of consecutive BAND INDICES separated by a true band gap.

    A gap between band n and n+1 exists iff

        min_k eps[k, n+1] - max_k eps[k, n] > gap_thres

    Returns [(n_start, n_stop, e_lo, e_hi), ...], n_stop inclusive.
    """
    lo_n = eps_kn.min(axis=0)
    hi_n = eps_kn.max(axis=0)
    nb = len(lo_n)
    cuts = [n for n in range(nb - 1) if lo_n[n + 1] - hi_n[n] > gap_thres]
    starts = [0] + [c + 1 for c in cuts]
    stops = cuts + [nb - 1]
    return [(a, b, float(lo_n[a]), float(hi_n[b]))
            for a, b in zip(starts, stops)]

def emax_from_band_count(eps_kn, emin, nwann, K=1.2):
    """Smallest emax with N_k(emin, emax) >= ceil(K * nwann) at EVERY k.

    Exact per k.  Integrating a smeared DOS to K*nwann is a BZ-AVERAGED proxy,
    and an average can be satisfied while one k-point fails.
    """
    target = int(np.ceil(K * nwann))
    lim = -np.inf
    for eps_n in eps_kn:
        e = np.sort(eps_n[eps_n >= emin])
        if len(e) < target:
            raise ValueError(
                f"only {len(e)} bands above {emin:.3f} eV at one k-point but "
                f"{target} needed for nwann={nwann}, K={K} -- increase nbands "
                "in the NSCF")
        lim = max(lim, e[target - 1])
    return float(lim)

def cap_frozen_window(eps_kn, emin, emax, nwann, pad=1e-6):
    """Largest emax' <= emax with N_k(emin, emax') <= nwann at every k.

    Degeneracy-safe: the cut lands strictly below the first band that would push
    any k over the count, so a degenerate multiplet is never split.
    """
    lim = float(emax)
    for eps_n in eps_kn:
        e = np.sort(eps_n[eps_n >= emin])
        if len(e) > nwann:
            lim = min(lim, float(e[nwann]) - pad)
    return max(lim, float(emin))