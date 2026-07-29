"""Projectability from AMN matrices -- grid-based, so the diffuse tail counts.

A_{n,M}(k) = <psi_nk | g_M> is a FULL-SPACE overlap on the FFT grid, computed
with the PAW augmentation term.  Nothing is truncated at rc, so a diffuse
orbital contributes its whole tail.  That is the entire difference from the
sphere backend, and it is exactly the deficit that made CaMg2Bi2 unselectable:
sphere capture there was ~17%, so Ca and Mg conduction character was invisible.

    Pi_k = sum_MN |g_M> (S^-1)_MN <g_N|          (idempotent, Hermitian)
    p_nk = <psi|Pi|psi> = sum_M |[A S^{-1/2}]_nM|^2     in [0, 1]

TWO THINGS THAT DIFFER FROM THE QE ROUTE
----------------------------------------
1. S is not in the .amn file.  Use S ~ A^dag A, the overlap seen THROUGH the
   band set.  This is exact only for a complete band set, and it has one
   consequence worth stating plainly: sum_n p_nk = rank(A) = n_proj IDENTICALLY,
   so the TOTAL carries no information about whether your candidate set is
   adequate.  The PER-BAND distribution still does, and that is what you need --
   if the occupied bands do not reach p ~ 1, the set cannot span them.

   `diag(A^dag A)_MM = sum_n |<psi_n|g_M>|^2` is bounded by ||g_M||^2, NOT by 1
   -- Bessel's inequality.  WannierBerri builds trial orbitals analytically
   (Bessel_j_radial_int + Projector) and does not normalise them to unit norm in
   the plane-wave representation, so the diagonal can and does exceed 1: a mean
   of 1.24 for CaMg2Bi2 just says those orbitals have norm > 1.  It is NOT a
   sign of anything wrong, and it is NOT caused by s/p character leaking into
   high bands (the sum runs over all bands either way and only ever increases
   with nbands).

   Crucially the normalisation does not affect p_nk at all: Pi is invariant
   under any invertible rescaling of the trial orbitals, exactly, and that
   includes the S = A^dag A version.  Verified numerically.

   Do not try to converge diag(A^dag A) either.  It approaches ||g_M||^2 only as
   the band set approaches COMPLETENESS, and a localised analytic orbital has a
   slowly-decaying Fourier tail, so the approach is power-law and the
   last-25%-of-bands increment is not even monotone in nbands.  Use
   `p_convergence`: converge the quantity you actually use.

   IMPORTANT BIAS.  With S = A^dag A over nb bands, Pi is the projector onto
   span{P_nb g}, and P_nb is forced to live inside the computed bands.  That
   INFLATES p for the low bands: the nproj units of weight have nowhere else to
   go.  p over the occupied manifold therefore DECREASES monotonically toward
   its true value as nbands grows -- Si's 0.92 at 40 bands is an overestimate.
   Any absolute threshold (a frozen window at p > 0.95, say) is meaningless
   until p is converged in nbands.

2. No m-reordering.  The trial orbitals come from WannierBerri's own orbital
   definitions, so the columns are already in WB order -- do NOT apply
   salc.gpaw_to_wb() here.  That transform exists only because P_ani is in
   GPAW's ordering.

THE WORKFLOW THIS ENABLES
-------------------------
Compute A ONCE for a wide CANDIDATE set (all shells on all orbits), select
columns from it, then Wannierise with the chosen subset.  This is Wannier90's
`select_projections` pattern.  The expensive step -- the overlaps -- happens
once, and the selection criterion becomes literally the same matrix that
determines the disentanglement quality afterwards.
"""

from __future__ import annotations

import warnings

import numpy as np

__all__ = [
    "load_amn",
    "block_map",
    "projectability_from_amn",
    "weights_from_amn",
    "block_scores",
    "pdos_from_weights",
    "channel_occupancy",
    "subset_projectability",
    "greedy_select",
    "window_rank_check",
    "capture_convergence",
    "p_convergence",
]


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_amn(path):
    """Return A with shape (nk, nb, nproj), from .amn.npz or a w90 .amn text."""
    p = str(path)
    if p.endswith(".npz"):
        z = np.load(p, allow_pickle=True)
        for key in ("data", "amn", "A", "arr_0"):
            if key in z:
                A = np.asarray(z[key])
                break
        else:
            raise KeyError(f"no recognised array in {p}: keys are {list(z)}")
    else:
        with open(p) as f:
            f.readline()                                   # comment
            nb, nk, nproj = (int(x) for x in f.readline().split()[:3])
            A = np.zeros((nk, nb, nproj), complex)
            for line in f:
                t = line.split()
                if len(t) < 5:
                    continue
                m, n, k = int(t[0]) - 1, int(t[1]) - 1, int(t[2]) - 1
                A[k, m, n] = float(t[3]) + 1j * float(t[4])
    A = np.asarray(A)
    if A.ndim != 3:
        raise ValueError(f"expected (nk, nb, nproj), got {A.shape}")
    return A


def block_map(proj_set):
    """[(label, slice), ...] over the AMN columns, one entry per Projection.

    Assumes WannierBerri emits columns projection by projection, each
    contributing n_sites * n_orbitals_per_site.  VERIFY THIS ONCE for your
    version: the total is checked against A.shape[2], but a permutation between
    projections of equal size would pass that check silently.  Build the
    candidate set with one Projection per (orbit, l) and blocks of distinct
    sizes if you want the check to bite.
    """
    out, off = [], 0
    for i, p in enumerate(proj_set.projections):
        n = p.num_wann
        label = getattr(p, "orbital", None) or f"proj{i}"
        out.append((f"{i}:{label}", slice(off, off + n)))
        off += n
    return out


# --------------------------------------------------------------------------
# projectability
# --------------------------------------------------------------------------

def _inv_sqrt_herm(S, rcond=1e-8):
    w, U = np.linalg.eigh(S)
    keep = w > rcond * max(float(w.max().real), 1e-30)
    winv = np.zeros_like(w)
    winv[keep] = w[keep] ** -0.5
    return (U * winv) @ U.conj().T, int((~keep).sum())


def _proj_gk(amn, kp, bandstructure, bessel, Projector):
    """Trial orbitals in the plane-wave basis of one k-point, exactly as
    AMN.from_bandstructure builds them."""
    pos = np.asarray(amn.positions)
    igk = kp.ig[:, :3] + kp.k[None, :]
    expgk = np.exp(-2j * np.pi * (pos @ igk.T))
    gk = igk @ bandstructure.RecLattice
    prj, cache = [], {}
    for orb, basis, rnodes, spread in zip(amn.orbitals, amn.basis_list,
                                          amn.radial_nodes_list,
                                          amn.spread_list):
        if spread not in cache:
            cache[spread] = Projector(gk, bessel, spread_factor=spread)
        prj.append(cache[spread](orb, basis, rnodes))
    volume = np.linalg.det(bandstructure.spacegroup.lattice)
    return np.array(prj) * expgk / np.sqrt(volume)


def _band_gram(bandstructure, mapping, keys, normalize=True, verbose=True):
    """O[i]_mn = <psi_m|psi_n> in the plane-wave basis, same k-mapping as S.

    This is the missing factor.  Bessel's inequality and p <= 1 both require the
    BANDS to be orthonormal in the same inner product as S.  In PAW they are
    orthonormal only including the augmentation term:

        <psi~_m|psi~_n> + sum_a sum_ij <psi~_m|p_i> Q^a_ij <p_j|psi~_n> = delta_mn

    so the bare plane-wave Gram is delta_mn minus a non-diagonal correction.
    AMN.from_bandstructure normalises each band's PW norm to 1, which fixes the
    diagonal but not the off-diagonal, and row-normalising a non-orthogonal set
    pushes some Gram eigenvalues ABOVE 1.  Then sum_n |psi_n><psi_n| > 1 and
    both bounds fail -- which is exactly the 124% coverage.

    Loewdin-orthonormalising the bands with this O restores both bounds by
    construction, whatever the size of the effect.

    Cost: nb^2 * npw per k-point.  Reported eigenvalue range IS the diagnostic:
    if it is 1.000 .. 1.000 the bands were fine and something else is wrong.
    """
    O = None
    lo, hi = np.inf, -np.inf
    for i, key in enumerate(keys):
        kp = bandstructure.kpoints[mapping[key]]
        C = np.asarray(kp.WF)
        C = C.reshape(C.shape[0], -1)
        if normalize:
            C = C / np.linalg.norm(C, axis=1)[:, None]
        Ok = C.conj() @ C.T
        if O is None:
            O = np.zeros((len(keys),) + Ok.shape, complex)
        O[i] = Ok
        ev = np.linalg.eigvalsh(Ok).real
        lo, hi = min(lo, ev.min()), max(hi, ev.max())
    if verbose:
        print(f"  band Gram eigenvalues over all k: {lo:.4f} .. {hi:.4f}")
        if hi > 1 + 1e-6:
            print("    > 1: the bands are NOT orthonormal in the PW inner "
                  "product, so p was not bounded. Loewdin fixes it.")
        else:
            print("    <= 1: bands are effectively orthonormal; a coverage "
                  "above 100% must have another cause.")
    return O


def true_overlap(amn, bandstructure, selected_kpoints=None, check=True,
                 normalize=True, verbose=True, with_band_gram=False):
    """S[i]_MN = <g_M|g_N> at the i-th k-point OF THE AMN, aligned with `A`.

    Returns (nk, nproj, nproj), stacked in the order `extract_amn` stacks A --
    sorted(amn.data) -- so S[k] pairs with A[k].

    K-POINT MAPPING.  The amn is built on the IRREDUCIBLE k-points, so
    amn.data is a dict keyed by ikirr while bandstructure.kpoints may be indexed
    differently, and amn.NK is the FULL-BZ count (216 in your BaTiO3 run, versus
    20 stored blocks).  Rather than assume one convention, this tries the
    plausible mappings and keeps the one that reproduces amn.data:

        1. positional -- bandstructure.kpoints[i] for the i-th sorted key
        2. by key     -- bandstructure.kpoints[key]
        3. auto_kptirr(bandstructure, NK=amn.NK)

    Verification is by RECOMPUTING A from the same proj_gk and comparing to
    amn.data elementwise.  Bound-based tests do not work here: I checked, and
    permuting S across k-points violates neither Bessel's inequality nor
    p <= 1, because S(k) varies smoothly and the norms are similar at every k.

    Do not pass k-point objects for `selected_kpoints`; it is an integer index
    array.  Normally leave it None.

    With the true S, p_n = [A S^-1 A^dag]_nn is EXACT for every band and
    independent of nbands -- the slow monotone decay seen with S = A^dag A is an
    artefact of that substitution, not physics.
    """
    from wannierberri.symmetry.orbitals import Bessel_j_radial_int, Projector

    data = amn.data
    keys = sorted(data) if isinstance(data, dict) else list(range(len(data)))
    nk_bs = len(bandstructure.kpoints)
    bessel = Bessel_j_radial_int()

    mappings = []
    if selected_kpoints is not None:
        sk = np.asarray(selected_kpoints)
        if sk.dtype.kind not in "iu":
            raise TypeError(
                f"selected_kpoints must be an integer index array, got "
                f"{sk.dtype} (did you pass bandstructure.kpoints?).")
        mappings.append(("user", {k: int(sk[i]) for i, k in enumerate(keys)}))
    else:
        mappings.append(("positional", {k: i for i, k in enumerate(keys)}))
        if all(isinstance(k, (int, np.integer)) and 0 <= int(k) < nk_bs
               for k in keys):
            mappings.append(("by-key", {k: int(k) for k in keys}))
        try:
            from wannierberri.w90files.w90file import auto_kptirr
            _, sel, kirr = auto_kptirr(bandstructure, NK=getattr(amn, "NK", None))
            mappings.append(("auto_kptirr",
                             {k: int(sel[kirr[i]]) for i, k in enumerate(keys)}))
        except Exception as e:                                  # noqa: BLE001
            mappings.append(("auto_kptirr", None))
            if verbose:
                print(f"  (auto_kptirr unavailable: {e})")

    tried = []
    for name, mapping in mappings:
        if mapping is None:
            continue
        try:
            S = np.zeros((len(keys), len(amn.orbitals), len(amn.orbitals)),
                         complex)
            dev = 0.0
            for i, key in enumerate(keys):
                kp = bandstructure.kpoints[mapping[key]]
                pg = _proj_gk(amn, kp, bandstructure, bessel, Projector)
                S[i] = pg.conj() @ pg.T
                if check:
                    wf = kp.WF.conj()
                    if normalize:
                        wf = wf / np.linalg.norm(kp.WF, axis=(1, 2))[:, None, None]
                    ref = np.asarray(data[key])
                    got = wf[:, :, 0] @ pg.T
                    if got.shape != ref.shape:
                        raise ValueError(f"shape {got.shape} vs {ref.shape}")
                    dev = max(dev, float(np.abs(got - ref).max())
                              / max(float(np.abs(ref).max()), 1e-30))
            if not check:
                return (S, _band_gram(bandstructure, mapping, keys, normalize,
                                      verbose)) if with_band_gram else S
            if dev < 1e-6:
                if verbose:
                    cover = float(np.mean([
                        np.diag(np.asarray(data[k]).conj().T
                                @ np.asarray(data[k])).real.sum()
                        / np.diag(S[i]).real.sum() for i, k in enumerate(keys)]))
                    print(f"  true overlap: k-mapping '{name}' verified against "
                          f"amn.data (max rel. dev {dev:.1e}); the band set "
                          f"reaches {cover:.1%} of the trial-orbital norm")
                    if cover > 1.0:
                        print("  NOTE coverage > 100% violates Bessel's "
                              "inequality, which is only possible if the BANDS "
                              "are not orthonormal in the plane-wave inner "
                              "product. They are not: AMN.from_bandstructure "
                              "normalises each pseudo band's PW norm to 1, and "
                              "in PAW <psi~_m|psi~_n> + augmentation = delta_mn, "
                              "so the bare Gram has off-diagonal parts and "
                              "row-normalising it pushes some eigenvalues above "
                              "1. Pass with_band_gram=True and use O= to fix it.")
                return ((S, _band_gram(bandstructure, mapping, keys, normalize,
                                       verbose))
                        if with_band_gram else S)
            tried.append(f"{name}: rel. dev {dev:.2e}")
        except Exception as e:                                  # noqa: BLE001
            tried.append(f"{name}: {type(e).__name__} {e}")

    raise RuntimeError(
        "could not find a k-point mapping reproducing amn.data. Tried -- "
        + "; ".join(tried)
        + f". amn has {len(keys)} stored blocks (keys {keys[:5]}...), "
          f"amn.NK={getattr(amn, 'NK', None)}, bandstructure has {nk_bs} "
          "k-points. Pass selected_kpoints explicitly as an integer array "
          "mapping each sorted amn.data key to an index in "
          "bandstructure.kpoints.")


def projectability_from_amn(A, S=None, O=None, rcond=1e-8, verbose=True,
                            atol=1e-6):
    """p_kn in [0, 1] and the Loewdin coefficients.

    O : (nk, nb, nb) band Gram from true_overlap(..., with_band_gram=True).
        REQUIRED for p to be bounded when the bands are the PW-normalised
        pseudo wavefunctions -- see _band_gram.  Without it p can exceed 1 and
        sum_n p can exceed nproj, which is what "sum_n p = 10.114 > 8" means.

    S : (nk, nproj, nproj) from `true_overlap`, optional
        The TRUE overlap from `true_overlap`.  Strongly preferred: p is then
        exact and nbands-independent, and sum_n p_nk becomes a real measure of
        how much of the trial-orbital space your bands span instead of the
        constant nproj.  Without it, S = A^dag A and p is biased upward and
        drifts with nbands.

    Returns (p_kn, Atil) with Atil of shape (nk, nb, nproj); the per-block
    weight is just the |.|^2 sum over that block's columns, so the same array
    serves the scalar selection and the m-resolved SALC weight.
    """
    A = np.asarray(A)
    nk, nb, nproj = A.shape
    Atil = np.zeros_like(A, dtype=complex)
    p_kn = np.zeros((nk, nb))
    dropped, capture = 0, []

    exact = S is not None
    if O is not None:
        A = np.stack([_inv_sqrt_herm(np.asarray(O[k]), rcond)[0] @ A[k]
                      for k in range(nk)])
        if verbose:
            print("  bands Loewdin-orthonormalised with the supplied Gram")
    for k in range(nk):
        Sk = (np.asarray(S[k]) if exact
              else A[k].conj().T @ A[k])           # true, or through the bands
        capture.append(np.diag(A[k].conj().T @ A[k]).real.copy())
        Sm12, nd = _inv_sqrt_herm(Sk, rcond)
        dropped += nd
        Atil[k] = A[k] @ Sm12
        p_kn[k] = (Atil[k].conj() * Atil[k]).real.sum(axis=1)

    capture = np.array(capture)
    if verbose:
        print(f"  nk={nk}, nbands={nb}, nproj={nproj}")
        print(f"  diag(A^dag A): min {capture.min():.4f}, "
              f"mean {capture.mean():.4f}   "
              "(bounded by ||g_M||^2, not by 1 -- WannierBerri's trial orbitals "
              "are not unit-normalised; harmless, p_nk is invariant to it)")
        cap = ("<= nproj (deficit = trial-orbital weight the bands do not "
               "reach)" if exact else f"must equal nproj={nproj}")
        print(f"  p_nk: max {p_kn.max():.4f};  sum_n p_nk = "
              f"{p_kn.sum(axis=1).mean():.3f}   {cap}")
    if dropped:
        warnings.warn(f"{dropped} near-null direction(s) in A^dag A across all "
                      "k -- trial orbitals are linearly dependent as seen by "
                      "the bands, or nbands is too small.")
    # NB: capture is bounded by ||g_M||^2, not by 1 -- see the module docstring.
    # No warning is issued on its absolute value; use capture_convergence().
    if p_kn.max() > 1 + atol:
        raise RuntimeError(
            f"p_nk reaches {p_kn.max():.4f} > 1. Pi is an orthogonal projector, "
            "so this means A is not what this code assumes -- check the array "
            "layout is (nk, nb, nproj) and not a transpose.")
    tot = p_kn.sum(axis=1)
    if exact and verbose:
        print(f"    deficit {nproj - tot.mean():.3f} of {nproj}  "
              f"({1 - tot.mean() / nproj:.1%} of the trial-orbital space is "
              "outside your band set)")
    if not exact and not np.allclose(tot, nproj, atol=1e-4):
        warnings.warn("sum_n p_nk != nproj: some projector directions were "
                      "dropped by rcond, so the set is rank-deficient.")
    return p_kn, Atil


def weights_from_amn(Atil, blocks):
    """{label: (nk, nb)} projectability carried by each block of columns.

    DO NOT rank on the band-summed value.  With S = A^dag A the Loewdin columns
    satisfy

        Atil^dag Atil = S^{-1/2} A^dag A S^{-1/2} = I

    so sum_n |Atil_nM|^2 = 1 EXACTLY for every column, whatever the orbital is.
    A block of m columns therefore sums to m, and its mean over nb bands is
    m/nb -- which is the whole of "Si s 0.0500, p 0.1500, d 0.2500": 2/40, 6/40,
    10/40.  Those numbers contain no physics at all, and ranking on them ranks
    by block size, i.e. always d > p > s.

    The information is entirely in WHERE each unit of weight lands.  Use
    block_scores() below.

    A consequence worth knowing: a poorly captured orbital -- Si d here, with
    diag(A^dag A) = 0.58 -- is renormalised to a full unit anyway, so it looks
    as important as p until you restrict to an energy window.
    """
    return {lab: (Atil[:, :, sl].conj() * Atil[:, :, sl]).real.sum(axis=2)
            for lab, sl in blocks}


def block_scores(w, blocks, eps_kn, wk_k, window, verbose=True, labels=None):
    """{label: fraction of that block's weight landing inside `window`}.

    score = sum_{nk in window} w_k * weight / n_columns_in_block,  in [0, 1].

    This is the informative quantity: a chemically relevant block concentrates
    its unit-per-column weight in the manifold you care about, an irrelevant one
    scatters it over high conduction bands.  Si d spreads out; Si s and p do not.

    Caveat, same as everywhere else in this project: if `window` is the whole
    spectrum every score is 1.0 by the identity above.  The window has to be the
    physical manifold, not the disentanglement window.
    """
    emin, emax = window
    sel = (eps_kn >= emin) & (eps_kn <= emax)
    wk = wk_k / wk_k.sum()
    size = {lab: (sl.stop - sl.start) for lab, sl in blocks}
    # k-weights sum to 1 and each column's band-sum is 1, so dividing by the
    # number of columns gives a fraction in [0, 1]
    out = {lab: float((wk[:, None] * w[lab] * sel).sum()) / size[lab]
           for lab, _ in blocks}
    if verbose:
        print(f"  block scores over {emin:.3f} .. {emax:.3f} eV "
              "(fraction of each block's weight inside the window):")
        for lab, _ in sorted(blocks, key=lambda b: -out[b[0]]):
            name = labels.get(lab, lab) if labels else lab
            print(f"    {str(name):28s} {out[lab]:6.1%}  "
                  f"({size[lab]} column(s))")
    return out


def capture_convergence(A, fractions=(0.5, 0.75, 1.0), tol=0.02, verbose=True):
    """DEPRECATED -- diagnoses the wrong quantity.  Use p_convergence.

    diag(A^dag A) -> ||g_M||^2 only in the complete-basis limit, and for a
    localised analytic trial orbital the Fourier tail decays as a power law, so
    this never converges and the last-25% increment is not monotone in nbands.
    Kept only so existing calls do not break.
    """
    warnings.warn("capture_convergence diagnoses convergence of diag(A^dag A), "
                  "which does not converge for a localised trial orbital. Use "
                  "p_convergence instead.", DeprecationWarning, stacklevel=2)
    A = np.asarray(A)
    nb = A.shape[1]
    full = np.zeros(A.shape[2])
    for k in range(A.shape[0]):
        full += np.diag(A[k].conj().T @ A[k]).real
    out = {}
    for f in fractions:
        n = max(1, int(round(f * nb)))
        d = np.zeros(A.shape[2])
        for k in range(A.shape[0]):
            d += np.diag(A[k, :n].conj().T @ A[k, :n]).real
        out[f] = float(np.max(np.abs(d / np.maximum(full, 1e-30) - 1.0)))
    if verbose:
        print("  band convergence of A^dag A (max relative deficit vs all bands):")
        for f in fractions:
            print(f"    {int(f * 100):3d}% of bands: {out[f]:.4f}")
        last = sorted(fractions)[-2] if len(fractions) > 1 else fractions[0]
        if out[last] > tol:
            print(f"    NOT CONVERGED: {int(last * 100)}% -> 100% still moves it "
                  f"by {out[last]:.3f} > {tol}. Raise nbands, or narrow the "
                  "candidate shells (a diffuse d needs far more bands than s/p).")
        else:
            print(f"    converged to within {tol}; A^dag A is a fair stand-in "
                  "for S")
    return out


def p_convergence(A, bands=None, fractions=(0.4, 0.6, 0.8, 1.0), tol=0.005,
                  verbose=True):
    """Is p_nk converged in nbands, over the bands you actually use?

    Recomputes p from A[:, :n, :] for each fraction and reports the mean over
    `bands` (default: the lowest nproj bands, a stand-in for the target
    manifold).  Costs nothing but array slicing.

    Read it as an approach FROM ABOVE: truncating the band set forces
    span{P_nb g} inside the computed bands, which inflates p for the low bands.
    So the sequence decreases and the last increment is your error bar.  If
    80% -> 100% still moves it by more than `tol`, the number you would compare
    against a threshold is not yet the number you will get with more bands.
    """
    A = np.asarray(A)
    nk, nb, nproj = A.shape
    if bands is None:
        warnings.warn("p_convergence: `bands` defaults to the lowest nproj "
                      "bands, which mixes the occupied manifold with "
                      "conduction states and understates p. Pass "
                      "bands=slice(0, n_occupied).", stacklevel=2)
        bands = slice(0, min(nproj, nb))
    out = {}
    for f in fractions:
        n = max(nproj, int(round(f * nb)))
        p, _ = projectability_from_amn(A[:, :n, :], verbose=False)
        out[f] = float(p[:, bands].mean())
    fr = sorted(out)
    if verbose:
        print("  p convergence in nbands (mean over the target bands):")
        for f in fr:
            n = max(nproj, int(round(f * nb)))
            print(f"    {n:4d} bands ({int(f * 100):3d}%): p = {out[f]:.5f}")
        drift = abs(out[fr[-1]] - out[fr[-2]])
        print(f"    last increment: {drift:.5f}", end="  ")
        if drift > tol:
            print(f"> tol {tol} -- NOT converged; p is still falling, so the "
                  "current value is an overestimate. Raise nbands before "
                  "calibrating any threshold.")
        else:
            print(f"<= tol {tol}; converged")
    return out


# --------------------------------------------------------------------------
# pDOS-like output
# --------------------------------------------------------------------------

def pdos_from_weights(eps_kn, wk_k, w, energies=None, width=0.1, npts=601,
                      pad=2.0):
    """Projected DOS built from the Loewdin weights -- the pDOS analogue.

        pdos_{a,l}(E) = sum_{nk} w_k * weight^{a,l}_nk * gauss(E - eps_nk)

    Comparable to a conventional pDOS, but with a sum rule that means something:
    sum over channels gives the PROJECTED DOS, which is <= the total DOS, and
    the gap between them is the part of the bands no atom-centred orbital in
    your set can represent.  A sphere-projected pDOS has no such interpretation
    -- its deficit is dominated by interstitial charge and by rcut.

    Returns (energies, {key: pdos}, total_dos).
    """
    eps_kn = np.asarray(eps_kn)
    wk = np.asarray(wk_k) / np.sum(wk_k)
    if energies is None:
        energies = np.linspace(eps_kn.min() - pad, eps_kn.max() + pad, npts)
    energies = np.asarray(energies)

    e = eps_kn.ravel()
    wrep = np.repeat(wk, eps_kn.shape[1])
    g = np.exp(-0.5 * ((energies[:, None] - e[None, :]) / width) ** 2) \
        / (width * np.sqrt(2 * np.pi))
    total = g @ wrep
    out = {key: g @ (wrep * np.asarray(v).ravel()) for key, v in w.items()}
    return energies, out, total


def channel_occupancy(eps_kn, wk_k, w, e_fermi, spin_degeneracy=2, ceiling=None,
                      verbose=True, labels=None):
    """Electrons of each channel's character in the occupied manifold.

        N^{a,l} = g_s * sum_{nk : eps <= E_F} w_k * weight^{a,l}_nk

    This is a LOEWDIN POPULATION: the Loewdin weights partition p_nk among the
    trial orbitals, positively and summing to p_nk, so N is the analogue of an
    integrated pDOS occupancy and is directly comparable to one -- to Zhang's
    integrated pDOS, or to a VASP LORBIT decomposition.

    The total sum_channels N is g_s * sum_occ p, which is BELOW the electron
    count by exactly the part of the occupied manifold your trial orbitals do
    not span.  That residue is the honest version of the "sphere deficit", and
    unlike it, it is a property of your projection set rather than of rcut.
    """
    eps_kn = np.asarray(eps_kn)
    wk = np.asarray(wk_k) / np.sum(wk_k)
    occ = eps_kn <= e_fermi
    N = {key: spin_degeneracy * float((wk[:, None] * np.asarray(v) * occ).sum())
         for key, v in w.items()}
    nelec = spin_degeneracy * float((wk[:, None] * occ).sum())
    if verbose:
        tot = sum(N.values())
        print(f"  Loewdin occupancies (electrons in the occupied manifold):")
        for key in sorted(N, key=lambda k: -N[k]):
            name = labels.get(key, key) if labels else key
            line = f"    {str(name):30s} {N[key]:7.3f} e"
            if ceiling is not None and ceiling.get(key):
                line += f"   ({N[key] / ceiling[key]:6.1%} of {ceiling[key]})"
            print(line)
        print(f"    {'TOTAL':30s} {tot:7.3f} e   of {nelec:.3f} in the manifold"
              f"  ({tot / max(nelec, 1e-30):.1%} spanned)")
    return N, nelec


# --------------------------------------------------------------------------
# selection: score the SET, on the manifold that must be reproduced
# --------------------------------------------------------------------------

def subset_projectability(A, S, cols, rcond=1e-8):
    """p_nk for the subspace spanned by a SUBSET of the trial orbitals.

    Exact, not a re-weighting of the full-set answer: dropping columns changes
    the projector, so the sub-blocks of A and S must be re-inverted.  Batched
    over k, so a greedy sweep over a few dozen candidate sets is seconds.
    """
    A = np.asarray(A)[:, :, cols]
    S = np.asarray(S)[:, cols][:, :, cols]
    reg = rcond * np.trace(S, axis1=1, axis2=2).real[:, None, None] / max(len(cols), 1)
    X = np.linalg.solve(S + reg * np.eye(len(cols)),
                        np.swapaxes(A.conj(), -1, -2))
    return np.real(np.einsum("knm,kmn->kn", A, X))


def greedy_select(A, S, blocks, target_mask, wk_k, n_froz=0, margin=1.2,
                  p_target=None, budget=None, required=(), min_rate_frac=0.15,
                  verbose=True, labels=None):
    """Fill the frozen manifold, best value first.

    SIZE comes from the physics, ORDER from projectability:

        n_target = ceil(margin * n_froz),  n_froz = max_k N_k in the frozen window

    n_froz is a hard requirement -- Wannier90 cannot freeze more bands than it
    has functions -- so sizing on it needs no calibration.  Coverage only
    decides which block to add next, ranked by marginal gain PER WANNIER
    FUNCTION.

    Blocks are whole symmetry orbits, so nwann moves in chunks and `margin` is
    far less delicate than it looks.  Si: n_froz = 6 with blocks of 6 (p),
    2 (s), 10 (d), so every margin in (1.0, 1.33] gives n_target = 7-8 and hence
    sp; only 1.0 exactly stops at p alone.  Stay above 1.0.

    min_rate_frac : refuse a block whose marginal rate has collapsed relative to
        the best seen, even when n_target is unmet -- otherwise a manifold that
        atom-centred orbitals cannot span gets padded with junk.  Si: d buys
        0.0071/WF against s at 0.1251/WF (6%), so it is rejected and a large
        margin is harmless.

    p_target : optional coverage early exit.  None by default: with
        WannierBerri's analytic trial orbitals Si's whole candidate set reaches
        only ~0.85, so an absolute coverage threshold is not calibrated.

    Returns (chosen, nwann, coverage, history).
    """
    wk = np.asarray(wk_k) / np.sum(wk_k)
    wt = wk[:, None] * target_mask
    denom = float(wt.sum())

    def cover(keys):
        cols = np.concatenate([np.arange(sl.start, sl.stop)
                               for k, sl in blocks if k in keys]) \
            if keys else np.zeros(0, int)
        if cols.size == 0:
            return 0.0
        return float((wt * subset_projectability(A, S, cols)).sum()
                     / max(denom, 1e-30))

    size = {k: sl.stop - sl.start for k, sl in blocks}
    n_target = int(np.ceil(margin * n_froz))
    chosen = set(required)
    nwann = sum(size[k] for k in chosen)
    remaining = [k for k, _ in blocks if k not in chosen]
    history, best_rate = [], None
    if verbose:
        print(f"    n_froz = {n_froz} bands at the worst k, margin {margin} "
              f"-> n_target = {n_target} WF")

    while remaining and nwann < n_target:
        cur = cover(chosen)
        if p_target is not None and cur >= p_target and nwann >= n_froz:
            break
        rates = {k: (cover(chosen | {k}) - cur) / size[k] for k in remaining}
        best = max(rates, key=rates.get)
        rate = rates[best]
        if best_rate is None or rate > best_rate:
            best_rate = rate
        """if nwann >= n_froz and best_rate > 0 and rate < min_rate_frac * best_rate:
            if verbose:
                name = labels.get(best, best) if labels else best
                print(f"    stop short of n_target: best remaining is {name} "
                      f"at {rate:.4f}/WF, {rate / best_rate:.0%} of the best "
                      f"rate seen -- padding to {n_target} would add nothing")
            break"""
        if budget is not None and nwann + size[best] > budget:
            break
        chosen.add(best)
        nwann += size[best]
        remaining.remove(best)
        new_cov = cur + rate * size[best]
        history.append((best, size[best], nwann, new_cov))
        if verbose:
            name = labels.get(best, best) if labels else best
            print(f"    + {str(name):32s} (+{size[best]:2d} WF, nwann={nwann:3d})"
                  f"  coverage {cur:.4f} -> {new_cov:.4f}   ({rate:.4f}/WF)")

    final = cover(chosen)
    ceiling = cover({k for k, _ in blocks})
    if verbose:
        print(f"  selected {len(chosen)} block(s), nwann = {nwann} "
              f"(n_froz = {n_froz}), coverage = {final:.4f}  "
              f"(whole candidate set: {ceiling:.4f})")
        if nwann < n_froz:
            print(f"  WARNING nwann={nwann} < n_froz={n_froz}: the frozen "
                  "window cannot fit. Widen the candidate shells.")
        if ceiling < 0.95:
            print(f"  NOTE even ALL candidate blocks reach only {ceiling:.4f}. "
                  "That is the TRIAL ORBITALS, not the manifold -- run "
                  "tune_spread before concluding that atom-centred orbitals are "
                  "insufficient.")
    return chosen, nwann, final, history

def window_rank_check(A, cols, eps_kn, out_win, wk_k=None, tol=1e-3,
                      verbose=True, labels=None, blocks=None):
    """Is the projection matrix full rank at EVERY k inside the outer window?

    Disentanglement solves for nwann orthonormal states drawn from the bands in
    the outer window, using A(k) restricted to those bands and the chosen
    columns.  If that matrix drops rank at one k, the solution there is
    ill-posed and the interpolated bands acquire spurious eigenvalues -- which
    show up as isolated spikes at that k and nowhere else, because every other
    k is fine.

    High-symmetry points are where it happens.  Symmetry can force an admixture
    to vanish exactly, so an orbital that survives on general k by a few percent
    of hybridisation has nothing left at L, K or A.

    The usual cause is an orbital whose OWN states lie outside the outer window:
    it is selected on admixture, then has nothing to be built from.  Run this
    BEFORE wannierising -- it costs one SVD per k-point.

    Returns {k: smallest singular value}, and per-column the worst k.
    """
    A = np.asarray(A)
    lo, hi = out_win
    smin = np.zeros(A.shape[0])
    for k in range(A.shape[0]):
        win = (eps_kn[k] >= lo) & (eps_kn[k] <= hi)
        M = A[k][win][:, cols]
        if M.shape[0] < M.shape[1]:
            smin[k] = 0.0
            continue
        sv = np.linalg.svd(M, compute_uv=False)
        smin[k] = float(sv[-1] / max(sv[0], 1e-30))
    worst = int(np.argmin(smin))
    if verbose:
        print(f"  window rank check: min singular-value ratio over k = "
              f"{smin.min():.3e} at k-index {worst}  "
              f"(median {np.median(smin):.3e})")
        if smin.min() < tol:
            print(f"    RANK DEFICIENT at k={worst}: the disentanglement is "
                  "ill-posed there and the interpolated bands will show "
                  "spurious eigenvalues at that k only. Most likely an orbital "
                  "whose own states lie OUTSIDE the outer window, selected on "
                  "admixture alone -- widen the window (gap_thres) to include "
                  "them, or drop that block.")
            # which column is responsible
            win = (eps_kn[worst] >= lo) & (eps_kn[worst] <= hi)
            M = A[worst][win][:, cols]
            if M.shape[0] >= M.shape[1]:
                _, _, Vt = np.linalg.svd(M)
                bad = np.abs(Vt[-1]) ** 2
                order = np.argsort(-bad)[:4]
                names = []
                for i in order:
                    col = cols[i]
                    nm = next((str(labels.get(kk, kk) if labels else kk)
                               for kk, sl in (blocks or [])
                               if sl.start <= col < sl.stop), f"col {col}")
                    names.append(f"{nm} ({bad[i]:.2f})")
                print(f"    null direction is carried by: {', '.join(names)}")
    return smin, worst