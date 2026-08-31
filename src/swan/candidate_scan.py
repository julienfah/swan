"""Candidate scan: one AMN pass over a wide projection set, then rank.
"""

from __future__ import annotations

import numpy as np

import warnings
from pathlib import Path
from typing import NamedTuple

from gpaw.mpi import serial_comm, world

from .utils.windows import band_blocks, cap_frozen_window, emax_from_band_count
from .projectability.amn_projectability import (channel_occupancy, frozen_from_projectability,
                                greedy_select, select_by_occupancy,
                                window_rank_check,
                                subset_projectability,
                                pdos_from_weights, projectability_from_amn,
                                true_overlap, weights_from_amn)

L_NUM = {v: k for k, v in {0: "s", 1: "p", 2: "d", 3: "f"}.items()}

__all__ = ["build_candidate_set", "extract_amn", "extract_eig", "describe_amn",
           "candidate_scan", "score_candidates", "CandidateScan",
           "gpaw_orbital_ldos", "empty_site_candidates",
           "add_empty_sites", "amn_projection_method", "cols_of",
           "occupied_block"]

L_NAME = {0: "s", 1: "p", 2: "d", 3: "f"}


class CandidateScan(NamedTuple):
    """Result of candidate_scan.

    NOTE the return type changed from a 4-tuple: unpacking
    `p_kn, w, blocks, cand = candidate_scan(...)` now raises ValueError rather
    than failing silently.  Use `res = candidate_scan(...)` and res.p_kn etc.
    """
    p_kn: object            # (nk, nb) total projectability
    w: dict                 # {(orbit_rep, l): (nk, nb)} Loewdin weights
    blocks: list            # [(key, slice), ...] over the amn columns
    proj_set: object        # the candidate ProjectionsSet
    eps_kn: object          # (nk, nb) eigenvalues, aligned with p_kn
    wk_k: object            # (nk,) k-point weights
    occupancy: dict         # {key: electrons} Loewdin populations
    n_states: float         # states (NOT electrons) in the occupancy window
    spanned: float          # fraction of them the candidate set spans
    pdos: tuple             # (energies, {key: pdos}, total_dos)
    A: object               # (nk, nb, nproj) raw amn
    S: object               # (nk, nproj, nproj) true overlap, or None
    e_fermi: float


def build_candidate_set(atoms, spacegroup, shells=(0, 1, 2), symprec=1e-4,
                        rotate_basis=True):
    """One Projection per (symmetry orbit, l).  Returns (ProjectionsSet, blocks).

    Blocks are [(key, slice), ...] with key = (orbit_representative_atom, l),
    which is exactly the key format `select_by_rank` consumes.
    """
    import spglib
    from wannierberri.symmetry.projections import Projection, ProjectionsSet

    cell = (np.array(atoms.cell[:]), atoms.get_scaled_positions(),
            atoms.get_atomic_numbers())
    ds = spglib.get_symmetry_dataset(cell, symprec=symprec)
    scaled = atoms.get_scaled_positions()

    projs, blocks, off = [], [], 0
    for rep in sorted(set(ds.equivalent_atoms)):
        members = [i for i, e in enumerate(ds.equivalent_atoms) if e == rep]
        positions = [scaled[i] for i in members]
        for l in shells:
            p = Projection(position_num=positions, orbital=L_NAME[l],
                           spacegroup=spacegroup, rotate_basis=rotate_basis)
            n = len(members) * (2 * l + 1)
            projs.append(p)
            blocks.append(((rep, l), slice(off, off + n)))
            off += n
    return ProjectionsSet(projections=projs), blocks


def _stack_payload(obj, block_ndim, depth=0):
    """Coerce a WannierBerri file payload into a stacked array.

    block_ndim = 2 for amn (per-k block is (nb, nproj)), 1 for eig ((nb,)).
    Handles a plain array, a dict keyed by k-index (sorted, so insertion order
    does not matter), a list, or an object that is only indexable -- all of
    which np.asarray() turns into a 0-d OBJECT array.
    """
    if depth > 2:
        return None
    want = block_ndim + 1
    if isinstance(obj, np.ndarray) and obj.ndim == want and obj.dtype != object:
        return obj
    if isinstance(obj, dict):
        try:
            keys = sorted(obj)
        except TypeError:
            keys = list(obj)
        blocks = [np.asarray(obj[k]) for k in keys]
        return np.stack(blocks) if blocks and blocks[0].ndim == block_ndim else None
    if isinstance(obj, (list, tuple)):
        blocks = [np.asarray(x) for x in obj]
        return np.stack(blocks) if blocks and blocks[0].ndim == block_ndim else None
    try:
        blocks = [np.asarray(obj[i]) for i in range(len(obj))]
        if blocks and blocks[0].ndim == block_ndim:
            return np.stack(blocks)
    except Exception:                                       # noqa: BLE001
        pass
    return None


def _as_knm(obj, depth=0):
    """Coerce a WannierBerri file payload into a (nk, nb, nproj) array.

    The payload is not always a plain array: W90_file subclasses commonly hold
    per-k blocks in a dict keyed by k-index, or a list, or expose them only via
    __getitem__.  np.asarray() on any of those returns a 0-d OBJECT array, which
    is what the previous version reported as "ndim 0".
    """
    return _stack_payload(obj, 2, depth)


def describe_amn(wandata):
    """Print the structure of the amn payload.  Run this once per WannierBerri
    version; the layout differs and guessing has cost us a cycle already."""
    amn = getattr(wandata, "amn", None)
    print(f"wandata.amn : {type(amn)}")
    print(f"  attributes: {[a for a in dir(amn) if not a.startswith('__')][:40]}")
    for name in ("data", "_data", "amn", "A"):
        if hasattr(amn, name):
            v = getattr(amn, name)
            print(f"  .{name}: {type(v)}", end="")
            if isinstance(v, dict):
                k0 = next(iter(v))
                print(f"  dict[{len(v)}], key {k0!r} -> "
                      f"{np.asarray(v[k0]).shape}")
            elif isinstance(v, (list, tuple)):
                print(f"  seq[{len(v)}], first -> {np.asarray(v[0]).shape}")
            elif isinstance(v, np.ndarray):
                print(f"  ndarray {v.shape} {v.dtype}")
            else:
                print()
    try:
        print(f"  len(amn) = {len(amn)}; amn[0] -> {np.asarray(amn[0]).shape}")
    except Exception as e:                                  # noqa: BLE001
        print(f"  not indexable: {e}")


def extract_eig(wandata):
    """(nk, nb) eigenvalues in eV, stacked in the SAME order as extract_amn.

    Read them from the eig file rather than from calc.get_eigenvalues(): the amn
    lives on WannierBerri's irreducible k-set, whose ordering need not match
    GPAW's, and pairing p_nk with the wrong eigenvalues would misplace every
    band silently.  Request files=["amn", "eig"] -- eig costs nothing.
    """
    eig = getattr(wandata, "eig", None)
    if eig is None:
        raise AttributeError("no eig on the WannierData -- add \"eig\" to files=")
    for obj in (getattr(eig, "data", None), eig):
        if obj is None:
            continue
        E = _stack_payload(obj, 1)
        if E is not None:
            return np.asarray(E).real
    raise AttributeError(
        f"could not coerce the eig payload ({type(getattr(eig, 'data', eig))}) "
        "to (nk, nb).")


def extract_amn(wandata):
    """Pull the (nk, nb, nproj) array out of a WannierData, defensively."""
    amn = getattr(wandata, "amn", wandata)
    tried = []
    for name, obj in [("amn", amn),
                      ("amn.data", getattr(amn, "data", None)),
                      ("amn._data", getattr(amn, "_data", None)),
                      ("amn.A", getattr(amn, "A", None)),
                      ("amn.amn", getattr(amn, "amn", None))]:
        if obj is None:
            tried.append(f"{name}: absent")
            continue
        A = _as_knm(obj)
        if A is not None:
            return A
        tried.append(f"{name}: {type(obj).__name__}"
                     + (f"[{len(obj)}]" if hasattr(obj, "__len__") else ""))
    describe_amn(wandata)
    raise AttributeError(
        "could not coerce the amn payload to (nk, nb, nproj). Tried: "
        + "; ".join(tried) + ". The structure dump above says what it is -- "
        "add that path to _as_knm.")


def candidate_scan(calc, spacegroup, from_gpaw, spin_channel=0,
                   shells=(0, 1, 2), symprec=1e-4, verbose=True,
                   pdos_width=0.1, pdos_path=None, occ_window=None, **kw):
    """Build a wide candidate set, compute ONLY the amn, return ranked weights.

    from_gpaw : the WannierData.from_gpaw callable (injected so this module does
        not depend on a particular WannierBerri import path).

    Returns (p_kn, w, blocks, proj_set) with
        p_kn : (nk, nb) total projectability of the candidate set
        w    : {(orbit_rep, l): (nk, nb)}  -- the format select_by_rank wants
    """
    proj_set, blocks = build_candidate_set(calc.atoms, spacegroup,
                                           shells=shells, symprec=symprec)
    nproj = proj_set.num_wann
    nb = calc.wfs.bd.nbands
    if verbose:
        print(f"candidate set: {len(proj_set.projections)} projections, "
              f"{nproj} trial orbitals, {nb} bands")
    if nproj > nb:
        raise ValueError(
            f"candidate set has {nproj} trial orbitals but only {nb} bands. "
            "Increase nbands in the NSCF, or narrow `shells`.")

    out = from_gpaw(
        calculator=calc,
        spin_channel=spin_channel,
        projections=proj_set,
        irreducible=True,
        files=["amn", "eig"],          # eig is free; no mmn, no unk
        return_bandstructure=True,     # needed for the TRUE overlap
        **kw,
    )
    wandata, bandstructure = out if isinstance(out, tuple) else (out, None)

    A = extract_amn(wandata)
    if A.shape[2] != nproj:
        raise RuntimeError(
            f"amn has {A.shape[2]} columns but the candidate set declares "
            f"{nproj} -- the block map would be wrong. Check the array layout "
            "is (nk, nb, nproj).")

    S = O = None
    if bandstructure is not None:
        try:
            S, O = true_overlap(wandata.amn, bandstructure, verbose=verbose,
                                with_band_gram=True)
        except Exception as e:                                  # noqa: BLE001
            print(f"  WARNING: true_overlap failed ({e})")
            print("  falling back to S = A^dag A: p will be biased upward and "
                  "will drift with nbands, so do not calibrate any absolute "
                  "threshold on it.")
    p_kn, Atil = projectability_from_amn(A, S=S, O=O, verbose=verbose)
    w = weights_from_amn(Atil, blocks)

    # eigenvalues from the eig file so they are on the amn's k-ordering
    eps_kn = extract_eig(wandata)
    if eps_kn.shape != p_kn.shape:
        raise RuntimeError(
            f"eig is {eps_kn.shape} but p_kn is {p_kn.shape} -- the eig and amn "
            "k-sets or band counts disagree.")
    wk_k = np.asarray(calc.get_k_point_weights())
    if len(wk_k) != p_kn.shape[0]:
        # the amn is on WannierBerri's irreducible set; fall back to uniform
        if verbose:
            print(f"  NOTE {len(wk_k)} GPAW k-weights vs {p_kn.shape[0]} amn "
                  "k-points -- using uniform weights over the amn set")
        wk_k = np.full(p_kn.shape[0], 1.0 / p_kn.shape[0])

    e_fermi = calc.get_fermi_level()
    labels = {(rep, l): f"atom {rep} {calc.atoms[rep].symbol} l={L_NAME[l]}"
              for (rep, l), _ in blocks}
    g_s = 1 if getattr(calc, "get_number_of_spins", lambda: 1)() == 2 else 2
    occ, n_states = channel_occupancy(eps_kn, wk_k, w, e_fermi,
                                      window=occ_window, labels=labels,
                                      blocks=blocks, spin_degeneracy=g_s,
                                      verbose=verbose)
    spanned = sum(occ.values()) / max(n_states, 1e-30)
    pdos = pdos_from_weights(eps_kn, wk_k, w, width=pdos_width)

    if verbose and spanned < 0.95:
        print(f"  the candidate set spans only {spanned:.1%} of the occupied "
              "manifold -- add shells, or tune spread_factor (see tune_spread). "
              "This is a property of the PROJECTION SET, not of any cutoff.")
    if pdos_path is not None:
        g_pdos, g_total = gpaw_orbital_ldos(calc, blocks, pdos[0], spin=spin_channel,
                                            width=pdos_width, symprec=symprec,
                                            verbose=verbose)
        _plot_pdos(pdos_path, *pdos, e_fermi=e_fermi, labels=labels,
                   title=f"{calc.atoms.get_chemical_formula()} candidate pDOS",
                   gpaw_pdos=g_pdos, gpaw_total=g_total)
        if verbose:
            print(f"  pDOS written to {pdos_path}")

    if verbose:
        print(f"  blocks reconstruct p_nk: {np.allclose(sum(w.values()), p_kn)}")
    return CandidateScan(p_kn=p_kn, w=w, blocks=blocks, proj_set=proj_set,
                         eps_kn=eps_kn, wk_k=wk_k, occupancy=occ,
                         n_states=n_states,
                         spanned=spanned, pdos=pdos, A=A, S=S, e_fermi=e_fermi)


# --------------------------------------------------------------------------
# diagnostics and trial-orbital tuning
# --------------------------------------------------------------------------

def per_band_profile(p_kn, wk_k, eps_kn=None, verbose=True):
    """BZ-weighted mean projectability per band, and the largest drop.

    Use the k WEIGHTS.  A plain sum over irreducible k-points weights every
    point equally, which skews the profile toward whatever the IBZ oversamples.

    The largest drop between consecutive bands is the natural manifold edge.
    """
    wk = np.asarray(wk_k) / np.sum(wk_k)
    p_n = (wk[:, None] * p_kn).sum(axis=0)
    drops = np.diff(p_n)
    edge = int(np.argmin(drops))
    if verbose:
        print(f"  per-band projectability (BZ-weighted): "
              f"max {p_n.max():.3f}, min {p_n.min():.3f}")
        print(f"  largest drop at band {edge} -> {edge + 1}: "
              f"{p_n[edge]:.3f} -> {p_n[edge + 1]:.3f}  "
              f"=> {edge + 1} bands are well described")
        if eps_kn is not None:
            print(f"  that edge sits near {eps_kn[:, edge + 1].min():.3f} eV")
    return p_n, edge


def tune_spread(calc, spacegroup, from_gpaw, e_fermi, eps_kn,
                spreads=(0.6, 0.8, 1.0, 1.2, 1.5, 2.0), shells=(0, 1),
                spin_channel=0, symprec=1e-4, verbose=True, **kw):
    """Sweep spread_factor, score by mean projectability over OCCUPIED bands.

    WannierBerri builds trial orbitals analytically (Bessel_j_radial_int +
    Projector), which is a hydrogenic-type guess rather than the
    pseudopotential's own atomic orbitals.  
    Tuning the spread recovers most of the gap and costs one amn per value,
    which is cheap because mmn and unk are not computed.  Do it once per
    element, not per compound.
    """
    from wannierberri.symmetry.projections import Projection, ProjectionsSet
    import spglib

    cell = (np.array(calc.atoms.cell[:]), calc.atoms.get_scaled_positions(),
            calc.atoms.get_atomic_numbers())
    ds = spglib.get_symmetry_dataset(cell, symprec=symprec)
    scaled = calc.atoms.get_scaled_positions()
    wk = np.asarray(calc.get_k_point_weights())
    occ = eps_kn <= e_fermi

    results = {}
    for sf in spreads:
        projs = []
        for rep in sorted(set(ds.equivalent_atoms)):
            pos = [scaled[i] for i in range(len(scaled))
                   if ds.equivalent_atoms[i] == rep]
            for l in shells:
                try:
                    projs.append(Projection(position_num=pos, orbital=L_NAME[l],
                                            spacegroup=spacegroup,
                                            rotate_basis=True,
                                            spread_factor=sf))
                except TypeError as e:
                    raise TypeError(
                        "Projection does not accept spread_factor in this "
                        f"WannierBerri version ({e}); check the constructor "
                        "signature and adjust.") from e
        pset = ProjectionsSet(projections=projs)
        wandata = from_gpaw(calculator=calc, spin_channel=spin_channel,
                            projections=pset, irreducible=True, files=["amn"],
                            return_bandstructure=False, **kw)
        if isinstance(wandata, tuple):
            wandata = wandata[0]
        A = extract_amn(wandata)
        p_kn, _ = projectability_from_amn(A, verbose=False)
        num = float((wk[:, None] * p_kn * occ).sum())
        den = float((wk[:, None] * occ).sum())
        results[sf] = num / max(den, 1e-30)
        if verbose:
            print(f"  spread_factor {sf:5.2f}  ->  mean p over occupied bands "
                  f"= {results[sf]:.4f}")
    best = max(results, key=results.get)
    if verbose:
        print(f"  best: spread_factor = {best} (p_occ = {results[best]:.4f})")
        if results[best] < 0.97:
            print("  still below 0.97 -- the analytic orbital shape is the "
                  "limit, not the spread. Add radial_nodes, or accept a lower "
                  "p_thr for this orbital family.")
    return best, results


def gpaw_orbital_ldos(calc, blocks, energies, spin=0, width=0.1, npts=1001,
                      symprec=1e-4, verbose=True):
    """GPAW's own get_orbital_ldos on the same grid, summed over each orbit.

    Returns ({key: ldos}, total_dos) or (None, None) if the call fails -- this
    is a comparison panel, so it must never take the scan down with it.

    Read the two panels as measuring DIFFERENT things, not as a consistency
    check.  get_orbital_ldos is |<p_i|psi~>|^2 with no metric: the projectors
    are duals of the partial waves, so rescaling phi_i -> 2 phi_i changes the
    number by 4x.  Its vertical scale is a setup convention, so its channels
    cannot be compared to the total DOS or across elements, and its deficit says
    nothing about your projections.  The upper panel's deficit does.

    What IS comparable is the SHAPE: where each channel has weight in energy.
    If the two panels disagree about which channel dominates a manifold, one of
    them is wrong -- most likely the m-ordering or the block map.
    """
    from .projectability.sphere_projectability import equivalent_atoms

    try:
        eq = equivalent_atoms(calc.atoms, symprec)
        e_ref, _ = calc.get_orbital_ldos(a=0, spin=spin, angular="s",
                                         npts=npts, width=width)
        out = {}
        for (rep, l), _ in blocks:
            members = [i for i, r in enumerate(eq) if r == rep]
            acc = None
            for i in members:
                e_i, d_i = calc.get_orbital_ldos(a=i, spin=spin,
                                                 angular=L_NAME[l],
                                                 npts=npts, width=width)
                acc = d_i if acc is None else acc + d_i
            out[(rep, l)] = np.interp(energies, e_ref, acc, left=0.0, right=0.0)
        e_t, d_t = calc.get_dos(spin=spin, npts=npts, width=width)
        total = np.interp(energies, e_t, d_t, left=0.0, right=0.0)
        return out, total
    except Exception as e:                                      # noqa: BLE001
        if verbose:
            print(f"  (GPAW LDOS panel skipped: {type(e).__name__}: {e})")
        return None, None


def _plot_pdos(path, energies, pdos, total, e_fermi, labels=None, title="",
               gpaw_pdos=None, gpaw_total=None):
    """Upper: projectability pDOS from the amn.  Lower: GPAW's get_orbital_ldos.

    Upper panel: the gap between the summed channels and the total DOS is the
    part of the bands NO atom-centred orbital in the candidate set can
    represent.  That reading exists because the Loewdin weights partition a
    quantity bounded by 1.

    Lower panel: the same channels from GPAW's projector weights, which carry a
    setup-dependent scale.  Compare the SHAPES, not the heights.

    Both panels report max(sum/total) and max(single channel/total), and the
    region where the sum EXCEEDS the total is shaded red.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    two = gpaw_pdos is not None
    fig, axes = plt.subplots(2 if two else 1, 1, figsize=(7, 8 if two else 4.5),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]

    def draw(ax, chans, tot, tot_label, head):
        summed = sum(chans.values())
        if tot is not None:
            ax.fill_between(energies, tot, color="0.85", label=tot_label,
                            zorder=0)
            # A decomposition cannot exceed the whole. Mark where it does.
            live = tot > 0.02 * float(np.max(tot))
            ratio = float(np.max(summed[live] / tot[live])) if live.any() else 0.0
            single = max((float(np.max(v[live] / tot[live])) if live.any() else 0.0)
                         for v in chans.values())
            over = live & (summed > tot * 1.001)
            if over.any():
                ax.fill_between(energies, tot, summed, where=over,
                                color="red", alpha=0.30, zorder=1,
                                label="sum EXCEEDS total (not a partition)")
            ax.text(0.015, 0.95,
                    f"max(sum/total) = {ratio:.2f}\n"
                    f"max(single channel/total) = {single:.2f}",
                    transform=ax.transAxes, va="top", fontsize=7,
                    bbox=dict(fc="white", ec="0.6", alpha=0.85))
        ax.plot(energies, summed, color="k", lw=1.2, label="sum of channels")
        for key, v in sorted(chans.items()):
            ax.plot(energies, v, lw=1.0,
                    label=str(labels.get(key, key) if labels else key))
        ax.axvline(e_fermi, color="red", ls="--", lw=1, label="$E_F$")
        ax.set_ylabel("DOS (states / eV)")
        ax.set_title(head, fontsize=10)

    draw(axes[0], pdos, total,
         "total DOS (from the amn eigenvalues)",
         "projectability pDOS (Loewdin, amn) -- sum <= total by construction; "
         "the gap is unrepresentable weight")
    axes[0].legend(fontsize=7, ncol=2)
    if two:
        draw(axes[1], gpaw_pdos, gpaw_total, "total DOS (GPAW)",
             "GPAW get_orbital_ldos -- no sum rule; red = parts exceed the whole")
        axes[1].legend(fontsize=7, ncol=2)
    axes[-1].set_xlabel("energy (eV)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# generalisation: trial orbitals off the atoms
# --------------------------------------------------------------------------

def empty_site_candidates(atoms, spacegroup, n_grid=24, min_dist=1.0,
                          max_sites=6, symprec=1e-4, verbose=True):
    """High-symmetry EMPTY positions, ranked by site-symmetry order.

    Atom-centred orbitals cannot span every manifold.  Interstitial states in
    open or electride-like structures, bond-centred Wannier functions in some
    covalent solids, and free-electron bands in simple metals all live where
    there is no atom.  Vitale et al. handle this by adding hydrogenic projectors
    at extra positions; `extend_to_energy_window` in your own workflow already
    does something similar by hand.

    Nothing else in the pipeline needs to change.  A Projection at an empty
    Wyckoff position produces amn columns exactly like an atomic one, so
    greedy_select scores it identically -- and `build` already derives
    symmetry-adapted orbitals at ANY position from its site group, so the SALC
    layer generalises for free too.  The only new part is proposing WHERE, which
    is what this does.

    Method: sample the cell, keep local maxima of the distance to the nearest
    atom (the interstitial voids) plus all symmetrised bond midpoints, then rank
    by |G_q|.  A geometric search rather than a Wyckoff-table lookup, so it
    needs no database and also finds bond centres, which are not always special
    positions.  If WannierBerri's symmetry.wyckoff_position module can enumerate
    the empty orbits of your space group, prefer that -- it is exact.

    Returns [(frac_position, |G_q|, distance_to_nearest_atom), ...].
    """
    import spglib

    cell = (np.array(atoms.cell[:]), atoms.get_scaled_positions(),
            atoms.get_atomic_numbers())
    C = np.array(atoms.cell[:])
    pos = atoms.get_positions()

    g = np.linspace(0, 1, n_grid, endpoint=False)
    F = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1)
                       for k in (-1, 0, 1)])
    cart = F @ C
    images = (pos[None, :, :] + (shifts @ C)[:, None, :]).reshape(-1, 3)
    d = np.linalg.norm(cart[:, None, :] - images[None, :, :], axis=2).min(axis=1)

    order = np.argsort(-d)
    picked = []
    for idx in order:
        if d[idx] < min_dist:
            break
        p = cart[idx]
        if all(np.linalg.norm(p - q) > min_dist for q in picked):
            picked.append(p)
        if len(picked) >= 4 * max_sites:
            break

    # bond midpoints (not generally special positions, but often where a
    # covalent Wannier centre wants to sit)
    from .projectability.sphere_projectability import equivalent_atoms
    eq = equivalent_atoms(atoms, symprec)
    for i in sorted(set(eq)):
        v = pos[None, :, :] + (shifts @ C)[:, None, :] - pos[i]
        v = v.reshape(-1, 3)
        r = np.linalg.norm(v, axis=1)
        near = v[(r > 1e-3) & (r < r[r > 1e-3].min() + 0.15)]
        for b in near:
            m = pos[i] + 0.5 * b
            dm = np.linalg.norm(m - images, axis=1).min()
            if dm > 0.4 * min_dist and all(np.linalg.norm(m - q) > 0.5 * min_dist
                                           for q in picked):
                picked.append(m)

    from .salc import site_group
    out = []
    seen = []
    for p in picked:
        q = np.linalg.solve(C.T, p) % 1.0
        if any(np.linalg.norm(((q - r + .5) % 1) - .5) < 10 * symprec
               for r in seen):
            continue
        seen.append(q)
        ops = site_group(cell, q, symprec)
        dist = float(np.linalg.norm(p - images, axis=1).min())
        out.append((q, len(ops), dist))
    out.sort(key=lambda t: (-t[1], -t[2]))
    out = out[:max_sites]

    if verbose:
        print(f"  empty-site candidates (ranked by site symmetry):")
        for q, n, dist in out:
            print(f"    {np.round(q, 4)}   |G_q| = {n:3d}   "
                  f"{dist:.2f} A from the nearest atom")
    return out


def add_empty_sites(proj_set, blocks, sites, spacegroup, shells=(0,),
                    rotate_basis=True):
    """Append Projections at empty positions; returns (proj_set, blocks).

    Start with s only: a spherically symmetric blob is what an interstitial
    state usually wants, and it costs one WF per site.
    """
    from wannierberri.symmetry.projections import Projection, ProjectionsSet

    projs = list(proj_set.projections)
    blocks = list(blocks)
    off = blocks[-1][1].stop if blocks else 0
    for j, (q, nops, _) in enumerate(sites):
        for l in shells:
            p = Projection(position_num=[q], orbital=L_NAME[l],
                           spacegroup=spacegroup, rotate_basis=rotate_basis)
            projs.append(p)
            blocks.append(((f"empty{j}", l), slice(off, off + p.num_wann)))
            off += p.num_wann
    return ProjectionsSet(projections=projs), blocks


# --------------------------------------------------------------------------
# the driver: same contract as sphere_projection_method
# --------------------------------------------------------------------------

def cols_of(chosen, blocks):
    """Column indices of a set of block keys."""
    parts = [np.arange(sl.start, sl.stop) for k, sl in blocks if k in chosen]
    return np.concatenate(parts) if parts else np.zeros(0, int)


def occupied_block(eps_kn, e_fermi, gap_thres=5.0, verbose=True):
    """(emin, emax) of the band block holding the occupied manifold.

    Band-INDEX gaps, so it is immune to k-mesh density, and semicore sits in its
    own block and is excluded.  gap_thres only decides what counts as connected:
    ~5 eV crosses an intra-valence gap.
    """
    blks = band_blocks(eps_kn, gap_thres)
    occupied = np.where((eps_kn <= e_fermi).any(axis=0))[0]
    if occupied.size == 0:
        raise ValueError("no states below E_F")
    n_hi = int(occupied.max())
    blk = next(b for b in blks if b[0] <= n_hi <= b[1])
    if verbose and world.rank == 0:
        print(f"  band blocks (gap_thres={gap_thres} eV):")
        for a, b, e0, e1 in blks:
            mark = "  <- occupied manifold" if (a, b) == blk[:2] else ""
            print(f"    bands {a:3d}-{b:3d}   {e0:9.3f} .. {e1:9.3f} eV{mark}")
    return float(blk[2]), float(blk[3])


def amn_projection_method(
    calc=None,
    spacegroup=None,
    from_gpaw=None,
    seed=None,
    in_dir="test",
    out_dir="test",
    K=1.2,
    gap_thres=5.0,
    objective_wd=None,
    maximize_fw=False,
    shells=(0, 1, 2),
    spin_channel=0,
    select="occupancy",
    alpha=0.45,
    renormalize=False,
    margin=1.2,
    p_target=None,
    p_froz=None,
    symprec=1e-4,
    comm=serial_comm,
    plot=True,
    verbose=True,
    return_details=False,
    **kw,
):
    """Select projections and windows from ONE candidate amn pass.

    Same 4-tuple as sphere_projection_method, so it drops into get_proj_set:

        selected_orbitals, out_win, frozen_win, nwann

    with selected_orbitals = [(iatom, (n, l_str)), ...] expanded over every
    member of each chosen symmetry orbit.

    p_froz : None (default) means the projectability gate on the frozen window
        is OFF and p is reported as a diagnostic only; the window is the target
        constrained by N_k <= nwann.

    margin : nwann target as a multiple of n_froz, the number of bands the
        frozen window must hold at the worst k.  This is the SIZE criterion and
        it rests on a hard requirement -- Wannier90 cannot freeze more bands
        than it has functions -- so it needs no calibration.  Projectability
        only decides the ORDER blocks are added in.  Blocks are whole orbits so
        nwann moves in chunks: Si has n_froz = 6 with blocks of 6/2/10, and any
        margin in (1.0, 1.33] gives 8 WF (sp).  Keep it above 1.0; exactly 1.0
        stops at p alone.  NOTE this is unrelated to `K`, which sizes the OUTER
        window.

    p_target : optional coverage-based early exit.  None by default -- with
        WannierBerri's analytic trial orbitals the whole candidate set reaches
        only ~0.85 on Si, so an absolute coverage threshold is not calibrated.

    gap_thres : what counts as a connected band block.  With `margin` these are
        the only two free numbers.

    ORDER MATTERS.  Windows come from the band structure alone (step 1), the amn
    is computed once (step 2), the SET is chosen by coverage (step 3), and only
    then are the windows sized from the chosen set (step 4).  Nothing feeds back
    into the selection window, which is what made the earlier fixed point
    fragile.

    Requires: calc loaded with communicator=serial_comm and written with
    mode='all'; from_gpaw = WannierData.from_gpaw; spacegroup as passed to build.
    """
    if from_gpaw is None:
        raise ValueError("pass from_gpaw=WannierData.from_gpaw")
    if calc is None:
        from gpaw import GPAW
        calc = GPAW(f"{in_dir}/{seed}/{seed}-nscf-irred.gpw", txt=None,
                    communicator=comm)
    if spacegroup is None:
        from irrep.spacegroup import SpaceGroup
        spacegroup = SpaceGroup.from_gpaw(calc)

    def say(*a):
        if verbose and world.rank == 0:
            print(*a)

    # ---- 2. one candidate amn (needed before the windows, since eps_kn must be
    #         on the amn's k-ordering rather than GPAW's)
    pdos_path = None
    if plot and seed is not None:
        d = Path(f"{out_dir}/{seed}")
        d.mkdir(parents=True, exist_ok=True)
        pdos_path = d / f"{seed}-candidate-pdos.png"
    say("\ncandidate amn pass:")
    res = candidate_scan(calc, spacegroup, from_gpaw, spin_channel=spin_channel,
                         shells=shells, symprec=symprec, verbose=verbose,
                         pdos_path=pdos_path, **kw)
    eps_kn, wk_k, e_fermi = res.eps_kn, res.wk_k, res.e_fermi
    if res.S is None:
        warnings.warn(
            "no true overlap: p is biased upward and drifts with nbands, so "
            f"p_target={p_target} and p_froz={p_froz} are not calibrated. Fix "
            "true_overlap before trusting the windows.")

    # ---- 1. windows from the band structure alone
    say(f"\nFermi level: {e_fermi:.4f} eV")
    emin, _ = occupied_block(eps_kn, e_fermi, gap_thres, verbose=verbose)
    if objective_wd is not None:
        emin = objective_wd[0]
        froz_hi = objective_wd[1]
        target_name = "objective_wd"
    elif maximize_fw:
        froz_hi = float(eps_kn.max())
        target_name = "maximize_fw"
    else:
        froz_hi = e_fermi + 2.0
        target_name = "E_F + 2 eV"

    # ---- 3. select the SET by coverage of the manifold to be reproduced
    target = (eps_kn >= emin) & (eps_kn <= froz_hi)
    n_min = int(target.sum(axis=1).max())
    labels = {(rep, l): f"atom {rep} {calc.atoms[rep].symbol} l={L_NAME[l]}"
              for (rep, l), _ in res.blocks if isinstance(rep, (int, np.integer))}
    say(f"\ntarget manifold {emin:.3f} .. {froz_hi:.3f} eV "
        f"({n_min} bands at the worst k):")
    if select == "occupancy":
        g_s = 1 if getattr(calc, "get_number_of_spins", lambda: 1)() == 2 else 2
        # select over the WHOLE window, conduction included -- an
        # occupied-only rule drops essential empty shells (Ti 3d in BaTiO3)
        occ_w, n_st = channel_occupancy(eps_kn, wk_k, res.w, e_fermi,
                                        window=(emin, froz_hi),
                                        clip_to_fermi=False,
                                        renormalize=renormalize, labels=labels,
                                        blocks=res.blocks, spin_degeneracy=g_s,
                                        verbose=(verbose and world.rank == 0))
        chosen, nwann = select_by_occupancy(occ_w, res.blocks, n_st,
                                            alpha=alpha,
                                            verbose=(verbose and world.rank == 0),
                                            labels=labels)
        coverage, history = float("nan"), []
        if nwann < n_min:
            warnings.warn(
                f"nwann={nwann} < n_froz={n_min}: the frozen window cannot fit. "
                "Lower alpha, or use select='greedy', which sizes on n_froz.")
    else:
        chosen, nwann, coverage, history = greedy_select(
            res.A, res.S if res.S is not None else
            np.stack([res.A[k].conj().T @ res.A[k] for k in range(len(wk_k))]),
            res.blocks, target, wk_k, n_froz=n_min, margin=margin,
            p_target=p_target, verbose=(verbose and world.rank == 0),
            labels=labels)

    # ---- 4. windows from the chosen set
    cols = cols_of(chosen, res.blocks)
    p_sel = subset_projectability(
        res.A, res.S if res.S is not None else
        np.stack([res.A[k].conj().T @ res.A[k] for k in range(len(wk_k))]), cols)

    # Scan from E_F UPWARD only.  The occupied bands are what the Wannier
    # functions must reproduce, so they are frozen whatever their p; a dip below
    # E_F must never truncate the window.  Scanning from emin is what collapsed
    # the frozen window to a point (-6.6514 .. -6.6514) on Si.
    scan_from = max(emin, e_fermi)
    diag_p = p_froz if p_froz is not None else 0.95
    froz_diag = frozen_from_projectability(eps_kn, p_sel, scan_from, froz_hi,
                                           p_froz=diag_p)
    occ_p = p_sel[eps_kn <= e_fermi]
    say(f"  occupied p: min {occ_p.min():.4f}, mean {occ_p.mean():.4f}; "
        f"at p_froz={diag_p} the first state below it above E_F is at "
        f"{froz_diag:.3f} eV")
    if p_froz is None:
        say("  projectability gate OFF (diagnostic only) -- frozen window set "
            "by the target and the N_k <= nwann cap")
        froz_p = froz_hi
    else:
        if froz_diag < froz_hi:
            say(f"  gate ON -> frozen window narrowed from {froz_hi:.3f} to "
                f"{froz_diag:.3f}")
        froz_p = max(froz_diag, scan_from)
    capped = cap_frozen_window(eps_kn, emin, froz_p, nwann)
    bound_by = target_name if capped >= froz_p - 1e-6 else \
        f"N_k <= nwann={nwann} cap"
    if capped < froz_p - 1e-6:
        say(f"  N_k <= nwann={nwann} caps it at {capped:.3f} eV -- if that is "
            "far below your target the PROJECTION SET is too small")
    frozen_win = (float(emin), float(capped))
    out_hi = emax_from_band_count(eps_kn, emin, nwann, K=K)
    pad = 1e-4
    out_win = (float(min(emin, frozen_win[0]) - pad),
               float(max(out_hi, frozen_win[1]) + pad))
    assert out_win[0] < frozen_win[0] and frozen_win[1] < out_win[1]

    # ---- expand orbits to the [(iatom, (n, l_str))] contract
    from .projectability.sphere_projectability import bound_channels, equivalent_atoms
    eq = equivalent_atoms(calc.atoms, symprec)
    selected_orbitals, extra = [], []
    for key in sorted(chosen, key=str):
        rep, l = key
        if not isinstance(rep, (int, np.integer)):
            extra.append(key)
            continue
        for a in [i for i, r in enumerate(eq) if r == rep]:
            ns = [n for n, _ in bound_channels(calc.setups[a]).get(l, [])]
            selected_orbitals.append((a, (ns[-1] if ns else None, L_NAME[l])))
    if extra:
        warnings.warn(
            f"{len(extra)} non-atomic block(s) selected ({extra}) -- these are "
            "empty-Wyckoff projections and cannot be expressed as "
            "(iatom, (n, l)). They are NOT in selected_orbitals; build the "
            "ProjectionsSet from `details.chosen` instead of from the 4-tuple.")

    window_rank_check(res.A, cols, eps_kn, out_win, verbose=verbose,
                      labels=labels, blocks=res.blocks)

    inside = (eps_kn >= frozen_win[0]) & (eps_kn <= frozen_win[1])
    say(f"\nSelected orbitals: "
        f"{[(calc.atoms[a].symbol, a, ls) for a, (n, ls) in selected_orbitals]}")
    say(f"Outer window : {out_win[0]:.4f} .. {out_win[1]:.4f} eV")
    say(f"Frozen window: {frozen_win[0]:.4f} .. {frozen_win[1]:.4f} eV "
        f"(target {target_name}; bound by {bound_by})")
    say(f"nwann = {nwann};  coverage of the target manifold = {coverage:.4f};  "
        f"mean p in the frozen window = {float(p_sel[inside].mean()):.4f}")

    if seed is not None and world.rank == 0:
        d = Path(f"{out_dir}/{seed}")
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "selection.txt", "w") as f:
            f.write(f"p_target={p_target} p_froz={p_froz} K={K} "
                    f"gap_thres={gap_thres}\n")
            f.write(f"outer  = {out_win}\nfrozen = {frozen_win}\n")
            f.write(f"nwann  = {nwann}\ncoverage = {coverage:.4f}\n")
            f.write(f"candidate set spans {res.spanned:.1%} of the occupancy "
                    "window\n\ngreedy order:\n")
            for key, sz, nw, cov in history:
                f.write(f"  + {key}  +{sz} WF  nwann={nw}  coverage={cov:.4f}\n")

    if return_details:
        return selected_orbitals, out_win, frozen_win, nwann, dict(
            scan=res, chosen=chosen, coverage=coverage, p_sel=p_sel,
            history=history, cols=cols)
    return selected_orbitals, out_win, frozen_win, nwann