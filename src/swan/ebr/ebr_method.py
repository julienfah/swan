"""Performs EBR projection search with an `alphabet` switch with four modes.

  'shells'        plain s/p/d only in the SEARCH, hybridised before validation.
  'shells+hyb'    build_at's blocks (hyb + rests) PLUS whole s/p/d shells --
                  formerly 'as_is'; no hybridisation step, the hybrids are
                  already in the alphabet.
  'isotypic'      per-shell irrep blocks in the SEARCH, hybridised before
                  validation.
  'isotypic+hyb'  per-shell irrep blocks PLUS build_at's blocks; no
                  hybridisation step, for the same reason as 'shells+hyb'.s
"""

from irrep.bandstructure import BandStructure
from irrep.spacegroup import SpaceGroup
import irrep.spacegroup as irrep_spacegroup
import sympy
from sympy import symbols, sympify
from wannierberri.symmetry.sawf import SymmetrizerSAWF
from wannierberri.symmetry.projections_searcher import EBRsearcher
from wannierberri.symmetry.projections import Projection, ProjectionsSet
from wannierberri.symmetry.wyckoff_position import WyckoffPosition
from wannierberri.symmetry import orbitals as wb_orb
from wannierberri.w90files.amn import AMN
import numpy as np
from gpaw import GPAW
from math import ceil
from pathlib import Path as Path_
from swan.utils.utils import find_emax_from_dos
from swan.auto_proj_and_windows import initial_DOS_energy_scan
from swan.symmetry import (NAME_OF, build_at, describe_orbital,
                           isotypic_components, parse_position, parse_shells,
                           register, shell_rep, site_group)
from swan.projectability.amn_projectability import orbital_window_fraction, true_overlap
from swan.ebr.candidate_scan import _stack_payload
from swan.ebr.ebr_select import (dedupe_combinations, rank_combinations,
                              score_combinations)
from swan.ebr.ebr_log import write_selection_log
from swan.ebr.validate_candidates import combination_tag

x, y, z = symbols('x y z')
_locals = {'x': x, 'y': y, 'z': z}

import re
import copy
from sympy import sympify as _sympify_orig

def _sympify_implicit_mult(s, *args, **kwargs):
    if isinstance(s, str):
        s = re.sub(r'(?<=[\d)])(?=[a-zA-Z])', '*', s)
    return _sympify_orig(s, *args, **kwargs)
irrep_spacegroup.sympify = _sympify_implicit_mult


def eig_from_bandstructure(amn, bandstructure):
    """(nk, nb) eigenvalues in the SAME order _stack_payload stacks amn.data."""
    keys = sorted(amn.data) if isinstance(amn.data, dict) else range(len(amn.data))
    out = []
    for i, _ in enumerate(keys):
        kp = bandstructure.kpoints[i]
        for name in ("Energy", "energies", "Energy_raw", "E"):
            if hasattr(kp, name):
                out.append(np.asarray(getattr(kp, name)).real.ravel())
                break
        else:
            raise AttributeError(
                f"no energy attribute on {type(kp)}; tried Energy/energies/E.")
    return np.array(out)


# ---------------------------------------------------------------------------
# alphabet helpers
# ---------------------------------------------------------------------------

def _block_rows(name, shells_l):
    """Rows of a registered or plain orbital block in the joint shell layout."""
    labels = [o for l in shells_l for o in wb_orb.orbitals_sets_dic[NAME_OF[l]]]
    rows = []
    for member in wb_orb.orbitals_sets_dic[str(name)]:
        coef = wb_orb.hybrids_coef.get(member)
        if coef is None:
            rows.append([1.0 if o == member else 0.0 for o in labels])
        else:
            rows.append([float(coef.get(o, 0.0)) for o in labels])
    return np.array(rows, float)


def shell_component_blocks(cell, pos_str, shells, prefix="CMP"):
    """Irrep blocks of EACH SHELL separately at one site.

    Per shell, not on the joint s+p+d space: a joint decomposition merges
    everything carrying the same irrep across shells (at C_3v a1 lives in s,
    p_z AND d_z2), and pure s stops being selectable. Per shell every block sits
    inside one shell, so each shell is exactly the sum of its own components --
    nothing a plain shell could reach is lost, and a shell that splits gains
    resolution it did not have (at O_h, d -> e_g + t_2g).

    Returns [(orbital_name, dim), ...]; a shell that does not split keeps its
    plain name, so nothing is registered for it.
    """
    q = parse_position(pos_str.split(',')) % 1.0
    ops = site_group(cell, q)
    out = []
    for l in parse_shells(shells):
        comps = isotypic_components([shell_rep([l], R) for R in ops])
        if len(comps) == 1:
            out.append((NAME_OF[l], 2 * l + 1))
            continue
        for k, (M, _d, _n) in enumerate(comps):
            # `__` is load bearing: validate_candidates._orb_short splits on it,
            # so only 'd0' reaches the tag and hence the candidate directory.
            nm = f"{prefix}_{pos_str}__{NAME_OF[l]}{k}"
            register(nm, M, [l], ops)
            out.append((nm, M.shape[0]))
    return out


def hybridize_combination(c, trial_projections, site_of, atoms, spacegroup,
                          shells_l, prefer_plain_shell=True, verbose=False):
    """Rewrite a selected combination as bond hybrids, preserving every span.

    A site is hybridised only when its selected blocks are EXACTLY a sum of
    whole shells -- build_at needs the full (2l+1)-dimensional space to solve
    the intertwiner in, and a partial selection (t_2g alone) has no full shell
    to re-express. Anything else is passed through untouched.
    """
    c = np.asarray(c, int)
    by_site = {}
    for j, s in enumerate(site_of):
        if c[j] > 0:
            by_site.setdefault(s, []).append(j)

    out = ProjectionsSet()
    for site, idxs in sorted(by_site.items()):
        rows = np.vstack([_block_rows(o, shells_l)
                          for j in idxs
                          for o in trial_projections.projections[j].orbitals])
        P = rows.T @ rows
        # which shells are covered in FULL, and is the span exactly those?
        full, Q, off = [], np.zeros_like(P), 0
        for l in shells_l:
            n = 2 * l + 1
            if np.allclose(P[off:off + n, off:off + n], np.eye(n), atol=1e-6):
                full.append(l)
                Q[off:off + n, off:off + n] = np.eye(n)
            off += n
        done = False
        if full and np.allclose(P, Q, atol=1e-6):
            # The span is exactly whole shells, so THREE descriptions of it
            # exist and they are not equally safe. In order:
            #
            #  1. bond hybrid   -- localised lobes, the best initial guess;
            #  2. plain shell   -- one orbital per projection, so D_wann is a
            #     single block and no column-ordering question arises;
            #  3. components    -- several orbitals, split blocks.
            pset, res = build_at(atoms=atoms, position=site.split(','),
                                 shells=[NAME_OF[l] for l in full],
                                 spacegroup=spacegroup,
                                 label=f"HYB_{site}_", verbose=False,
                                 fallback="minimal_shell")
            if getattr(res, "hybrid", False):
                for pr in pset.projections:
                    out.add(pr)
                done = True
                if verbose:
                    print(f"    hybridised {site}: "
                          f"{''.join(NAME_OF[l] for l in full)}")
            elif prefer_plain_shell:
                for l in full:
                    out.add(Projection(position_sym=site, orbital=NAME_OF[l],
                                       spacegroup=spacegroup))
                done = True
                if verbose:
                    print(f"    {site}: no bond hybrid -> plain "
                          f"{''.join(NAME_OF[l] for l in full)}")
        if not done:
            for j in idxs:
                out.add(copy.deepcopy(trial_projections.projections[j]))
    return out



def _add_block(trial_projections, site_of, seen_spans, proj, p, shells_l,
               prefer, verbose=True):
    """Add a projection unless a block with the SAME SPAN is already at this site.
    
    `prefer` decides which of two equal-span blocks is kept: 'hybrid' keeps the
    build_at block, 'component' keeps the CMP/plain one. They span the same
    space, so this changes no coverage and no admissibility -- only the initial
    guess.
    """
    rows = np.vstack([_block_rows(str(o), shells_l) for o in proj.orbitals])
    P = rows.T @ rows
    is_hyb = any('WP_' in str(o) for o in proj.orbitals)
    if prefer is None:
        trial_projections.add(proj)
        site_of.append(p)
        return True
    for k, (site, Q, kept_hyb, j) in enumerate(seen_spans):
        if site != p or not np.allclose(P, Q, atol=1e-6):
            continue
        take_new = (is_hyb and not kept_hyb) if prefer == 'hybrid' \
            else (kept_hyb and not is_hyb)
        if take_new:
            trial_projections.projections[j] = proj
            seen_spans[k] = (p, Q, is_hyb, j)
            if verbose:
                print(f"    {p}: {proj.orbitals[0]} replaces an equal-span "
                      f"block (prefer={prefer!r})")
        elif verbose:
            print(f"    {p}: {proj.orbitals[0]} dropped, equal span to a "
                  "block already added")
        return False
    trial_projections.add(proj)
    site_of.append(p)
    seen_spans.append((p, P, is_hyb, len(trial_projections.projections) - 1))
    return True


ALPHABETS = ('shells', 'shells+hyb', 'isotypic', 'isotypic+hyb')


def EBR_method(in_dir, out_dir, seed, ecut, only_on_site=True, calc=None,
               verbose=False, comm=None, K=1.2, p_min=None, margin=2,
               max_score=10000, min_gain=0.02, use_in_window=False, gap_thres=1,
               objective_wd=None, include_empty=False, empty_shells=('s',),
               dedupe_prefer='hybrid', dedupe_rank_first=True,
               dedupe_by='span', band_gram=True,
               validate=False, validate_n_max=5, validate_kwargs=None,
               eta_ok=20.0, spread_ok=10.0,
               per_size_window=True,
               alphabet='isotypic+hyb',
               block_dedupe='hybrid', k_values=(1.5,1.8),
               prefer_plain_shell=False):
    """Uses an EBRsearcher to find symmetry adapted projections for a given system

    alphabet : one of ALPHABETS.  See the module docstring.
        The '+hyb' modes put build_at's blocks in the SEARCH and skip the
        post-selection hybridisation (the hybrids are already selectable);
        the others hybridise the chosen set, span preserving, before
        validation.

    k_values : None (default) -> one wannierisation per candidate, with the
        pipeline's own K. A sequence such as (1.2, 1.5) validates each
        candidate at each K, stopping at the first that is good enough, and
        records every attempt. The winner's K is then used for the returned
        outer window, so the window shipped is the window measured.

    block_dedupe : 'hybrid' | 'component'. Which of two equal-span blocks at
        the same site to keep. The '+hyb' alphabets contain literal duplicates
        -- on GaN, CMP_d0 == WP_rest0 and CMP_d1 == WP_rest1 at both sites,
        20 redundant amn columns out of 72. They span the same space, so the
        choice changes no coverage and no admissibility, only the initial
        guess. Set None to keep both (the old behaviour).

    prefer_plain_shell : when a site's span is a complete shell but build_at
        finds no bond hybrid, ship the plain shell instead of the component
        blocks. Only used by the non-'+hyb' modes.

    per_size_window : run one EBRsearcher per candidate SIZE, each with the
        outer window that size will actually be wannierised in.
    """
    if alphabet not in ALPHABETS:
        raise ValueError(f"alphabet must be one of {ALPHABETS}, got {alphabet!r}")
    if calc is None:
        calc = GPAW(f"{in_dir}/{seed}/{seed}-nscf-irred.gpw", txt=None, communicator=comm)
    print("Building bandstructure...")
    bandstructure = BandStructure.from_gpaw(calculator_gpaw=calc, code="gpaw",
                                Ecut=ecut, include_TR=True, spin_channel=0)
    spacegroup = bandstructure.spacegroup
    if verbose:
        print(f"spacegroup: {spacegroup.number} {spacegroup.name}")
        print(SpaceGroup.from_gpaw(calc).name)
        print(calc.wfs.gd.N_c)
    print("building symmetrizer...")
    n_bands = calc.get_number_of_bands()
    n_skipped_bands = max(2, ceil(n_bands*0.05))
    if verbose:
        print(f"n_bands: {n_bands}, n_skipped_bands: {n_skipped_bands}")
    symmetrizer = SymmetrizerSAWF.from_irrep(bandstructure, irreducible=True,unitary_params=dict(error_threshold=0.1, warning_threshold=0.01, nbands_upper_skip=n_skipped_bands))

    trial_projections = ProjectionsSet()
    cell = calc.atoms.cell
    atomic_positions = calc.atoms.get_scaled_positions()
    numbers = calc.atoms.numbers
    lattice = (cell, atomic_positions, numbers)
    wps = SpaceGroup.wyckoff_positions(lattice)
    print(wps)
    WP =parse_wp_strings(wps)
    selected_positions = []

    if only_on_site:
        import spglib
        _ds = spglib.get_symmetry_dataset(
            (np.array(cell), atomic_positions, numbers), symprec=1e-4)
        for atom_idx in sorted(set(_ds.equivalent_atoms)):
            atomic_pos = atomic_positions[atom_idx]
            candidates = []
            for pos in WP:
                wpos = WyckoffPosition(
                    position_str=",".join(str(p) for p in pos),
                    spacegroup=spacegroup,
                )
                if wyckoff_contains(wpos, atomic_pos):
                    candidates.append((wpos.num_points, pos, wpos))
            if candidates:
                mult, pos, wpos = min(candidates, key=lambda c: c[0])
                if wpos.num_free_vars > 0:
                    pos = [float(v) for v in atomic_pos]
                    print(f"  {calc.atoms[atom_idx].symbol}{atom_idx}: Wyckoff "
                          f"position has {wpos.num_free_vars} free parameter(s)"
                          f" -> pinned to the atom at "
                          f"{np.round(atomic_pos, 6).tolist()}")
                selected_positions.append(list(pos))
        seen, uniq = set(), []
        for p in selected_positions:
            k = ",".join(str(v) for v in p)
            if k not in seen:
                seen.add(k)
                uniq.append(p)
        selected_positions = uniq
    else:
        selected_positions = WP

    empty_positions = []
    if include_empty:
        occupied = {tuple(str(v) for v in p) for p in selected_positions}
        for pos in WP:
            if any(getattr(e, "free_symbols", set()) for e in pos):
                continue
            if tuple(str(v) for v in pos) in occupied:
                continue
            wpos = WyckoffPosition(position_str=",".join(str(p) for p in pos),
                                   spacegroup=spacegroup)
            if any(wyckoff_contains(wpos, ap) for ap in atomic_positions):
                continue
            empty_positions.append(pos)

    selected_positions_str = [",".join(str(y) for y in x) for x in selected_positions]

    print("Selected positions for projections:")
    for p in selected_positions_str:
        print(p)

    # ---- ALPHABET ---------------------------------------------------------
    shells_l = parse_shells(['s', 'p', 'd'])
    cell_t = (np.array(cell[:]), atomic_positions, numbers)
    site_of = []                    # site label per projection index
    seen_spans = []                 # (site, projector, is_hybrid, index)
    with_hyb = alphabet.endswith('+hyb')
    base = alphabet[:-4] if with_hyb else alphabet
    print(f"alphabet = {alphabet!r}")
    for p in selected_positions_str:
        # RADIAL family: whole shells, or their per-shell irrep blocks.
        if base == 'shells':
            for l in ['s', 'p', 'd']:
                _add_block(trial_projections, site_of, seen_spans,
                           Projection(position_sym=p, orbital=l,
                                      spacegroup=spacegroup),
                           p, shells_l, block_dedupe, verbose)
        else:                                   # 'isotypic'
            q_p = parse_position(p.split(',')) % 1.0
            for name, dim in shell_component_blocks(cell_t, p, ['s', 'p', 'd']):
                if name in ('s', 'p', 'd'):
                    # a whole shell is FRAME INDEPENDENT: position_sym, no
                    # rotate_basis, every site of the orbit using the same
                    # global p_x/p_y/p_z.
                    pr = Projection(position_sym=p, orbital=name,
                                    spacegroup=spacegroup)
                else:
                    # a split block is not: it must be rotated into each site's
                    # own frame.
                    pr = Projection(position_num=[q_p], orbital=name,
                                    spacegroup=spacegroup, rotate_basis=True)
                _add_block(trial_projections, site_of, seen_spans, pr,
                           p, shells_l, block_dedupe, verbose)
        # GEOMETRIC family: bond lobes plus irrep-adapted complements. Adds the
        # cross-shell and within-shell mixtures no radial decomposition can
        # produce (CaMg2Bi2's `-0.0149|s> -0.9999|dz2>` and its 2-dim piece
        # inside a merged 4-dim d block).
        if with_hyb:
            pset, _ = build_at(atoms=calc.atoms, position=p.split(','),
                               shells=['s', 'p', 'd'],
                               spacegroup=spacegroup, label=f"WP_{p}_",
                               verbose=True, fallback="minimal_shell")
            for proj in pset.projections:
                _add_block(trial_projections, site_of, seen_spans, proj,
                           p, shells_l, block_dedupe, verbose)

    if empty_positions:
        print(f"Empty Wyckoff positions added, shells {tuple(empty_shells)}:")
    for pos in empty_positions:
        p = ",".join(str(v) for v in pos)
        print(f"  {p}")
        for l in empty_shells:
            trial_projections.add(Projection(position_sym=p, orbital=l,
                                             spacegroup=spacegroup, rotate_basis=True))
            site_of.append(p)

    Emin_0, Emax_0, pdos, candidates = initial_DOS_energy_scan(
        calc=calc, out_dir=out_dir, in_dir=in_dir, seed=seed, dos_kwargs={"spin": 0, "npts": 1001, "width": 0.05}, gap_thres=gap_thres
    )
    froz_min = Emin_0
    if objective_wd is not None:
        froz_max = objective_wd[1]
        Emax_0 = max(objective_wd[1], Emax_0)
        Emin_0 = min(objective_wd[0], Emin_0)
    else:
        froz_max = calc.get_fermi_level() + 2.

    print(f"Initial energy windows: froz_min={froz_min}, froz_max={froz_max}, outer_min={Emin_0}, outer_max={Emax_0}")
    print(trial_projections.write_with_multiplicities(orbit=False))

    eps_all = np.array([calc.get_eigenvalues(kpt=k, spin=0)
                        for k in range(len(calc.get_k_point_weights()))])
    n_froz = int(max(((e >= froz_min) & (e <= froz_max)).sum() for e in eps_all))
    num_wann_max = int(ceil(margin * n_froz))
    print(f"Frozen manifold holds {n_froz} bands at the worst k "
          f"-> num_wann_max = {num_wann_max}")

    energies, dos = calc.get_dos(spin=0, npts=1001, width=0.05)

    nw_of_proj = [p.num_wann for p in trial_projections.projections]

    def nwann_of(c):
        return int(sum(int(v) * n for v, n in zip(np.asarray(c, int), nw_of_proj)))

    def _searcher(outer_max):
        return EBRsearcher(
            symmetrizer=symmetrizer,
            trial_projections_set=trial_projections,
            froz_min=froz_min,
            froz_max=froz_max,
            outer_min=Emin_0,
            outer_max=outer_max,
            debug=False,
        )

    outer_max_of = {}
    if per_size_window:
        print("Running EBRsearcher per size (window sized from K x nwann)...")
        combinations, seen = [], set()
        for nw in range(n_froz, num_wann_max + 1):
            emax_nw = find_emax_from_dos(energies=energies, dos_total=dos,
                                         n_wann=nw, emin=Emin_0, K=K)
            outer_max_of[nw] = emax_nw
            found = _searcher(emax_nw).find_combinations(num_wann_max=nw)
            new = 0
            for c in found:
                if nwann_of(c) != nw:
                    continue
                key = tuple(int(v) for v in np.asarray(c, int))
                if key in seen:
                    continue
                seen.add(key)
                combinations.append(c)
                new += 1
            print(f"  nwann={nw:3d}  outer_max={emax_nw:8.3f} eV  "
                  f"-> {new} combinations")
        if not outer_max_of:
            raise RuntimeError(
                f"the size sweep produced no windows: n_froz={n_froz} > "
                f"num_wann_max={num_wann_max}?")
        ebrsearcher = _searcher(max(outer_max_of.values()))
    else:
        print("Running EBRsearcher...")
        ebrsearcher = _searcher(Emax_0)
        combinations = ebrsearcher.find_combinations(num_wann_max=num_wann_max)
        for nw in range(n_froz, num_wann_max + 1):
            outer_max_of[nw] = Emax_0
    print(f"Found {len(combinations)} combinations")

    print("Computing the amn for the union of trial projections...")
    amn = AMN.from_bandstructure(bandstructure, trial_projections)
    A = _stack_payload(amn.data, 2)
    if band_gram:
        S, O = true_overlap(amn, bandstructure, with_band_gram=True)
    else:
        S, O = true_overlap(amn, bandstructure), None
        print("  WARNING band_gram=False: coverage is NOT bounded by 1 and the "
              "ranking is not trustworthy")
    S = np.asarray(S)
    if not np.isfinite(S).all():
        cols_bad = np.where(~np.isfinite(S).all(axis=(0, 1)))[0]
        raise RuntimeError(
            f"the trial-orbital overlap has NaN/Inf in columns "
            f"{cols_bad.tolist()}. Inspect the projections covering those "
            "columns -- most likely one of them is malformed.")
    if not np.isfinite(A).all():
        raise RuntimeError("the amn contains NaN/Inf; the projections are "
                           "malformed, not the selection")

    eps_kn = eig_from_bandstructure(amn, bandstructure)
    if eps_kn.shape != A.shape[:2]:
        raise RuntimeError(f"eig {eps_kn.shape} does not match amn {A.shape[:2]}")

    blocks, off = [], 0
    for j, p in enumerate(trial_projections.projections):
        blocks.append((j, slice(off, off + p.num_wann)))
        off += p.num_wann
    if off != A.shape[2]:
        raise RuntimeError(f"projections declare {off} columns, amn has {A.shape[2]}")
    if len(site_of) != len(trial_projections.projections):
        raise RuntimeError("site_of out of sync with trial_projections -- "
                           "the per-site hybridisation would regroup wrongly")

    wk_k = np.full(A.shape[0], 1.0 / A.shape[0])
    target = (eps_kn >= froz_min) & (eps_kn <= froz_max)

    def outer_mask_for(nw):
        top = outer_max_of.get(nw, max(outer_max_of.values()))
        return (eps_kn >= Emin_0) & (eps_kn <= top)

    outer = outer_mask_for(max(outer_max_of))

    kept = dedupe_combinations(S, blocks, combinations, trial_projections,
                               prefer=dedupe_prefer,
                               rank_first=dedupe_rank_first,
                               dedupe_by=dedupe_by, searcher=ebrsearcher)
    cand = sorted((np.asarray(c, int) for c, _ in kept),
                  key=lambda c: int(sum(ci * (sl.stop - sl.start)
                                        for ci, (j, sl) in zip(c, blocks))))
    if len(cand) > max_score:
        # `cand` is sorted by nwann, so the cut discards the LARGEST candidates
        # and the Pareto front cannot reach past the last size that survives.
        # On CaMg2Bi2 that capped the front at 17 WF while num_wann_max was 20,
        # which reads as the search failing rather than the scorer truncating.
        cut_at = nwann_of(cand[max_score])
        print(f"  WARNING scoring only the {max_score} smallest of {len(cand)} "
              f"distinct combinations: nothing at or above {cut_at} WF is "
              f"scored, so the Pareto front cannot extend past it "
              f"(num_wann_max={num_wann_max}). Raise max_score to see them.")
        cand = cand[:max_score]

    in_window = orbital_window_fraction(A, S, blocks, outer, O=O)
    by_size = {}
    for c in cand:
        by_size.setdefault(nwann_of(c), []).append(c)
    scored = []
    for nw in sorted(by_size):
        iw_nw = (in_window if not per_size_window
                 else orbital_window_fraction(A, S, blocks, outer_mask_for(nw),
                                              O=O))
        scored += score_combinations(A, S, blocks, by_size[nw], target, wk_k,
                                     O=O, in_window=iw_nw)

    labels = [str(p) for p in trial_projections.projections]
    ranked, info = rank_combinations(scored, p_min=p_min, min_gain=min_gain,
                                     use_in_window=use_in_window, labels=labels)
    selected_combination = ranked[0][0]

    # ---- span-preserving hybridisation, BEFORE validation -----------------
    def _pset_of(c):
        if with_hyb:
            # The hybrids were selectable in the search, so the chosen set is
            # already the one to ship -- but do NOT join it. join_same_wyckoff()
            # groups blocks sharing a site under one Projection and orders their
            # columns SITE-MAJOR, while WannierBerri builds D_wann
            # block-diagonal PER ORBITAL; on a multi-point orbit the two
            # layouts disagree and the blocks land on the wrong columns. See
            # the long note in hybridize_combination.
            #
            # The old 'as_is' path joined and got away with it only because
            # build_at's multi-orbital sets happened to land on 1-point orbits
            # (MnTe's Mn, SrTiO3's Ti), where the two orderings coincide.
            # Adding components puts several orbitals on SrTiO3's 3-point O
            # orbit, and then it does not.
            return trial_projections.get_combination(np.asarray(c, int))
        s = hybridize_combination(c, trial_projections, site_of, calc.atoms,
                                  spacegroup, shells_l,
                                  prefer_plain_shell=prefer_plain_shell,
                                  verbose=verbose)
        # compare against nwann_of(c), which is arithmetic on the sizes captured
        # when the alphabet was built -- not against a set re-derived from
        # trial_projections, which is exactly the thing a mutation would corrupt.
        expect = nwann_of(c)
        if s.num_wann != expect:
            raise RuntimeError(
                f"hybridisation changed num_wann {expect} -> {s.num_wann}"
                "; it must be span preserving")
        return s

    selected_proj_set = _pset_of(selected_combination)
    print("Selected projection set:")
    print(selected_proj_set.write_with_multiplicities(orbit=False))

    refined_emax = find_emax_from_dos(energies=energies, dos_total=dos,
                                      n_wann=selected_proj_set.num_wann,
                                      emin=Emin_0, K=K)
    froz_window = (froz_min, froz_max)
    outer_window = (Emin_0, refined_emax)
    log_orbitals(selected_proj_set,outer_window,froz_window,selected_proj_set.num_wann, seed, out_dir)
    write_selection_log(
        f"{out_dir}/{seed}/ebr_selection.log",
        trial_projections=trial_projections, blocks=blocks, scored=scored,
        front=info["front"], chosen=info["chosen"],
        froz_window=froz_window, outer_window=outer_window,
        selected_proj_set=selected_proj_set,
        A=A, S=S, O=O, target_mask=target, outer_mask=outer, wk_k=wk_k,
        n_froz=n_froz, num_wann_max=num_wann_max,
        n_raw=len(combinations), n_dedup=len(kept), n_scored=len(cand),
        describe_orbital=describe_orbital, steps=info["steps"],
        min_gain=min_gain, rel_gain=0.15, use_in_window=use_in_window)
    print(f"selection log written to {out_dir}/{seed}/ebr_selection.log")

    if not validate:
        print("  validation disabled (validate=False): returning the coverage "
              "pick without measuring eta.")
    if validate:
        print(f"  validating: accept the pick if eta <= {eta_ok} and max spread "
              f"<= {spread_ok}, otherwise sweep up to {validate_n_max} "
              "alternatives")
        from swan.ebr.ebr_select import shortlist_for_validation
        from swan.ebr.validate_candidates import validate_candidates
        from swan.wannier import wannierize, interpolate_bands
        from swan.utils.metrics import least_square_deviation_within_frozen

        short = shortlist_for_validation(scored, S, blocks, trial_projections,
                                         in_window=in_window,
                                         n_max=validate_n_max, labels=labels)
        picked = next((t for t in scored
                       if np.array_equal(t[0], selected_combination)), None)
        if picked is not None:
            short = [picked] + [t for t in short
                                if not np.array_equal(t[0], selected_combination)]

        def outer_for(nwann, K=K):
            # K defaults to the pipeline's; validate_candidates overrides it
            # per attempt when k_values is set.
            emax = find_emax_from_dos(energies=energies, dos_total=dos,
                                      n_wann=nwann, emin=Emin_0, K=K)
            return (Emin_0, emax)

        results = validate_candidates(
            short, blocks, trial_projections, calc, seed, out_dir, in_dir,
            froz_window, outer_window,
            wannierize_fn=wannierize, interpolate_fn=interpolate_bands,
            metric_fn=least_square_deviation_within_frozen,
            outer_window_fn=outer_for, metric_outer=outer_window,
            atoms=calc.atoms, eta_ok=eta_ok, spread_ok=spread_ok,
            k_values=k_values,
            # ALWAYS pass pset_fn. With pset_fn=None validate_candidates
            # rebuilds the set itself -- get_combination followed by
            # join_same_wyckoff -- so the set it wannierises is JOINED
            # regardless of what _pset_of returns. That is the third place the
            # join lives, and the one validation actually uses: it is why
            # 'isotypic' (which passed the lambda) worked while 'isotypic+hyb'
            # (which passed None) did not, on the same material and the same
            # alphabet family.
            pset_fn=(lambda t: _pset_of(t[0])),
            **(validate_kwargs or {}),unitary_params=dict(error_threshold=0.1, warning_threshold=0.01, nbands_upper_skip=n_skipped_bands),ecut_pw=ecut)
        best = min((r for r in results if np.isfinite(r["eta"])),
                   key=lambda r: r["eta"], default=None)
        K_best = (best or {}).get("K", K)
        cur_tag = combination_tag(selected_combination, blocks,
                                  trial_projections, calc.atoms)
        if best is not None and best["tag"].split("@K")[0] != cur_tag:
            print(f"  eta prefers {best['tag']} ({best['nwann']} WF, "
                  f"eta={best['eta']:.3f}) over {cur_tag} "
                  f"({selected_proj_set.num_wann} WF) -- switching")
            for t in short:
                if combination_tag(t[0], blocks, trial_projections,
                                   calc.atoms) == best["tag"].split("@K")[0]:
                    selected_combination = t[0]
                    break
            else:
                raise RuntimeError(
                    f"eta winner {best['tag']!r} is not in the shortlist -- "
                    "tags disagree between the sweep and the switch")
            selected_proj_set = _pset_of(selected_combination)
        if best is not None and K_best != K:
            print(f"  eta prefers K = {K_best:g} over {K:g}")
        if best is not None:
            # the window that was MEASURED is the window that is shipped
            refined_emax = find_emax_from_dos(
                energies=energies, dos_total=dos,
                n_wann=selected_proj_set.num_wann, emin=Emin_0, K=K_best)
            outer_window = (Emin_0, refined_emax)
            log_orbitals(selected_proj_set, outer_window, froz_window,
                         selected_proj_set.num_wann, seed, out_dir)

    return selected_proj_set, froz_window, outer_window

def parse_wp_strings(strings, max_den=1000):
    result = []
    for s in strings:
        exprs = []
        for part in (s if isinstance(s, (list, tuple)) else s.split(',')):
            if isinstance(part, (int, float)):
                expr = sympy.Rational(part).limit_denominator(max_den)
            else:
                expr = sympify(part, locals=_locals, rational=True)
                if expr.is_Rational:
                    expr = expr.limit_denominator(max_den)
            exprs.append(expr)
        result.append(exprs)
    return result

def wyckoff_contains(wpos, position, tol=1e-4):
    """Correct membership test: checks the FULL orbit, not just the first
    representative, when the Wyckoff position has no free variables."""
    if wpos.num_free_vars == 0:
        diffs = np.array(wpos.positions) - np.array(position)
        diffs -= np.round(diffs)
        return bool(np.any(np.all(np.abs(diffs) < tol, axis=1)))
    else:
        return wpos.contains_position(position) is not None

def log_orbitals(proj_set,outer_win,frozen_win,nwann, seed, out_dir):
    log_file_path = Path_(f"{out_dir}/{seed}/orbitals_and_windows.txt")
    with open(log_file_path, "w") as f:
        f.write(f"Outer window: {outer_win}\n")
        f.write(f"Frozen window: {frozen_win}\n")
        f.write(f"Number of Wannier functions: {nwann}\n")
    with open(log_file_path, "a") as f:
        f.write("Selected Projections (+ non-atomic-centered ones):\n")
        for p in proj_set.projections:
            f.write(str(p) + "\n")
            for orb in p.orbitals:
                f.write(describe_orbital(orb) + "\n")

if __name__ == "__main__":
    seed = "Si2"
    dir = "test_min_shells_full"
    calc = GPAW(f"{dir}/{seed}/{seed}-nscf-irred.gpw", txt=None)
    EBR_method(in_dir=f"{dir}",out_dir=f"{dir}",seed=f"{seed}",ecut=500,calc=calc,comm=None,only_on_site=True,verbose=True,alphabet='shells+hyb')