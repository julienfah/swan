from irrep.bandstructure import BandStructure
from irrep.spacegroup import SpaceGroup
import irrep.spacegroup as irrep_spacegroup
import sympy
from sympy import symbols, sympify
from wannierberri.symmetry.sawf import SymmetrizerSAWF
from wannierberri.symmetry.projections_searcher import EBRsearcher
from wannierberri.symmetry.projections import Projection, ProjectionsSet
from wannierberri.symmetry.wyckoff_position import WyckoffPosition
from wannierberri.w90files.amn import AMN
import numpy as np
from gpaw import GPAW
from math import ceil
from pathlib import Path as Path_
from pawan.utils import find_emax_from_dos
from pawan.auto_proj_and_windows import initial_DOS_energy_scan
from pawan.salc_M4 import build_at, describe_orbital
from pawan.amn_projectability import orbital_window_fraction, true_overlap
from pawan.candidate_scan import _stack_payload
from pawan.ebr_select import (dedupe_combinations, rank_combinations,
                              score_combinations)
from pawan.ebr_log import write_selection_log
from pawan.validate_candidates import combination_tag

x, y, z = symbols('x y z')
_locals = {'x': x, 'y': y, 'z': z}
"""
irrep PR:
sympy.sympify('2x')   # -> SympifyError
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
parse_expr('2x', local_dict={'x': sympy.Symbol('x')},
           transformations=standard_transformations + (implicit_multiplication_application,))
# -> 2*x, works fine

WB PR:
uncomment line in clear_cached_properties in ProjectionsSet to clear _free_vars and num_wann_per_site, which are cached properties that depend on the free variables of the Wyckoff position. This is necessary because the free variables can change when the Wyckoff position is modified, and we want to ensure that the cached properties are updated accordingly.

"""
import re
from sympy import sympify as _sympify_orig

def _sympify_implicit_mult(s, *args, **kwargs):
    if isinstance(s, str):
        s = re.sub(r'(?<=[\d)])(?=[a-zA-Z])', '*', s)  # "2x" -> "2*x", "2(x+y)" -> "2*(x+y)"
    return _sympify_orig(s, *args, **kwargs)
irrep_spacegroup.sympify = _sympify_implicit_mult


def eig_from_bandstructure(amn, bandstructure):
    """(nk, nb) eigenvalues in the SAME order _stack_payload stacks amn.data.

    Taken from the bandstructure rather than from calc.get_eigenvalues: the amn
    lives on the irreducible k-set of `bandstructure`, whose ordering need not
    match GPAW's, and pairing projectabilities with the wrong eigenvalues would
    misplace every band silently. Uses the positional mapping, which is the one
    true_overlap verifies against amn.data.
    """
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
                f"no energy attribute on {type(kp)}; tried Energy/energies/E. "
                f"Available: {[a for a in dir(kp) if 'ner' in a.lower()]}")
    return np.array(out)


def EBR_method(in_dir, out_dir, seed, ecut, only_on_site=True, calc=None,
               verbose=False, comm=None, K=1.2, p_min=None, margin=2,
               max_score=2000, min_gain=0.02, use_in_window=True, gap_thres=1,
               objective_wd=None, include_empty=False, empty_shells=('s',),
               dedupe_prefer='hybrid', dedupe_rank_first=True,
               dedupe_by='span', band_gram=True,
               validate=False, validate_n_max=5, validate_kwargs=None,
               eta_ok=20.0, spread_ok=10.0,hybrids=True):
    """Uses an EBRsearcher to find symmetry adapted projections for a given system"""
    if calc is None:
        calc = GPAW(f"{in_dir}/{seed}/{seed}-nscf-irred.gpw", txt=None, communicator=comm)
    print("Building bandstructure...")
    bandstructure = BandStructure(calculator_gpaw=calc, code="gpaw",
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
    #select positions to consider for projections
    cell = calc.atoms.cell
    atomic_positions = calc.atoms.get_scaled_positions()
    numbers = calc.atoms.numbers
    lattice = (cell, atomic_positions, numbers)
    wps = SpaceGroup.wyckoff_positions(lattice)
    print(wps)
    WP =parse_wp_strings(wps)
    selected_positions = []

    if only_on_site:
        # One projection per symmetry ORBIT, not per atom and not per Wyckoff
        # letter. A Projection built with position_num generates the whole orbit
        # from the space group, so symmetry-equivalent atoms need a single
        # entry; two INEQUIVALENT atoms sharing a letter with different free
        # parameters are different orbits and each need one.
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
                    # PIN the free parameter to this atom. The atom is a point
                    # of the orbit, so the site group at its coordinates is
                    # exactly the Wyckoff position's and the SALCs are right.
                    # Symbolic positions cannot be handed to build_at -- (x,0,1/2)
                    # has no single site group until x is chosen -- so without
                    # this those sites get plain shells and no hybrids.
                    # Side effect: num_free_vars becomes 0, hence
                    # proj_max_multiplicity 1 and nothing for maximize_distance
                    # to move, which is what you want for an on-site projection.
                    pos = [float(v) for v in atomic_pos]
                    print(f"  {calc.atoms[atom_idx].symbol}{atom_idx}: Wyckoff "
                          f"position has {wpos.num_free_vars} free parameter(s)"
                          f" -> pinned to the atom at "
                          f"{np.round(atomic_pos, 6).tolist()}")
                selected_positions.append(list(pos))
        # safety: two orbits can still land on the same fixed position
        seen, uniq = set(), []
        for p in selected_positions:
            k = ",".join(str(v) for v in p)
            if k not in seen:
                seen.add(k)
                uniq.append(p)
        selected_positions = uniq
    else:
        selected_positions = WP

    # EMPTY Wyckoff positions. The frozen manifold can need an irrep that no
    # ATOMIC orbital supplies well: BaTiO3 needs one A1g beyond Ti t2g + O p,
    # and every atomic candidate for it is unreachable (Ti s at 16.6%
    # in-window, Ba s at 10.1%). An interstitial s at an empty site can carry
    # it with a far higher in-window fraction -- the obstructed-atomic-limit
    # case, and the reason EBRsearcher supports empty positions at all.
    #
    # FIXED positions only: one with a free parameter has no single site group
    # until x is chosen, so build_at cannot act on it and EBRsearcher would have
    # to optimise x afterwards. `empty_shells` defaults to s alone, to keep the
    # alphabet -- and hence the combination count -- small.
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
                continue                      # coincides with an atom
            empty_positions.append(pos)

    selected_positions_str = [",".join(str(y) for y in x) for x in selected_positions]

    print("Selected positions for projections:")
    for p in selected_positions_str:
        print(p)
    # select orbitals for each position
    for p in selected_positions_str:
        if hybrids:
            pset,_ = build_at(atoms=calc.atoms,position=p.split(','),shells=['s','p','d'],spacegroup=spacegroup,label=f"WP_{p}_",verbose=True,fallback="minimal_shell")
            """for o in ['d','pz', 'sp2']:
                proj = Projection(position_sym=p, orbital=o, spacegroup=spacegroup)
                trial_projections.add(proj)"""
            for proj in pset.projections:
                trial_projections.add(proj)
        for l in ['s', 'p', 'd']:
            proj = Projection(position_sym=p, orbital=l, spacegroup=spacegroup)
            trial_projections.add(proj)

    if empty_positions:
        print(f"Empty Wyckoff positions added, shells {tuple(empty_shells)}:")
    for pos in empty_positions:
        p = ",".join(str(v) for v in pos)
        print(f"  {p}")
        for l in empty_shells:
            trial_projections.add(Projection(position_sym=p, orbital=l,
                                             spacegroup=spacegroup))
    #initial selection windows

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
    print("Running EBRsearcher...")
    ebrsearcher = EBRsearcher(
        symmetrizer=symmetrizer,
        trial_projections_set=trial_projections,
        froz_min=froz_min,
        froz_max=froz_max,
        outer_min=Emin_0,
        outer_max=Emax_0,
        debug=False # set to True to see more printed information
    )

    # Cap nwann by the FROZEN MANIFOLD, not by n_bands. n_bands never binds, so
    # the search enumerates every subset the irreps allow: BaTiO3 gave 879224.
    # You never want more Wannier functions than ~margin x the bands that must
    # be reproduced, and the subset count falls steeply with the cap
    # (21 projections: 762k subsets at 40, 38k at 20).
    eps_all = np.array([calc.get_eigenvalues(kpt=k, spin=0)
                        for k in range(len(calc.get_k_point_weights()))])
    n_froz = int(max(((e >= froz_min) & (e <= froz_max)).sum() for e in eps_all))
    num_wann_max = int(ceil(margin * n_froz))
    print(f"Frozen manifold holds {n_froz} bands at the worst k "
          f"-> num_wann_max = {num_wann_max}")
    combinations = ebrsearcher.find_combinations(num_wann_max=num_wann_max)
    print(f"Found {len(combinations)} combinations")

    if verbose and len(combinations) <= 50:
        print(f"Found {len(combinations)} combinations of projections:")
        for c in combinations:
            print(("+" * 80 + "\n") * 2)
            print(trial_projections.write_with_multiplicities(c))
            newset = trial_projections.get_combination(c)
            newset.join_same_wyckoff()
            newset.maximize_distance()
            print(newset.write_wannier90(mod1=True))

    # --- choose among the combinations instead of taking combinations[0].
    # One amn for the UNION of the trial projections, then every combination is
    # a column subset: symmetry says which sets are admissible, projectability
    # says which are buildable.
    print("Computing the amn for the union of trial projections...")
    amn = AMN.from_bandstructure(bandstructure, trial_projections)
    A = _stack_payload(amn.data, 2)
    # with_band_gram: AMN.from_bandstructure normalises each pseudo band's PW
    # norm to 1, which does not restore orthogonality (the PAW augmentation is
    # not diagonal), so p is unbounded -- hence "the band set reaches 105.7% of
    # the trial-orbital norm". O Loewdin-orthonormalises the bands and restores
    # the bound; without it every coverage in the ranking is inflated.
    if band_gram:
        S, O = true_overlap(amn, bandstructure, with_band_gram=True)
    else:
        # DIAGNOSTIC ONLY. Without O, p_nk is not bounded by 1 -- MnTe reported
        # "the band set reaches 105.7% of the trial-orbital norm" -- and the
        # inflation is band-dependent, not a uniform rescale, so it can reorder
        # combinations rather than just shifting them. Use it to reproduce a
        # pre-correction run, not to select.
        S, O = true_overlap(amn, bandstructure), None
        print("  WARNING band_gram=False: coverage is NOT bounded by 1 and the "
              "ranking is not trustworthy")
    S = np.asarray(S)
    if not np.isfinite(S).all():
        bad = sorted({j for j, p in enumerate(trial_projections.projections)
                      for k in range(S.shape[0])
                      if not np.isfinite(S[k]).all()})
        cols_bad = np.where(~np.isfinite(S).all(axis=(0, 1)))[0]
        raise RuntimeError(
            f"the trial-orbital overlap has NaN/Inf in columns "
            f"{cols_bad.tolist()}. Inspect the projections covering those "
            "columns -- most likely one of them is malformed (a pinned "
            "free-parameter position, or a spread_factor that underflows). "
            "The selection cannot proceed on a non-finite overlap.")
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

    wk_k = np.full(A.shape[0], 1.0 / A.shape[0])
    target = (eps_kn >= froz_min) & (eps_kn <= froz_max)
    outer = (eps_kn >= Emin_0) & (eps_kn <= Emax_0)

    kept = dedupe_combinations(S, blocks, combinations, trial_projections,
                               prefer=dedupe_prefer,
                               rank_first=dedupe_rank_first,
                               dedupe_by=dedupe_by, searcher=ebrsearcher)
    # score the smallest first and stop: each score is a dense solve per k, so
    # tens of thousands is not affordable, and the answer is always among the
    # compact sets anyway
    cand = sorted((np.asarray(c, int) for c, _ in kept),
                  key=lambda c: int(sum(ci * (sl.stop - sl.start)
                                        for ci, (j, sl) in zip(c, blocks))))
    if len(cand) > max_score:
        print(f"  scoring the {max_score} smallest of {len(cand)} distinct "
              "combinations")
        cand = cand[:max_score]
    in_window = orbital_window_fraction(A, S, blocks, outer, O=O)
    scored = score_combinations(A, S, blocks, cand, target, wk_k, O=O,
                                in_window=in_window)
    labels = [str(p) for p in trial_projections.projections]
    ranked, info = rank_combinations(scored, p_min=p_min, min_gain=min_gain,
                                     use_in_window=use_in_window, labels=labels)
    selected_combination = ranked[0][0]
    selected_proj_set = trial_projections.get_combination(selected_combination)
    selected_proj_set.join_same_wyckoff()
    #selected_proj_set.maximize_distance()
    print("Selected projection set:")
    print(selected_proj_set.write_with_multiplicities(orbit=False))
    #refined_emax = emax_from_band_count(emin=Emin_0,nwann=selected_proj_set.num_wann)
    energies,dos =calc.get_dos(spin=0,npts=1001,width=0.05)
    refined_emax = find_emax_from_dos(energies=energies,dos_total=dos, n_wann=selected_proj_set.num_wann, emin=Emin_0,K=K)
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

    # OPTIONAL: settle the top few by actually wannierising them. Coverage
    # ranks by span; eta_2 measures interpolation, which depends on
    # localisation -- on MnTe the set with 0.006 LOWER coverage gave half the
    # band distance. Off by default: this costs ~5 wannierisations and is a
    # calibration step, not something to run in production.
    if not validate:
        print("  validation disabled (validate=False): returning the coverage "
              "pick without measuring eta. Pass validate=True to wannierise it "
              f"and fall back to the shortlist if eta > {eta_ok} meV.")
    if validate:
        print(f"  validating: accept the pick if eta <= {eta_ok} and max spread "
              f"<= {spread_ok}, otherwise sweep up to {validate_n_max} "
              "alternatives")
        from pawan.ebr_select import shortlist_for_validation
        from pawan.validate_candidates import validate_candidates
        from pawan.wannier import wannierize, interpolate_bands
        from pawan.metrics import least_square_deviation_within_frozen

        short = shortlist_for_validation(scored, S, blocks, trial_projections,
                                         in_window=in_window,
                                         n_max=validate_n_max, labels=labels)
        # The pipeline's own pick goes FIRST and is evaluated whether or not it
        # survived the shortlist filters: if it is already good the sweep stops
        # after one wannierisation, and if it is not, its eta is the baseline
        # everything else is compared against.
        picked = next((t for t in scored
                       if np.array_equal(t[0], selected_combination)), None)
        if picked is not None:
            short = [picked] + [t for t in short
                                if not np.array_equal(t[0], selected_combination)]

        def outer_for(nwann):
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
            **(validate_kwargs or {}),unitary_params=dict(error_threshold=0.1, warning_threshold=0.01, nbands_upper_skip=n_skipped_bands))
        best = next((r for r in results if np.isfinite(r["eta"])), None)
        cur_tag = combination_tag(selected_combination, blocks,
                                  trial_projections, calc.atoms)
        if best is not None and best["tag"] != cur_tag:
            print(f"  eta prefers {best['tag']} ({best['nwann']} WF, "
                  f"eta={best['eta']:.3f}) over {cur_tag} "
                  f"({selected_proj_set.num_wann} WF) -- switching")
            for t in short:
                if combination_tag(t[0], blocks, trial_projections,
                                   calc.atoms) == best["tag"]:
                    selected_combination = t[0]
                    break
            else:
                raise RuntimeError(
                    f"eta winner {best['tag']!r} is not in the shortlist -- "
                    "tags disagree between the sweep and the switch")
            selected_proj_set = trial_projections.get_combination(selected_combination)
            selected_proj_set.join_same_wyckoff()
            refined_emax = find_emax_from_dos(
                energies=energies, dos_total=dos,
                n_wann=selected_proj_set.num_wann, emin=Emin_0, K=K)
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
    """Correct membership test: checks the FULL orbit, not just the
    first representative, when the Wyckoff position has no free variables."""
    if wpos.num_free_vars == 0:
        diffs = np.array(wpos.positions) - np.array(position)
        diffs -= np.round(diffs)   # wrap to nearest periodic image
        return bool(np.any(np.all(np.abs(diffs) < tol, axis=1)))
    else:
        return wpos.contains_position(position) is not None
def log_orbitals(proj_set,outer_win,frozen_win,nwann, seed, out_dir):
# log the selected orbitals and the windows
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
#test
if __name__ == "__main__":
    seed = "Si2"
    dir = "test_min_shells_full"
    calc = GPAW(f"{dir}/{seed}/{seed}-nscf-irred.gpw", txt=None)
    EBR_method(in_dir=f"{dir}",out_dir=f"{dir}",seed=f"{seed}",ecut=500,calc=calc,comm=None,only_on_site=True,verbose=True)