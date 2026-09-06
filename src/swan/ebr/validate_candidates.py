"""Wannierise a shortlist of candidates and rank them by band distance.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from gpaw.mpi import serial_comm

import numpy as np

__all__ = ["dft_reference_bands", "validate_candidates",
           "combination_tag"]


def _max_spread(path):
    """Largest Wannier spread from the file wannierize() writes.

    A set can reach an acceptable band distance while leaving one function
    delocalised; eta averages that away, the spread does not.
    """
    vals = []
    for line in Path(path).read_text().splitlines():
        if "Spread:" in line:
            try:
                vals.append(float(line.split("Spread:")[1].strip().rstrip(",")))
            except ValueError:
                pass
    return max(vals) if vals else None


def _orb_short(name):
    """'WP_0,0,0__rest0' -> 'r0';  'WP_1/2,0,0__hyb' -> 'hyb';  's' -> 's'."""
    n = str(name)
    if "__" in n:
        n = n.split("__", 1)[1]
    return n.replace("rest", "r")


def _site_label(proj, atoms=None, symprec=1e-3):
    """Element symbol for a projection's position, else the position string."""
    pos = None
    try:
        pos = np.asarray(proj.positions[0], dtype=float)
    except Exception:                                           # noqa: BLE001
        head = str(proj).splitlines()[0]
        try:
            pos = np.array([float(x) for x in
                            head.replace("Projection ", "").split(":")[0].split(",")])
        except Exception:                                       # noqa: BLE001
            return head.replace("Projection ", "").split(":")[0].replace(" ", "")
    if atoms is not None:
        sp = atoms.get_scaled_positions()
        d = sp - pos
        d -= np.round(d)
        hit = np.where(np.all(np.abs(d) < symprec, axis=1))[0]
        if hit.size:
            return atoms[int(hit[0])].symbol
    return ",".join(f"{x:g}" for x in pos)


def combination_tag(c, blocks, trial_projections, atoms=None):
    """Readable, run-stable identifier, e.g. 'Mn-hyb+r0+r1|Te-p'.

    Index-based tags are NOT stable: the alphabet order follows the iteration
    order of `selected_positions`, and MnTe silently swapped Mn and Te between
    two runs, so '0-1-2-10' named different projections each time and cached
    results were incomparable. Derived from site and orbital instead, and sorted
    so the same set always produces the same string.
    """
    per_site = {}
    for (j, sl), v in zip(blocks, np.asarray(c, int)):
        if v <= 0:
            continue
        p = trial_projections.projections[j]
        site = _site_label(p, atoms)
        orbs = [_orb_short(o) for o in getattr(p, "orbitals", [])] or ["?"]
        per_site.setdefault(site, []).extend(orbs)
    return "|".join(f"{s}-{'+'.join(sorted(set(o)))}"
                    for s, o in sorted(per_site.items()))


def dft_reference_bands(in_dir,seed,calc=None, bands_gpw=None, npoints=200, cache=None,
                        spin=0, comm=None, txt=None):
    """(energies (1, nk, nbands), kpts) for the DFT reference path.

    bands_gpw : path to an existing band-structure .gpw -- the normal case, and
        much the cheaper one. The k-points come from that calculation, so the
        path is whatever you actually computed rather than whatever
        cell.bandpath() would guess; the two need not agree, and the metric
        aligns on k-points, so taking them from the file avoids a silent
        mismatch.

    calc : fall back to a fixed_density run on the ASE band path if no file is
        given.

    Cached to `cache` (npz): every candidate for this material compares against
    the same curve, so this is the only DFT work the whole sweep needs.
    """
    if cache is not None and Path(cache).exists():
        z = np.load(cache)
        return z["energies"], z["kpts"]
    else:
        from gpaw import GPAW
        if txt is not None:
            kw = {"txt": txt}
        else:
            kw = {}
        if comm is not None:
            kw["communicator"] = comm
        bs = GPAW(f"{in_dir}/{seed}/{seed}-bands.gpw", **kw).band_structure()
        e = np.asarray(bs.energies)          # (nspin, nk, nbands)
        kpts = np.asarray(bs.path.kpts)
    """elif calc is not None:
        print("DFT reference bands not found, computing new ones...")
        path = calc.atoms.cell.bandpath(npoints=npoints)
        bs = calc.fixed_density(kpts=path.kpts, symmetry="off",
                                txt=txt).band_structure()
        e = np.asarray(bs.energies)
        kpts = np.asarray(bs.path.kpts)
    else:
        raise ValueError("pass bands_gpw or calc")"""
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, energies=e, kpts=kpts)
    return e, kpts


def validate_candidates(shortlist, blocks, trial_projections, calc, seed,
                        out_dir, in_dir, froz_window, outer_window,
                        wannierize_fn, interpolate_fn, metric_fn,
                        outer_window_fn=None, metric_outer=None, atoms=None,
                        pset_fn=None, k_values=None,
                        stop_when_good=True, eta_ok=20.0, spread_ok=10.0,
                        dft_energies=None, dft_kpts=None, bands_gpw=None,
                        npoints=200, spin_channel=0, cache=True, verbose=True,
                        **wkw):
    """Wannierise each candidate, score it, return them sorted best-first.

    metric_fn(dft_energies, dft_kpts, wann_energies, wann_kpts,
              outer_win, froz_window) -> float

    outer_window_fn : callable(nwann) -> (lo, hi), used for the
        WANNIERISATION of each candidate. The outer window is sized from
        K * nwann, so an 18-WF and a 24-WF candidate need different ones and
        sharing a single window would starve the larger set of bands. Defaults
        to the fixed `outer_window`.

    pset_fn : callable(t) -> ProjectionsSet, the set actually wannierised for
        shortlist entry `t`. Defaults to
        trial_projections.get_combination(t[0]) followed by join_same_wyckoff,
        i.e. the raw trial projections, so passing nothing reproduces the old
        behaviour exactly.

        It exists so the set MEASURED is the set SHIPPED. The span fixes the
        disentangled subspace and Omega_I, but not Omega_D + Omega_OD, and
        maximal localisation is non-convex -- so a hybridised initial guess and
        a plain one can land in different minima with different spreads, and
        eta, an interpolation measure, inherits that. Validating the
        un-hybridised set would score something other than what is returned.

        nwann is taken from the returned set rather than from t[2], since a
        variant may legitimately change it.

    k_values : None (default) -> unchanged behaviour, one wannierisation per
        candidate with the window outer_window_fn(nwann) gives.

        A sequence, e.g. (1.2, 1.5), sweeps the outer-window size factor K per
        candidate: outer_window_fn is called as outer_window_fn(nwann, K=k) and
        each K is wannierised, tagged and recorded separately. The sweep for a
        candidate STOPS at the first K that is good enough (eta <= eta_ok and
        max spread <= spread_ok), so the extra cost is paid only where the
        first K does not work. They must be BIGGER than the value past to 
        build the EBRsearcher.

        K sets how many bands the disentanglement gets per Wannier function.
        Too small starves it; too large drags in states that do not belong to
        the manifold. Which is right is material dependent and there is no
        criterion for it here, so the honest thing is to measure both and let
        eta choose -- every attempt lands in the results, not just the winner.

    metric_outer : the window the METRIC uses, the same for every candidate.
        It must be fixed: it selects which DFT bands enter the comparison, and
        varying it per candidate would score them over different band sets.
        Defaults to `outer_window`.

    stop_when_good : evaluate shortlist[0] -- which the caller should make the
        pipeline's own pick -- and RETURN IMMEDIATELY if it is good enough. The
        sweep then costs one wannierisation on the materials where the
        selection already works, and only pays for the rest when it does not.
        It also removes the need for the pick to survive the shortlist filters:
        it is evaluated whether or not it would have passed them.

    eta_ok : band distance below which the first candidate is accepted, in the
        units metric_fn returns (meV for least_square_deviation_within_frozen,
        which uses scale=1000 with root=True). MnTe's good sets sat at 17-50
        and its bad ones at 91+, so 20 is a tight pass.
    spread_ok : also require max Wannier spread below this (Angstrom^2). A set
        can interpolate acceptably with one delocalised function, and that
        shows up here rather than in eta.

    bands_gpw : the DFT band-structure .gpw. Defaults to
        {in_dir}/{seed}/{seed}-bands.gpw if that exists. Read through
        .band_structure(), which already returns energies as
        (nspin, nk, nbands) and the path k-points -- the shape the metric
        expects.

    Each candidate gets its OWN output directory: wannierize() writes fixed
    filenames under {out_dir}/{seed}, so candidates run in the same directory
    would overwrite each other's chk and spreads.

    Results are cached per candidate in results.json, so an interrupted sweep
    resumes and a rerun after changing only the selection costs nothing.

    A candidate that fails to converge is recorded with eta=inf rather than
    aborting the sweep -- failure to wannierise IS the measurement for that set.
    """
    root = Path(out_dir) / seed / "candidates"
    root.mkdir(parents=True, exist_ok=True)
    store = root / "results.json"
    done = json.loads(store.read_text()) if (cache and store.exists()) else {}

    if dft_energies is None:
        if bands_gpw is None:
            guess = Path(in_dir) / seed / f"{seed}-bands.gpw"
            bands_gpw = guess if guess.exists() else None
        if verbose:
            print(f"  DFT reference: {bands_gpw or 'fixed_density run'} "
                  "(computed once, shared by every candidate)")
        dft_energies, dft_kpts = dft_reference_bands(in_dir, seed,
            calc=calc, bands_gpw=bands_gpw, npoints=npoints,
            spin=spin_channel, cache=root / "dft_bands.npz",comm=serial_comm)
    print("got DFT reference bands, validating candidates...")

    def _good(rec):
        sp = rec.get("max_spread")
        return (rec["error"] is None and rec["eta"] <= eta_ok
                and (sp is None or sp <= spread_ok))

    ks = list(k_values) if k_values else [None]
    out = []
    for i_cand, t in enumerate(shortlist):
      c, cov, nwann0 = t[0], t[1], t[2]
      base_tag = combination_tag(c, blocks, trial_projections, atoms)
      accepted = False
      for k in ks:
        nwann = nwann0
        # the K goes in the tag, so each attempt caches and writes separately
        tag = base_tag + ("" if k is None else f"@K{k:g}")
        if tag in done:
            rec = done[tag]
            if verbose:
                print(f"  [{tag}] cached: eta = {rec['eta']}")
            out.append(rec)
            if _good(rec):
                accepted = True
                break
            continue

        d = root / tag
        d.mkdir(exist_ok=True)
        rec = dict(tag=tag, nwann=int(nwann), coverage=float(cov),
                   eta=float("inf"), max_spread=None, error=None)
        if k is not None:
            rec["K"] = float(k)
        try:
            if pset_fn is None:
                pset = trial_projections.get_combination(np.asarray(c, int))
                pset.join_same_wyckoff()
            else:
                # the set that gets measured is the set that gets shipped
                pset = pset_fn(t)
                nwann = int(pset.num_wann)
                rec["nwann"] = nwann
            if verbose:
                print(f"  [{tag}] wannierising {nwann} WF "
                      f"(coverage {cov:.4f})"
                      + ("" if k is None else f", K={k:g}") + f" -> {d}")
            if outer_window_fn is None:
                ow = outer_window
            elif k is None:
                ow = outer_window_fn(int(nwann))
            else:
                ow = outer_window_fn(int(nwann), K=k)
            rec["outer_window"] = [float(ow[0]), float(ow[1])]
            try:
                wannierize_fn(proj_set=pset, outer_win=ow,
                            frozen_win=froz_window, seed=seed,
                            out_dir=str(root.parent.parent), in_dir=in_dir,
                            calc_nscf_irred=calc, spin_channel=spin_channel, recompute_files=False,
                            **wkw)
            except AssertionError as e:
                print(f"  [{tag}] wannierize() failed with {type(e).__name__}: {e}")
                print("  retrying with recompute_files=True")
                wannierize_fn(proj_set=pset, outer_win=ow,
                                            frozen_win=froz_window, seed=seed,
                                            out_dir=str(root.parent.parent), in_dir=in_dir,
                                            calc_nscf_irred=calc, spin_channel=spin_channel, recompute_files=True,
                                            **wkw)
            bands, wb_path = interpolate_fn(
                seed=seed, out_dir=str(root.parent.parent), in_dir=in_dir,
                calc_nscf_irred=calc, npoints=npoints)
            # WannierBerri returns the interpolated eigenvalues as
            # bands.Enk.data, shape (nk, nwann); the k-points come from
            # wb_path.get_kpoints(). Reaching for `.energies` or `.K_list`
            # yields the wrapper object and a 1-D array, and the metric then
            # dies on y_pred.shape[1].
            energies = np.asarray(bands.Enk.data)
            kpts_w = np.asarray(wb_path.get_kpoints())
            if energies.ndim != 2:
                raise RuntimeError(
                    f"interpolated energies have shape {energies.shape}, "
                    "expected (nk, nwann) -- check bands.Enk.data")
            rec["eta"] = float(metric_fn(dft_energies, dft_kpts,
                                         energies, kpts_w,
                                         metric_outer or outer_window,
                                         froz_window))
            d = Path(root.parent.parent)
            spreads = Path(root.parent.parent) / seed / f"{seed}_wannier_spreads.txt"
            if spreads.exists():
                (d / "spreads.txt").write_text(spreads.read_text())
                rec["max_spread"] = _max_spread(spreads)
        except Exception as e:                                  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
            if verbose:
                print(f"  [{tag}] FAILED: {rec['error']}")
                traceback.print_exc()
        done[tag] = rec
        if cache:
            store.write_text(json.dumps(done, indent=2))
        out.append(rec)

        sp = rec.get("max_spread")
        if _good(rec):
            accepted = True
            if verbose:
                print(f"  [{tag}] eta = {rec['eta']:.2f} <= {eta_ok}"
                      + (f", max spread {sp:.2f} <= {spread_ok}"
                         if sp is not None else "")
                      + " -- good enough"
                      + ("" if k is None or k == ks[-1]
                         else f"; not trying K > {k:g}"))
            break
        if verbose and rec["error"] is None:
            print(f"  [{tag}] eta = {rec['eta']:.2f}"
                  + (f", max spread {sp:.2f}" if sp is not None else "")
                  + f" -- not good enough (eta_ok={eta_ok}, "
                    f"spread_ok={spread_ok})")
      # end of the K sweep for this candidate
      if stop_when_good and i_cand == 0 and accepted:
            if verbose:
                print("  first candidate accepted, skipping the rest of the "
                      "sweep")
            return out

    out.sort(key=lambda r: r["eta"])
    if verbose:
        print(f"\n  {'eta':>10} {'nwann':>6} {'coverage':>9} {'spread':>8}"
              "   candidate")
        for r in out:
            note = "" if r["error"] is None else f"   ({r['error'][:40]})"
            sp = r.get("max_spread")
            print(f"  {r['eta']:>10.3f} {r['nwann']:>6} {r['coverage']:>9.4f}"
                  f" {('%8.2f' % sp) if sp is not None else '       -'}"
                  f"   {r['tag']}{note}")
        finite = [r for r in out if np.isfinite(r["eta"])]
        if len(finite) > 1:
            best, second = finite[0], finite[1]
            if best["coverage"] < second["coverage"]:
                print(f"\n  NOTE the winner has LOWER coverage than the runner-up"
                      f" ({best['coverage']:.4f} vs {second['coverage']:.4f}). "
                      "That is the MnTe situation: coverage measures span, eta "
                      "measures interpolation, and interpolation depends on "
                      "localisation, which coverage cannot see.")
    return out
