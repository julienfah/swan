"""Choosing among EBRsearcher combinations.

EBRsearcher returns every EBR combination whose irreps cover the frozen window
and fit inside the outer window. That is a symmetry filter: it says which sets
are ADMISSIBLE, not which is buildable. Three stages narrow it down, cheapest
first.

1. DEDUPLICATE BY IRREP CONTENT. Two combinations with the same irrep vector at
   every k span the same space -- they differ only in gauge. At a T_d site sp3
   spans A1 + T2, so [sp3 @ 8a] and [s @ 8a, p @ 8a] are the same solution
   written twice. Adding plain s, p, d to the alphabet multiplies these
   duplicates, which is why Si2 appears to sprout extra combinations when you
   widen the alphabet: they are not new physics. Among equivalents keep the
   hybrid, since the trial projection doubles as the initial guess and a
   bond-pointing lobe is a better starting gauge than separate s and p.

   Adding plain shells is still worth doing: a composite locks its irreps in a
   fixed ratio, so sp3 cannot express a manifold needing 2 x A1 + 1 x T2 while
   separate s and p can. You gain expressiveness, and dedup removes the cost.

2. SCORE BY COVERAGE. One amn for the union alphabet, then each combination is a
   column subset -- a small dense solve, microseconds. Coverage of the frozen
   manifold says whether the admissible set is actually BUILDABLE: symmetry
   permits an orbital, projectability says whether any localised function there
   overlaps the bands.

3. BAND DISTANCE on the survivors. The only ground truth, and now applied to a
   handful of candidates instead of all of them.
"""

from __future__ import annotations

import numpy as np

from .amn_projectability import subset_projectability

__all__ = ["combination_irreps", "span_projector", "group_by_span",
           "_full_rank",
           "dedupe_combinations",
           "score_combinations",
           "rank_combinations",
    "shortlist_for_validation", "combination_condition"]


def combination_irreps(searcher, c):
    """Hashable signature of a combination's irrep content, over all k-blocks.

    Two combinations with equal signatures cover exactly the same irreps at
    every k, so they span the same space.
    """
    vecs = getattr(searcher, "irreps_per_projection_vectors", None)
    if vecs is None:
        raise AttributeError(
            "searcher has no irreps_per_projection_vectors; check the "
            "WannierBerri version and adjust the attribute name")
    c = np.asarray(c, int)
    return tuple(tuple(int(v) for v in (c @ np.asarray(block)))
                 for block in vecs)


def _n_hybrid(trial_set, c):
    """How many chosen projections are registered hybrids (heuristic tie-break).

    A hybrid orbital name is not a plain shell, so anything not in 'spdf' or a
    single-orbital label counts as a composite/SALC type.
    """
    plain = {"s", "p", "d", "f"}
    n = 0
    for cj, proj in zip(np.asarray(c, int), trial_set.projections):
        if cj <= 0:
            continue
        for o in getattr(proj, "orbitals", []):
            if str(o) not in plain:
                n += int(cj)
                break
    return n


def _pinv_herm(A, rcond=1e-10):
    """Pseudo-inverse of a Hermitian PSD matrix, via eigh.

    np.linalg.pinv uses gesdd, which raises "SVD did not converge" on
    non-finite input and is the wrong routine for a Gram matrix anyway. eigh is
    the right algorithm here and is far more robust.
    """
    A = 0.5 * (np.asarray(A) + np.asarray(A).conj().T)
    w, U = np.linalg.eigh(A)
    keep = w > rcond * max(float(np.abs(w).max()), 1e-30)
    winv = np.zeros_like(w)
    winv[keep] = 1.0 / w[keep]
    return (U * winv) @ U.conj().T


def span_projector(S, blocks, c, rcond=1e-10, trial_set=None):
    """The projector onto span{g_c}, in the full trial-orbital basis.

        M_c = S[:, c] @ pinv(S[c, c]) @ S[c, :]

    M_c[i, j] = <g_i|P|g_j>, so two column sets span the same subspace iff their
    M agree. Note trace(M) is sum_i <g_i|P|g_i> over ALL trial functions, which
    is NOT the dimension of the span unless they are orthonormal -- in this
    alphabet d is represented twice, so it is counted twice. It is still a valid
    invariant (equal spans give equal traces) and is used only as a cheap
    pre-filter.
    """
    S0 = np.asarray(S)[0]
    cols = np.concatenate([np.arange(sl.start, sl.stop)
                           for (j, sl), cj in zip(blocks, np.asarray(c, int))
                           if cj > 0]) if np.any(c) else np.zeros(0, int)
    if cols.size == 0:
        return np.zeros_like(S0), 0.0
    Scc = S0[np.ix_(cols, cols)]
    if not np.isfinite(Scc).all():
        bad = [j for (j, sl), cj in zip(blocks, np.asarray(c, int)) if cj > 0
               and not np.isfinite(S0[sl, sl]).all()]
        names = ([str(trial_set.projections[j]).splitlines()[0] for j in bad]
                 if trial_set is not None else bad)
        raise ValueError(
            "the trial-orbital overlap S contains NaN/Inf for projection(s) "
            f"{names}. That is a problem in the overlap itself, not in the "
            "selection -- check true_overlap and the projections at that site "
            "(a pinned free-parameter Wyckoff position with a degenerate "
            "coordinate is one way to get it).")
    M = S0[:, cols] @ _pinv_herm(Scc, rcond) @ S0[cols, :]
    return M, float(np.trace(M).real)


def group_by_span(S, blocks, combinations, atol=1e-6, rcond=1e-10):
    """Cluster combinations by the subspace they span. Returns [[idx, ...], ...].

    Compares projectors with a TOLERANCE. Hashing rounded floats fails here for
    a structural reason: M is 36x36 for this alphabet and two genuinely
    equivalent projectors differ by ~1e-9, so with ~1300 entries the odds that
    at least one straddles a 1e-6 rounding boundary are ~92%. That reported 439
    distinct spans out of 439, when the log plainly showed pairs with identical
    coverage AND identical cond -- e.g. [1,2,3,10] and [3,5,10], differing only
    by d == rest0 + rest1.

    The trace is used as an O(1) pre-filter with the same tolerance, never as a
    bucket key, so there is no boundary to straddle.
    """
    proj = [span_projector(S, blocks, c, rcond) for c in combinations]
    groups, reps = [], []                    # reps: (group index, M, trace)
    for i, (M, tr) in enumerate(proj):
        for gi, Mr, trr in reps:
            if abs(tr - trr) < 1e-6 and np.allclose(M, Mr, atol=atol):
                groups[gi].append(i)
                break
        else:
            reps.append((len(groups), M, tr))
            groups.append([i])
    return groups


def _full_rank(S, blocks, c, tol=1e-3):
    """Does this combination span as many dimensions as it has columns?

    A rank-deficient set spans a SMALLER space than its column count, and
    span_projector uses pinv, so it groups with the full-rank set spanning that
    smaller space. It must never be the representative: it would be rejected
    downstream by the conditioning check and take the good set with it. On
    BaTiO3 that is exactly how Ti t2g + O p (12 WF) disappeared -- a dependent
    13-WF partner won the tie-break and was then thrown out.
    """
    S0 = np.asarray(S)[0]
    cols = np.concatenate([np.arange(sl.start, sl.stop)
                           for (j, sl), cj in zip(blocks, np.asarray(c, int))
                           if cj > 0]) if np.any(c) else np.zeros(0, int)
    if cols.size == 0:
        return False
    Scc = S0[np.ix_(cols, cols)]
    d = np.sqrt(np.clip(np.diag(Scc).real, 1e-30, None))
    Sn = (Scc / d[:, None]) / d[None, :]
    return bool(np.linalg.eigvalsh(Sn).min().real >= tol)


def dedupe_combinations(S, blocks, combinations, trial_set, prefer="hybrid",
                        rank_first=True, dedupe_by="span", searcher=None,
                        atol=1e-6, verbose=True):
    """Group equivalent combinations; keep one representative each.

    dedupe_by : 'span' (default) groups combinations spanning the SAME subspace,
        which is the only sound criterion. 'irreps' groups by irrep content and
        is the LEGACY behaviour, kept for reproducing older runs.

        Grouping by irreps is WRONG and was a real bug: different Wyckoff
        positions can induce identical irrep multiplicities at every k while
        spanning genuinely different subspaces -- that is the composite band
        representation ambiguity. On ZnS, Zn(sp3)+Zn(dT2)+Zn(dE)+S(p) and
        Zn(dT2)+Zn(dE)+S(sp3)+S(dT2) were grouped and one was silently
        discarded. On MnTe it plausibly hid [3,5,7,9,10] behind [0,1,2,10],
        because the hybrid preference keeps whichever member has more composite
        projections and the other member is then never scored. If switching to
        'irreps' restores an old answer, that answer came from this.

    prefer : 'hybrid' | 'fewest' | 'first' -- which member to keep.
    rank_first : put FULL RANK ahead of `prefer`. A rank-deficient
        representative is rejected downstream and takes its whole group with
        it; that is how Ti t2g + O p vanished on BaTiO3. Set False only to
        reproduce the pre-fix ordering.
    """
    combinations = [np.asarray(c, int) for c in combinations]
    if dedupe_by == "irreps":
        if searcher is None:
            raise ValueError("dedupe_by='irreps' needs the EBRsearcher instance")
        keys = {}
        for i, c in enumerate(combinations):
            keys.setdefault(combination_irreps(searcher, c), []).append(i)
        groups = list(keys.values())
    elif dedupe_by == "span":
        groups = group_by_span(S, blocks, combinations, atol=atol)
    else:
        raise ValueError(f"dedupe_by must be 'span' or 'irreps', got {dedupe_by!r}")

    def pref_key(c):
        if prefer == "hybrid":
            return (_n_hybrid(trial_set, c), -int(c.sum()))
        if prefer == "fewest":
            return (-int(c.sum()),)
        return (0,)

    out = []
    for g in groups:
        members = [combinations[i] for i in g]
        if prefer == "first" and not rank_first:
            rep = members[0]
        elif rank_first:
            rep = max(members, key=lambda c: (_full_rank(S, blocks, c),) + pref_key(c))
        else:
            rep = max(members, key=lambda c: pref_key(c) + (_full_rank(S, blocks, c),))
        out.append((rep, members))
    if verbose:
        dup = sum(len(m) - 1 for _, m in out)
        what = "spans" if dedupe_by == "span" else "irrep contents (LEGACY)"
        print(f"  {len(combinations)} combinations -> {len(out)} distinct {what} "
              f"({dup} duplicates removed)")
        if dedupe_by == "irreps":
            n_mixed = sum(1 for _, m in out
                          if len({span_projector(S, blocks, c)[0].tobytes()[:64]
                                  for c in m}) > 1)
            print(f"  WARNING dedupe_by='irreps': at least {n_mixed} group(s) "
                  "contain members spanning DIFFERENT subspaces; all but the "
                  "representative are discarded without being scored")
        n_bad = sum(1 for rep, _ in out if not _full_rank(S, blocks, rep))
        if n_bad:
            print(f"  WARNING {n_bad} representative(s) are rank deficient")
    return out


def combination_condition(S, blocks, c, tol=1e-4):
    """Smallest eigenvalue of the NORMALISED overlap of the chosen columns.

    EBRsearcher matches irrep content, which cannot see that two projections
    overlap. On MnTe it picked `rest0` (the hybrid complement, effectively
    -1.000|dz2>) together with the full `d` shell at the same site: 6 trial
    functions spanning 5 dimensions, smallest normalised eigenvalue 4.5e-06.
    That set has no well-defined projector, and `subset_projectability`
    regularises rather than failing, so the coverage it reports is silently
    computed on a lower-rank subspace.

    Returns the minimum over k of lambda_min(D^-1/2 S D^-1/2), which is 1 for an
    orthogonal set and ~0 for a dependent one.
    """
    S = np.asarray(S)
    cols = np.concatenate([np.arange(sl.start, sl.stop)
                           for (j, sl), cj in zip(blocks, np.asarray(c, int))
                           if cj > 0]) if np.any(c) else np.zeros(0, int)
    if cols.size == 0:
        return 0.0
    worst = np.inf
    for k in range(S.shape[0]):
        Sk = S[k][np.ix_(cols, cols)]
        d = np.sqrt(np.clip(np.diag(Sk).real, 1e-30, None))
        Sn = (Sk / d[:, None]) / d[None, :]
        worst = min(worst, float(np.linalg.eigvalsh(Sn).min().real))
    return worst


def score_combinations(A, S, blocks, combinations, target_mask, wk_k,
                       O=None, in_window=None, trial_set=None, verbose=True):
    """Coverage of the target manifold for each combination.

    blocks : [(j, slice), ...] over the amn columns, one entry per projection in
        the SAME order as trial_set.projections.

    Combinations with any multiplicity > 1 are skipped: repeating a fixed-site
    projection would reuse identical columns, and for a free-parameter position
    the repeats sit at DIFFERENT coordinates only after maximize_distance(), so
    the amn must be recomputed for them. proj_max_multiplicity is 1 for every
    fixed Wyckoff position, so in practice this rarely bites.
    """
    wk = np.asarray(wk_k) / np.sum(wk_k)
    wt = wk[:, None] * target_mask
    denom = float(wt.sum())
    out = []
    for c in combinations:
        c = np.asarray(c, int)
        if (c > 1).any():
            out.append((c, float("nan"), int(c.sum()), float("nan"), 1.0))
            continue
        cols = np.concatenate([np.arange(sl.start, sl.stop)
                               for (j, sl), cj in zip(blocks, c) if cj > 0]) \
            if c.any() else np.zeros(0, int)
        if cols.size == 0:
            out.append((c, 0.0, 0, 0.0, 1.0))
            continue
        p = subset_projectability(A, S, cols, O=O)
        iw = (min(in_window[j] for (j, sl), cj in zip(blocks, c) if cj > 0)
              if in_window else 1.0)
        out.append((c, float((wt * p).sum() / max(denom, 1e-30)), int(cols.size),
                    combination_condition(S, blocks, c), iw))
    if verbose:
        n_skip = sum(1 for t in out if not np.isfinite(t[1]))
        if n_skip:
            print(f"  {n_skip} combination(s) skipped (multiplicity > 1; "
                  "recompute the amn after maximize_distance)")
    return out


def rank_combinations(scored, cond_min=1e-3, min_gain=0.02, rel_gain=0.15,
                      p_min=None, use_in_window=False, verbose=True,
                      labels=None):
    """Pareto front on (nwann, coverage), then stop where the gain flattens.

    Neither pure criterion works, and MnTe shows both failures in one table:

        22 WF  coverage 0.5853
        24 WF  coverage 0.7333     <- +0.148 for 2 WF
        26 WF  coverage 0.7349     <- +0.0016 for 2 WF

    Sorting by nwann takes 22 and misses that the next 2 orbitals buy almost a
    sixth of the manifold. Sorting by coverage takes the largest set, since
    coverage is monotone in the set -- adding orbitals can never lower it.

    So: build the Pareto front (each entry the best coverage at its size),
    then walk it from the smallest and stop at the first step whose gain PER
    WANNIER FUNCTION falls below `rel_gain` of the best step seen. On MnTe that
    accepts 22->24 (0.074/WF) and rejects 24->26 (0.0008/WF, 1% of it), landing
    on 24 -- the set the pDOS route found.

    cond_min rejects linearly dependent sets first: EBRsearcher matches irreps
    and cannot see that two projections overlap, so it will offer e.g. a hybrid
    complement together with the full shell it came from.

    min_gain is the coverage a single added Wannier function must buy (0.02 =
    2% of the manifold), measured with lookahead over the whole remaining front.
    rel_gain is accepted for backward compatibility and no longer used: the
    lookahead makes a relative knee test redundant.

    p_min, if given, is a floor on absolute coverage applied after the walk.
    Leave it None -- WannierBerri's analytic trial orbitals cap coverage near
    0.8, so an absolute threshold mostly just fires or does not depending on the
    material rather than on the choice.
    """
    # optional in-window weighting: effective = coverage x min over the chosen
    # blocks of the fraction of that orbital's own weight inside the OUTER
    # window. Coverage is monotone in the set and so cannot penalise an orbital
    # the disentanglement is unable to build; this can. min, not mean, because
    # one unreachable trial function compromises the whole set -- it has nothing
    # to be built from and drags the others through the orthonormalisation.
    #
    # Ad hoc: the product has no derivation, only the property that each factor
    # is in [0, 1] and measures something the other misses. Validate with the
    # band distance before trusting it, and keep it OFF for a set you already
    # know Wannierises well.
    if use_in_window:
        scored = [(t[0], t[1] * (t[4] if len(t) > 4 else 1.0)) + tuple(t[2:])
                  for t in scored]
    live = [t for t in scored if np.isfinite(t[1])]
    ok = [t for t in live if len(t) < 4 or not np.isfinite(t[3])
          or t[3] >= cond_min]
    n_dep = len(live) - len(ok)
    if not ok:
        ok = live
    if not ok:
        raise ValueError("no scorable combinations")

    # Pareto front: best coverage at each nwann, keeping only strict improvements
    best_at = {}
    for t in ok:
        if t[2] not in best_at or t[1] > best_at[t[2]][1]:
            best_at[t[2]] = t
    front, top = [], -np.inf
    for nw in sorted(best_at):
        t = best_at[nw]
        if t[1] > top + 1e-12:
            front.append(t)
            top = t[1]

    # Walk the front with LOOKAHEAD: from the current point, consider every
    # larger set on the front, not just the next one, and jump to whichever has
    # the best coverage gain PER WANNIER FUNCTION. Repeat until nothing pays.
    #
    # Stepping only to the neighbour truncates on a non-monotone profile. Real
    # case, ZnS:
    #     9 -> 12 : +0.0034/WF   (below the floor -> old rule stopped here)
    #    12 -> 13 : +0.0484/WF   (well above it, never evaluated)
    # The knee was at 13 and the neighbour-only walk could not see past the flat
    # step in front of it. With lookahead, 9 -> 13 is scored directly as
    # +0.0147/WF, which is the honest average cost of getting there.
    #
    # min_gain is the coverage one added Wannier function must buy (0.02 = 2% of
    # the manifold). It is what distinguishes "the gain flattened" from "there
    # was never any gain": Si2's front rises by ~0.005/WF the whole way and must
    # stop at the smallest set, while MnTe has a genuine 0.074/WF first step.
    chosen = front[0]
    steps = []
    while True:
        cand = [(t, (t[1] - chosen[1]) / (t[2] - chosen[2]))
                for t in front if t[2] > chosen[2]]
        if not cand:
            break
        best, rate = max(cand, key=lambda x: x[1])
        steps.append((chosen[2], best[2], rate))
        if rate < min_gain:
            break
        chosen = best

    if p_min is not None and chosen[1] < p_min:
        better = [t for t in front if t[1] >= p_min]
        if better:
            chosen = better[0]

    pool = [chosen] + [t for t in sorted(ok, key=lambda t: (t[2], -t[1]))
                       if t is not chosen]
    self_info = dict(front=front, steps=steps, chosen=chosen, n_dependent=n_dep)
    if verbose:
        if n_dep:
            print(f"  {n_dep} combination(s) rejected as linearly dependent "
                  f"(min normalised overlap eigenvalue < {cond_min})")
        print(f"  Pareto front (best coverage at each size):")
        for t in front:
            mark = "  <- selected" if t is chosen else ""
            print(f"    {t[2]:>4} WF   coverage {t[1]:.4f}{mark}")
        for a, b, r in steps:
            print(f"    {a} -> {b}: {r:+.4f} per WF"
                  + ("   (below min_gain, stopped here)" if r < min_gain else ""))
        nz = [i for i, v in enumerate(chosen[0]) if v > 0]
        print(f"  selected {chosen[2]} WF, coverage {chosen[1]:.4f}: "
              f"{[labels[i] for i in nz] if labels else nz}")
    return pool, self_info


def shortlist_for_validation(scored, S, blocks, trial_set, in_window=None,
                             n_max=6, cov_frac=0.90, iw_min=0.30,
                             cond_min=1e-3, verbose=True, labels=None):
    """Candidates worth Wannierising: FILTER on both criteria, then order.

    Measured on MnTe over 13 wannierisations, neither scalar ranks:

        corr(min in-window, log eta) = -0.94 on the first 8, +0.38 on all 13
        corr(coverage,      log eta) = +0.66 on the first 8, -0.79 on all 13

    Both flipped sign when a second family of candidates was added, so neither
    is a ranking function. As a CONJUNCTION they separated the sample perfectly:

        coverage > 0.70 AND min(in-window) > 0.3  ->  eta 17-50   (5 sets)
        either one failing                        ->  eta 91-5573 (8 sets)

    They fail differently, which is why the conjunction works. Low coverage
    means the set does not span the manifold at all -- the Te-centred sets sat
    at 0.54 and gave eta ~5500. Low in-window means one orbital cannot be built
    from the available states -- every set containing Te s (0.207) landed in
    91-240 despite the HIGHEST coverages in the sample.

    cov_frac : keep combinations within this fraction of the best coverage
        (relative, so it transfers between materials).
    iw_min : absolute floor on min(in-window). 0.30 sits in the gap between
        0.207 (Te s, always bad) and 0.377 (always fine) -- fitted to one
        material, so treat it as provisional.

    Within the survivors, ordering barely matters -- they span eta 17-50 -- so
    they are returned largest-coverage first with span-diversity enforced.
    """
    live = [t for t in scored
            if np.isfinite(t[1]) and (len(t) < 4 or not np.isfinite(t[3])
                                      or t[3] >= cond_min)]
    if not live:
        return []

    def min_iw(c):
        if not in_window:
            return 1.0
        vals = [in_window[j] for (j, sl), v in zip(blocks, np.asarray(c, int))
                if v > 0 and j in in_window]
        return min(vals) if vals else 0.0

    best_cov = max(t[1] for t in live)
    passed = [t for t in live
              if t[1] >= cov_frac * best_cov and min_iw(t[0]) >= iw_min]
    n_cov = sum(1 for t in live if t[1] < cov_frac * best_cov)
    n_iw = sum(1 for t in live if min_iw(t[0]) < iw_min)
    if not passed:
        if verbose:
            print(f"  no combination passes both filters "
                  f"(coverage >= {cov_frac:.2f} x {best_cov:.4f}, "
                  f"min in-window >= {iw_min}); falling back to coverage order")
        passed = sorted(live, key=lambda t: -t[1])

    passed.sort(key=lambda t: (-t[1], -_n_hybrid(trial_set, t[0])))
    out, seen = [], []
    for t in passed:
        if len(out) >= n_max:
            break
        M, _ = span_projector(S, blocks, t[0])
        if any(np.allclose(M, Mp, atol=1e-6) for Mp in seen):
            continue
        seen.append(M)
        out.append(t)

    if verbose:
        print(f"  shortlist ({len(out)} of {len(live)}): rejected {n_cov} on "
              f"coverage < {cov_frac:.2f} x best, {n_iw} on min in-window < {iw_min}")
        print(f"    {'nwann':>6} {'coverage':>9} {'min iw':>7} {'hyb':>4}   projections")
        for t in out:
            nz = [j for (j, sl), v in zip(blocks, np.asarray(t[0], int)) if v > 0]
            print(f"    {t[2]:>6} {t[1]:>9.4f} {min_iw(t[0]):>7.3f} "
                  f"{_n_hybrid(trial_set, t[0]):>4}   "
                  f"{[labels[i] for i in nz] if labels else nz}")
    return out