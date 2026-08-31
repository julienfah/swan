import numpy as np
from gpaw import GPAW
from gpaw.mpi import serial_comm, world
from ase.dft.bandgap import bandgap

from .amn_projectability import (pdos_from_weights, projectability_from_amn,
                                 true_overlap, weights_from_amn)
from ..candidate_scan import build_candidate_set, extract_amn, extract_eig
from .sphere_projectability import equivalent_atoms
from ..utils.utils import find_emax_from_dos, safety_check_windows

L_NAME = {0: "s", 1: "p", 2: "d", 3: "f"}


def complete_lone_p(calc, selected, nwann, candidates, l_num):
    """Add s at any site whose only selected shell is p.

    p-block valence bonding is sp: sp, sp2, sp3. An isolated p shell at a bonded
    site is almost never the right basis, and with p alone `build` has 3
    orbitals and no totally-symmetric partner, so it cannot form directed lobes
    at all. One extra orbital per site fixes that.

    NOT applied to a lone d. Transition-metal d is genuinely atomic -- the MnTe
    reference set keeps Mn as plain d -- and adding s there buys a diffuse,
    poorly localised function for nothing. Restricting to p is what makes this
    rule agree with the reference:

        raw                    +s on lone p           +s on lone p or d
        Si   sp            8   sp              8      sp               8
        GaN  Ga sp, N p   14   Ga sp, N sp    16      Ga sp, N sp     16
        MnTe Mn d, Te p   16   Mn d, Te sp    18      Mn sd, Te sp    20
                                    ^ matches Mn d + Te sp2+pz = 18

    At the -6m2 Te site {s, p} splits into A1'(s), A2''(pz) and E'(px, py), and
    sp2 spans A1' + E' -- the same space the reference sp2 + pz spans, so `build`
    recovers that gauge without being told.
    """
    have = {}
    for iatom, (n, l_str) in selected:
        have.setdefault(iatom, set()).add(l_str)
    by_atom = {}
    for iatom, (n, l_str) in candidates:
        by_atom.setdefault(iatom, {})[l_str] = n
    for iatom, ls in sorted(have.items()):
        if ls == {"p"} and "s" in by_atom.get(iatom, {}):
            selected.append((iatom, (by_atom[iatom]["s"], "s")))
            nwann += 1
            if world.rank == 0:
                print(f"Completed lone p with s: "
                      f"{calc.atoms[iatom].symbol}{iatom}")
    return selected, nwann


def coverage_cutoff(energies, pdos, dos_total, e_fermi, frac=0.7, tiny=1e-2):
    """Energy above E_F where the projection stops describing the bands.

    The candidate set covers the valence manifold well and the free-electron
    region hardly at all -- on MnTe the summed channels track the total DOS up
    to roughly 10 eV and then fall to a fifth of it. Selecting over a window
    that runs into that region dilutes every enrichment by states no
    atom-centred orbital can represent, which is what lets marginal channels
    creep in.

    Cut where the covered fraction drops below `frac` of its value over the
    occupied manifold. This keeps the antibonding partners a hybrid needs, which
    a fixed E_F + 2 would lose, without paying for the region where the
    measurement is meaningless.
    """
    summed = sum(pdos.values())
    ok = dos_total > tiny * np.max(dos_total)
    occ = energies <= e_fermi
    if not occ.any() or np.trapezoid(dos_total[occ], energies[occ]) <= 0:
        return float(energies[-1])
    # DOS-weighted, not a median of pointwise ratios: most grid points in the
    # occupied range sit in gaps where both curves are ~0 and the ratio is
    # noise, which drags a median far below the real covered fraction
    ref = (np.trapezoid(summed[occ], energies[occ])
           / np.trapezoid(dos_total[occ], energies[occ]))
    ratio = np.where(ok, summed / np.maximum(dos_total, 1e-30), np.inf)
    above = np.flatnonzero(ok & (energies > e_fermi) & (ratio < frac * ref))
    return float(energies[above[0]]) if above.size else float(energies[-1])


def band_count_max(calc, emin, emax, spin=0):
    n = 0
    for k in range(len(calc.get_k_point_weights())):
        e = calc.get_eigenvalues(kpt=k, spin=spin)
        n = max(n, int(((e >= emin) & (e <= emax)).sum()))
    return n


def amn_pdos(calc, spacegroup, from_gpaw, dos_kwargs, shells=("s", "p", "d"),
             symprec=1e-4, spin_channel=0):
    """pDOS from the amn, on the same energy grid `calc.get_dos` would use."""
    l_num = {"s": 0, "p": 1, "d": 2, "f": 3}
    proj_set, blocks = build_candidate_set(
        calc.atoms, spacegroup, shells=[l_num[l] for l in shells],
        symprec=symprec)
    out = from_gpaw(calculator=calc, spin_channel=spin_channel,
                    projections=proj_set, irreducible=True,
                    files=["amn", "eig"], return_bandstructure=True)
    wandata, bandstructure = out if isinstance(out, tuple) else (out, None)

    A = extract_amn(wandata)
    eps_kn = extract_eig(wandata)
    S, O = true_overlap(wandata.amn, bandstructure, with_band_gram=True)
    p_kn, Atil = projectability_from_amn(A, S=S, O=O)
    w = weights_from_amn(Atil, blocks)

    wk_k = np.asarray(calc.get_k_point_weights())
    if len(wk_k) != eps_kn.shape[0]:
        wk_k = np.full(eps_kn.shape[0], 1.0 / eps_kn.shape[0])

    eq = equivalent_atoms(calc.atoms, symprec)
    w_atom = {}
    for (rep, l), v in w.items():
        members = [i for i, r in enumerate(eq) if r == rep]
        for a in members:
            w_atom[(a, L_NAME[l])] = v / len(members)

    energies, pdos, dos_total = pdos_from_weights(
        eps_kn, wk_k, w_atom, width=dos_kwargs.get("width", 0.05),
        npts=dos_kwargs.get("npts", 1001))
    candidates = [(iatom, (4, l)) for l in shells
                  for iatom in range(len(calc.atoms))]
    return energies, pdos, dos_total, candidates


def Zhang_projection_method_amn(
    K=1.2,
    out_dir="test",
    in_dir="test",
    seed=None,
    calc=None,
    dos_kwargs={"spin": 0, "npts": 1001, "width": 0.05},
    gap_thres=0.1,
    maximize_fw=False,
    objective_wd=None,
    comm=serial_comm,
    spacegroup=None,
    from_gpaw=None,
    margin=2.0,
    cover_frac=0.7,
    complete_sp=False,
):
    if calc is None:
        calc = GPAW(f"{in_dir}/{seed}/{seed}-nscf-irred.gpw", txt=None, communicator=comm)
    if spacegroup is None:
        from irrep.spacegroup import SpaceGroup
        spacegroup = SpaceGroup.from_gpaw(calc)
    e_fermi = calc.get_fermi_level()
    energies, pdos, dos_total, candidates = amn_pdos(
        calc, spacegroup, from_gpaw, dos_kwargs)

    def integrate(dos, emin, emax):
        mask = (energies >= emin) & (energies <= emax)
        return np.trapezoid(dos[mask], energies[mask], dx=energies[1] - energies[0])

    l_conversion = {0: "s", 1: "p", 2: "d", 3: "f"}
    l_num = {"s": 0, "p": 1, "d": 2, "f": 3}
    alpha_initial = 0.45
    alpha_max = 1.5

    Emin_0, Emax_0 = initial_DOS_energy_scan(
        calc=calc, out_dir=out_dir, in_dir=in_dir, seed=seed, dos_kwargs=dos_kwargs,
        gap_thres=gap_thres, energies=energies, pdos=pdos, dos_total=dos_total
    )
    if objective_wd is not None:
        Emin_0, Emax_0 = objective_wd[0], max(objective_wd[1], Emax_0)  # override with user-defined window
    print(f"Initial outer window: {Emin_0} to {Emax_0} eV")
    print(f"Candidates for projections: "
          f"{[(calc.atoms[i].symbol + str(i), l) for i, (n, l) in candidates]}")

    ncol = sum(2 * l_num[l_str] + 1 for iatom, (n, l_str) in candidates)
    n_froz = band_count_max(calc, Emin_0,
                            objective_wd[1] if objective_wd is not None else e_fermi + 2)
    nwann_max = int(margin * n_froz)
    print(f"Frozen manifold holds {n_froz} bands at the worst k -> nwann_max = {nwann_max}")

    def alpha_selection(alpha, emin, emax):
        selected_orbitals = []
        nwann = 0
        n_states = integrate(dos_total, emin, emax)
        uniform = n_states / ncol
        ranked = []
        for iatom, (n, l_str) in candidates:
            deg = 2 * l_num[l_str] + 1
            integrated = integrate(pdos[(iatom, l_str)], emin, emax)
            ranked.append(((integrated / deg) / uniform if uniform > 0 else 0.0,
                           iatom, n, l_str, deg))
        for enrichment, iatom, n, l_str, deg in sorted(ranked, key=lambda r: -r[0]):
            sym = calc.atoms[iatom].symbol
            if enrichment > alpha and nwann + deg <= nwann_max:
                if world.rank == 0:
                    print(
                        f"Selected orbital: {sym}{iatom}, n={n}, l={l_str}, enrichment={enrichment:.3f} > alpha={alpha:.3f}"
                    )
                selected_orbitals.append((iatom, (n, l_str)))
                nwann += deg
            else:
                why = "alpha" if enrichment <= alpha else f"nwann_max={nwann_max}"
                if world.rank == 0:
                    print(
                        f"Rejected orbital: {sym}{iatom}, n={n}, l={l_str}, enrichment={enrichment:.3f} ({why})"
                    )
        if complete_sp:
            selected_orbitals, nwann = complete_lone_p(
                calc, selected_orbitals, nwann, candidates, l_num)
        return selected_orbitals, nwann

    alpha = alpha_initial
    alpha_increments = (alpha_max - alpha_initial) / 10.0

    e_sel_max = min(Emax_0, coverage_cutoff(energies, pdos, dos_total, e_fermi,
                                            frac=cover_frac))
    print(f"Selection window: {Emin_0} to {e_sel_max} eV "
          f"(coverage cutoff; Emax_0 was {Emax_0})")
    selected_orbitals, nwann = alpha_selection(alpha, Emin_0, e_sel_max)

    if not selected_orbitals:
        print("No orbitals selected with initial alpha thresholds. Decreasing alpha.")
        alpha = alpha_initial / 2
        selected_orbitals, nwann = alpha_selection(alpha, Emin_0, e_sel_max)
        if not selected_orbitals:
            raise ValueError(
                "No orbitals selected even after decreasing alpha thresholds. Check the DOS and PDOS data."
            )
    # now refine the outer window based on the selected orbitals and their pDOS
    emax_refined = find_emax_from_dos(energies, dos_total, Emin_0, nwann, K=K)
    steps = 0
    while emax_refined is None and steps < 10:
        alpha = alpha + alpha_increments
        selected_orbitals, nwann = alpha_selection(alpha, Emin_0, e_sel_max)
        emax_refined = find_emax_from_dos(energies, dos_total, Emin_0, nwann, K=K)
        steps += 1

    if emax_refined is None:
        raise ValueError("E_max not found — increase nbands in NSCF")

    # Selection runs on the OUTER window so antibonding hybrid partners above
    # E_F + 2 are visible, and nwann_max bounds it. Without that bound the map
    # W -> nwann -> W' is increasing and runs away: on GaN the state target went
    # 7.2 -> 16.8 -> 28.8 -> 40.8 with nwann 6 -> 14 -> 24 -> 34.
    prev_nwann = nwann
    for _ in range(10):
        selected_orbitals, nwann = alpha_selection(
            alpha, Emin_0, min(emax_refined, e_sel_max))
        if not selected_orbitals:
            raise ValueError(
                "No orbitals selected within the refined outer window. Check the DOS and PDOS data."
            )
        if nwann == prev_nwann:
            break  # selection and outer window are now mutually consistent
        emax_new = find_emax_from_dos(energies, dos_total, Emin_0, nwann, K=K)
        if emax_new is None:
            raise ValueError("E_max not found — increase nbands in NSCF")
        emax_refined = emax_new
        prev_nwann = nwann
    print(f"Refined projections: "
          f"{[(calc.atoms[i].symbol + str(i), l) for i, (n, l) in selected_orbitals]}"
          f", nwann={nwann}")

    e_froz_max_0 = e_fermi + 2
    e_froz_min_0 = Emin_0

    if maximize_fw:
        e_froz_max_0 = emax_refined  # ensure frozen window is below outer window
    if objective_wd is not None:
        e_froz_max_0 = objective_wd[1]  # use objective frozen window if provided
        e_froz_min_0 = objective_wd[0]
    print(
        f"before safety check: Outer window: {Emin_0} to {emax_refined} eV, Frozen window: {e_froz_min_0} to {e_froz_max_0} eV"
    )
    out_win, frozen_win = safety_check_windows(calc, nwann, (Emin_0, emax_refined), (e_froz_min_0, e_froz_max_0))
    # could do all checks in one kpoints loop, more efficient, but this is clearer for now

    print(f"Selected orbitals: "
          f"{[(calc.atoms[i].symbol + str(i), l) for i, (n, l) in selected_orbitals]}")
    print(f"Outer window: {out_win[0]} to {out_win[1]} eV")
    print(f"Frozen window: {frozen_win[0]} to {frozen_win[1]} eV")

    return selected_orbitals, out_win, frozen_win, nwann


def initial_DOS_energy_scan(
    calc=None,
    out_dir="test",
    in_dir="test",
    seed=None,
    dos_kwargs={"spin": 0, "npts": 1001, "width": 0.05},
    gap_thres=0.1,
    comm=serial_comm,
    energies=None,
    pdos=None,
    dos_total=None,
):
    if calc is None:
        calc = GPAW(f"{in_dir}/{seed}/{seed}-nscf-irred.gpw", txt=None, communicator=comm)
    e_fermi = calc.get_fermi_level()
    print(f"Fermi level: {e_fermi} eV")

    # estimate gap for delta scaling
    gap, p1, p2 = bandgap(calc, output=None)

    # plot the total DOS and the PDOS for each candidate orbital
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))
    plt.plot(dos_total, energies, label="Total DOS", color="black", linewidth=2)
    for (iatom, l), dos in pdos.items():
        plt.plot(dos, energies, label=f"{calc.atoms[iatom].symbol}, l={l}", alpha=0.7)
    plt.axhline(e_fermi, color="red", linestyle="--", label="Fermi level")
    plt.ylabel("Energy (eV)")
    plt.xlabel("DOS (states/eV)")
    plt.title(f"DOS and PDOS for {seed}")
    plt.legend()
    plt.savefig(f"{out_dir}/{seed}/{seed}-dos_pdos.png", dpi=200)
    plt.close()
    # analyze energy ranges where candidate PDOS is non-zero
    emin_list, emax_list = [], []

    for (iatom, l), dos in pdos.items():
        emin_ij, emax_ij = find_zero_dos_window(energies, dos, e_fermi, gap=gap, threshold=1e-10, gap_thres=gap_thres)
        emin_list.append(emin_ij)
        emax_list.append(emax_ij)
    return min(emin_list), max(emax_list)


def find_zero_dos_window(energies, dos, e_fermi, gap=0.0, threshold=1e-6, gap_thres=0.1):
    delta = gap / 2 + 0.2  # must span at least the gap to ensure a valid target window
    if world.rank == 0:
        print(f"Gap: {gap} eV, Delta for target window: {delta} eV")
    target_low = e_fermi - delta
    target_high = e_fermi + delta

    intervals = get_nonzero_intervals(energies, dos, threshold)
    # filter out narrow noise spikes
    intervals = [(s, e) for s, e in intervals if (e - s) >= 1e-3]
    # merge very close intervals
    intervals = merge_intervals(intervals, threshold=gap_thres)
    if world.rank == 0:
        print(f"Raw non-zero-DOS energy intervals: \n {intervals}")

    # nearest interval start at or below target_low
    candidates_low = [s for s, e in intervals if s <= target_low]
    merged_start = max(candidates_low) if candidates_low else target_low

    # nearest interval end at or above target_high
    candidates_high = [e for s, e in intervals if e >= target_high]
    merged_end = min(candidates_high) if candidates_high else target_high

    return merged_start, merged_end


def get_nonzero_intervals(energies, dos, threshold):
    intervals = []
    in_nonzero = dos[0] >= threshold
    start = energies[0] if in_nonzero else None
    for i in range(1, len(energies)):
        if dos[i] >= threshold and not in_nonzero:
            start = energies[i]
            in_nonzero = True
        elif dos[i] < threshold and in_nonzero:
            intervals.append((start, energies[i - 1]))
            in_nonzero = False
    if in_nonzero:
        intervals.append((start, energies[-1]))
    return intervals


def merge_intervals(intervals, threshold):
    merged = []
    for s, e in sorted(intervals):
        if merged and s - merged[-1][1] <= threshold:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged