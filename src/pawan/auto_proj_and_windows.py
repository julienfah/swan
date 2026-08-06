from gpaw import GPAW
from pathlib import Path as Path_
import numpy as np
from pawan.utils import find_emax_from_dos
from irrep.spacegroup import SpaceGroup
from collections import defaultdict
from wannierberri.symmetry.wyckoff_position import split_into_orbits
from wannierberri.symmetry.projections import Projection, ProjectionsSet
from ase.dft.bandgap import bandgap
from gpaw.mpi import serial_comm, world

from pawan.salc_M3 import build,describe_orbital,projectability_from_gpaw,site_group
from pawan.extend_proj_set import extend_to_energy_window
from pawan.utils import safety_check_windows
#from pawan.candidate_scan import candidate_scan, amn_projection_method
from pawan.zhang_amn import Zhang_projection_method_amn
from wannierberri.w90files import WannierData

def get_proj_set(
    calc = None,
    K=1.2,
    seed=None,
    out_dir="test",
    in_dir="test",
    dos_kwargs={"spin": 0, "npts": 1001, "width": 0.05},
    gap_thres=7,
    maximize_fw=False,
    objective_wd=None,
    hybridize_on_site=False,
    comm=serial_comm,
):
    if calc is None:
        calc = GPAW(f"{in_dir}/{seed}/{seed}-nscf-irred.gpw", txt=None, communicator=comm)
    selected_orbitals, outer_win, frozen_win, nwann = Zhang_projection_method(#sphere_projection_method(#
        K=K,
        out_dir=out_dir,
        in_dir=in_dir,
        seed=seed,
        calc=calc,
        dos_kwargs=dos_kwargs,
        gap_thres=gap_thres,
        maximize_fw=maximize_fw,
        objective_wd=objective_wd,
        comm=comm,
        #from_gpaw=WannierData.from_gpaw,
        #alpha=0.3,
        #backend="sphere"
    )
    space_group = SpaceGroup.from_gpaw(calc)
    if hybridize_on_site:
        """#perform hybridization for each atom
        selected_orbitals_l_list = defaultdict(list) #convert to list of {iatom : list of all l values of iatom}
        hybrydized_orbitals = []
        for i, (n,l) in selected_orbitals:
            selected_orbitals_l_list[i].append(l)
        for i, l_list in selected_orbitals_l_list.items():
            selected_hybridized_orbitals = hybridize_orbitals(calc.atoms, calc.atoms.positions[i], orbitals=l_list)
            hybrydized_orbitals.extend([(i, (4,l)) for l in selected_hybridized_orbitals])
        selected_orbitals = hybrydized_orbitals"""
        print("-"*80,"\n")
        """p_kn, w, blocks, cand_set = candidate_scan(
            calc=calc,
            spacegroup=space_group,
            from_gpaw=WannierData.from_gpaw,
            spin_channel=0,
            shells=(0,1 ),
            unitary_params=dict(nbands_upper_skip=2)
        )"""
        """res = candidate_scan(calc=calc, spacegroup=space_group,
                     from_gpaw=WannierData.from_gpaw, spin_channel=0,
                     shells=(0, 1,2), pdos_path=f"{out_dir}/{seed}/{seed}-pdos.png",unitary_params=dict(error_threshold=0.1, warning_threshold=0.01, nbands_upper_skip=2)
        )
        p_kn, w, blocks, = res.eps_kn, res.w, res.blocks
        #print("candidate set:", cand_set)
        print("sum of p_kn:", np.sum(p_kn,axis=1), np.sum(p_kn,axis=0))"""
        """selected_orbitals, outer_win, frozen_win, nwann = amn_projection_method(
            calc=calc, spacegroup=space_group, from_gpaw=WannierData.from_gpaw,
            seed=seed, in_dir=in_dir, out_dir=out_dir,
            K=K, gap_thres=5.0, objective_wd=objective_wd, shells=(0, 1, 2),margin=1.1,verbose=True,alpha=0.33)
        print("selected orbitals after amn_projection_method:", selected_orbitals)
        print("outer_win:", outer_win, "frozen_win:", frozen_win, "nwann:", nwann)
        print("-"*80,"\n")"""
        
        #exit()
        shells_dict = {calc.atoms[iatom].symbol: [l for iatom2, (n, l) in selected_orbitals if iatom2 == iatom] for iatom, (n, l) in selected_orbitals}
        weight_func = projectability_from_gpaw(calc, window=outer_win,n_select="valence")
        proj_set, salc_sites = build(calc.atoms, shells_dict, prefix=f"{seed}_",weight_fn=None,fallback="best_hybrid")
        assert proj_set.num_wann == nwann, \
        f"projection set has {proj_set.num_wann} WF but windows were sized for {nwann}"
        #proj_set = build(calc.atoms, shells_dict, prefix=f"{seed}_")

    else:
        projs = []
        # group atoms by species
        species_positions = defaultdict(list)
        for atom, pos in zip(calc.atoms, space_group.positions):
            species_positions[atom.symbol].append(pos)

        seen = set()
        for iatom, (n, l) in selected_orbitals:
            symbol = calc.atoms[iatom].symbol
            if (symbol, n, l) in seen:
                continue  # this species+shell already handled via orbit splitting
            seen.add((symbol, n, l))
            positions = species_positions[symbol]
            orbits_ind = split_into_orbits(positions, space_group)
            for orbit_indices in orbits_ind:
                orbit_positions = [positions[i] for i in orbit_indices]
                proj = Projection(position_num=orbit_positions, orbital=l, spacegroup=space_group, rotate_basis=True)
                projs.append(proj)
        proj_set = ProjectionsSet(projections=projs)
    if objective_wd is not None and frozen_win[1] < objective_wd[1]:
        energies, dos_total = calc.get_dos(**dos_kwargs)
        print(
            f"Warning: The frozen window was capped to {frozen_win[1]} eV, which is below the objective frozen window of {objective_wd[1]} eV. Adding s orbitals from the next smallest-multiplicity Wyckoff position to extend the projection set."
        )
        proj_set, outer_win, frozen_win, nwann = extend_to_energy_window(
            calc, energies, dos_total, K, nwann, proj_set, outer_win, frozen_win, objective_wd
        )

    # log the selected orbitals and the windows
    log_file_path = Path_(f"{out_dir}/{seed}/orbitals_and_windows.txt")
    with open(log_file_path, "w") as f:
        f.write(
            f"Selected orbitals (atom symbol, atom index, orbital): {[(calc.atoms[i].symbol, i, l) for i, (n, l) in selected_orbitals]},\n"
        )
        f.write(f"Outer window: {outer_win}\n")
        f.write(f"Frozen window: {frozen_win}\n")
        f.write(f"Number of Wannier functions: {nwann}\n")
    with open(log_file_path, "a") as f:
        f.write("Selected Projections (+ non-atomic-centered ones):\n")
        for p in proj_set.projections:
            f.write(str(p) + "\n")
            for orb in p.orbitals:
                f.write(describe_orbital(orb) + "\n")
    #with open(log_file_path, "a") as f:
    #    f.write(f"Selected Projections (+ non-atomic-centered ones):\n {'\n'.join(map(str, proj_set.projections))},\n")

    return proj_set, outer_win, frozen_win, nwann


def Zhang_projection_method(
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
):
    """Placeholder for Zhang's projection method, which will be implemented in the future."""
    if calc is None:
        calc = GPAW(f"{in_dir}/{seed}/{seed}-nscf-irred.gpw", txt=None, communicator=comm)
    e_fermi = calc.get_fermi_level()
    energies, dos_total = calc.get_dos(**dos_kwargs)
    def integrate(dos, emin, emax):
        mask = (energies >= emin) & (energies <= emax)
        return np.trapezoid(dos[mask], energies[mask], dx=energies[1] - energies[0])
    print(f"dos_total HHH {integrate(dos_total, energies[0], energies[-1])}")
    tot = 0
    for l in ["s", "p", "d"]:
        for iatom in range(len(calc.atoms)):
            e, dos = calc.get_orbital_ldos(a=iatom, angular=l, **dos_kwargs)
            tot += integrate(dos, energies[0], energies[-1])
    print(tot)
    l_conversion = {0: "s", 1: "p", 2: "d", 3: "f"}
    l_num = {"s": 0, "p": 1, "d": 2, "f": 3}
    alpha_initial = {l: (2 * l_num[l] + 1) * 0.6 for l in l_conversion.values()}  # 60% threshold
    alpha_max = {l: 2 * l_num[l] + 1 for l in l_conversion.values()}  # 100% threshold

    Emin_0, Emax_0, pdos, candidates = initial_DOS_energy_scan(
        calc=calc, out_dir=out_dir, in_dir=in_dir, seed=seed, dos_kwargs=dos_kwargs, gap_thres=gap_thres
    )
    if objective_wd is not None:
        Emin_0, Emax_0 = objective_wd[0], max(objective_wd[1], Emax_0)  # override with user-defined window
    print(f"Initial outer window: {Emin_0} to {Emax_0} eV")
    print(f"Candidates for projections: {candidates}")
    # then integrate each orbital's pDOS and compare with occupation tolerance alpha

    def integrate(dos, emin, emax):
        mask = (energies >= emin) & (energies <= emax)
        return np.trapezoid(dos[mask], energies[mask], dx=energies[1] - energies[0])

    def alpha_selection(alpha, emin, emax):
        selected_orbitals = []
        nwann = 0
        print(sum(integrate(d, energies[0], energies[-1]) for d in pdos.values()),
      "should equal", integrate(dos_total, energies[0], energies[-1]))
        for iatom, (n, l_str) in candidates:
            integrated = integrate(pdos[(iatom, l_str)], emin, emax)
            total = integrate(pdos[(iatom, l_str)], energies[0], energies[-1])
            if world.rank == 0:
                print(f"Total integrated pDOS for {calc.atoms[iatom]}, {l_str}: {total}")
            if integrated > alpha[l_str]:
                if world.rank == 0:
                    print(
                        f"Selected orbital: Atom {iatom}, n={n}, l={l_str}, integrated pDOS={integrated:.3f} > alpha={alpha[l_str]:.3f}\n\n"
                    )
                selected_orbitals.append((iatom, (n, l_str)))
                nwann += 2 * l_num[l_str] + 1
            else:
                if world.rank == 0:
                    print(
                        f"Rejected orbital: Atom {iatom}, n={n}, l={l_str}, integrated pDOS={integrated:.3f} <= alpha={alpha[l_str]:.3f}\n\n"
                    )
        return selected_orbitals, nwann
    
    alpha = alpha_initial
    alpha_increments = {
        l: (alpha_max[l] - alpha_initial[l]) / 10.0 for l in l_conversion.values()
    }  # 10 steps to reach max

    selected_orbitals, nwann = alpha_selection(alpha, Emin_0, Emax_0)

    if not selected_orbitals:
        print("No orbitals selected with initial alpha thresholds. Decreasing alpha.")
        alpha = {l: (2 * l_num[l] + 1) * 0.5 for l in l_conversion.values()}
        selected_orbitals, nwann = alpha_selection(alpha, Emin_0, Emax_0)
        if not selected_orbitals:
            raise ValueError(
                "No orbitals selected even after decreasing alpha thresholds. Check the DOS and PDOS data."
            )
    # now refine the outer window based on the selected orbitals and their pDOS
    emax_refined = find_emax_from_dos(energies, dos_total, Emin_0, nwann, K=K)
    steps = 0
    while emax_refined is None and steps < 10:
        alpha = {l: alpha[l] + alpha_increments[l] for l in l_conversion.values()}
        selected_orbitals, nwann = alpha_selection(alpha, Emin_0, Emax_0)
        emax_refined = find_emax_from_dos(energies, dos_total, Emin_0, nwann, K=K)
        steps += 1

    if emax_refined is None:
        raise ValueError("E_max not found — increase nbands in NSCF")

    # Refine the orbital selection on the widened window, keeping nwann and the
    # outer window mutually consistent: the refined window (Emax_0 -> emax_refined)
    # lowers the integrated occupations and can remove orbitals, which in
    # turn requires the outer window to be re-solved for the new count so that
    # int(rho) dE = K * nwann still holds. Iterate to a fixed point.
    selected_orbitals, nwann = alpha_selection(alpha, Emin_0, emax_refined)
    """prev_nwann = nwann
    for _ in range(10):
        selected_orbitals, nwann = alpha_selection(alpha, Emin_0, emax_refined)
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
        prev_nwann = nwann"""
    print(f"Refined projections: {selected_orbitals}, nwann={nwann}")

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

    print(f"Selected orbitals: {selected_orbitals}")
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
):
    if calc is None:
        calc = GPAW(f"{in_dir}/{seed}/{seed}-nscf-irred.gpw", txt=None, communicator=comm)
    e_fermi = calc.get_fermi_level()
    energies, dos_total = calc.get_dos(**dos_kwargs)
    print(f"Fermi level: {e_fermi} eV")

    # estimate gap for delta scaling
    # homo, lumo = calc.get_homo_lumo()
    gap, p1, p2 = bandgap(calc)
    # gap = max(0.0, lumo - homo)

    l_conversion = {0: "s", 1: "p", 2: "d", 3: "f"}
    ef_idx = np.searchsorted(energies, e_fermi)
    setups = calc.setups
    atoms = calc.atoms

    # identify candidates (all valence shells + first empty shells)
    candidates = []

    candidates = [(iatom, (4, l)) for l in ["s", "p", "d"] for iatom, atom in enumerate(atoms)]

    # compute orbital PDOS for each candidate
    pdos = {}
    for iatom in range(len(calc.atoms)):
        for i, nl in candidates:
            if iatom == i and (iatom, nl[1]) not in pdos:
                e, dos = calc.get_orbital_ldos(a=iatom, angular=nl[1], **dos_kwargs)
                pdos[(iatom, nl[1])] = dos
    # plot the total DOS and the PDOS for each candidate orbital
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))
    plt.plot(dos_total, energies, label="Total DOS", color="black", linewidth=2)
    for (iatom, l), dos in pdos.items():
        #if iatom == 0:  # only plot for the first atom of each species to avoid clutter
        plt.plot(dos, energies, label=f"{calc.atoms[iatom].symbol}, l={l}", alpha=0.7)
    plt.axhline(e_fermi, color="red", linestyle="--", label="Fermi level")
    plt.ylabel("Energy (eV)")
    #plt.ylim(0,37)
    #plt.xlim(0, 5)
    plt.xlabel("DOS (states/eV)")
    plt.title(f"DOS and PDOS for {seed}")
    plt.legend()
    plt.savefig(f"{out_dir}/{seed}/{seed}-dos_pdos.png", dpi=200)
    plt.close()
    # analyze energy ranges where candidate PDOS is non-zero
    emin_list, emax_list = [], []
    threshold = 1e-6
    ef_idx = np.searchsorted(energies, e_fermi)

    for (iatom, l), dos in pdos.items():
        emin_ij, emax_ij = find_zero_dos_window(energies, dos, e_fermi, gap=gap, threshold=1e-10, gap_thres=gap_thres)
        emin_list.append(emin_ij)
        emax_list.append(emax_ij)
    return min(emin_list), max(emax_list), pdos, candidates


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
    # add a param for threshold, like 1ev? e.g. would like to be able to get s bamd of GaAs
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

