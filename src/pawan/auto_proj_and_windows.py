from gpaw import GPAW
from pathlib import Path as Path_
import numpy as np
from pawan.utils import find_emax_from_dos
from irrep.spacegroup import SpaceGroup
from collections import defaultdict
from wannierberri.symmetry.wyckoff_position import split_into_orbits
from wannierberri.symmetry.projections import Projection, ProjectionsSet
from ase.dft.bandgap import bandgap
from gpaw.mpi import serial_comm,world

def get_proj_set(K=1.2,seed=None,dir="test",dos_kwargs={'spin': 0, 'npts': 1001, 'width': 0.05},gap_thres=0.1,comm=serial_comm):
    calc = GPAW(f'{dir}/{seed}/{seed}-nscf-irred.gpw', txt=None, communicator=comm)
    selected_orbitals, outer_win, frozen_win,nwann = Zhang_projection_method(K=K,dir=dir,seed=seed,calc=calc,dos_kwargs=dos_kwargs,gap_thres=gap_thres)

    space_group = SpaceGroup.from_gpaw(calc)

    #log the selected orbitals and the windows
    log_file_path = Path_(f"{dir}/{seed}/orbitals_and_windows.txt")
    with open(log_file_path, "w") as f:
        f.write(f"Selected orbitals (atom symbol, atom index, orbital): {[(calc.atoms[i].symbol,i,l) for i,(n,l) in selected_orbitals]},\n")
        f.write(f"Outer window: {outer_win}\n")
        f.write(f"Frozen window: {frozen_win}\n")
        f.write(f"Number of Wannier functions: {nwann}\n")
        
    projs = []
    # group atoms by species
    species_positions = defaultdict(list)
    for atom, pos in zip(calc.atoms, space_group.positions):
        species_positions[atom.symbol].append(pos)

    seen = set()
    for iatom,(n,l) in selected_orbitals:
        symbol = calc.atoms[iatom].symbol
        if (symbol, n, l) in seen:
            continue          # this species+shell already handled via orbit splitting
        seen.add((symbol, n, l))
        positions = species_positions[symbol]
        orbits_ind = split_into_orbits(positions, space_group)
        for orbit_indices in orbits_ind:
            orbit_positions = [positions[i] for i in orbit_indices]
            proj = Projection(
                position_num=orbit_positions,
                orbital=l,
                spacegroup=space_group,
                rotate_basis=True
            )
            projs.append(proj)
    with open(log_file_path, "a") as f:
        f.write(f"Selected Projections:\n {'\n'.join(map(str, projs))},\n")
    return ProjectionsSet(projections=projs), outer_win, frozen_win, nwann

def Zhang_projection_method(K=1.2, dir="test", seed=None, calc=None, dos_kwargs={'spin': 0, 'npts': 1001, 'width': 0.05},gap_thres=0.1, comm=serial_comm):
    '''Placeholder for Zhang's projection method, which will be implemented in the future.'''
    if calc is None:
        calc = GPAW(f'{dir}/{seed}/{seed}-nscf-irred.gpw', txt=None, communicator=comm)
    e_fermi = calc.get_fermi_level()
    energies, dos_total = calc.get_dos(**dos_kwargs)

    l_conversion = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
    l_num = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
    alpha_initial = {l: (2*l_num[l]+1)*0.6 for l in l_conversion.values()}  # 60% threshold
    alpha_max    = {l:  2*l_num[l]+1      for l in l_conversion.values()}  # 100% threshold

    Emin_0, Emax_0,pdos,candidates = initial_DOS_energy_scan(calc=calc, dir=dir, seed=seed, dos_kwargs=dos_kwargs, gap_thres=gap_thres) 
    print(f"Initial outer window: {Emin_0} to {Emax_0} eV")
    print(f"Candidates for projections: {candidates}")
    # then integrate each orbital's pDOS and compare with occupation tolerance alpha
    def integrate(dos,emin,emax):
        mask = (energies >= emin) & (energies <= emax)
        return np.trapezoid(dos[mask], energies[mask],dx=energies[1]-energies[0])
    def alpha_selection(alpha,emin,emax):
        selected_orbitals = []
        nwann = 0
        for (iatom, (n, l_str)) in candidates:
            integrated = integrate(pdos[(iatom, l_str)],emin,emax)
            total = integrate(pdos[(iatom,l_str)], energies[0], energies[-1])
            if world.rank == 0:
                print(f"Total integrated pDOS for {calc.atoms[iatom]}, {l_str}: {total}")
            if integrated > alpha[l_str]:
                if world.rank == 0:
                    print(f"Selected orbital: Atom {iatom}, n={n}, l={l_str}, integrated pDOS={integrated:.3f} > alpha={alpha[l_str]:.3f}\n\n")
                selected_orbitals.append((iatom,( n, l_str)))
                nwann += 2*l_num[l_str] + 1
            else:
                if world.rank == 0:
                    print(f"Rejected orbital: Atom {iatom}, n={n}, l={l_str}, integrated pDOS={integrated:.3f} <= alpha={alpha[l_str]:.3f}\n\n")
        return selected_orbitals, nwann

    alpha = alpha_initial
    alpha_increments = {l: (alpha_max[l]-alpha_initial[l])/10. for l in l_conversion.values()}  # 10 steps to reach max

    selected_orbitals, nwann = alpha_selection(alpha, Emin_0, Emax_0)

    if not selected_orbitals:
        print("No orbitals selected with initial alpha thresholds. Decreasing alpha.")
        selected_orbitals, nwann = alpha_selection({l: (2*l_num[l]+1)*0.5 for l in l_conversion.values()}, Emin_0, Emax_0)
        if not selected_orbitals:
            raise ValueError("No orbitals selected even after decreasing alpha thresholds. Check the DOS and PDOS data.")
    # now refine the outer window based on the selected orbitals and their pDOS
    #print(pdos)
    emax_refined = find_emax_from_dos(energies, dos_total, Emin_0, nwann, K=K)
    steps =0
    while emax_refined is None and steps < 10:
        alpha = {l: alpha[l] + alpha_increments[l] for l in l_conversion.values()}
        selected_orbitals,nwann = alpha_selection(alpha, Emin_0, Emax_0)
        emax_refined = find_emax_from_dos(energies, dos_total, Emin_0, nwann, K=K)
        steps += 1

    if emax_refined is None:
        raise ValueError("E_max not found — increase nbands in NSCF")
    
    #refine the orbital selection based on the refined outer window
    selected_orbitals, nwann = alpha_selection(alpha, Emin_0, emax_refined)
    print(f"Refined projections: {selected_orbitals}, nwann={nwann}")
    '''# ensure that at least nwann bands are inside the window for any k point
    eigs = np.array([calc.get_eigenvalues(kpt=k) for k in range(len(calc.get_ibz_k_points()))])
    for k_eigs in eigs:
        in_window = k_eigs[(k_eigs >= Emin_0) & (k_eigs <= emax_refined)]
        if len(in_window) < nwann:
            above_emin = k_eigs[k_eigs >= Emin_0]
            emax_refined = max(emax_refined, above_emin[nwann-1])  # set emax to include at least nwann bands
            print(f"Adjusted emax to {emax_refined} eV to include at least {nwann} bands at any k-point.")

    # ensure that there is no more than nwann bands in the frozen window for any k point
    froz_max = e_fermi + 2  # start with a guess
    #froz_max = calc.get_homo_lumo()[1]+2  # bottom of conduction band if there is one
    for k_eigs in eigs:
        in_outer = k_eigs[(k_eigs >= Emin_0) & (k_eigs <= emax_refined)]
        in_frozen = in_outer[(in_outer >= Emin_0) & (in_outer <= froz_max)]
        if len(in_frozen) > nwann:
            froz_max = min(froz_max, in_frozen[nwann] - 0.01)
            print(f"Frozen window capped to {froz_max:.3f} eV (nfrozen must be < nwann={nwann})")

    # check if there is at least one band btw froz_max and emax_refined at any k point
    for k_eigs in eigs:
        in_free = k_eigs[(k_eigs > froz_max) & (k_eigs <= emax_refined)]
        if len(in_free) == 0:
            above_frozen = k_eigs[k_eigs > froz_max]
            if len(above_frozen) > 0:
                emax_refined = max(emax_refined, above_frozen[0] + 0.01)
                print(f"Extended emax to {emax_refined:.3f} eV to ensure free bands exist at all k-points.")
    
    out_win = (Emin_0, emax_refined)
    frozen_win = (Emin_0, froz_max)'''
    out_win, frozen_win = safety_check_windows(calc,nwann,(Emin_0, emax_refined),(Emin_0, e_fermi + 2))
    #could do all checks in one kpoints loop, more efficient, but this is clearer for now

    print(f"Selected orbitals: {selected_orbitals}")
    print(f"Outer window: { out_win[0]} to {out_win[1]} eV")
    print(f"Frozen window: {frozen_win[0]} to {frozen_win[1]} eV")

    return selected_orbitals, out_win, frozen_win, nwann

def safety_check_windows(calc,nwann,outer_win, frozen_win):
    emin_0, emax_0 = outer_win
    emax_refined = emax_0
    # ensure that at least nwann bands are inside the window for any k point
    eigs = np.array([calc.get_eigenvalues(kpt=k) for k in range(len(calc.get_ibz_k_points()))])
    for k_eigs in eigs:
        in_window = k_eigs[(k_eigs >= emin_0) & (k_eigs <= emax_0)]
        if len(in_window) < nwann:
            above_emin = k_eigs[k_eigs >= emin_0]
            emax_refined = max(emax_refined, above_emin[nwann-1])  # set emax to include at least nwann bands
            print(f"Adjusted emax to {emax_refined} eV to include at least {nwann} bands at any k-point.")

    # ensure that there is no more than nwann bands in the frozen window for any k point
    froz_max = frozen_win[1]  # start with the provided frozen window max
    #froz_max = calc.get_homo_lumo()[1]+2  # bottom of conduction band if there is one
    for k_eigs in eigs:
        in_outer = k_eigs[(k_eigs >= emin_0) & (k_eigs <= emax_refined)]
        in_frozen = in_outer[(in_outer >= emin_0) & (in_outer <= froz_max)]
        if len(in_frozen) > nwann:
            froz_max = min(froz_max, in_frozen[nwann] - 0.01)
            print(f"Frozen window capped to {froz_max:.3f} eV (nfrozen must be < nwann={nwann})")

    # check if there is at least one band btw froz_max and emax_refined at any k point
    for k_eigs in eigs:
        in_free = k_eigs[(k_eigs > froz_max) & (k_eigs <= emax_refined)]
        if len(in_free) == 0:
            above_frozen = k_eigs[k_eigs > froz_max]
            if len(above_frozen) > 0:
                emax_refined = max(emax_refined, above_frozen[0] + 0.01)
                print(f"Extended emax to {emax_refined:.3f} eV to ensure free bands exist at all k-points.")
    return (emin_0, emax_refined),(emin_0, froz_max)

def initial_DOS_energy_scan(calc=None, dir="test", seed=None, dos_kwargs={'spin': 0, 'npts': 1001, 'width': 0.05}, gap_thres=0.1, comm=serial_comm):
    if calc is None:
        calc = GPAW(f'{dir}/{seed}/{seed}-nscf-irred.gpw', txt=None, communicator=comm)
    e_fermi = calc.get_fermi_level()
    energies, dos_total = calc.get_dos(**dos_kwargs)
    print(f"Fermi level: {e_fermi} eV")

    # estimate gap for delta scaling
    #homo, lumo = calc.get_homo_lumo()
    gap, p1, p2 = bandgap(calc)
    #gap = max(0.0, lumo - homo)

    l_conversion = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
    ef_idx = np.searchsorted(energies, e_fermi)
    setups = calc.setups
    atoms  = calc.atoms

    # identify candidates (all valence shells + first empty shells)
    candidates = []
    '''for iatom, atom in enumerate(atoms):
        setup = setups[iatom]
        occupied_ls = set(
            (n,l_conversion[l]) for n, l, f in zip(setup.n_j, setup.l_j, setup.f_j) if f > 0
        )
        for nl_str in occupied_ls:
            candidates.append((iatom, nl_str))#(iatom, (n, l_str)))
    '''
    candidates = [(iatom, (4,l)) for l in ['s', 'p', 'd'] for iatom, atom in enumerate(atoms)]

    # compute orbital PDOS for each candidate
    pdos={}
    for iatom in range(len(calc.atoms)):
        for i,nl in candidates:  
            if iatom == i and (iatom, nl[1]) not in pdos:  
                e, dos = calc.get_orbital_ldos(a=iatom, angular=nl[1], npts=1001, width=0.05)
                pdos[(iatom, nl[1])] = dos
    #plot the total DOS and the PDOS for each candidate orbital
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,6))
    plt.plot(dos_total, energies, label='Total DOS', color='black', linewidth=2)
    for (iatom, l), dos in pdos.items():
        plt.plot(dos, energies, label=f'{calc.atoms[iatom].symbol}, l={l}',alpha=0.7)
    plt.axhline(e_fermi, color='red', linestyle='--', label='Fermi level')
    plt.ylabel('Energy (eV)')
    plt.xlabel('DOS (states/eV)')
    plt.title(f'DOS and PDOS for {seed}')
    plt.legend()
    plt.savefig(f"{dir}/{seed}/{seed}-dos_pdos.png", dpi=200)
    plt.close()
    # analyze energy ranges where candidate PDOS is non-zero
    emin_list, emax_list = [], []
    threshold = 1e-6  
    ef_idx = np.searchsorted(energies, e_fermi)

    for (iatom, l), dos in pdos.items():
        '''# scan downward from E_F to find VBM for this orbital
        below = dos[:ef_idx][::-1]
        nonzero_below = np.where(below > threshold)[0]
        if len(nonzero_below) == 0:
            continue
        # start scanning from the top of the valence band, not from E_F
        vbm_idx = ef_idx - nonzero_below[0]

        # now scan downward from VBM to find where DOS goes to zero
        below_vbm = dos[:vbm_idx][::-1]
        zero_below = np.where(below_vbm < threshold)[0]
        emin_ij = energies[vbm_idx - zero_below[0]] if len(zero_below) > 0 else energies[0]

        # scan upward from E_F
        above = dos[ef_idx:]
        zero_above = np.where(above < threshold)[0]
        emax_ij = energies[ef_idx + zero_above[0]] if len(zero_above) > 0 else energies[-1] #correct??
        emin_list.append(emin_ij)
        emax_list.append(emax_ij)'''  
        emin_ij, emax_ij = find_zero_dos_window(energies, dos, e_fermi, gap=gap, threshold=1e-10, gap_thres=gap_thres)
        emin_list.append(emin_ij)
        emax_list.append(emax_ij)
    return min(emin_list), max(emax_list),pdos,candidates
    
def find_zero_dos_window(energies, dos, e_fermi, gap=0.0, threshold=1e-6,gap_thres=0.1):
    delta = gap / 2 + 0.2  # must span at least the gap to ensure a valid target window
    if world.rank == 0:
        print(f"Gap: {gap} eV, Delta for target window: {delta} eV")
    target_low = e_fermi - delta
    target_high = e_fermi + delta
    
    intervals = get_nonzero_intervals(energies, dos, threshold)
    # filter out narrow noise spikes
    intervals = [(s, e) for s, e in intervals if (e - s) >= 1e-3]
    #merge very close intervals
    intervals = merge_intervals(intervals, threshold=gap_thres)# add a param for threshold, like 1ev? e.g. would like to be able to get s bamd of GaAs
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
            intervals.append((start, energies[i-1]))
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