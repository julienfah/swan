from gpaw import GPAW
from pathlib import Path as Path_
import numpy as np
from pawan.utils import find_emax_from_dos
from irrep.spacegroup import SpaceGroup
from collections import defaultdict
from wannierberri.symmetry.wyckoff_position import split_into_orbits
from wannierberri.symmetry.projections import Projection, ProjectionsSet

def get_proj_set(K=1.2,seed=None,dir="test",dos_kwargs={'spin': 0, 'npts': 1001, 'width': 0.05}):
    calc = GPAW(f'{dir}/{seed}/{seed}-nscf-irred.gpw', txt=None)
    selected_orbitals, outer_win, frozen_win,nwann = Zhang_projection_method(K=K,dir=dir,seed=seed,calc=calc,dos_kwargs=dos_kwargs)

    space_group = SpaceGroup.from_gpaw(calc)

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
    return ProjectionsSet(projections=projs), outer_win, frozen_win, nwann

def Zhang_projection_method(K=1.2, dir="test", seed=None, calc=None, dos_kwargs={'spin': 0, 'npts': 1001, 'width': 0.05}):
    '''Placeholder for Zhang's projection method, which will be implemented in the future.'''
    if calc is None:
        calc = GPAW(f'{dir}/{seed}/{seed}-nscf-irred.gpw', txt=None)
    e_fermi = calc.get_fermi_level()
    energies, dos_total = calc.get_dos(**dos_kwargs)
    l_conversion = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
    l_num = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
    alpha_initial = {l: (2*l_num[l]+1) / 2 for l in l_conversion.values()}  # 50% threshold
    alpha_max    = {l:  2*l_num[l]+1      for l in l_conversion.values()}  # 100% threshold

    Emin_0, Emax_0,pdos,candidates = initial_DOS_energy_scan(calc=calc, dir=dir, seed=seed, dos_kwargs=dos_kwargs) 
    print(f"Initial outer window: {Emin_0} to {Emax_0} eV")
    print(f"Candidates for projections: {candidates}")
    # then integrate each orbital's pDOS and compare with occupation tolerance alpha
    def integrate(dos,emin,emax):
        mask = (energies >= emin) & (energies <= emax)
        return np.trapezoid(dos[mask], energies[mask],dx=energies[1]-energies[0])
    def alpha_selection(alpha):
        selected_orbitals = []
        nwann = 0
        for (iatom, (n, l_str)) in candidates:
            integrated = integrate(pdos[(iatom, l_str)],Emin_0,Emax_0)
            if integrated > alpha[l_str]:
                selected_orbitals.append((iatom,( n, l_str)))
                nwann += 2*l_num[l_str] + 1
        return selected_orbitals, nwann

    alpha = alpha_initial
    alpha_increments = {l: (alpha_max[l]-alpha_initial[l])/10. for l in l_conversion.values()}  # 10 steps to reach max

    selected_orbitals, nwann = alpha_selection(alpha)

    # now refine the outer window based on the selected orbitals and their pDOS
    #print(pdos)
    emax_refined = find_emax_from_dos(energies, dos_total, Emin_0, nwann, K=K)
    steps =0
    while emax_refined is None and steps < 10:
        alpha = {l: alpha[l] + alpha_increments[l] for l in l_conversion.values()}
        selected_orbitals,nwann = alpha_selection(alpha)
        emax_refined = find_emax_from_dos(energies, dos_total, Emin_0, nwann, K=K)
        steps += 1

    if emax_refined is None:
        raise ValueError("E_max not found — increase nbands in NSCF")
    
    # ensure that at least nwann bands are inside the window for any k point
    eigs = np.array([calc.get_eigenvalues(kpt=k) for k in range(len(calc.get_ibz_k_points()))])
    for k_eigs in eigs:
        in_window = k_eigs[(k_eigs >= Emin_0) & (k_eigs <= emax_refined)]
        if len(in_window) < nwann:
            above_emin = k_eigs[k_eigs >= Emin_0]
            emax_refined = max(emax_refined, above_emin[nwann-1])  # set emax to include at least nwann bands
            print(f"Adjusted emax to {emax_refined} eV to include at least {nwann} bands at any k-point.")

    # ensure that there is less than nwann bands in the frozen window for any k point
    froz_max = e_fermi + 2  # start with a guess
    for k_eigs in eigs:
        in_outer = k_eigs[(k_eigs >= Emin_0) & (k_eigs <= emax_refined)]
        in_frozen = in_outer[(in_outer >= Emin_0) & (in_outer <= froz_max)]
        if len(in_frozen) >= nwann:
            froz_max = min(froz_max, in_frozen[nwann - 1] - 0.01)
            print(f"Frozen window capped to {froz_max:.3f} eV (nfrozen must be < nwann={nwann})")

    # check if there is at least one disentangled band at any k point
    for k_eigs in eigs:
        in_free = k_eigs[(k_eigs > froz_max) & (k_eigs <= emax_refined)]
        if len(in_free) == 0:
            above_frozen = k_eigs[k_eigs > froz_max]
            if len(above_frozen) > 0:
                emax_refined = max(emax_refined, above_frozen[0] + 0.01)
                print(f"Extended emax to {emax_refined:.3f} eV to ensure free bands exist at all k-points.")
    
    out_win = (Emin_0, emax_refined)
    frozen_win = (Emin_0, froz_max)
    #could do all checks in one kpoints loop, more efficient, but this is clearer for now

    print(f"Selected orbitals: {selected_orbitals}")
    print(f"Outer window: { out_win[0]} to {out_win[1]} eV")
    print(f"Frozen window: {frozen_win[0]} to {frozen_win[1]} eV")

    return selected_orbitals, out_win, frozen_win, nwann

def initial_DOS_energy_scan(calc=None, dir="test", seed=None, dos_kwargs={'spin': 0, 'npts': 1001, 'width': 0.05}):
    if calc is None:
        calc = GPAW(f'{dir}/{seed}/{seed}-nscf-irred.gpw', txt=None)
    e_fermi = calc.get_fermi_level()
    energies, dos_total = calc.get_dos(**dos_kwargs)
    print(f"Fermi level: {e_fermi} eV")

    l_conversion = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
    ef_idx = np.searchsorted(energies, e_fermi)
    setups = calc.setups
    atoms  = calc.atoms

    # identify candidates (all valence shells + first empty shells)
    candidates = []
    for iatom, atom in enumerate(atoms):
        setup = setups[iatom]
        occupied_ls = set(
            (n,l_conversion[l]) for n, l, f in zip(setup.n_j, setup.l_j, setup.f_j) if f > 0
        )
        for l_str in occupied_ls:
            candidates.append((iatom, l_str))

    # compute orbital PDOS for each candidate
    pdos={}
    for iatom in range(len(calc.atoms)):
        for i,nl in candidates:  
            if iatom == i and (iatom, nl[1]) not in pdos:  
                e, dos = calc.get_orbital_ldos(a=iatom, angular=nl[1], npts=1001, width=0.05)
                pdos[(iatom, nl[1])] = dos

    # analyze energy ranges where candidate PDOS is non-zero
    emin_list, emax_list = [], []
    threshold = 1e-6  
    ef_idx = np.searchsorted(energies, e_fermi)

    for (iatom, l), dos in pdos.items():
        # scan downward from E_F to find VBM for this orbital
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
        emax_list.append(emax_ij)  
    return min(emin_list), max(emax_list),pdos,candidates
    
