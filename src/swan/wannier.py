from gpaw import GPAW
from wannierberri.w90files import WannierData
from wannierberri import System_R, Path, evaluate_k_path
from gpaw.mpi import serial_comm
from pathlib import Path as Path_ # avoid name conflict with wannierberri.grid.Path
import numpy as np

def wannierize(
    proj_set,
    outer_win,
    frozen_win,
    seed,
    out_dir,
    in_dir,
    calc_nscf_irred=None,
    spin_channel=0,
    unitary_params=dict(error_threshold=0.1, warning_threshold=0.01, nbands_upper_skip=2),
    wannierization_params=dict(
        num_iter=100,
        conv_tol=1e-8,
        print_progress_every=20,
        sitesym=True,
        localise=True,
    ),
    recompute_files=True,
    ecut_pw=500,
    comm=serial_comm,
):
    """
    Proper wannierization of the system, using the previously determined parameters.

    :param proj_set: ProjectionsSet object containing the projections to be used for the wannierization.
    :param outer_win: Tuple containing the outer energy window boundaries.
    :param frozen_win: Tuple containing the frozen energy window boundaries.
    """
    if calc_nscf_irred is None:
        calc_nscf_irred = GPAW(f"{in_dir}/{seed}/{seed}-nscf-irred.gpw", txt=None, communicator=comm)
    all_files_exist = all((Path_(out_dir) / Path_(f"{seed}/{seed}_wannier_data.{ext}.npz")).exists() for ext in ["amn", "mmn", "eig", "sawf"])
    if recompute_files or not all_files_exist:
        #should recompute amn always, others never
        print(f"Computing wannierization files for {seed}...")
        wandata, bandstructure = WannierData.from_gpaw(
            calculator=calc_nscf_irred,
            spin_channel=spin_channel,
            projections=proj_set,
            irreducible=True,
            ecut_pw=ecut_pw,
            files=["amn", "mmn", "eig", "symmetrizer"],#, "unk"],
            unk_grid=tuple(calc_nscf_irred.wfs.gd.N_c),
            unitary_params=unitary_params,
            return_bandstructure=True,
        )
        wandata.to_npz(f"{out_dir}/{seed}/{seed}_wannier_data")
    else:
        print(f"Loading wannierization files for {seed}...")
        wandata = WannierData.from_npz(
            seedname=f"{out_dir}/{seed}/{seed}_wannier_data",
            files=["mmn", "eig", "chk", "symmetrizer"],
            ignore_missing_files=False,
            irreducible=True,
        )
        from irrep.bandstructure import BandStructure

        bs = BandStructure.from_gpaw(
            calculator_gpaw=calc_nscf_irred,
            Ecut=ecut_pw,
            irreducible=True,
            spin_channel=spin_channel,
            include_TR=True,
        )
        wandata.set_projections(projections=proj_set,bandstructure=bs)
        wandata.to_npz(f"{out_dir}/{seed}/{seed}_wannier_data")

    wandata.wannierise(
        froz_min=frozen_win[0],
        froz_max=frozen_win[1],
        outer_min=outer_win[0],
        outer_max=outer_win[1],  # np.inf,#
        savechk=False,
        **wannierization_params,
    )
    wandata.chk.to_npz(f"{out_dir}/{seed}/{seed}_wannier_data.chk.npz")
    # log spreads
    """plot_wannier(
        seed, out_dir, sc=(0, 0), select_WF=[i for i in range(proj_set.num_wann)],
        reduce_r_points=1, wannier_data=wandata,atoms=calc_nscf_irred.atoms
    )"""
    with open(f"{out_dir}/{seed}/{seed}_wannier_spreads.txt", "w") as f:
        for center, spread in zip(wandata.chk.wannier_centers_cart, wandata.chk.wannier_spreads):
            f.write(f"Center: {center}, Spread: {spread}\n")

def index_to_label(labels, points, atol=1e-2):
    """labels: {label: pos_array} (subset), points: full list of positions.
    Returns {index_in_points: label}, with every matching index included,
    sorted by index."""
    result = {}
    for label, pos in labels.items():
        for i, p in enumerate(points):
            if np.allclose(pos, p, atol=atol):
                result[i] = label
    return dict(sorted(result.items()))
def interpolate_bands(seed, out_dir, in_dir, calc_nscf_irred=None,wannier_data=None, npoints=200, comm=serial_comm):
    """
    Use of the Wannier functions to interpolate the bands.
    """
    if calc_nscf_irred is None:
        calc_nscf_irred = GPAW(f"{in_dir}/{seed}/{seed}-nscf-irred.gpw", txt=None, communicator=comm)
    atoms = calc_nscf_irred.atoms
    #path = atoms.cell.bandpath()
    path = atoms.cell.bandpath(npoints=npoints)
    kpts = path.kpts
    if wannier_data is None:
        wannier_data = WannierData.from_npz(
            seedname=f"{out_dir}/{seed}/{seed}_wannier_data",
            files=["amn", "mmn", "eig", "chk", "symmetrizer"],
            ignore_missing_files=False,
            irreducible=True,
    )
    wandata = wannier_data
    system = System_R.from_wannierdata(wandata=wandata, berry=True)

    kpoints = path.special_points  # dict of label: kcoords
    wb_labels = index_to_label(kpoints, kpts, atol=1e-6)
    x, X, xlabels = path.get_linear_kpoint_axis()
    breaks = [i for i in range(len(x) - 1) if np.isclose(x[i], x[i + 1])]

    wb_path = Path(system=system, k_list=kpts, labels=wb_labels, breaks=breaks)
    bands_wannier = evaluate_k_path(system, path=wb_path)
    return bands_wannier, wb_path



def unfold_unk(wandata):
    unk, sym, chk = wandata.unk, wandata.symmetrizer, wandata.chk
    N = np.array(unk.grid_size, dtype=int)

    # real-space gather maps: out(r) <- src(W^{-1}(r - w)), one per sym op
    grid = np.indices(tuple(N))
    index_maps = []
    for symop in sym.spacegroup.symmetries:
        W = np.asarray(symop.rotation)                 # integer, lattice basis
        w = np.asarray(symop.translation)
        Winv = np.rint(np.linalg.inv(W)).astype(int)
        M = N[:, None] * Winv / N[None, :]
        toff = N * (Winv @ w)
        assert np.allclose(M, np.rint(M)),   f"rot incommensurate with grid {N.tolist()}"
        assert np.allclose(toff, np.rint(toff)), f"transl incommensurate with grid {N.tolist()}"
        M, toff = np.rint(M).astype(int), np.rint(toff).astype(int)
        J = (np.tensordot(M, grid, (1, 0)) - toff[:, None, None, None]) % N[:, None, None, None]
        index_maps.append((J[0], J[1], J[2]))

    for ik in range(chk.num_kpts):
        if ik in unk.data:
            continue
        ikirr = sym.kpt2kptirr[ik]
        isym  = sym.kpt_from_kptirr_isym[ik]           # the op with kptirr2kpt[ikirr,isym]==ik
        u_src = unk.data[int(sym.kptirr[ikirr])]        # (NB, *grid, 1)

        i0, i1, i2 = index_maps[isym]
        u_rot = u_src[:, i0, i1, i2, :]

        if sym.time_reversals[isym]:
            u_rot = u_rot.conj()

        # band mixing: apply d_band blocks along axis 0, exactly as rotate_U does for amn rows
        NB = u_rot.shape[0]
        flat = u_rot.reshape(NB, -1)                    # (NB, ngrid)
        out  = np.zeros_like(flat)
        NB = u_rot.shape[0]
        flat = u_rot.reshape(NB, -1)
        out = np.zeros_like(flat)
        for (s, e), blk in zip(sym.d_band_block_indices[ikirr],
                               sym.d_band_blocks[ikirr][isym]):
            out[s:e] = blk @ flat[s:e]
        unk.data[ik] = out.reshape(u_rot.shape)
        unk.data[ik] = out.reshape(u_rot.shape)

    assert len(unk.data) == chk.num_kpts, f"filled {len(unk.data)}/{chk.num_kpts}"
    return wandata
def plot_wannier(seed, out_dir, sc=(-2, 2), select_WF=None, reduce_r_points=1,
                 wannier_data=None,atoms=None):
    wandata = wannier_data  # pass the in-memory object from wannierize()
    if len(wandata.unk.data) < wandata.chk.num_kpts:
        unfold_unk(wandata)
    if len(wandata.chk.v_matrix) < wandata.chk.num_kpts:
        sym = wandata.symmetrizer
        U_irr = [wandata.chk.v_matrix[int(sym.kptirr[i])] for i in range(sym.NKirr)]
        U_full = sym.U_to_full_BZ(U_irr)          # list of NK arrays
        wandata.chk.v_matrix = {ik: U_full[ik] for ik in range(wandata.chk.num_kpts)}
    if atoms is not None:
        kw = dict(atoms_cart=atoms.get_positions(),
                  atoms_names=atoms.get_chemical_symbols())
    return wandata.plotWF(sc_min=sc[0], sc_max=sc[1], select_WF=select_WF,
                          reduce_r_points=reduce_r_points,
                          path=f"{out_dir}/{seed}/{seed}.WF", **kw)
