"""test general functionalities"""

 
def test_grid():
    from swan.utils.utils import adaptative_high_sym_k_grid,standardize_cell
    from ase.build import bulk
    atoms = standardize_cell(bulk('Si', 'diamond', a=5.43))
    g_grid = adaptative_high_sym_k_grid(atoms,nk_length=20, multiplier=1, tol=1e-5)
    print(f"g_grid = {g_grid}")
    assert g_grid == (8, 8, 8)

def test_point_group_from_atoms():
    from swan.utils.utils import pointgroup_from_atoms,standardize_cell
    from ase.build import bulk
    atoms = standardize_cell(bulk('Si', 'diamond', a=5.43))
    pg = pointgroup_from_atoms(atoms)
    print(f"point group = {pg}")
    assert pg.size == 48*2

def test_hybrid():
    from swan.symmetry import build_at, build # build_at works only on a provided atomic site of the atomic structure, build does all of them at once.
    from swan.utils.utils import standardize_cell
    from ase import Atoms
    import numpy as np
    from gpaw import GPAW
    from ase.io import read

    atoms = standardize_cell(read("test/Si2/Si2.cif"))
    pset_0,res_0 = build_at(atoms=atoms, position=atoms.get_scaled_positions()[0],shells="sp", verbose=True, fallback="minimal_shell",label="Si")
    pset_1,res_1 = build_at(atoms=atoms, position=atoms.get_scaled_positions()[1],shells="sp", verbose=True, fallback="minimal_shell",label="Si_second")
    print(pset_0.projections[0].orbitals)
    from wannierberri.symmetry.orbitals import orbitals_sets_dic,hybrids_coef
    orb_names = orbitals_sets_dic[pset_0.projections[0].orbitals[0]]
    for i_orb in range(len(orb_names)):
        #check sp3
        for orb_name in orb_names:
            coeff = hybrids_coef[orb_names[i_orb]]
            print(f"coeff = {coeff}")
            assert np.allclose(np.abs(list(coeff.values())), 0.5)
    #check that the two psets are correctly related by symmetry (xaxis=[-1,0,0] on second atom)
    for orb_0,orb_1 in zip(orbitals_sets_dic[pset_0.projections[0].orbitals[0]], orbitals_sets_dic[pset_1.projections[0].orbitals[0]]):
        coeff_0 = hybrids_coef[orb_0]
        coeff_1 = hybrids_coef[orb_1]

        assert np.isclose(coeff_0["px"], -coeff_1["px"])

def todo_test_ebr_method():
    from swan.ebr.ebr_method import EBR_method
    from ase.build import bulk
    from swan.utils.utils import standardize_cell
    raise NotImplementedError("This test is not implemented yet, needs a GPAW calculator.")

def test_full_run():
    import subprocess
    subprocess.run([
        "mpirun",
        "-np", "6", # adjust the number of processes as a function of your hardware
        "uv", "run", "swan",
        "test/Ga2N2/Ga2N2.cif",
        "--auto-nk-grid",
        "--output-dir", "test_tutorial",
        "--create-xsf",

    ], check=True)
    assert True  # If the subprocess runs without errors, the test passes
    from json import load
    import numpy as np
    with open("test_tutorial/Ga2N2/candidates/results.json", "r") as f: 
        val_result = load(f)
        assert np.isclose(val_result["Ga-hyb|N-p@K1.2"]["eta"],15.529060988437665), "eta value does not match expected for Ga-hyb|N-p@K1.2"
        assert np.isclose(val_result["Ga-hyb|N-p@K1.2"]["max_spread"],2.3779553019705015), "max_spread value does not match expected for Ga-hyb|N-p@K1.2"

