from ase import Atoms
from ase.build import bulk
from ase.io import write
import numpy as np

## Si
a = 5.43  # Lattice constant in angstroms
# Define the silicon crystal structure using ASE
lattice = a*(np.ones ((3,3))-np.eye(3))/2 # each row is a basis vector here, in units of a
positions = np.array([[0,0,0],[1,1,1]])/4
Si = Atoms("Si2",cell=lattice,pbc=[1,1,1],scaled_positions=positions)
Si = bulk('Si', 'diamond', a=5.43)
### Cu
Cu = bulk('Cu', 'fcc', a=3.61)              # broken, one garbage WF (the s one?)
GaAs = bulk('GaAs', 'zincblende', a=5.65)   # good in frozen window
MgO = bulk('MgO', 'rocksalt', a=4.21)       # works well
NaCl = bulk('NaCl', 'rocksalt', a=5.64)     # good in frozen window, but it does not include the lower conduction band, go to adaptative frozen win?
Al   = bulk('Al', 'fcc', a=4.05)            # one of the WF outside the FW is garbage, and even inside it is not perfect
Ag   = bulk('Ag', 'fcc', a=4.09)            # full garbage, even the dft looks wrong
tungsten = bulk('W', 'bcc', a=3.16) 

tests= [Si, Cu, GaAs, MgO, NaCl, Al, Ag, tungsten]

for test in tests:
    write(f"{test.get_chemical_formula()}.cif", test)