from ase import Atoms
from ase.build import bulk
from ase.io import write
import numpy as np


Si = bulk('Si', 'diamond', a=5.43)          # works well, but squeering the outer wdw with zhang algo made it slightly worse (see the two png)
Cu = bulk('Cu', 'fcc', a=3.61)              # broken, one garbage WF (the s one?)
GaAs = bulk('GaAs', 'zincblende', a=5.65)   # good in frozen window
MgO = bulk('MgO', 'rocksalt', a=4.21)       # works well
NaCl = bulk('NaCl', 'rocksalt', a=5.64)     # good in frozen window, but it does not include the lower conduction band, go to adaptative frozen win?
Al   = bulk('Al', 'fcc', a=4.05)            # one of the WF outside the FW is garbage, and even inside it is not perfect
Ag   = bulk('Ag', 'fcc', a=4.09)            # full garbage, even the dft looks wrong
tungsten = bulk('W', 'bcc', a=3.16)         # broken, more bands in frozen window than WF -> need to adaptative frozen window?
Vanadium = bulk('V', 'bcc', a=3.03)         # broken, requires 1/2 translation but real grid from GPAW is odd, should be adaptative?
Ga = bulk('Ga', 'orthorhombic', a=4.51, b=4.52, c=7.66)
ke  = bulk('K',  'bcc', a=5.23)             # failes, frozen window is bigger than outer !!! -> need to get the E_min close to Ef, not at the bottom of the bands


tests= [Ga,ke]

for test in tests:
    write(f"{test.get_chemical_formula()}.cif", test)