# swan - PAW dft computation with GPAW into WannierBerri powered wanierisation 
This is a very first draft to create an automated workflow for wannierisation, based on https://github.com/TMM-TUDA/Automatic-wannier-flow and the afferent paper by Zhang et al., in GPAW + WannierBerri.

## Repo structure
```
swan/
|
|-test/
|   |-Si2/
|   |   |-Si2.cif
|   |   |-Si2-wannierized_bands.png
|   |-...
|
|-src/swan
|       |-auto_proj_and_window.py
|       |-dft.py
|       |-extend_proj_set.py
|       |-main.py
|       |-utils.py
|       |-wannier.py
|
|-create_input_files.py
|-run_all_tests.sh
```
- ```test```: contains folder for different crystals used as tests, each one with a .cif input and a .png plot containing the comparison between the Wannier interpolated bands and the DFT ones
- ```main.py``` : contains the core pipeline
- ```dft.py``` contains the automatized SCF and NSCF computations
- ```wannier.py``` contains the wannierization and band interpolation step
- ```auto_proj_and_windows.py``` contains the implementation of the algorithm of Zhang for finding the energy windows and the projections, called in ```main.py```
- ```utils.py``` : contains utility functions, such as the argument parser and the determination of the crystal system
- ```create_input_files``` can be used to create .cif inputs from a python ASE Atoms object declaration
- ```run_all_tests.sh``` : utility script for  running multiple tests consecutively. The tests that will be ran are written in the first lines (can be modified)
## How to use 
Input: a .cif file containing the structure of the crystal of interest (can be generated via python in ```create_input_files.py```) 
Optional inputs : run with --help to get their description.

How to run: 

If you are using uv simply run ```uv run swan <input.cif>```

If you use a standard pip managed python, create a venv with the provided requirements.txt, activate it and run ```python3 -m swan <input.cif>```

If you have an MPI-enabled installation of GPAW (to check : ```gpaw info```), you can run the code with ```mpirun -np <number of processes>``` to allow parallelization over multiple cores. 
Outputs: logs and output files, including a plot of the bands appear in ```test/<chemical formula>```