# PAWAN - PAW dft computation with GPAW into WannierBerri powered wanierisation 
This is a very first draft to create an automated workflow for wannierisation, based on https://github.com/TMM-TUDA/Automatic-wannier-flow and the afferent paper by Zhang et al., in GPAW + WannierBerri.

All the core code is contained in ```main.py```
## How to use 
Input: a .cif file containing the structure of the crystal of interest (can be generated via python in ```create_input_files.py```) 
Optional inputs : run with --help to get their description.

How to run: 

If you are using uv simply run ```uv run main.py <input.cif>```

If you use a standard pip managed python, create a venv with the provided requirements.txt, activate it and run ```python3 main.py <input.cif>```

Outputs: logs and output files, including a plot of the bands appear in ```test/<chemical formula>```