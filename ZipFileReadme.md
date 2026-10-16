ZipFile Readme.md

# SU(8) Unified Radiance‑Channel Pipeline (Referee‑Optimized)

This ZIP contains the complete SU(8) radiance‑channel inference framework and the stand‑alone 
deterministic pipeline used to generate all AE.1–AE.5 diagnostics in the accompanying manuscript.

## How to Run
From a terminal:

python su8_pipeline.py


You will be prompted:

Enter path to GRMHDsu8_radiance_framework.zip:


Provide the path to this ZIP file.  
All outputs will be written to:

./su8_diagnostics/


## Required Input
The only required input is:

notebooks/Table_AE1_posteriorsummary.csv


All other artifacts (feature‑space NPZ, posterior NPZ, diagnostics) are regenerated deterministically.

## License
Released under **CC BY‑NC 4.0**.