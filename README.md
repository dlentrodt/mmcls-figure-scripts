# Readme

This software is an archive of the code used to create the scientific figures of the article

- D. Lentrodt, O. Diekmann, C. H. Keitel, S. Rotter, J. Evers "Certifying Multimode Light-Matter Interaction in Lossy Resonators" *Phys. Rev. Lett.* **130**, 263602 (2023); Journal: https://doi.org/10.1103/PhysRevLett.130.263602, Preprint: https://doi.org/10.48550/arXiv.2107.11775.

Requirements:
- numpy, scipy, matplotlib, jupyter, possibly other standard python packages
- pynuss: A software package for nuclear resonance scattering. Currently unpublished and available from the author K. P. Heeg.
- pyrot: A software package implementing linear dispersion theory and Parratt's formalism for general one-dimensional cavities given by a refractive index profile, created for this paper. Available via pip (see https://github.com/dlentrodt/pyrot).

Contents:
- License
- Readme file
- ML_numerical.py: A class implementing a simple numerical scheme to compute Mittag-Leffler pole expansions of a given function that can be numerically evaluated in the complex plane.
- cav_functions.py: Helper for functions for evaluation of certain quantities in the complex plane.
- Subfolders containing scripts creating each figure as pdf output. Additional plots for benchmarking are commented out in the code.
