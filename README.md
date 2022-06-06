# Readme

This software is an archive of the code used to create the scientific figures
of the article

- Dominik Lentrodt, Oliver Diekmann, Christoph H. Keitel, Stefan Rotter, Jörg Evers "Certifying multi-mode light-matter interaction in lossy resonators without a fit" (Preprint at https://doi.org/10.48550/arXiv.2107.11775).

Contents:
- License
- Readme file
- ML_numerical.py: A class implementing a simple numerical scheme to compute Mittag-Leffler pole expansions of a given function that can be numerically evaluated in the complex plane.
- cavity1d.py: Classes implementing linear dispersion theory and Parratt's formalism for general one-dimensional cavities given by a refractive index profile.
- cav_functions.py: Helper for functions for evaluation of certain quantities in the complex plane.
- Subfolders containing scripts creating each figure as pdf output. Additional plots for benchmarking are commented out in the code.