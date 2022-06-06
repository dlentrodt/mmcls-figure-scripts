import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pynuss
import pygreenfn.green_functions as pynuss_gf

import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters

class MLexpansion():

    def __init__(self, function):
        self.function = function
        self.poles    = None
        self.residues = None

    def setPoles(self, poles):
        self.poles = poles # for example when finding poles from reflection coefficient and then expanding Green's function

    def findPoles(self, reArg, imArg, *args, refineParams=None, setAttribute=False):
        poles = []

        neighborhood_size, threshold = args

        ### evaluate function and get grid ###
        reArgGrid, imArgGrid = np.meshgrid(reArg, imArg, indexing='ij')
        cArgGrid = reArgGrid + 1j*imArgGrid
        data = np.abs(self.function(cArgGrid))**2

        ### find poles ###
        data_max = filters.maximum_filter(data, neighborhood_size)
        maxima = (data == data_max)
        data_min = filters.minimum_filter(data, neighborhood_size)
        diff = ((data_max - data_min) > threshold)
        maxima[diff == 0] = 0

        labeled, num_objects = ndimage.label(maxima)
        slices = ndimage.find_objects(labeled)

        im_idx, re_idx = [], []
        for dy,dx in slices:
            x_center = (dx.start + dx.stop - 1)/2
            im_idx.append(x_center)
            y_center = (dy.start + dy.stop - 1)/2    
            re_idx.append(y_center)
        im_idx = list(map(int, im_idx))
        re_idx = list(map(int, re_idx))
        re_poles, im_poles = [reArg[re_idx], imArg[im_idx]]
        poles = re_poles + 1j*im_poles

        ### refine poles if specified ###
        if not (refineParams is None):
            n_fine_re, range_re, n_fine_im, range_im  = refineParams
            poles_fine = []
            for pole in poles:
                # fine grid
                reArg_fine = np.linspace(np.real(pole)-range_re, np.real(pole)+range_re, n_fine_re)
                imArg_fine = np.linspace(np.imag(pole)-range_im, np.imag(pole)+range_im, n_fine_im)
                reArgGrid, imArgGrid = np.meshgrid(reArg_fine, imArg_fine, indexing='ij')
                cArgGrid = reArgGrid + 1j*imArgGrid

                # evaluate function
                data = np.abs(self.function(cArgGrid))**2

                # find poles
                data_max = filters.maximum_filter(data, neighborhood_size)
                maxima = (data == data_max)
                data_min = filters.minimum_filter(data, neighborhood_size)
                diff = ((data_max - data_min) > threshold)
                maxima[diff == 0] = 0

                labeled, num_objects = ndimage.label(maxima)
                slices = ndimage.find_objects(labeled)

                im_idx, re_idx = [], []
                for dy,dx in slices:
                    x_center = (dx.start + dx.stop - 1)/2
                    im_idx.append(x_center)
                    y_center = (dy.start + dy.stop - 1)/2    
                    re_idx.append(y_center)
                im_idx = list(map(int, im_idx))
                re_idx = list(map(int, re_idx))
                re_poles, im_poles = [reArg_fine[re_idx], imArg_fine[im_idx]]
                pole_fine = re_poles + 1j*im_poles
                
                if len(pole_fine) > 1:
                    #continue
                    print(pole)
                    print(pole_fine)
                    raise ValueError("more than one pole in refined grid") # maybe raise warning and continue here instead
                if not (len(pole_fine) == 0):
                    poles_fine.append(pole_fine[0])
            poles = np.asarray(poles_fine)

        ### set attribute ###
        if setAttribute:
            self.poles = poles
        return poles

    def findResidues(self, radius, nIntegral, poles=None, setAttribute=False):
        if poles is None:
            poles = self.poles

        residues = []
        ### for each pole... ###
        for pole in poles:
            # helpers
            phis = np.linspace(0, 2.*np.pi, nIntegral)
            circleGrid = pole + radius*(np.cos(phis)+1j*np.sin(phis))
            deltaPath = 2.*np.pi*radius/nIntegral
            # ...compute residue via line integral around pole
            function_vals = self.function(circleGrid)
            phase_vals    = np.exp(1j*phis)
            residue = 1./(2j*np.pi) * deltaPath * 1j * np.sum(function_vals*phase_vals)
            residues.append(residue)

        residues = np.asarray(residues)

        ### set attribute ###
        if setAttribute:
            self.residues = residues
        return residues

    def MLexpansion(self, cArgGrid, poleIdxsSlice, cArgRef=0.+0.j, poleIdxsAll=None, ExcludeConst=False):
        if not (poleIdxsAll is None):
            poles    = self.poles[poleIdxsAll]
            residues = self.residues[poleIdxsAll]
        else:
            poles    = self.poles
            residues = self.residues
        constant  = self.function(cArgRef)
        constant += np.einsum('p,p->', residues, 1./(poles-cArgRef))
        # slice selected poles
        poles    = self.poles[poleIdxsSlice]
        residues = self.residues[poleIdxsSlice]
        cArgGrid_ = np.einsum('p,r...->pr...', np.ones_like(poles, dtype=np.complex128), cArgGrid )
        poles_    = np.einsum('p,r...->pr...', poles, np.ones_like(cArgGrid, dtype=np.complex128))
        if ExcludeConst:
            MLexp_    = np.einsum('p,pr...->r...', residues, 1./(cArgGrid_-poles_) )
        else:
            MLexp_    = constant + np.einsum('p,pr...->r...', residues, 1./(cArgGrid_-poles_) )
        return MLexp_, constant

    def MLexpansion_poleRemoval(self, cArgGrid, poleIdxsSlice, cArgRef=0.+0.j, removeConsts=False):
        # slice selected poles
        poles    = self.poles[poleIdxsSlice]
        residues = self.residues[poleIdxsSlice]
        cArgGrid_    = np.einsum('p,r...->pr...', np.ones_like(poles, dtype=np.complex128), cArgGrid )
        poles_       = np.einsum('p,r...->pr...', poles, np.ones_like(cArgGrid, dtype=np.complex128))
        if removeConsts:
            MLexp_remove = np.einsum('p,pr...->r...', residues, 1./(poles_-cArgRef) + 1./(cArgGrid_-poles_) )
        else:
            MLexp_remove = np.einsum('p,pr...->r...', residues, 1./(cArgGrid_-poles_) )
        # compute full function
        full_function = self.function(cArgGrid)
        # return with selected poles removed
        return full_function - MLexp_remove



