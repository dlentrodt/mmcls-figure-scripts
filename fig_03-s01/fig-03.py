# for calculations
import numpy as np
import scipy as sp
import itertools
from scipy.signal import argrelextrema
import copy

# for plotting
import matplotlib
import matplotlib.pylab as plt
from matplotlib.patches import ConnectionPatch

# suppress text output of functions
from IPython.utils import io

# for fitting
from scipy.optimize import curve_fit

# add path
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# import module functions
from cavity1d import linear_dispersion_scattering as _linearScattering_linearDispersion #phaseAngle
import ML_numerical as ML

### analytic version of Green function for numerical efficiency ###
def rij(betai, betaj):
    return (betai-betaj)/(betai+betaj)

def beta_j(k, n):
    #return np.asarray([np.sqrt((k*i)**2-kpar**2 + 0.j) for i in n])
    return np.asarray([k*i for i in n])

def r_total_5layers(theta, k, ns, ds, k0):
    # TODO
    return r_total_kp(theta, k, ns[2], ds[1], k0)

def greenfunction5layers(z, k, ns, ds): # k=[keV], ds=[1/keV]
    #bare quantities
    betas = beta_j(k, ns)
    beta0, beta1, beta2, beta3, beta4, beta5, beta6 = betas
    r01, r12, r23, r34, r45, r56 = rij(betas[0:6], betas[1:7])
    _, d1, d2, d3, d4, d5, __ = ds
    phase_ = np.einsum('n,n...->n...', ds[1:6], betas[1:6])
    e1, e2, e3, e4, e5 = np.exp(2j*phase_)
    #effective quantities
    r20 = -(r12+r01*e1)/(1+r12*r01*e1)
    r30 = (-r23+r20*e2)/(1-r23*r20*e2)
    r46 = (r45+r56*e5)/(1+r45*r56*e5)
    r36 = (r34+r46*e4)/(1+r34*r46*e4)
    D = 1- r30*r36*e3
    return ((2j*np.pi/beta3)*(1/D)*(1+r36*np.exp(1j*beta3*d3))*
            (1+r30*np.exp(1j*beta3*d3)))

def prefactor(k, *atom_params, fixed_omTrans=True):
    _, atom_dPol, atom_om, _ = atom_params
    if fixed_omTrans:
        return -np.abs(atom_dPol)**2 * atom_om**2 / (4.*np.pi)
    return -np.abs(atom_dPol)**2 * k**2 / (4.*np.pi)

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
    
def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

####################
### barrier scan ###
####################

barrier_ns = np.linspace(1.2, 20.0, 200) # 200
mm_quantifiers = []

for barrier_n in barrier_ns:

    ##%%time
    R0 =   1.0  # thickness of single layer, [arb. units]

    tSmall = 0.01 #thin layer
    N = [1.0, barrier_n, 1.0, barrier_n, 1.0] # Refractive index of each layer
    T = [ -1, tSmall, 1.0, tSmall,  -1] # Thicknesses of each layer

    k = np.linspace(0.001, 15., 200000)

    ### Atom params ###
    R_ref = -tSmall
    # Two-level atom parameters #
    atom_pos   =   0.5
    # Position of the atom
    atom_dPol  =   0.05 #05#0.03  # dipole moment
    atom_om    =   0.0   # transition energy
    atom_gamma =   0.0  # spontaneous decay rate into non-radiative channels
    atom_params = [atom_pos-R_ref, atom_dPol, atom_om, atom_gamma]
    # NOTE: atom_dPol and atom_om get scaled in loop to stay in one regime and on resonance

    # Linear dispersion constants #
    preFact   = 0.07
    gamma     = atom_gamma
    constants = [preFact, gamma]

    ##%%time
    atom_params_empty = copy.copy(atom_params)
    atom_params_empty[1] = 0.

    _, _, S_linDisp_empty = _linearScattering_linearDispersion(k, N, T, atom_params_empty, phase_zero_offset=-k*R0) # R_ref=0.0,

    ##%%time
    max_inds = argrelextrema(np.abs(S_linDisp_empty[0,1,:])**2, np.less)
    ##print(max_inds)
    atom_params[2] = k[max_inds[0][1]]#-0.5

    _, _, S_linDisp = _linearScattering_linearDispersion(k, N, T, atom_params, phase_zero_offset=-k*R0) # R_ref=0.0, 

    # # plot

    # plt.figure()

    # plt.axvline(atom_params[2], color='k', dashes=[1,1], lw=1.0)
    # plt.plot(k, np.abs(S_linDisp[0,1,:])**2)
    # plt.plot(k, np.abs(S_linDisp_empty[0,1,:])**2, '--')

    # zoom_dif = 0.05
    # plt.xlim([atom_params[2]-zoom_dif, atom_params[2]+zoom_dif])

    # plt.close()

    # plt.figure()

    # plt.axvline(atom_params[2], color='k', dashes=[1,1], lw=1.0)
    # plt.plot(k, np.abs(S_linDisp[0,1,:])**2)
    # plt.plot(k, np.abs(S_linDisp_empty[0,1,:])**2, '--')

    # zoom_dif = 0.05
    # #plt.xlim([atom_params[2]-zoom_dif, atom_params[2]+zoom_dif])

    # plt.close()

    # for i, (pole, residue) in enumerate(zip(poles_green, residues_green)):
    #     print("{:02.0f}   {:02.2f} {:02.2f}j   {:02.8f} {:02.8f}j".format(i,
    #                                                     pole.real, pole.imag, residue.real, residue.imag))

    def rij(betai, betaj):
        return (betai-betaj)/(betai+betaj)

    def beta_j(k, n):
        #return np.asarray([np.sqrt((k*i)**2-kpar**2 + 0.j) for i in n])
        return np.asarray([k*i for i in n])

    def r_total_5layers(theta, k, ns, ds, k0):
        # TODO
        return r_total_kp(theta, k, ns[2], ds[1], k0)

    def greenfunction5layers(z, k, ns, ds): # k=[keV], ds=[1/keV]
        #bare quantities
        betas = beta_j(k, ns)
        beta0, beta1, beta2, beta3, beta4, beta5, beta6 = betas
        r01, r12, r23, r34, r45, r56 = rij(betas[0:6], betas[1:7])
        _, d1, d2, d3, d4, d5, __ = ds
        phase_ = np.einsum('n,n...->n...', ds[1:6], betas[1:6])
        e1, e2, e3, e4, e5 = np.exp(2j*phase_)
        #effective quantities
        r20 = -(r12+r01*e1)/(1+r12*r01*e1)
        r30 = (-r23+r20*e2)/(1-r23*r20*e2)
        r46 = (r45+r56*e5)/(1+r45*r56*e5)
        r36 = (r34+r46*e4)/(1+r34*r46*e4)
        D = 1- r30*r36*e3
        return ((2j*np.pi/beta3)*(1/D)*(1+r36*np.exp(1j*beta3*d3))*
                (1+r30*np.exp(1j*beta3*d3)))

    def prefactor(k, *atom_params, fixed_omTrans=True):
        _, atom_dPol, atom_om, _ = atom_params
        if fixed_omTrans:
            return -np.abs(atom_dPol)**2 * atom_om**2 / (4.*np.pi)
        return -np.abs(atom_dPol)**2 * k**2 / (4.*np.pi)

    ns = copy.copy(N)
    ds = copy.copy(T)
    ns.insert(1, 1.0)
    ds.insert(1, 0.001)
    ns.insert(-1, 1.0)
    ds.insert(-1, 0.001)
    #print(ns, ds)

    Theta = np.pi/2.
    green_fn = greenfunction5layers(atom_pos, k, ns, ds)
    compLevShift       = prefactor(k, *atom_params, fixed_omTrans=True) * green_fn
    compLevShift_fixed = prefactor(k, *atom_params, fixed_omTrans=False) * green_fn

    # plt.figure()

    # plt.axvline(atom_params[2], color='k', dashes=[1,1], lw=1.0)
    # plt.plot(k, np.abs(S_linDisp[0,1,:])**2)
    # plt.plot(k, np.abs(S_linDisp_empty[0,1,:])**2, '--')

    # zoom_dif = 0.05
    # #plt.xlim([atom_params[2]-zoom_dif, atom_params[2]+zoom_dif])

    # plt.close()

    # #
    # plt.figure()

    # plt.axvline(atom_params[2], color='k', dashes=[1,1], lw=1.0)
    # plt.axhline(0., color='k', dashes=[1,1], lw=1.0)
    # plt.plot(k, np.real(compLevShift))
    # plt.plot(k, -2.*np.imag(compLevShift))
    # plt.plot(k, np.real(compLevShift_fixed), '--')
    # plt.plot(k, -2.*np.imag(compLevShift_fixed), '--')

    # zoom_dif = 0.05
    # #plt.xlim([atom_params[2]-zoom_dif, atom_params[2]+zoom_dif])
    # y_u = 0.1
    # y_l = 0.05
    # plt.ylim([-y_l, y_u])

    # plt.close()

    ### Green's function in complex plane ###
    k_re = np.linspace(k[0], k[-1], 100)
    k_im = np.linspace(-10.0, 0.0, 2001)

    Kre, Kim = np.meshgrid(k_re, k_im, indexing='ij')
    Kcomp = Kre + 1j*Kim

    green_comp        = greenfunction5layers(atom_pos, Kcomp, ns, ds)
    compLevShift_comp = prefactor(Kcomp, *atom_params, fixed_omTrans=True) * green_comp
    re_ax_idx = find_nearest_idx(k_im, 0.0)

    ##print('Done!')

    ### Find poles and residues ##################################################################################

    ##%%time

    def comp_lev_shift_for_ML(atom_pos, ns, ds, *atom_params):
        def fun_(kComp):
            GF = greenfunction5layers(atom_pos, kComp, ns, ds)
            result = prefactor(kComp, *atom_params, fixed_omTrans=True) * GF
        #         if len(kComp.shape)==2:
        #             plt.figure()
        #             plt.imshow(np.abs(result.T)**2, origin='lower', aspect='auto',
        #                                   norm=matplotlib.colors.LogNorm())
        #             plt.colorbar()
        #             plt.close()
            return result
        return fun_

    ### initialize ###
    comp_lev_shift = comp_lev_shift_for_ML(atom_pos, ns, ds, *atom_params)
    MLexp_compLevShift = ML.MLexpansion(comp_lev_shift)

    ### find poles ###
    k_re_pS = np.linspace(2.0, 7.0, 100)
    k_im_pS = np.linspace( -10.0,  0.1, 201)

    neighborhood_size = 5
    threshold         = 0.000001
    args = [neighborhood_size, threshold]

    #refineParams=None
    refineParams=[1000, k_re_pS[2]-k_re_pS[0], 1001, k_im_pS[2]-k_im_pS[0]]

    poles_green = MLexp_compLevShift.findPoles(k_re_pS, k_im_pS, *args, refineParams=refineParams, setAttribute=True)

    ##print('Done! (1)')

    ### find residues ###
    radius  = (k_im_pS[1]-k_im_pS[0])
    samples = 50000
    residues_green = MLexp_compLevShift.findResidues(radius, samples, poles=None, setAttribute=True)

    zero_pole_idx = 0

    ### eval function for test ###
    Kre_, Kim_ = np.meshgrid(k_re_pS, k_im_pS, indexing='ij')
    Kcomp_ = Kre_+1j*Kim_
    compLevShift_comp_MLtest = MLexp_compLevShift.function(Kcomp_)
    re_ax_idx_ = find_nearest_idx(k_im_pS, 0.0)

    ##print("Done! (2)")

    # plt.figure(figsize=(14,10))

    # #
    # plt.subplot(211)
    # plt.axvline(atom_params[2], color='k', dashes=[1,1], lw=1.0)
    # plt.axhline(0., color='k', dashes=[1,1], lw=1.0)
    # plt.plot(k, np.real(compLevShift))
    # plt.plot(k, -2.*np.imag(compLevShift))
    # # plt.plot(k_re, np.real(compLevShift_comp[:, re_ax_idx]), '--')
    # # plt.plot(k_re, -2.*np.imag(compLevShift_comp[:, re_ax_idx]), '--')
    # plt.plot(k_re_pS, np.real(compLevShift_comp_MLtest[:, re_ax_idx_]), '--')
    # plt.plot(k_re_pS, -2.*np.imag(compLevShift_comp_MLtest[:, re_ax_idx_]), '--')

    # plt.xlim(k_re_pS[0],k_re_pS[-1])
    # zoom_dif = 0.05
    # #plt.xlim([atom_params[2]-zoom_dif, atom_params[2]+zoom_dif])
    # y_u = 0.02
    # y_l = 0.01
    # plt.ylim([-y_l, y_u])

    # #
    # plt.subplot(212)
    # # plt.imshow(np.abs(compLevShift_comp.T)**2, origin='lower', aspect='auto', norm=matplotlib.colors.LogNorm(),
    # #                                            extent=[k_re[0], k_re[-1], k_im[0], k_im[-1]])
    # plt.imshow(np.abs(compLevShift_comp_MLtest.T)**2, origin='lower', aspect='auto', norm=matplotlib.colors.LogNorm(),
    #                                            extent=[k_re_pS[0], k_re_pS[-1], k_im_pS[0], k_im_pS[-1]])
    # plt.plot(np.real(poles_green), np.imag(poles_green), 'mx')

    # plt.close()

    ### Mittag-Leffler ##########################################################################################

    main_pole_idx = find_nearest_idx(poles_green, 4.536645754863973)
    ##print(main_pole_idx, poles_green[main_pole_idx])

    # for i, (pole, residue) in enumerate(zip(poles_green, residues_green)):
    #     print("{:02.0f}   {:02.2f} {:02.2f}j   {:02.8f} {:02.8f}j".format(i,
    #                                                     pole.real, pole.imag, residue.real, residue.imag))

    main_pole_idx = find_nearest_idx(poles_green, 4.536645754863973)
    ##print(main_pole_idx, poles_green[main_pole_idx])

    poleIdxsSlice = [main_pole_idx]

    cArgRef=4.0

    green_re_ML_numerical_single, G0 = MLexp_compLevShift.MLexpansion(k, poleIdxsSlice, cArgRef=cArgRef,
                                                               poleIdxsAll=None, ExcludeConst=True)
    ##print("G0: {}".format(G0))

    # ### Plot: ML expansion for CLS and Sup ###

    # xrange=0.02
    # yrange=50
    # sc = 0.7

    γ0 = atom_params[2]*atom_params[1]**2
    # natural linewidth for the 1D case; see also check notebook on free space linewidtbh

    # find index where main pole and full result are zero
    main_pole_zeros_idxs = np.where(np.diff(np.sign(np.real(green_re_ML_numerical_single))))[0]
    full_func_zeros_idxs = np.where(np.diff(np.sign(np.real(compLevShift))))[0]

    main_pole_zeros_kvals = k[main_pole_zeros_idxs]
    full_func_zeros_kvals = k[full_func_zeros_idxs]

    main_pole_zeros_Δvals = np.real(green_re_ML_numerical_single)[main_pole_zeros_idxs]
    full_func_zeros_Δvals = np.real(compLevShift)[full_func_zeros_idxs]

    # print(main_pole_zeros_idxs)
    # print(full_func_zeros_idxs)
    # print(main_pole_zeros_kvals)
    # print(full_func_zeros_kvals)
    # print(main_pole_zeros_Δvals)
    # print(full_func_zeros_Δvals)

    main_pole_zero_idx = main_pole_zeros_idxs[find_nearest_idx(main_pole_zeros_kvals, np.real(poles_green[main_pole_idx]))]
    full_func_zero_idx = full_func_zeros_idxs[find_nearest_idx(full_func_zeros_kvals, np.real(poles_green[main_pole_idx]))]

    # print(main_pole_zero_idx)
    # print(full_func_zero_idx)

    # extract quantifiers for complex residue and multi-pole effects
    main_pole         = poles_green[main_pole_idx]
    main_pole_residue = residues_green[main_pole_idx]
    main_pole_zero    = k[main_pole_zero_idx]
    full_func_zero    = k[full_func_zero_idx]
    sup_at_full_func_zero = -2.*np.imag(compLevShift)[full_func_zero_idx]

    print(main_pole)
    print(main_pole_residue)
    print(main_pole_zero)
    print(full_func_zero)
    
    mm_quantifiers.append([main_pole, main_pole_residue, main_pole_zero, full_func_zero,
                           γ0, sup_at_full_func_zero, atom_params[2]]) # note that atom_params[2] corresponds to ω_min after the ω**2 rescaling, since we choose the atom frequency to sit at the minimum
    
#     plt.figure(figsize=(14*sc*2,9*sc))

#     ## Lamb shift ##
#     plt.subplot(121, xlabel='$k$ [arb. units]', ylabel='CLS [arb. units]')
#     plt.axhline(0, color='k', dashes=[], linewidth=1)
#     plt.axvline(np.real(mm_quantifiers[-1][0]), color='k', dashes=[], linewidth=1, label='pole loc') # main_pole loc
#     plt.axvline(np.real(mm_quantifiers[-1][2]), color='m', dashes=[], linewidth=1, label='single pole zero') # main_pole zero
#     plt.axvline(np.real(mm_quantifiers[-1][3]), color='g', dashes=[4,4], linewidth=1, label='full zero') # full func zero
#     plt.axvline(np.real(mm_quantifiers[-1][6]), color='r', dashes=[1,1], linewidth=1, label='minimum') # full func zero
#     plt.plot(k, np.real(compLevShift))
#     plt.plot(k, np.real(green_re_ML_numerical_single), '--', label="{}")
#     plt.ylim([-0.005, 0.005])
#     plt.xlim([1., 8.])
#     plt.legend(fontsize=16)

#     plt.subplot(122, xlabel='$k$ [arb. units]', ylabel='CLS [arb. units]')
#     plt.axhline(-2.*np.imag(G0), color='r', dashes=[], linewidth=1)
#     plt.plot(k, -2.*np.imag(compLevShift))
#     plt.plot(k, -2.*np.imag(green_re_ML_numerical_single), '--')
#     plt.ylim([-0.0005, 0.03])
#     plt.xlim([1., 8.])

#     plt.tight_layout()
#     ###plt.savefig('plots/InvCav_mode6.pdf', transparent=True, dpi=1200, bbox_inches='tight')
#     plt.close()

mm_quantifiers = np.asarray(mm_quantifiers)

plt.figure()

plt.plot(barrier_ns, mm_quantifiers[:, 3]-atom_params[2], '-', label='full shift')
plt.plot(barrier_ns, mm_quantifiers[:, 2]-atom_params[2], '--', label='main pole shift (complex residue)')
plt.plot(barrier_ns, mm_quantifiers[:, 3]-mm_quantifiers[:, 2], '--', label='multi-pole shift')
plt.legend(loc=1)

plt.close()

##################
### multi-mode ###
##################

R0 =   1.0  # thickness of single layer, [arb. units]

tSmall = 0.01 #thin layer
N = [1.0,    4.0, 1.0,    4.0, 1.0] # Refractive index of each layer
T = [ -1, tSmall, 1.0, tSmall,  -1] # Thicknesses of each layer

k = np.linspace(0.001, 15., 200000)

### Atom params ###
R_ref = -tSmall
# Two-level atom parameters #
atom_pos   =   0.5
# Position of the atom
atom_dPol  =   0.05 #05#0.03  # dipole moment
atom_om    =   0.0   # transition energy
atom_gamma =   0.0  # spontaneous decay rate into non-radiative channels
atom_params = [atom_pos-R_ref, atom_dPol, atom_om, atom_gamma]
# NOTE: atom_dPol and atom_om get scaled in loop to stay in one regime and on resonance

# Linear dispersion constants #
preFact   = 0.07
gamma     = atom_gamma
constants = [preFact, gamma]

### no atom ###
atom_params_empty = copy.copy(atom_params)
atom_params_empty[1] = 0.

_, _, S_linDisp_empty = _linearScattering_linearDispersion(k, N, T, atom_params_empty, phase_zero_offset=-k*R0) # R_ref=0.0,

### with atom ###
max_inds = argrelextrema(np.abs(S_linDisp_empty[0,1,:])**2, np.less)
print(max_inds)
atom_params[2] = k[max_inds[0][1]]#-0.5

_, _, S_linDisp = _linearScattering_linearDispersion(k, N, T, atom_params, phase_zero_offset=-k*R0) # R_ref=0.0,

# plot

plt.figure()

plt.axvline(atom_params[2], color='k', dashes=[1,1], lw=1.0)
plt.plot(k, np.abs(S_linDisp[0,1,:])**2)
plt.plot(k, np.abs(S_linDisp_empty[0,1,:])**2, '--')

zoom_dif = 0.05
plt.xlim([atom_params[2]-zoom_dif, atom_params[2]+zoom_dif])

plt.close()

plt.figure()

plt.axvline(atom_params[2], color='k', dashes=[1,1], lw=1.0)
plt.plot(k, np.abs(S_linDisp[0,1,:])**2)
plt.plot(k, np.abs(S_linDisp_empty[0,1,:])**2, '--')

zoom_dif = 0.05
#plt.xlim([atom_params[2]-zoom_dif, atom_params[2]+zoom_dif])

plt.close()

### Green function ###
ns = copy.copy(N)
ds = copy.copy(T)
ns.insert(1, 1.0)
ds.insert(1, 0.001)
ns.insert(-1, 1.0)
ds.insert(-1, 0.001)
print(ns, ds)

Theta = np.pi/2.
green_fn = greenfunction5layers(atom_pos, k, ns, ds)
compLevShift       = prefactor(k, *atom_params, fixed_omTrans=True) * green_fn
compLevShift_fixed = prefactor(k, *atom_params, fixed_omTrans=False) * green_fn

plt.figure()

plt.axvline(atom_params[2], color='k', dashes=[1,1], lw=1.0)
plt.plot(k, np.abs(S_linDisp[0,1,:])**2)
plt.plot(k, np.abs(S_linDisp_empty[0,1,:])**2, '--')

zoom_dif = 0.05
#plt.xlim([atom_params[2]-zoom_dif, atom_params[2]+zoom_dif])

plt.close()

#
plt.figure()

plt.axvline(atom_params[2], color='k', dashes=[1,1], lw=1.0)
plt.axhline(0., color='k', dashes=[1,1], lw=1.0)
plt.plot(k, np.real(compLevShift))
plt.plot(k, -2.*np.imag(compLevShift))
plt.plot(k, np.real(compLevShift_fixed), '--')
plt.plot(k, -2.*np.imag(compLevShift_fixed), '--')

zoom_dif = 0.05
#plt.xlim([atom_params[2]-zoom_dif, atom_params[2]+zoom_dif])
y_u = 0.1
y_l = 0.05
plt.ylim([-y_l, y_u])

plt.close()

### Green's function in complex plane ###
k_re = np.linspace(k[0], k[-1], 500)
k_im = np.linspace(-3.0, 0.0, 501)

Kre, Kim = np.meshgrid(k_re, k_im, indexing='ij')
Kcomp = Kre + 1j*Kim

green_comp        = greenfunction5layers(atom_pos, Kcomp, ns, ds)
compLevShift_comp = prefactor(Kcomp, *atom_params, fixed_omTrans=True) * green_comp
re_ax_idx = find_nearest_idx(k_im, 0.0)

print('Done!')

### Fin poles and residues ###
def comp_lev_shift_for_ML(atom_pos, ns, ds, *atom_params):
    def fun_(kComp):
        GF = greenfunction5layers(atom_pos, kComp, ns, ds)
        result = prefactor(kComp, *atom_params, fixed_omTrans=True) * GF
#         if len(kComp.shape)==2:
#             plt.figure()
#             plt.imshow(np.abs(result.T)**2, origin='lower', aspect='auto',
#                                   norm=matplotlib.colors.LogNorm())
#             plt.colorbar()
#             plt.close()
        return result
    return fun_

### initialize ###
comp_lev_shift = comp_lev_shift_for_ML(atom_pos, ns, ds, *atom_params)
MLexp_compLevShift = ML.MLexpansion(comp_lev_shift)

### find poles ###
k_re_pS = np.linspace(-400.0, 400.0, 20000)
k_im_pS = np.linspace( -2.5,  0.1,  201)

neighborhood_size = 5
threshold         = 0.000001
args = [neighborhood_size, threshold]

#refineParams=None
refineParams=[1000, k_re_pS[2]-k_re_pS[0], 1001, k_im_pS[2]-k_im_pS[0]]

poles_green = MLexp_compLevShift.findPoles(k_re_pS, k_im_pS, *args, refineParams=refineParams, setAttribute=True)

print('Done! (1)')

### find residues ###
radius  = (k_im_pS[1]-k_im_pS[0])
samples = 50000
residues_green = MLexp_compLevShift.findResidues(radius, samples, poles=None, setAttribute=True)

zero_pole_idx = 0

### eval function for test ###
Kre_, Kim_ = np.meshgrid(k_re_pS, k_im_pS, indexing='ij')
Kcomp_ = Kre_+1j*Kim_
compLevShift_comp_MLtest = MLexp_compLevShift.function(Kcomp_)
re_ax_idx_ = find_nearest_idx(k_im_pS, 0.0)

print("Done! (2)")

plt.figure(figsize=(14,10))

#
plt.subplot(211)
plt.axvline(atom_params[2], color='k', dashes=[1,1], lw=1.0)
plt.axhline(0., color='k', dashes=[1,1], lw=1.0)
plt.plot(k, np.real(compLevShift))
plt.plot(k, -2.*np.imag(compLevShift))
# plt.plot(k_re, np.real(compLevShift_comp[:, re_ax_idx]), '--')
# plt.plot(k_re, -2.*np.imag(compLevShift_comp[:, re_ax_idx]), '--')
plt.plot(k_re_pS, np.real(compLevShift_comp_MLtest[:, re_ax_idx_]), '--')
plt.plot(k_re_pS, -2.*np.imag(compLevShift_comp_MLtest[:, re_ax_idx_]), '--')

plt.xlim(k_re_pS[0],k_re_pS[-1])
zoom_dif = 0.05
#plt.xlim([atom_params[2]-zoom_dif, atom_params[2]+zoom_dif])
y_u = 0.02
y_l = 0.01
plt.ylim([-y_l, y_u])

#
plt.subplot(212)
# plt.imshow(np.abs(compLevShift_comp.T)**2, origin='lower', aspect='auto', norm=matplotlib.colors.LogNorm(),
#                                            extent=[k_re[0], k_re[-1], k_im[0], k_im[-1]])
plt.imshow(np.abs(compLevShift_comp_MLtest.T)**2, origin='lower', aspect='auto', norm=matplotlib.colors.LogNorm(),
                                           extent=[k_re_pS[0], k_re_pS[-1], k_im_pS[0], k_im_pS[-1]])
plt.plot(np.real(poles_green), np.imag(poles_green), 'mx')

plt.close()

### Mittag-Leffler ###
zero_pole_idx = find_nearest_idx(poles_green, 0.)
print(zero_pole_idx)
print(zero_pole_idx-2, poles_green[zero_pole_idx-2])
print(zero_pole_idx-1, poles_green[zero_pole_idx-1])
print(zero_pole_idx, poles_green[zero_pole_idx])
print(zero_pole_idx+1, poles_green[zero_pole_idx+1])

poleIdxsSlices = [
                    [zero_pole_idx+1],
                    #[zero_pole_idx],
                    [zero_pole_idx+1,zero_pole_idx+2],
                    np.arange(zero_pole_idx-10,zero_pole_idx+10),
                    np.arange(0,len(poles_green)),
                    #np.arange(1,10),
                    #np.arange(2,10)
                 ]

cArgRef=4.0

green_re_ML_numerical = []
for poleIdxsSlice in poleIdxsSlices:
    ML_single_, G0 = MLexp_compLevShift.MLexpansion(k, poleIdxsSlice, cArgRef=cArgRef, poleIdxsAll=None,
                                                                                    ExcludeConst=True)
    green_re_ML_numerical.append(ML_single_)
    print("G0: {}".format(G0))

### Plot: ML expansion for CLS and Sup ###

xrange=0.02
yrange=50
sc = 0.7

plt.figure(figsize=(14*sc*2,9*sc))

## Lamb shift ##
plt.subplot(121, xlabel='$k$ [arb. units]', ylabel='CLS [arb. units]')
plt.axvline(np.real(poles_green[zero_pole_idx+1]), color='m', dashes=[], linewidth=1)
plt.axvline(atom_params[2], color='k', dashes=[], linewidth=1)
plt.axhline(0., color='k', dashes=[], linewidth=1)
plt.axhline(np.real(G0), color='r', dashes=[], linewidth=1)
plt.plot(k, np.real(compLevShift))
for i_, ML_single_ in enumerate(green_re_ML_numerical):
    plt.plot(k, np.real(ML_single_), '--', label="{}".format(len(poleIdxsSlices[i_])))
plt.ylim([-0.005, 0.005])
plt.xlim([1., 8.])
plt.legend()

plt.subplot(122, xlabel='$k$ [arb. units]', ylabel='CLS [arb. units]')
plt.axhline(-2.*np.imag(G0), color='r', dashes=[], linewidth=1)
plt.plot(k, -2.*np.imag(compLevShift))
for ML_single_ in green_re_ML_numerical:
    plt.plot(k, -2.*np.imag(ML_single_), '--')
plt.ylim([-0.0005, 0.03])
plt.xlim([1., 8.])

plt.tight_layout()
###plt.savefig('plots/InvCav_mode6.pdf', transparent=True, dpi=1200, bbox_inches='tight')
plt.close()

γ0 = atom_params[2]*atom_params[1]**2
# natural linewidth for the 1D case; see also check notebook on free space linewidtbh

### Relabel ###
# Save quantities to plot in comparison under different name so they do not get overwritting in single mode part.
k_MULTI = copy.copy(k)
G0_MULTI = copy.copy(G0)

poles_green_MULTI = copy.copy(poles_green)
atom_params_MULTI = copy.copy(atom_params)

S_linDisp_empty_MULTI = copy.copy(S_linDisp_empty)
S_linDisp_MULTI = copy.copy(S_linDisp)

compLevShift_MULTI = copy.copy(compLevShift)
green_re_ML_numerical_MULTI = copy.copy(green_re_ML_numerical)

poleIdxsSlices_MULTI = copy.copy(poleIdxsSlices)

zero_pole_idx_MULTI = copy.copy(zero_pole_idx)

γ0_MULTI = copy.copy(γ0)

print('Done!')

###################
### single-mode ###
###################

R0 =   1.0  # thickness of single layer, [arb. units]

tSmall = 0.01 #thin layer
N = [1.0,   20.0, 1.0+0.0j, 20.0, 1.0] # Refractive index of each layer
T = [ -1, tSmall, 1.0, tSmall,  -1] # Thicknesses of each layer

k = np.linspace(0.001, 15., 200000)

### Atom params ###
R_ref = -tSmall
# Two-level atom parameters #
atom_pos   =   0.5
# Position of the atom
atom_dPol  =   0.001 #05#0.03  # dipole moment
atom_om    =   0.0   # transition energy
atom_gamma =   0.0  # spontaneous decay rate into non-radiative channels
atom_params = [atom_pos-R_ref, atom_dPol, atom_om, atom_gamma]
# NOTE: atom_dPol and atom_om get scaled in loop to stay in one regime and on resonance

# Linear dispersion constants #
preFact   = 0.07
gamma     = atom_gamma
constants = [preFact, gamma]

### no atom ###
atom_params_empty = copy.copy(atom_params)
atom_params_empty[1] = 0.

_, _, S_linDisp_empty = _linearScattering_linearDispersion(k, N, T, atom_params_empty, phase_zero_offset=-k*R0) # R_ref=0.0,

### with atom ###
max_inds = argrelextrema(np.abs(S_linDisp_empty[0,1,:])**2, np.less)
print(max_inds)
atom_params[2] = k[max_inds[0][1]]#-0.5

_, _, S_linDisp = _linearScattering_linearDispersion(k, N, T, atom_params, phase_zero_offset=-k*R0) # R_ref=0.0,

# plot

plt.figure()

plt.axvline(atom_params[2], color='k', dashes=[1,1], lw=1.0)
plt.plot(k, np.abs(S_linDisp[0,1,:])**2)
plt.plot(k, np.abs(S_linDisp_empty[0,1,:])**2, '--')

zoom_dif = 0.05
plt.xlim([atom_params[2]-zoom_dif, atom_params[2]+zoom_dif])

plt.close()

plt.figure()

plt.axvline(atom_params[2], color='k', dashes=[1,1], lw=1.0)
plt.plot(k, np.abs(S_linDisp[0,1,:])**2)
plt.plot(k, np.abs(S_linDisp_empty[0,1,:])**2, '--')

zoom_dif = 0.05
#plt.xlim([atom_params[2]-zoom_dif, atom_params[2]+zoom_dif])

plt.close()

ns = copy.copy(N)
ds = copy.copy(T)
ns.insert(1, 1.0)
ds.insert(1, 0.001)
ns.insert(-1, 1.0)
ds.insert(-1, 0.001)
print(ns, ds)

Theta = np.pi/2.
green_fn = greenfunction5layers(atom_pos, k, ns, ds)
compLevShift       = prefactor(k, *atom_params, fixed_omTrans=True) * green_fn
compLevShift_fixed = prefactor(k, *atom_params, fixed_omTrans=False) * green_fn

plt.figure()

plt.axvline(atom_params[2], color='k', dashes=[1,1], lw=1.0)
plt.plot(k, np.abs(S_linDisp[0,1,:])**2)
plt.plot(k, np.abs(S_linDisp_empty[0,1,:])**2, '--')

zoom_dif = 0.05
#plt.xlim([atom_params[2]-zoom_dif, atom_params[2]+zoom_dif])

plt.close()

#
plt.figure()

plt.axvline(atom_params[2], color='k', dashes=[1,1], lw=1.0)
plt.axhline(0., color='k', dashes=[1,1], lw=1.0)
plt.plot(k, np.real(compLevShift))
plt.plot(k, -2.*np.imag(compLevShift))
plt.plot(k, np.real(compLevShift_fixed), '--')
plt.plot(k, -2.*np.imag(compLevShift_fixed), '--')

zoom_dif = 0.05
#plt.xlim([atom_params[2]-zoom_dif, atom_params[2]+zoom_dif])
y_u = 0.0015
y_l = 0.0005
plt.ylim([-y_l, y_u])

plt.close()

### Green's function in complex plane ###
k_re = np.linspace(-10, 10, 1000)
k_im = np.linspace(-0.2, 0.2, 1001)

Kre, Kim = np.meshgrid(k_re, k_im, indexing='ij')
Kcomp = Kre + 1j*Kim

green_comp        = greenfunction5layers(atom_pos, Kcomp, ns, ds)
compLevShift_comp = prefactor(Kcomp, *atom_params, fixed_omTrans=True) * green_comp
re_ax_idx = find_nearest_idx(k_im, 0.0)

print('Done!')

plt.figure(figsize=(14,5))

#
plt.subplot(111)
plt.imshow(np.abs(compLevShift_comp.T)**2, origin='lower', aspect='auto', norm=matplotlib.colors.LogNorm(),
                                           extent=[k_re[0], k_re[-1], k_im[0], k_im[-1]])
# plt.imshow(np.abs(compLevShift_comp_MLtest.T)**2, origin='lower', aspect='auto', norm=matplotlib.colors.LogNorm(),
#                                            extent=[k_re_pS[0], k_re_pS[-1], k_im_pS[0], k_im_pS[-1]])

plt.close()

### Find poles and residues ###
def comp_lev_shift_for_ML(atom_pos, ns, ds, *atom_params):
    def fun_(kComp):
        GF = greenfunction5layers(atom_pos, kComp, ns, ds)
        result = prefactor(kComp, *atom_params, fixed_omTrans=True) * GF
#         if len(kComp.shape)==2:
#             plt.figure()
#             plt.imshow(np.abs(result.T)**2, origin='lower', aspect='auto',
#                                   norm=matplotlib.colors.LogNorm())
#             plt.colorbar()
#             plt.close()
        return result
    return fun_

### initialize ###
comp_lev_shift = comp_lev_shift_for_ML(atom_pos, ns, ds, *atom_params)
MLexp_compLevShift = ML.MLexpansion(comp_lev_shift)

### find poles ###
k_re_pS = np.linspace(-10.0, 10.0, 10000)
k_im_pS = np.linspace( -0.02,  0.02,  401)

neighborhood_size = 5
threshold         = 0.000001
args = [neighborhood_size, threshold]

#refineParams=None
refineParams=[1000, k_re_pS[2]-k_re_pS[0], 1001, k_im_pS[2]-k_im_pS[0]]

poles_green = MLexp_compLevShift.findPoles(k_re_pS, k_im_pS, *args, refineParams=refineParams, setAttribute=True)

print('Done! (1)')

### find residues ###
radius  = (k_im_pS[1]-k_im_pS[0])
samples = 50000
residues_green = MLexp_compLevShift.findResidues(radius, samples, poles=None, setAttribute=True)

zero_pole_idx = 0

### eval function for test ###
Kre_, Kim_ = np.meshgrid(k_re_pS, k_im_pS, indexing='ij')
Kcomp_ = Kre_+1j*Kim_
compLevShift_comp_MLtest = MLexp_compLevShift.function(Kcomp_)
re_ax_idx_ = find_nearest_idx(k_im_pS, 0.0)

print("Done! (2)")

plt.figure(figsize=(14,10))

#
plt.subplot(211)
plt.axvline(atom_params[2], color='k', dashes=[1,1], lw=1.0)
plt.axhline(0., color='k', dashes=[1,1], lw=1.0)
plt.plot(k, np.real(compLevShift))
plt.plot(k, -2.*np.imag(compLevShift))
# plt.plot(k_re, np.real(compLevShift_comp[:, re_ax_idx]), '--')
# plt.plot(k_re, -2.*np.imag(compLevShift_comp[:, re_ax_idx]), '--')
plt.plot(k_re_pS, np.real(compLevShift_comp_MLtest[:, re_ax_idx_]), '--')
plt.plot(k_re_pS, -2.*np.imag(compLevShift_comp_MLtest[:, re_ax_idx_]), '--')

plt.xlim(k_re_pS[0],k_re_pS[-1])
zoom_dif = 0.05
#plt.xlim([atom_params[2]-zoom_dif, atom_params[2]+zoom_dif])
y_u = 0.02
y_l = 0.01
plt.ylim([-y_l, y_u])

#
plt.subplot(212)
# plt.imshow(np.abs(compLevShift_comp.T)**2, origin='lower', aspect='auto', norm=matplotlib.colors.LogNorm(),
#                                            extent=[k_re[0], k_re[-1], k_im[0], k_im[-1]])
plt.imshow(np.abs(compLevShift_comp_MLtest.T)**2, origin='lower', aspect='auto', norm=matplotlib.colors.LogNorm(),
                                           extent=[k_re_pS[0], k_re_pS[-1], k_im_pS[0], k_im_pS[-1]])
plt.plot(np.real(poles_green), np.imag(poles_green), 'mx')
plt.plot(-np.real(poles_green), np.imag(poles_green), 'w.')
plt.axhline(0., color='w', dashes=[3,3], linewidth=1)

plt.close()

### Mittag-Leffler ###
zero_pole_idx = find_nearest_idx(poles_green, 0.)
print(zero_pole_idx)
print(zero_pole_idx-2, poles_green[zero_pole_idx-2])
print(zero_pole_idx-1, poles_green[zero_pole_idx-1])
print(zero_pole_idx, poles_green[zero_pole_idx])
print(zero_pole_idx+1, poles_green[zero_pole_idx+1])

poleIdxsSlices = [
                    [zero_pole_idx+1],
                    #[zero_pole_idx],
                    #np.arange(zero_pole_idx-10,zero_pole_idx+10),
                    np.arange(0,len(poles_green)),
                    #np.arange(1,10),
                    #np.arange(2,10)
                 ]

cArgRef=4.0

green_re_ML_numerical = []
for poleIdxsSlice in poleIdxsSlices:
    ML_single_, G0 = MLexp_compLevShift.MLexpansion(k, poleIdxsSlice, cArgRef=cArgRef, poleIdxsAll=None,
                                                                                    ExcludeConst=True)
    green_re_ML_numerical.append(ML_single_)
    print("G0: {}".format(G0))

### Plot: ML expansion for CLS and Sup ###

xrange=0.02
yrange=50
sc = 0.7

plt.figure(figsize=(14*sc*2,9*sc))

## Lamb shift ##
plt.subplot(121, xlabel='$k$ [arb. units]', ylabel='CLS [arb. units]')
plt.axvline(poles_green[zero_pole_idx+1], color='m', dashes=[], linewidth=1)
plt.axvline(atom_params[2], color='k', dashes=[], linewidth=1)
plt.axhline(0., color='k', dashes=[], linewidth=1)
plt.axhline(G0, color='r', dashes=[], linewidth=1)
plt.plot(k, np.real(compLevShift))
for ML_single_ in green_re_ML_numerical:
    plt.plot(k, np.real(ML_single_), '--')
plt.ylim([-0.00015, 0.00015])
plt.xlim([3.15, 3.4])

plt.subplot(122, xlabel='$k$ [arb. units]', ylabel='CLS [arb. units]')
plt.plot(k, -2.*np.imag(compLevShift))
for ML_single_ in green_re_ML_numerical:
    plt.plot(k, -2.*np.imag(ML_single_), '--')
plt.ylim([0.0, 0.00052])
plt.xlim([3.15, 3.4])

plt.tight_layout()
###plt.savefig('plots/InvCav_mode6.pdf', transparent=True, dpi=1200, bbox_inches='tight')
plt.close()

γ0 = atom_params[2]*atom_params[1]**2
# natural linewidth for the 1D case; see also check notebook on free space linewidth

############
### Plot ###
############

scale_y1 = γ0
scale_y2 = γ0_MULTI

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

# matplotlib settings
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Ellipse, Rectangle
import matplotlib.patches as patches

plt.rc('text', usetex=True)
plt.rc('font',**{'family':'serif','serif':['Times New Roman']})
params = {'text.latex.preamble' : [r'\usepackage{amssymb}', r'\usepackage{amsmath}']}
plt.rcParams.update(params)
font_size = 8
legend_font_size = 5.5
#font_size = 12
ylabelpad = 2
xlabelpad = 1
yticks = [2,4,6,8,10]
lw=2
mpl.rcParams.update({'font.size': font_size})

figW = 17.9219/2.
figH = 10
fig = plt.figure(figsize=cm2inch((figW, figH)))
botMarg = 0.1
vBetweenMarg = 0.14
topMarg = 0.05
leftMarg = 0.11
hBetweenMarg = 0.1
rightMarg = 0.022

caxh = 0.05
botMargCax = 0.05

h = (1.0-topMarg-vBetweenMarg-botMarg)/2.0
w = (1.0-leftMarg-hBetweenMarg-rightMarg)/2.0

### axes ###
ax1 = fig.add_axes([leftMarg, botMarg+h+vBetweenMarg, w, h],
                  xlabel=r'', ylabel='', title='Shift contributions')
# ax2 = fig.add_axes([leftMarg+w+hBetweenMarg, botMarg+h+vBetweenMarg, w, h],
#                   xlabel=r'', ylabel='')
ax3 = fig.add_axes([leftMarg+w+hBetweenMarg, botMarg+h+vBetweenMarg, w, h],
                  xlabel=r'$\omega_\mathrm{a}$ [$c/L$]', ylabel=r'$\Delta$ [$\gamma$]',
                   title='Single mode limit')
ax4 = fig.add_axes([leftMarg, botMarg, 2*w+hBetweenMarg, h],
                  xlabel=r'$\omega_\mathrm{a}$ [$c/L$]', ylabel=r'$\Delta$ [$\gamma$]',
                  title='Pole convergence in the multi-mode case')#r'Im$(\omega)$ [arb.~units]''

### ax1 ###
#yunits = mm_quantifiers[:, 5] # then y-axis is: [Gamma_peak]
yunits = 1 # then y-axis is: [c/L]
ax1.axhline(0, color='k', linewidth=0.7)
ax1.plot(barrier_ns, -(mm_quantifiers[:, 6]-np.real(mm_quantifiers[:, 0]))/yunits, 'C5-', label='off-resonant')
ax1.plot(barrier_ns, (mm_quantifiers[:, 3]-np.real(mm_quantifiers[:, 0]))/yunits, 'C6-', label='full resonant shift')
ax1.plot(barrier_ns, (mm_quantifiers[:, 2]-np.real(mm_quantifiers[:, 0]))/yunits, 'C9--', label='complex residue')
ax1.plot(barrier_ns, (mm_quantifiers[:, 3]-mm_quantifiers[:, 2])/yunits, 'C8--', label='multi-pole')
ax1.legend(loc=4, fontsize=legend_font_size)
ax1.set_xlim([1.0, 12])
###ax1.set_ylim([-50, 122])
ax1.set_ylim([-3, 2])
#ax1.set_yticks([0., 1.0])
ax1.set_xticks([1,5,10])
ax1.set_xlabel(r'$n_\mathrm{mirror}$')
ax1.set_ylabel(r'multi-mode shift [c/L]', labelpad=2)
ylabel_coords_1 = [-0.18,0.48]
ax1.get_yaxis().set_label_coords(*ylabel_coords_1)

ax1.tick_params(axis="y",direction="in")
ax1.tick_params(axis="x",direction="in")

### ax3 ###
#scale_factor = 1
scale_factor = (k/atom_params[2])**2

ax3.axvline(atom_params[2], color='grey', dashes=[], linewidth=1, label='')
ax3.axvline(np.real(poles_green[zero_pole_idx+1]), color='m', dashes=[2.5,2.5], linewidth=2) #, label=r'Re$(\tilde{\omega}_1)$'
ax3.axhline(0., color='k', dashes=[], linewidth=1)
#ax3.axhline(np.real(G0), color='r', dashes=[], linewidth=1)
ax3.plot(k, np.real(compLevShift)/scale_y1 * scale_factor, label='full')
for i_, ML_single_ in enumerate(green_re_ML_numerical[0:1]):
    ax3.plot(k, np.real(ML_single_)/scale_y1 * scale_factor, dashes=[2,1], label="{} pole".format(len(poleIdxsSlices[i_])))

ax3.legend(fontsize=legend_font_size, loc=4)
ax3.set_ylim([-0.00014/scale_y1, 0.00014/scale_y1])
ax3.set_xlim([3.15, 3.4])
ylabel_coords = [-0.15, 0.5]
ax3.get_yaxis().set_label_coords(*ylabel_coords)

ax3.tick_params(axis="y",direction="in")
ax3.tick_params(axis="x",direction="in")

ax3.text(3.22, 32, r'$\omega_\mathrm{min}$', fontsize=6, color='grey', ha='center')
ax3.text(3.22, 23, r'Re($\tilde{\omega}_\mathrm{pole}$)', fontsize=6, color='m', ha='center')

# arrow #
style = "Simple, tail_width=0.2, head_width=2.5, head_length=4"
kw = dict(arrowstyle=style, color="grey")
a3 = patches.FancyArrowPatch((3.22, 35), (3.272, 39),
                             connectionstyle="arc3,rad=-.4", **kw)
ax3.add_patch(a3)

# arrow 2 #
style = "Simple, tail_width=0.2, head_width=2.5, head_length=4"
kw = dict(arrowstyle=style, color="m")
a3 = patches.FancyArrowPatch((3.22, 26), (3.27, 28),
                             connectionstyle="arc3,rad=-.3", **kw)
ax3.add_patch(a3)

### ax4 ###

#scale_factor = 1
scale_factor = (k/atom_params_MULTI[2])**2

ax4.axvline(atom_params_MULTI[2], color='grey', dashes=[], linewidth=1)
ax4.axvline(np.real(poles_green_MULTI[zero_pole_idx_MULTI+1]), color='m', dashes=[2.5,2.5], linewidth=1) # , label=r'Re$(\tilde{\omega}_1)$'
ax4.axhline(0., color='k', dashes=[], linewidth=1)
#ax4.axhline(np.real(G0_MULTI), color='r', dashes=[], linewidth=1)
ax4.plot(k_MULTI, np.real(compLevShift_MULTI)/scale_y2 * scale_factor, label='full')
for i_, ML_single_ in enumerate(green_re_ML_numerical_MULTI):
    if i_==0:
        ax4.plot(k_MULTI, np.real(ML_single_)/scale_y2 * scale_factor, dashes=[2,1], label="{} pole".format(len(poleIdxsSlices_MULTI[i_])))
    else:
        ax4.plot(k_MULTI, np.real(ML_single_)/scale_y2 * scale_factor, dashes=[2,1], label="{} poles".format(len(poleIdxsSlices_MULTI[i_])))

ax4.legend(loc=4, fontsize=legend_font_size)
ax4.set_ylim([-0.003/scale_y2, 0.0105/scale_y2])
ax4.set_xlim([1., 8.])
ax4.set_yticks([-0.2,0,0.2,0.4,0.6,0.8])
ylabel_coords4 = [-0.08, 0.5]
ax4.get_yaxis().set_label_coords(*ylabel_coords4)

ax4.tick_params(axis="y",direction="in")
ax4.tick_params(axis="x",direction="in")

ax4.text(4.05, 0.88, r'$\omega_\mathrm{min}$', fontsize=6, color='grey', ha='center')
ax4.text(5.08, 0.88, r'Re($\tilde{\omega}_\mathrm{pole}$)', fontsize=6, color='m', ha='center')

######
xpos_label_a = 0.005
ypos_label_a = 0.97
xpos_label_c = 0.005
xpos_offset = 0.01
plt.gcf().text(xpos_label_a,                            ypos_label_a+0.002, '(a)', fontsize=8)
#plt.gcf().text(xpos_label_a+w+hBetweenMarg+xpos_offset, ypos_label_a, '(b)', fontsize=6)
plt.gcf().text(xpos_label_c+w+hBetweenMarg+xpos_offset, ypos_label_a+0.002, '(b)', fontsize=8, color='k')
plt.gcf().text(xpos_label_c,                            ypos_label_a-h-vBetweenMarg, '(c)', fontsize=8, color='k')

######
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes

axins = ax4.inset_axes([0.02, 0.65, 0.3, 0.3])
#axins = fig.add_axes([leftMarg+w+hBetweenMarg+w*0.1, botMarg+h*2/3, 0.1, 0.1])
axins.axvline(atom_params_MULTI[2], color='grey', dashes=[], linewidth=1) # , label=r'$\omega_\mathrm{a}$'
axins.axvline(np.real(poles_green_MULTI[zero_pole_idx_MULTI+1]), color='m', dashes=[2.5,2.5], linewidth=1)
axins.axhline(0., color='k', dashes=[], linewidth=1)
axins.plot(k_MULTI, np.real(compLevShift_MULTI)/scale_y2 * scale_factor, label='full')
for i_, ML_single_ in enumerate(green_re_ML_numerical_MULTI):
    if i_==0:
        axins.plot(k_MULTI, np.real(ML_single_)/scale_y2, 'C1--', label="{} pole".format(len(poleIdxsSlices_MULTI[i_])))

axins.set_ylim([-0.005/scale_y2/2, 0.0075/scale_y2/1.5])
axins.set_xlim([3.8, 5])
axins.set_xticks([])
axins.set_yticks([])

axins.tick_params(axis="y",direction="in")
axins.tick_params(axis="x",direction="in")

ax4.indicate_inset_zoom(axins, edgecolor="black")

fig.savefig('fig-03.pdf', dpi=1000) #
plt.show()




