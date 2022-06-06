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

# import module functions
from cavity1d import linear_dispersion_scattering as _linearScattering_linearDispersion #phaseAngle
import ML_numerical as ML

def parratt_maxwell1D_matrix_compK(N0, D0, kRange, phaseOfHell=None):
    """Input format:
                    - N0: [N0, N1, N2, ..., NN, NN+1]; where N0,NN+1 is the space to the left/right, respectively and the rest are the layers
                    - D0: [-1, D1, D2, ..., DN, -1]; left/right space do not have thicknesses, rest is layers
                    - kRange: just a 1D array/list of k-values to compute the spectra on
         NOTE: The elements of N0 can either be numbers or arrays. The latter corresponds to an energy dependence.
    """
    if not all(isinstance(x, float) for x in N0):
        return parratt_maxwell1D_matrix_eDep(N0, D0, kRange, phaseOfHell=None)
    N = np.asarray(N0, dtype=np.complex128)
    D = np.asarray(D0, dtype=np.complex128)
    k = np.asarray(kRange, dtype=np.complex128)
    transfer_matrix_tot = np.asarray([[np.ones_like(k), np.zeros_like(k)],
                                      [np.zeros_like(k), np.ones_like(k)]])
    for p in np.arange(len(N)-1):
        n0 = N[p]
        n1 = N[p+1]
        kZ0 = k*np.sqrt(n0**2)
        kZ1 = k*np.sqrt(n1**2)
        r01 = (kZ0-kZ1)/(kZ0+kZ1)
        t01 = 2.0*np.sqrt(kZ0*kZ1)/(kZ0+kZ1)
        transfer_matrix_interface = np.asarray([[1.0/t01, r01/t01],
                                                                                        [r01/t01, 1.0/t01]])
        if p<(len(N)-2):
            d1 =  D[p+1]
            phi1 = kZ1*d1
            transfer_matrix_layer = np.asarray([[np.exp(-1j*phi1), np.zeros_like(phi1)],
                                                                                    [np.zeros_like(phi1), np.exp(1j*phi1)]])
            transfer_matrix_tot = np.einsum('ijk...,jlk...,lmk...->imk...', transfer_matrix_tot,
                                                                                                                    transfer_matrix_interface,
                                                                                                                    transfer_matrix_layer)
        else:
            transfer_matrix_tot = np.einsum('ijk...,jlk...->ilk...', transfer_matrix_tot,
                                                                                                            transfer_matrix_interface)
    if not (phaseOfHell is None):
        transfer_to_hell = np.asarray([[np.exp(-1j*phaseOfHell), np.zeros_like(phi1)],
                                                                     [np.zeros_like(phi1), np.exp(1j*phaseOfHell)]])
        transfer_matrix_tot = np.einsum('ijk...,jlk...->ilk...', transfer_matrix_tot,
                                                                                                        transfer_to_hell)
    R1 = transfer_matrix_tot[1,0,:]/transfer_matrix_tot[0,0]
    R2 = -transfer_matrix_tot[0,1,:]/transfer_matrix_tot[0,0]
    T1 = 1.0/transfer_matrix_tot[0,0]
    T2 = -transfer_matrix_tot[1,0,:]*transfer_matrix_tot[0,1,:]/transfer_matrix_tot[0,0,:] \
             + transfer_matrix_tot[1,1,:]
    return np.asarray([[T1, R2], [R1, T2]])

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

### params ###
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

### without atom ###
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
print(re_ax_idx)

print('Done!')

### Reflection in complex plane ###
S_linDisp_comp = parratt_maxwell1D_matrix_compK(N, T, Kcomp, phaseOfHell=-Kcomp*R0)

print('Done!')

###############################
### Find poles and residues ###
###############################

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
k_re_pS = np.linspace(0.0, 14.0, 5000)
k_im_pS = np.linspace( -2.5,  0.0,  201)

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
print(re_ax_idx_)
print(k_im_pS[re_ax_idx_])

print("Done! (2)")

plt.figure(figsize=(14,10))

#
plt.subplot(211)
plt.axvline(atom_params[2], color='k', dashes=[1,1], lw=1.0)
plt.axhline(0., color='k', dashes=[1,1], lw=1.0)
plt.plot(k, np.real(compLevShift))
plt.plot(k, -2.*np.imag(compLevShift))
plt.plot(k_re, np.real(compLevShift_comp[:, re_ax_idx]), '--')
plt.plot(k_re, -2.*np.imag(compLevShift_comp[:, re_ax_idx]), '--')
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
plt.xlim([k_re_pS[0], k_re_pS[-1]])

plt.close()

plt.figure(figsize=(14,10))

#
plt.subplot(211)
plt.axvline(atom_params[2], color='k', dashes=[1,1], lw=1.0)
plt.axhline(0., color='k', dashes=[1,1], lw=1.0)
plt.plot(k, np.abs(S_linDisp_empty[0,1,:])**2)
plt.plot(k_re, np.abs(S_linDisp_comp[0,1,:, re_ax_idx])**2, '--')
plt.xlim([k_re[0], k_re[-1]])
plt.ylim([0,1])

#
plt.subplot(212)
# plt.imshow(np.abs(compLevShift_comp.T)**2, origin='lower', aspect='auto', norm=matplotlib.colors.LogNorm(),
#                                            extent=[k_re[0], k_re[-1], k_im[0], k_im[-1]])
plt.imshow(np.abs(S_linDisp_comp[0,1, :, :].T)**2, origin='lower', aspect='auto', norm=matplotlib.colors.LogNorm(),
                                           extent=[k_re[0], k_re[-1], k_im[0], k_im[-1]])
plt.plot(np.real(poles_green), np.imag(poles_green), 'mx')
plt.xlim([k_re[0], k_re[-1]])
plt.ylim([k_im[0], k_im[-1]])

plt.close()

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
font_size = 6
legend_font_size = 5
#font_size = 12
ylabelpad = 2
xlabelpad = 1
yticks = [2,4,6,8,10]
lw=2
mpl.rcParams.update({'font.size': font_size})

figW = 17.9219/2.
figH = 8
fig = plt.figure(figsize=cm2inch((figW, figH)))
botMarg = 0.2
vBetweenMarg = 0.0
topMarg = 0.02 * 7/figH
leftMarg = 0.061
hBetweenMarg = 0.07
rightMarg = 0.014

caxh = 0.05
botMargCax = 0.05

h = (1.0-topMarg-vBetweenMarg-botMarg)/2.0
w = (1.0-leftMarg-hBetweenMarg-rightMarg)/2.0

### axes ###
ax1 = fig.add_axes([leftMarg, botMarg+h+vBetweenMarg, w, h],
                  xlabel=r'Re$(\omega)$ [arb.~units]', ylabel='Reflectance')
ax2 = fig.add_axes([leftMarg+w+hBetweenMarg, botMarg+h+vBetweenMarg, w, h],
                  xlabel=r'Re$(\omega)$ [arb.~units]', ylabel='Level shift [arb.~units]')
ax3 = fig.add_axes([leftMarg, botMarg, w, h],
                  xlabel=r'Re$(\omega)$ [$c/L$]', ylabel=r'Im$(\omega)$ [$c/L$]')
ax4 = fig.add_axes([leftMarg+w+hBetweenMarg, botMarg, w, h],
                  xlabel=r'Re$(\omega_\mathrm{test})$ [$c/L$]', ylabel='Im$(\omega_\mathrm{test})$ [$c/L$]')#r'Im$(\omega)$ [arb.~units]''

cax  = fig.add_axes([leftMarg,                botMargCax, w, caxh])
cax2 = fig.add_axes([leftMarg+w+hBetweenMarg, botMargCax, w, caxh])

### ax1 ###
#ax1.axvline(atom_params[2], color='C3', dashes=[3,1], lw=1.0)
#ax1.axhline(0., color='k', dashes=[1,1], lw=1.0)

#ax1.plot(k, np.abs(S_linDisp_empty[0,1,:])**2)
ax1.plot(k_re, np.abs(S_linDisp_comp[0,1,:, re_ax_idx])**2, 'C1', label='no atom')#, '--'
ax1.plot(k, np.abs(S_linDisp[0,1,:])**2, 'C0--', label='with atom')

ax1.legend(fontsize=4, loc=1, handlelength=3)
ax1.set_xlim([k_re[0], k_re[-1]])
ax1.set_ylim([0,1])

ax1.set_xticks([0.0, 5, 10, 15])
ax1.set_xticklabels([])
ax1.set_yticks([0, 1])
ax1.set_yticklabels([0, 1])
ax1.tick_params(axis="y",direction="in")
ax1.tick_params(axis="x",direction="in")

ylabel_coords = [-0.08,0.5]
ax1.get_yaxis().set_label_coords(*ylabel_coords)

### ax2 ###
ax2.axvline(atom_params[2], color='C0', dashes=[3,1], lw=1.0)
ax2.axhline(0., color='k',  lw=1.0)#dashes=[1,1],
ax2.plot(k_re, np.real(compLevShift_comp[:, re_ax_idx]), 'C2', label=r'Re$[\tilde{\delta}(\omega_\mathrm{test})]$')
ax2.plot(k_re, -np.imag(compLevShift_comp[:, re_ax_idx]), 'C3', dashes=[6,1],
                                                               label=r'Im$[\tilde{\delta}(\omega_\mathrm{test})]$')
# ax2.plot(k_re_pS, np.real(compLevShift_comp_MLtest[:, re_ax_idx_]), '--')
# ax2.plot(k_re_pS, -np.imag(compLevShift_comp_MLtest[:, re_ax_idx_]), '--')

ax2.legend(fontsize=4, handlelength=3)
ax2.set_xlim([k_re[0], k_re[-1]])
y_u =  0.0125
y_l = -0.005
ax2.set_ylim([y_l,y_u])
ax2.get_yaxis().set_label_coords(*ylabel_coords)

ax2.tick_params(axis="y",direction="in")
ax2.tick_params(axis="x",direction="in")
ax2.set_xticks([0.0, 5, 10, 15])
ax2.set_xticklabels([])
ax2.set_yticks([0])
ax2.set_yticklabels([0])

### ax3 ###
im = ax3.imshow(np.abs(S_linDisp_comp[0,1, :, :].T)**2, origin='lower', aspect='auto', norm=matplotlib.colors.LogNorm(),
                                           extent=[k_re[0], k_re[-1], k_im[0], k_im[-1]])
ax3.plot(np.real(poles_green), np.imag(poles_green), 'm.', markersize=2,
                                label=r'poles of $\tilde{\delta}$')

ylim_lower=-2.0
ax3.set_xlim([k_re[0], k_re[-1]])
ax3.set_ylim([ylim_lower, k_im[-1]])
ax3.get_yaxis().set_label_coords(*ylabel_coords)

ax3.set_xticks([0.0, 5, 10, 15])
ax3.set_xticklabels([0, 5, 10, 15])
ax3.set_yticks([-2,0])
ax3.set_yticklabels([-2,''])
ax3.tick_params(axis="y",direction="in")
ax3.tick_params(axis="x",direction="in")
ax3.legend(fontsize=4, loc=4)

fig.colorbar(im, cax=cax, orientation='horizontal')
cax.tick_params(axis="x",direction="in")

### ax4 ###

im2 = ax4.imshow(np.abs(compLevShift_comp.T)**2, origin='lower', aspect='auto', norm=matplotlib.colors.LogNorm(),
                                           extent=[k_re[0], k_re[-1], k_im[0], k_im[-1]])
ax4.plot(np.real(poles_green), np.imag(poles_green), 'm.', markersize=2,
                                 label=r'poles of $\tilde{\delta}$')

ax4.legend(fontsize=4, loc=4)
ax4.set_xlim([k_re[0], k_re[-1]])
ax4.set_ylim([ylim_lower, k_im[-1]])
ax4.get_yaxis().set_label_coords(*ylabel_coords)

ax4.set_xticks([0.0, 5, 10, 15])
ax4.set_xticklabels([0, 5, 10, 15])
ax4.set_yticks([-2, 0])
ax4.set_yticklabels([-2, 0])
ax4.tick_params(axis="y",direction="in")
ax4.tick_params(axis="x",direction="in")

fig.colorbar(im2, cax=cax2, orientation='horizontal')
cax2.tick_params(axis="x",direction="in")

######
xpos_label_a = 0.066
ypos_label_a = 0.95
plt.gcf().text(xpos_label_a,                ypos_label_a, '(a)', fontsize=6)
plt.gcf().text(xpos_label_a+w+hBetweenMarg, ypos_label_a, '(c)', fontsize=6)
plt.gcf().text(xpos_label_a,                ypos_label_a-h-vBetweenMarg, '(b)', fontsize=6, color='w')
plt.gcf().text(xpos_label_a+w+hBetweenMarg, ypos_label_a-h-vBetweenMarg, '(d)', fontsize=6, color='w')

ax1.text(2.5, 0.45, "atomic \n peak", fontsize=6, color='C0', ha='center')
ax2.text(3.2, 0.0033, r'$\omega_\mathrm{a}$', fontsize=6, color='C0', ha='center')

# arrow for label
style = "Simple, tail_width=0.2, head_width=2.5, head_length=4"
kw = dict(arrowstyle=style, color="C0")
a3 = patches.FancyArrowPatch((2.5, 0.55), (4.3, 0.7),
                             connectionstyle="arc3,rad=-.5", **kw)
ax1.add_patch(a3)

style = "Simple, tail_width=0.2, head_width=2.5, head_length=4"
kw = dict(arrowstyle=style, color="C0")
a3 = patches.FancyArrowPatch((3.2, 0.004), (4.3, 0.006),
                             connectionstyle="arc3,rad=-.5", **kw)
ax2.add_patch(a3)

# Draw arc with arrow.
from matplotlib.patches import Arc
x0, y0 = np.real(poles_green[2]), np.imag(poles_green[2])
radius = 0.5
angle = 360
x_broad = 15./2. * 0.9
angle_rad = angle * np.pi / 180  # degrees to radians
arc_radius = radius / 4
arc = Arc((x0, y0),
          arc_radius*2*x_broad, arc_radius*2,  # ellipse width and height
          theta1=0, theta2=angle, linestyle='solid', linewidth=0.5, color='w')
ax4.add_patch(arc)
arc_arrow_length = 0.03
arc_arrow_dx = arc_arrow_length * np.cos(angle_rad + np.pi / 2)
arc_arrow_dy = arc_arrow_length * np.sin(angle_rad + np.pi / 2)
ax4.arrow(
    x0 + x_broad * arc_radius * np.cos(angle_rad) - arc_arrow_dx,
    y0 + arc_radius * np.sin(angle_rad) - arc_arrow_dy,
    # We want to define a vector,
    # but we don't want to draw any line besides arrow head,
    # so we make arrow "body" unnoticeable.
    -arc_arrow_dx * 0.00000001,
    +arc_arrow_dy * 0.00000001,
    head_width=0.07,
    head_length=0.02,
    color='w')

fig.savefig('fig-s01.pdf', dpi=1000) #
plt.show()








