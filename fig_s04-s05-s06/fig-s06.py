import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import ticker

import pynuss
import pygreenfn.green_functions as pynuss_gf

# add path
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# import module functions
from pyrot.cavity1d import linear_dispersion_scattering as _linearScattering_linearDispersion
import ML_numerical as ML
import cav_functions as CF

# global constants
keV_to_inv_m  = 1.6022*10**(-16) / (1.0545718*10**(-34) * 2.99792468*10**(8))
neV_to_keV = 10**(-12)
nm = 1e-9
keV_to_inv_nm  = 1.6022*10**(-16) / (1.0545718*10**(-34) * 2.99792468*10**(8)) * nm
c = 299792458 # [m/s]

### Functions ###
#define analytically derived 5-layer GF

# def rij(betai, betaj):
#     return (betai-betaj)/(betai+betaj)

# def beta_j(kpar, k, n):
#     return np.asarray([np.sqrt((k*i)**2-kpar**2) for i in n])

# def greenfunction5layers(z, k, theta, ns, ds, k0): # k=[keV], ds=[1/keV]
#     #bare quantities
#     kpar = k0*np.cos(theta)
#     betas = beta_j(kpar, k, ns)
#     beta0, beta1, beta2, beta3, beta4, beta5, beta6 = betas
#     r01, r12, r23, r34, r45, r56 = rij(betas[0:6], betas[1:7])
#     _, d1, d2, d3, d4, d5, __ = ds
#     phase_ = np.einsum('n,n...->n...', ds[1:6], betas[1:6])
#     e1, e2, e3, e4, e5 = np.exp(2j*phase_)
#     #effective quantities
#     r20 = -(r12+r01*e1)/(1+r12*r01*e1)
#     r30 = (-r23+r20*e2)/(1-r23*r20*e2)
#     r46 = (r45+r56*e5)/(1+r45*r56*e5)
#     r36 = (r34+r46*e4)/(1+r34*r46*e4)
#     D = 1- r30*r36*e3
#     return ((2j*np.pi/beta3)*(1/D)*(1+r36*np.exp(1j*beta3*d3))*
#             (1+r30*np.exp(1j*beta3*d3)))

#define analytically derived 7-layer GF

def r_total_7layers(theta, k, ns, ds, k0):
    # TODO
    return r_total_kp(theta, k, ns[2], ds[1], k0)

def rij(betai, betaj):
    return (betai-betaj)/(betai+betaj)

def beta_j(kpar, k, n):
    return np.asarray([np.sqrt((k*i)**2-kpar**2) for i in n])

def green_to_ClsSup_prefactor(ω, gr):
    dPol, rhoN, fLM = gr._dPol()
    k = ω*keV_to_inv_m
    k0 = eFe.TransitionEnergy*keV_to_inv_m
    #dPol = dPol * np.sqrt(k0/k)**3 # note: this is commented, because we keep the dipole moment constant with energy
    d0_effs = (dPol * np.sqrt(rhoN*fLM*Layers[4].Thickness
                              *mFe.Lattice[0].Element.Abundance))
    #omega_a = eFe.TransitionEnergy # [keV] # note: NOT ω, since we only want to expand Green fn part.
    omega_a = ω
    gamma = mFe.Lattice[0].Element.TransitionWidth * neV_to_keV * keV_to_inv_m 
    unit_factor = keV_to_inv_m**2/gamma /(4.*np.pi)
    prefactor = -omega_a**2 * unit_factor*d0_effs*d0_effs
    return prefactor/keV_to_inv_m

def greenfunction7layers(z, k, theta, ns, ds, k0):
    #bare quantities
    kpar = k0*np.cos(theta)
    betas = beta_j(kpar, k, ns)
    beta0, beta1, beta2, beta3, beta4, beta5, beta6, beta7, beta8 = betas
    r01, r12, r23, r34, r45, r56, r67, r78 = rij(betas[0:8], betas[1:9])
    _, d1, d2, d3, d4, d5, d6,d7,__ = ds
    phase_ = np.einsum('n,n...->n...', ds[1:8], betas[1:8])
    e1, e2, e3, e4, e5, e6, e7 = np.exp(2j*phase_) 
    #effective quantities
    r20 = -(r12+r01*e1)/(1+r12*r01*e1)
    r30 = (-r23+r20*e2)/(1-r23*r20*e2)
    r40 = (-r34+r30*e3)/(1-r34*r30*e3)
    r50 = (-r45+r40*e4)/(1-r45*r40*e4)
    r68 = (r67+r78*e7)/(1+r67*r78*e7)
    r58 = (r56+r68*e6)/(1+r56*r68*e6)
    D = 1- r50*r58*e5
    return ((2j*np.pi/beta5)*(1/D)*(1+r58*np.exp(1j*beta5*d5))*
            (1+r50*np.exp(1j*beta5*d5)))

def r_01_kp(theta, k, n, k0):
    k_par = k0*np.cos(theta)
    kz  = np.sqrt(      k**2 - k_par**2 + 0j)
    kz1 = np.sqrt( n**2*k**2 - k_par**2 + 0j)
    return ( kz - kz1 )/( kz + kz1 )

def r_total_kp(theta, k, n, d, k0):
    kappa     = k*d
    kappa_par = k0*d*np.cos(theta)
    r01 = r_01_kp(theta, k, n, k0)
    kappaz1 = np.sqrt( n**2*kappa**2 - kappa_par**2 )
    a = np.exp( 2j * kappaz1 )
    return (a-r01)/(r01*a-1)

#######################
### Setup and grids ###
#######################

### function setup ###

r_ = r_total_7layers
G_ = greenfunction7layers

### pynuss ###
Beam = pynuss.Beam()
Detector = pynuss.Detector(Beam)

mPt = pynuss.Material.fromSymbol('Pt')
mC = pynuss.Material.fromSymbol('C')
eFe = pynuss.ResonantElement.fromTemplate('Fe57')
mFe = pynuss.Material.fromElement(eFe)
mSi = pynuss.Material.fromSymbol('Si')
m56Fe = pynuss.Material.fromSymbol('Fe')

#pynuss Layer structure
nm = 1e-9
#cavity layout -- don't change number of layers
Layers = [pynuss.Layer(mPt,    3.0 * nm),
          pynuss.Layer(mC,     3.5 * nm),
          pynuss.Layer(m56Fe,  3.0 * nm),
          pynuss.Layer(mC,     7.5 * nm),
          #pynuss.Layer(m56Fe,  1.0 * nm),
          pynuss.Layer(mFe,    3.0 * nm),
          #pynuss.Layer(m56Fe,  1.0 * nm),
          pynuss.Layer(mC,    27.0 * nm),
          pynuss.Layer(mPt,   10.0 * nm),
          pynuss.Layer(mSi, -1)]

gr = pynuss_gf.GrazingIncidence(Beam, Detector, Layers)

### parameters ###

theta  = gr.ReflectionMinimum(6)*1e-3

thetas = 1.0e-03 * np.linspace(3.8, 4.5, 200)

eFe = pynuss.ResonantElement.fromTemplate('Fe57')
mFe = pynuss.Material.fromElement(eFe)
Fe_resEnergy = mFe.Lattice[0].Element.TransitionEnergy # [keV]
k0Fe57 = Fe_resEnergy*keV_to_inv_m # [1/m] resonant wavevector

n, d = pynuss_gf.gr_to_NT(gr, mFe)
n, d = np.array(n), np.array(d)*keV_to_inv_nm

kpar = Fe_resEnergy*np.cos(theta) # [keV]

offset = 0.0001
k_re = np.linspace(kpar+offset, kpar+offset+0.0002, 3000) # [keV]
k_im = np.linspace(-0.0001,  0.0001,  3001) # [keV]
k_im2 = np.linspace(-0.00002,  0.00002,  3001) # [keV]

detuning_unit_scale = 1e6
Detuning = (k_re-Fe_resEnergy)*detuning_unit_scale # [meV]

K_re, K_im = np.meshgrid(k_re, k_im) # [keV]
Θs, K_im2 = np.meshgrid(thetas, k_im2)

###################
### Check evals ###
###################

rtot_kp_re = r_(theta, k_re,         n, d, Fe_resEnergy)
rtot_kp    = r_(theta, K_re+1j*K_im, n, d, Fe_resEnergy)

rtot_angle_re = r_(thetas, Fe_resEnergy, n, d, Fe_resEnergy)
rtot_angle    = r_(thetas, Fe_resEnergy+1j*K_im2, n, d, Fe_resEnergy)

compLevShift_re = G_(0.5*d, k_re, theta, n, d, Fe_resEnergy)
compLevShift    = G_(0.5*d, K_re+1j*K_im, theta, n, d, Fe_resEnergy)

cls =     np.real(compLevShift_re)
sup = -2.*np.imag(compLevShift_re)

green_to_ClsSup_pref  = green_to_ClsSup_prefactor(Fe_resEnergy, gr)

print('Done!')

compLevShift_res_check, _ = gr.EffectiveLevelScheme(theta*1e3, subensembles=None)
cls_res_check =     np.real(compLevShift_res_check[0])
sup_res_check = -2.*np.imag(compLevShift_res_check[0])

print(compLevShift_res_check)

print(n, d/keV_to_inv_nm)
rocking_energy = CF.r_i_j(0, len(n)-1, n, d/keV_to_inv_nm, theta*1e3, k_re, Fe_resEnergy, pol='s')
# rocking_energy_res = CF.r_i_j(0, len(n)-1, n, d/keV_to_inv_nm, theta*1e3, Fe_resEnergy, Fe_resEnergy, pol='s')

# rocking_energy_res_check = gr.ReflectionCoefficientFromGreen(theta*1e3)

# print(rocking_energy_res)
# print(rocking_energy_res_check)
# print(rocking_energy_res-rocking_energy_res_check)

print('Done!')

#########################
### Pynuss comparison ###
#########################

# rocking
rocking_pynuss = gr.ReflectionIntensity(thetas*1e3)

# find first min
main_mode = 6
Theta0 = gr.ReflectionMinimum(main_mode)

# nuclear spectrum in first min
spectrum_nuc_pynuss = gr.ReflectionIntensity(Theta0, Detuning)

# field distribution
Depth0, Field = gr.FieldIntensity(Theta0)

# # 2D nuclear spectrum
# spectrum_nuc_2D_pynuss = np.empty((len(dTheta), len(Detuning)))
# for i, th in enumerate(Theta0+dTheta):
#     spectrum_nuc_2D_pynuss[i, :] = gr.ReflectionIntensity(th, Detuning)

print('Done!')

#############################################
### ML expansion of CLS and superradiance ###
#############################################

def green_function_for_ML(theta, n, d, k0):
    def fun_(kComp):
        return G_(0.5*d, kComp, theta, n, d, Fe_resEnergy)
    return fun_

### initialize ###
green_fun = green_function_for_ML(theta, n, d, Fe_resEnergy)
MLexp_green = ML.MLexpansion(green_fun)

### find poles ###
k_re_pS = np.linspace(kpar,        kpar+0.02, 12000) # [keV]
k_im_pS = np.linspace(    -0.002,       0.0001, 3001) # [keV]

neighborhood_size = 5
threshold         = 2
args = [neighborhood_size, threshold]

#refineParams=None
refineParams=[1000, k_re_pS[1]-k_re_pS[0], 1001, k_im_pS[1]-k_im_pS[0]]

poles_green = MLexp_green.findPoles(k_re_pS, k_im_pS, *args, refineParams=refineParams, setAttribute=True)

### find residues ###
radius  = (k_im_pS[1]-k_im_pS[0])
samples = 50000
residues_green = MLexp_green.findResidues(radius, samples, poles=None, setAttribute=True)

zero_pole_idx = 0

print("Done!")

#############################################
### ML expansion with varying pole number ###
#############################################

poleIdxsSlices = [
                    [5],
                    [4,5,6],
                    #np.arange(0,40),
                    np.arange(0,69)
                 ]

green_re_ML_numerical = []
for poleIdxsSlice in poleIdxsSlices:
    ML_single_, G0 = MLexp_green.MLexpansion(k_re, poleIdxsSlice, cArgRef=kpar+0.j, poleIdxsAll=None,
                                                                                    ExcludeConst=False)
    green_re_ML_numerical.append(ML_single_)
    print("G0: {}".format(G0))

single_pole = residues_green[5]/(k_re - poles_green[5])
zero_loc = np.real(poles_green[5]) + np.imag(poles_green[5])*np.imag(residues_green[5])/np.real(residues_green[5])

print("G0: {}".format(G0))
print('Done!')

### Plot ###
xrange=0.02*detuning_unit_scale/1e3
yrange=50
sc = 0.7
scale_y = green_to_ClsSup_pref
#scale_y = 1

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
figH = 12
fig = plt.figure(figsize=cm2inch((figW, figH)))
botMarg = 0.07
vBetweenMarg = 0.07
topMarg = 0.01
leftMarg = 0.1
hBetweenMarg = 0.1
rightMarg = 0.02

caxh = 0.05
botMargCax = 0.05

h = (1.0-topMarg-3*vBetweenMarg-botMarg)/4.0
w = (1.0-leftMarg-0*hBetweenMarg-rightMarg)/1.0

### axes ###
ax1 = fig.add_axes([leftMarg, botMarg+3.*(h+vBetweenMarg), w, h],
                  xlabel=r'Depth [nm]', ylabel=r'Field at $\theta_6$')
ax2 = fig.add_axes([leftMarg, botMarg+2.*(h+vBetweenMarg), w, h],
                  xlabel=r'$\hbar\omega - \hbar\omega^{[\mathrm{physical}]}_\mathrm{nuc}$ [meV]', ylabel=r'Cavity reflectance')
ax3 = fig.add_axes([leftMarg, botMarg+1.*(h+vBetweenMarg), w, h],
                  xlabel=r'$\hbar\omega_\mathrm{nuc} - \hbar\omega^{[\mathrm{physical}]}_\mathrm{nuc}$ [meV]', ylabel=r'$\Delta_\mathrm{LS}$ [$\gamma$]')
ax4 = fig.add_axes([leftMarg, botMarg+0.*(h+vBetweenMarg), w, h],
                  xlabel=r'$\hbar\omega_\mathrm{nuc} - \hbar\omega^{[\mathrm{physical}]}_\mathrm{nuc}$ [meV]', ylabel=r'$\Gamma$ [$\gamma$]')

### ax1 ###
ax1.plot(Depth0 / nm, Field, 'C0', label='field at $\omega_0$')

# Layer boundaries
Thicknesses = pynuss_gf.gr_to_NT(gr, mFe)[1][1:-1]
print(Thicknesses)
for i in np.arange(len(Thicknesses)):
    color_='red'
    alpha_ = 0.3
    str_ = ''
    if str(Layers[i].Material) == '<Material {B, C}>': #B4C
        color_ = 'gray'
        str_ = 'B4C'
    elif str(Layers[i].Material) == '<Material Si>':
        color_ = 'gray'
        alpha_ = 0.2
        str_ = 'Si'
    elif str(Layers[i].Material) == '<Material C>': #B4C
        alpha_ = 0.4
        color_='yellow'
        str_ = 'C'
    elif str(Layers[i].Material) == '<Material Pt>':   #Pt
        color_='red'
        str_ = 'Pt'
    elif str(Layers[i].Material) == '<Material Fe>': #B4C
        color_='black'
        str_ = r'$^{56}$Fe'
        if str(Layers[i].Material.Lattice[0].Element) == '<ResonantElement Fe>': #B4C
            str_ = r'$^{57}$Fe'
            alpha_ = 0.7
            color_='magenta'
    if i in [3,5,6]:
        str_ = None
    if i==len(Thicknesses)-1:
        ax1.axvspan(np.sum(Thicknesses[0:i]), Depth0[-1]/nm,
                    alpha=alpha_, color=color_, linewidth=0, label=str_)
    else:
        ax1.axvspan(np.sum(Thicknesses[0:i]), np.sum(Thicknesses[0:i+1]),
                    alpha=alpha_, color=color_, linewidth=0, label=str_)

ax1.set_xlim([Depth0[0]/nm, Depth0[-1]/nm])
ax1.legend(fontsize=5, loc=2)
ax1.tick_params(axis="x",direction="in")
ax1.tick_params(axis="y",direction="in")

ylabel_coords = [-0.08,0.5]
ax1.get_yaxis().set_label_coords(*ylabel_coords)

### ax2 ###
ax2.axvline(0., color='k', dashes=[1,1], linewidth=1, label='nuclear resonance')
ax2.axvline((np.real(poles_green[5])-Fe_resEnergy)*detuning_unit_scale, color='m', dashes=[2,1], linewidth=1,
                                                                        label='closest cavity pole')
ax2.plot(Detuning, np.abs(rocking_energy)**2)

ax2.set_xlim([-xrange,+xrange])
ax2.set_ylim([0,1])
ax2.set_yticks([0,1])
ax2.set_yticklabels([0,1])
ax2.tick_params(axis="x",direction="in")
ax2.tick_params(axis="y",direction="in")
ax2.legend(fontsize=5, loc=1)
ax2.get_yaxis().set_label_coords(*ylabel_coords)

### ax3 ###
ax3.axvline(0., color='k', dashes=[1,1], linewidth=1)
ax3.axvline((np.real(poles_green[5])-Fe_resEnergy)*detuning_unit_scale, color='m', dashes=[2,1], linewidth=1)#, label='pole'
#ax3.axvline((zero_loc-Fe_resEnergy)*detuning_unit_scale, color='r', dashes=[1,1], linewidth=1, label='predicted CLS zero')
#ax3.axhline(cls_res_check, color='g', dashes=[1,1], linewidth=1, label='resonant pynuss') # this should intersect the superradiance at zero detuning

ax3.axhline(0, color='k', dashes=[], linewidth=1)
#
ax3.plot(Detuning, cls*scale_y, linewidth=2, linestyle='-',label='full')
for i in range(len(poleIdxsSlices)):
    if i==0:
        ax3.plot(Detuning, np.real(green_re_ML_numerical[i])*scale_y, '--',
                 linewidth=2, label="{} pole".format(len(poleIdxsSlices[i])))
    else:
        ax3.plot(Detuning, np.real(green_re_ML_numerical[i])*scale_y, '--',
                     linewidth=2, label="{} poles".format(len(poleIdxsSlices[i])))

ax3.set_xlim([-xrange,+xrange])
ax3.set_ylim([-1,1.8])
ax3.legend(fontsize=5, loc=2, handlelength=3)
ax3.tick_params(axis="x",direction="in")
ax3.tick_params(axis="y",direction="in")
ax3.get_yaxis().set_label_coords(*ylabel_coords)

### ax4 ###
ax4.axvline(0., color='k', dashes=[1,1], linewidth=1)
ax4.axvline((np.real(poles_green[5])-Fe_resEnergy)*detuning_unit_scale, color='m', dashes=[2,1], linewidth=1)#, label='pole'
#ax4.axvline((zero_loc-Fe_resEnergy)*detuning_unit_scale, color='r', dashes=[1,1], linewidth=1, label='predicted CLS zero')
#ax4.axhline(sup_res_check, color='g', dashes=[1,1], linewidth=1, label='resonant pynuss') # this should intersect the superradiance at zero detuning
#
ax4.plot(Detuning, sup*scale_y, linewidth=2, linestyle='-',label='full')
for i in range(len(poleIdxsSlices)):
    if i==0:
        ax4.plot(Detuning, -2.*np.imag(green_re_ML_numerical[i])*scale_y, '--',
                 linewidth=2, label="{} pole".format(len(poleIdxsSlices[i])))
    else:
        ax4.plot(Detuning, -2.*np.imag(green_re_ML_numerical[i])*scale_y, '--',
                 linewidth=2, label="{} poles".format(len(poleIdxsSlices[i])))

ax4.set_xlim([-xrange,+xrange])
ax4.set_ylim([0,6.5])
ax4.legend(fontsize=5, loc=1, handlelength=3)
ax4.tick_params(axis="x",direction="in")
ax4.tick_params(axis="y",direction="in")
ax4.get_yaxis().set_label_coords(*ylabel_coords)

######
xpos_label_a = 0.005
ypos_label_a = 0.97
xpos_offset = 0.01
#plt.gcf().text(xpos_label_a,                            ypos_label_a, '(a)', fontsize=6)
#plt.gcf().text(xpos_label_a+w+hBetweenMarg+xpos_offset, ypos_label_a, '(b)', fontsize=6)
#plt.gcf().text(xpos_label_a,                            ypos_label_a-h-vBetweenMarg, '(c)', fontsize=6, color='k')
#plt.gcf().text(xpos_label_a+w+hBetweenMarg+xpos_offset, ypos_label_a-h-vBetweenMarg, '(d)', fontsize=6, color='k')

fig.savefig('fig-s06.pdf') #, dpi=1000
plt.show()

