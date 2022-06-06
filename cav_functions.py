import copy

import sys
import time

import warnings

# math
import math
import numpy as np
import numpy.linalg

# pynuss
import pynuss
from pynuss.common import isNumeric

# integrator for ordinary differential equations
from scipy.integrate import ode, complex_ode

# Fourier transforms
from scipy.fftpack import fft, ifft, fftshift, fftfreq

################################################################################

### helper functions ###

def find_nearest_idx(array, value):
    """
    Find the closest element in an array and return the corresponding index.
    """
    array = np.asarray(array)
    idx = (np.abs(array-value)).argmin()
    return idx

def progressBar(value, endvalue, bar_length=20):
    """
    Progress bar for loops
    """
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()

### conversion factors ###

keV_to_inv_m  = 1.6022*10**(-16) / (1.0545718*10**(-34) * 2.99792468*10**(8))
neV_to_keV = 10**(-12)
nm = 1.0e-09
fs = 1.0e-15
keV_to_inv_nm  = 1.6022*10**(-16) / (1.0545718*10**(-34) * 2.99792468*10**(8)) * nm
c0 = 299792458.0 # [m/s] # speed of light

### conversion functions###

def j_from_z(z, gr): # z in [nm]
    """
    Convert depth from cavity surface into layer index + depth from layer surface.

    Returns the layer index j and depth from the layer boundary z-z_j
    given the total depth z and a layer system gr.
        - j=0 corresponds to the first layer (vacuum in pynuss) where z<0.
          The distance to the upper layer boundary is not defined in this case
          and given as -z (TODO: check that field formula applies in this region)
        - j=1 is the first layer, with layer boundary position z_1 = 0
        - j=2 is the second layer, with layer boundary position z_2 = t_1
          (t_1: thickness of the first layer)
        - j>2 is treated analogously.
    """
    Thicknesses = gr_to_Thicknesses(gr)
    if z<0.:
        return 0, z
    if z==0.:
        return 1, z
    for j, t in enumerate(Thicknesses[0:-1]):
        if ( sum(Thicknesses[0:j]) < z ) and ( sum(Thicknesses[0:j+1]) >= z):
            return j+1, z-np.sum(Thicknesses[0:j])
    return j+2, z-np.sum(Thicknesses[0:j+1]) # returns index and sum of layer thicknesses above in [nm]

def gr_to_NT(gr, omega, omega0):
    """
    Gives a pynuss-independent list representation of the off-resonant layer
    properties.

    Returns (N, T), containing a list of refractive indices N and thicknesses T
    for the layer system.
    Unlike in pynuss, the uppermost layer is explicitly included. Uppermost and
    lowermost layer are taken to have thickness -1.
    """
    N = [1.] # initialize with vacuum on the outside
    T = [-1] # initialize with vacuum on the outside
    for layer in gr.Layer:
        N.append(layer.Material.RefractiveIndex(omega0))
        T.append(layer.Thickness/nm) # [nm]
    if not (gr.Layer[-1].Thickness == -1):
        # if last layer is not a substrate, pynuss includes a vacuum substrate
        # by default
        N.append(1.)
        T.append(-1)
    return N, T

def gr_to_Thicknesses(gr):
    Thicknesses = np.empty(len(gr.Layer)) #[nm]
    for i,l in enumerate(gr.Layer):
        Thicknesses[i] = l.Thickness/nm
    if not (gr.Layer[-1].Thickness == -1):
        Thicknesses_ = np.empty(len(gr.Layer)+1)
        Thicknesses_[0:-1] = Thicknesses
        Thicknesses_[-1] = -1
        Thicknesses = Thicknesses_
    return Thicknesses

def find_res_layer_idx(gr):
    """
    Finds the resonant layers.

    Returns list of resonant layer indices and the resonant isotope material.
    Note: pynuss only supports a single resonant element in the system.
    """
    l_idx_list = []
    for i_, layer in enumerate(gr.Layer):
        if isinstance(layer.Material.Lattice[0].Element, pynuss.ResonantElement):
            l_idx_list.append(i_)
            ResIso = layer.Material
    return l_idx_list, ResIso

### Tomas1995 functions###
def Εs_0(z, gr, Theta, omega, omega0):
    Field = []
    N, T = gr_to_NT(gr, omega, omega0)
    for i,zi in enumerate(z):
        n = len(N)-1
        j, z_offset = j_from_z(zi, gr)
        betaj = beta_j(j, N, T, Theta, omega, omega0)
        dj = gr.Layer[j-1].Thickness/nm # [nm]
        if j==0 or j==(len(N)-1):
            dj = 0.
        rs_j0 = r_i_j(j, 0, N, T, Theta, omega, omega0, pol='s') # = rs_j-
        rs_jn = r_i_j(j, n, N, T, Theta, omega, omega0, pol='s') # = rs_j+
        ts_0j = t_i_j(0, j, N, T, Theta, omega, omega0, pol='s')
        Dsj = 1. - rs_j0 * rs_jn * np.exp(2j*betaj*dj)
        zm = z_offset
        zp = dj - z_offset
        Field.append(ts_0j*np.exp(1j*betaj*dj)/Dsj * ( np.exp(-1j*betaj*zp) +  rs_jn*np.exp(+1j*betaj*zp) ))
    return z, np.asarray(Field)

def Εs_0_forward_backward(z, gr, Theta, omega, omega0):
    Field_forward  = []
    Field_backward = []
    N, T = gr_to_NT(gr, omega, omega0)
    for i,zi in enumerate(z):
        n = len(N)-1
        j, z_offset = j_from_z(zi, gr)
        betaj = beta_j(j, N, T, Theta, omega, omega0)
        dj = gr.Layer[j-1].Thickness/nm # [nm]
        if j==0 or j==(len(N)-1):
            dj = 0.
        rs_j0 = r_i_j(j, 0, N, T, Theta, omega, omega0, pol='s') # = rs_j-
        rs_jn = r_i_j(j, n, N, T, Theta, omega, omega0, pol='s') # = rs_j+
        ts_0j = t_i_j(0, j, N, T, Theta, omega, omega0, pol='s')
        Dsj = 1. - rs_j0 * rs_jn * np.exp(2j*betaj*dj)
        zm = z_offset
        zp = dj - z_offset
        Field_forward.append(ts_0j*np.exp(1j*betaj*dj)/Dsj * np.exp(-1j*betaj*zp) )
        Field_backward.append(ts_0j*np.exp(1j*betaj*dj)/Dsj * rs_jn*np.exp(+1j*betaj*zp) )
        print(i/len(z))
    return z, np.asarray(Field_forward), np.asarray(Field_backward)

def time_env_forward_backward(t, x, z, pulse_fn_time, gr, ThetaIn, omega0, t_s_conv=fs/nm): # t [fs], x,z [nm]
    c0_ = c0*t_s_conv
    N, T = gr_to_NT(gr, omega0, omega0)
    envs_forward  = []
    envs_backward = []
    for i,zi in enumerate(z):
        j, z_offset = j_from_z(zi, gr)
        dj = gr.Layer[j-1].Thickness/nm # [nm]
        if j==0 or j==(len(N)-1):
            dj = 0.
        z_forward  = z_offset
        z_backward = 2.*dj - z_offset
        betaj_0cen = beta_j(j, N, T, ThetaIn, omega0, omega0) # [1/nm]
        #
        t_shifted_forward  = t - x/c0_ * np.cos(ThetaIn*1e-3) - z_forward/c0_  * np.sqrt( N[j]**2*-(np.cos(ThetaIn*1e-3))**2 + 0.j )
        t_shifted_backward = t - x/c0_ * np.cos(ThetaIn*1e-3) - z_backward/c0_ * np.sqrt( N[j]**2*-(np.cos(ThetaIn*1e-3))**2 + 0.j )
        pulse_env_forward  = pulse_fn_time(t_shifted_forward)
        pulse_env_backward = pulse_fn_time(t_shifted_backward)
        phase_fac_forward  = np.exp(-1j*betaj_0cen*z_forward)
        phase_fac_backward = np.exp(-1j*betaj_0cen*z_backward)
        #
        envs_forward.append(pulse_env_forward*phase_fac_forward)
        envs_backward.append(pulse_env_backward*phase_fac_backward)
    return np.asarray(envs_forward), np.asarray(envs_backward)

def Εs_0_forward_backward2(z, gr, Theta, omega, omega0):
    Field_forward  = []
    Field_backward = []
    N, T = gr_to_NT(gr, omega, omega0)
    n = len(N)-1
    betajs = []
    rs_j0s = []
    rs_jns = []
    ts_0js = []
    Dsjs   = []
    for j_, n_ in enumerate(N):
        print(j_/len(N))
        betaj = beta_j(j_, N, T, Theta, omega, omega0)
        rs_j0 = r_i_j(j_, 0, N, T, Theta, omega, omega0, pol='s')
        rs_jn = r_i_j(j_, n, N, T, Theta, omega, omega0, pol='s')
        betajs.append(betaj)
        rs_j0s.append(rs_j0)
        rs_jns.append(rs_jn)
        ts_0js.append(t_i_j(0, j_, N, T, Theta, omega, omega0, pol='s'))
        dj = gr.Layer[j_-1].Thickness/nm # [nm]
        if j_==0 or j_==(len(N)-1):
            dj = 0.
        Dsjs.append(1. - rs_j0 * rs_jn * np.exp(2j*betaj*dj))
    for i,zi in enumerate(z):
        ###print(i/len(z))
        j, z_offset = j_from_z(zi, gr)
        dj = gr.Layer[j-1].Thickness/nm # [nm]
        if j==0 or j==(len(N)-1):
            dj = 0.
        betaj, rs_j0, rs_jn, ts_0j, Dsj = betajs[j], rs_j0s[j], rs_jns[j], ts_0js[j], Dsjs[j]
        zm = z_offset
        zp = dj - z_offset
        Field_forward.append(ts_0j*np.exp(1j*betaj*dj)/Dsj * np.exp(-1j*betaj*zp) )
        Field_backward.append(ts_0j*np.exp(1j*betaj*dj)/Dsj * rs_jn*np.exp(+1j*betaj*zp) )
    return z, np.asarray(Field_forward), np.asarray(Field_backward)

def beta_j(j, N, T, Theta, omega, omega0):
    ### omega = ResIsotope.TransitionEnergy # [keV]
    k = omega*keV_to_inv_nm # [1/nm]
    k0 = omega0*keV_to_inv_nm # [1/nm]
    k_parallel = k0*np.cos(Theta/1000.) # [1/nm]
    # ϵj_re = np.real(N[j]**2)
    # ϵj_im = np.imag(N[j]**2)
    # betaj_re = np.sqrt(0.5 * ( np.sqrt((ϵj_re*k**2 - k_parallel**2)**2 + (ϵj_im*k**2)**2) + (ϵj_re*k**2 - k_parallel**2)) )
    # betaj_im = np.sqrt(0.5 * ( np.sqrt((ϵj_re*k**2 - k_parallel**2)**2 + (ϵj_im*k**2)**2) - (ϵj_re*k**2 - k_parallel**2)) )
    # return betaj_re + 1j*betaj_im
    betaj = np.sqrt(N[j]**2*k**2-k_parallel**2)
    return betaj # [1/nm]


def D_j_i_k(j, i, k, N, T, Theta, omega, omega0, pol='s'):
    betaj = beta_j(j, N, T, Theta, omega, omega0)
    dj = T[j] # [m] TODO: units
    rj_i = r_i_j(j, i, N, T, Theta, omega, omega0, pol=pol)
    rj_k = r_i_j(j, k, N, T, Theta, omega, omega0, pol=pol)
    return 1. - rj_i*rj_k*np.exp(2.j*betaj*dj)

def gamma_ij(i, j, N, T, Theta, omega, omega0, pol='s'):
    ### single interface, abs(i-j)=1 ###
    if not (np.abs(i-j) == 1):
        raise ValueError('Not adjacent layers, gamma_ij not defined.')
    if pol=='s':
        return 1.+0.j
    ϵi = N[i]**2
    ϵj = N[j]**2
    return ϵi/ϵj

def r_ij(i, j, N, T, Theta, omega, omega0, pol='s'):
    if not (np.abs(i-j) == 1):
        raise ValueError('Not adjacent layers, r_ij not defined.')
    betai = beta_j(i, N, T, Theta, omega, omega0)
    betaj = beta_j(j, N, T, Theta, omega, omega0)
    gammaij = gamma_ij(i, j, N, T, Theta, omega, omega0, pol=pol)
    return (betai - gammaij*betaj)/(betai + gammaij*betaj)

def t_ij(i, j, N, T, Theta, omega, omega0, pol='s'):
    if not (np.abs(i-j) == 1):
        raise ValueError('Not adjacent layers, t_ij not defined.')
    gammaij = gamma_ij(i, j, N, T, Theta, omega, omega0, pol=pol)
    rij = r_ij(i, j, N, T, Theta, omega, omega0, pol=pol)
    return np.sqrt(gammaij)*(1. + rij)

def r_i_j_k(i, j, k, N, T, Theta, omega, omega0, pol='s'):
    ### recurrence relation ###
    betaj = beta_j(j, N, T, Theta, omega, omega0)
    dj = T[j] # [m] TODO: units
    Dj = D_j_i_k(j, i, k, N, T, Theta, omega, omega0, pol=pol)
    ri_j = r_i_j(i, j, N, T, Theta, omega, omega0, pol=pol)
    rj_i = r_i_j(j, i, N, T, Theta, omega, omega0, pol=pol)
    rj_k = r_i_j(j, k, N, T, Theta, omega, omega0, pol=pol)
    ti_j = t_i_j(i, j, N, T, Theta, omega, omega0, pol=pol)
    tj_i = t_i_j(j, i, N, T, Theta, omega, omega0, pol=pol)
    return 1./Dj * ( ri_j + (ti_j*tj_i - ri_j*rj_i) * rj_k * np.exp(2j*betaj*dj) )

def t_i_j_k(i, j, k, N, T, Theta, omega, omega0, pol='s'):
    ### recurrence relation ###
    betaj = beta_j(j, N, T, Theta, omega, omega0)
    dj = T[j] # [m] TODO: units
    Dj = D_j_i_k(j, i, k, N, T, Theta, omega, omega0, pol=pol)
    ti_j = t_i_j(i, j, N, T, Theta, omega, omega0, pol=pol)
    tj_k = t_i_j(j, k, N, T, Theta, omega, omega0, pol=pol)
    return 1./Dj * ti_j*tj_k * np.exp(1j*betaj*dj)

def r_i_j(i, j, N, T, Theta, omega, omega0, pol='s'):
    ### starts and ends the recurrence chain ###
    if np.abs(i-j) == 1:
        return r_ij(i, j, N, T, Theta, omega, omega0, pol=pol)
    if i==j:
        return 0.+0.j
    # choose middle index to start recurrence chain #
    if i>j:
        k=i-1
    else:
        k=i+1
    return r_i_j_k(i, k, j, N, T, Theta, omega, omega0, pol=pol)

def t_i_j(i, j, N, T, Theta, omega, omega0, pol='s'):
    ### starts and ends the recurrence chain ###
    if np.abs(i-j) == 1:
        return t_ij(i, j, N, T, Theta, omega, omega0, pol=pol)
    if i==j:
        return 1.+0.j
    # choose middle index to start recurrence chain #
    if i>j:
        k=i-1
    else:
        k=i+1
    return t_i_j_k(i, k, j, N, T, Theta, omega, omega0, pol=pol)

def GF(z, z0, gr, Theta, omega, omega0, pol='s'):
    N, T = gr_to_NT(gr, omega, omega0)
    xip = 1
    xis = -1
    if pol == 'p':
        xiq = xip
    else:
        xiq = xis
    n = len(N)-1
    betan = beta_j(n, N, T, Theta, omega, omega0)
    ts_0n = t_i_j(0, n, N, T, Theta, omega, omega0)
    # only single pol (s):
    zs,Es0_1 = Εs_0(z, gr, Theta, omega, omega0)
    zs,Esn_1 = Εs_n(z0, gr, Theta, omega, omega0)
    zs,Es0_2 = Εs_0(z0, gr, Theta, omega, omega0)
    zs,Esn_2 = Εs_n(z, gr, Theta, omega, omega0)
    Z0, Z = np.meshgrid(z0, z) # note reversed order for consistency with np.outer
    heavi_1 = np.heaviside(np.real(Z-Z0), 0.5)
    heavi_2 = np.heaviside(np.real(Z0-Z), 0.5)
    return 2j*np.pi/betan * xis/ts_0n * ( np.outer(Es0_1, Esn_1)*heavi_1 + np.outer(Esn_2, Es0_2)*heavi_2 ) # [TODO: units]

