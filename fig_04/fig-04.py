from datetime import datetime
startTime = datetime.now()

import matplotlib
import matplotlib.pyplot as plt
import pynuss
import numpy as np
from ipywidgets import interactive

Beam = pynuss.Beam()
Detector = pynuss.Detector(Beam)

eFe = pynuss.ResonantElement.fromTemplate('Fe57')
mFe = pynuss.Material.fromElement(eFe)
mPt = pynuss.Material.fromSymbol('Pt')
mPd = pynuss.Material.fromSymbol('Pd')
mC  = pynuss.Material.fromSymbol('C')
mB4C = pynuss.Material.fromChemicalFormula('B4C', 2100)
mSi  = pynuss.Material.fromSymbol('Si')
m56Fe = pynuss.Material.fromSymbol('Fe')

Theta = np.linspace(1.0, 8.0, 500)  # mrad
k_im     = np.linspace(-2.0, 0.0, 300) # keV

Modes = [4,6]
Detunings = [
    np.linspace(-15, 15, 200),
    np.linspace(-5,   5, 200)
]
dThetas_zooms = [
    np.linspace(-0.2, 0.2, 200),
    np.linspace(-0.2, 0.2, 200),
]

nm = 1e-9

n = len(Modes)
m = 7

theta_dashes = []
theta_dashes_2D = [3, 3]
theta_lw = 2.0
rmin_col = 'C3'
qzero_col = 'C2'
clszero_col = 'C9'
plot_lw = 3.0
spec_dashes = [5,5]
theta_dashes = []

### cavity structure ###
Layers = [pynuss.Layer(mPt,    3.0 * nm),
          pynuss.Layer(mC,     3.5 * nm),
          pynuss.Layer(m56Fe,  3.0 * nm),
          pynuss.Layer(mC,     7.5 * nm),
          pynuss.Layer(m56Fe,  1.0 * nm),
          pynuss.Layer(mFe,    1.0 * nm),
          pynuss.Layer(m56Fe,  1.0 * nm),
          pynuss.Layer(mC,    27.0 * nm),
          pynuss.Layer(mPt,   10.0 * nm),
          pynuss.Layer(mSi, -1)]

Thicknesses = np.empty(len(Layers)) #[nm]
for i,l in enumerate(Layers):
    Thicknesses[i] = l.Thickness/nm
    
gr = pynuss.GrazingIncidence(Beam, Detector, Layers)

# Rocking curve
Rocking = gr.ReflectionIntensity(Theta)

# Rocking min
Mode_4 = 4
Theta_min_4 = gr.ReflectionMinimum(Mode_4,stepsize=0.005)
Mode_6 = 6
Theta_min_6 = gr.ReflectionMinimum(Mode_6,stepsize=0.005)

# Field intensity
Depth_4, Field_4 = gr.FieldIntensity(Theta_min_4)
Depth_6, Field_6 = gr.FieldIntensity(Theta_min_6)

# Spectrum
Spectrum_4 = gr.ReflectionIntensity(Theta_min_4, Detunings[0])
Spectrum_6 = gr.ReflectionIntensity(Theta_min_6, Detunings[1])

############
### Plot ###
############

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

figW = 17.9219/1.9
figH = 7
fig = plt.figure(figsize=cm2inch((figW, figH)))
botMarg = 0.1
hBetweenMarg = 0.06
topMarg = 0.02
leftMarg = 0.09
vBetweenMarg = 0.12
rightMarg = 0.01
h = (1.0-topMarg-vBetweenMarg-botMarg)/2.0
w = (1.0-leftMarg-1*hBetweenMarg-rightMarg)/2.0

### axes ###
ax3 = fig.add_axes([leftMarg, botMarg+h+vBetweenMarg, w, h],
                  xlabel='Depth [nm]', ylabel='Field')
ax = fig.add_axes([leftMarg+w+hBetweenMarg, botMarg+h+vBetweenMarg, w, h],
                  xlabel=r'$\theta$ [mrad]', ylabel='Reflectance')
#ax2 = fig.add_axes([leftMarg+2*(w+hBetweenMarg), botMarg+h+vBetweenMarg, w, h],
#                  xlabel=r'$\Delta\,[\gamma]$', ylabel=r'')
ax1 = fig.add_axes([leftMarg, botMarg, w, h],
                  xlabel=r'$\omega-\omega_\mathrm{nuc}\,[\gamma_\mathrm{nuc}]$', ylabel='Reflectance')
ax2 = fig.add_axes([leftMarg+w+hBetweenMarg, botMarg, w, h],
                  xlabel=r'$\omega-\omega_\mathrm{nuc}\,[\gamma_\mathrm{nuc}]$', ylabel='')
#ax5 = fig.add_axes([leftMarg+2*(w+hBetweenMarg), botMarg, w, h],
#                  xlabel=r'$\theta$ [mrad]', ylabel=r'[peak superradiance]')

### axes ###
# ax = fig.add_axes([leftMarg, botMarg+h+vBetweenMarg, w, h],
#                   xlabel=r'$\theta$ [mrad]', ylabel='Reflectance')
# ax1 = fig.add_axes([leftMarg+w+hBetweenMarg, botMarg+h+vBetweenMarg, w, h],
#                   xlabel=r'$\Delta\,[\gamma]$', ylabel='')
# ax2 = fig.add_axes([leftMarg, botMarg, w, h],
#                   xlabel=r'$\Delta\,[\gamma]$', ylabel=r'$\theta-\theta_0$ [mrad]')
# ax3 = fig.add_axes([leftMarg+w+hBetweenMarg, botMarg, w, h],
#                   xlabel=r'Mode number', ylabel=r'CLS [$\gamma$]')

#cax1 = fig.add_axes([leftMarg+2*w+hBetweenMarg+cBetweenMarg+cW, botMarg+h+vBetweenMarg, cW, h])
#cax2 = fig.add_axes([leftMarg+2*w+hBetweenMarg+cBetweenMarg+cW, botMarg, cW, h])

###

# rocking
ax.plot(Theta, Rocking, 'C4')
ax.axvline(Theta_min_4, color='C2', dashes=[4,1],linewidth=1.0)
ax.axvline(Theta_min_6, color='C3', dashes=[4,1],linewidth=1.0)
ax.set_xlim([Theta[0], Theta[-1]])
ax.set_ylim([0.0,1.1])
plt.text(0.515, 0.9, r'$\theta_4$', horizontalalignment='center', verticalalignment='center',
         transform=ax.transAxes, color='C2')
plt.text(0.7, 0.9, r'$\theta_6$', horizontalalignment='center', verticalalignment='center',
         transform=ax.transAxes, color='C3')
ax.set_yticks([0,1])
ax.yaxis.labelpad = 0.01
ax.xaxis.labelpad = 1

# spectrum 1
ax1.plot(Detunings[0], Spectrum_4, 'C2', label=r'Spectrum at $\theta_4$')
ax1.axvline(0.0, color='k', dashes=[4,1], linewidth=1)
ax1.set_xlim([Detunings[0][0], Detunings[0][-1]])
ax1.set_ylim([0., 0.75])
ax1.legend(fontsize=5, loc='lower center')
ax1.annotate("", xy=(-2, 0.67), xytext=(0.3, 0.67), arrowprops=dict(arrowstyle="->",color='C2'))
plt.text(0.47, 0.95, r"$\Delta_4$", horizontalalignment='right', verticalalignment='center',
         transform=ax1.transAxes, color='C2')

# spectrum 2
ax2.plot(Detunings[1], Spectrum_6, 'C3', label=r'Spectrum at $\theta_6$')
ax2.axvline(0.0, color='k', dashes=[4,1], linewidth=1)
#ax2.set_xlim([props_list[1]['Detuning'][0], props_list[1]['Detuning'][-1]])
ax2.set_xlim([-3,3])
ax2.set_ylim([0., 0.43])
ax2.legend(fontsize=5, loc='lower center')
ax2.annotate("", xy=(0.4, 0.38), xytext=(-0.07, 0.38), arrowprops=dict(arrowstyle="->",color='C3'))
plt.text(0.53, 0.95, r"$\Delta_6$", horizontalalignment='left', verticalalignment='center',
         transform=ax2.transAxes, color='C3')

# Field
ax3.plot(Depth_4 / nm, Field_4, 'C2')
ax3.plot(Depth_6 / nm, Field_6, 'C3')

# Layer boundaries
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
        alpha_ = 0.2
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
    if i in [3,4,6,7,8]:
        str_ = None
    if i==len(Thicknesses)-1:
        ax3.axvspan(np.sum(Thicknesses[0:i]), Depth_4[-1]/nm,
                    alpha=alpha_, color=color_, linewidth=0, label=str_)
    else:
        ax3.axvspan(np.sum(Thicknesses[0:i]), np.sum(Thicknesses[0:i+1]),
                    alpha=alpha_, color=color_, linewidth=0, label=str_)

ax3.set_xlim([Depth_4[0]/nm, Depth_4[-1]/nm])
firstCavThickness = np.sum(Thicknesses[0:4])
ax3.legend(fontsize=4, loc=2)
ax3.xaxis.labelpad = 1


plot_lw = 1.5

for _ax in [ax,ax1,ax2,ax3]:
    _ax.tick_params(direction='in')
    #_ax.set_xticks([-40,-20,0,20,40])
    #_ax.yaxis.labelpad = ylabelpad
    #_ax.xaxis.labelpad = xlabelpad

tx_rel = -0.12
ty_rel =  1.02

lims_x = ax1.get_xlim()
lims_y = ax1.get_ylim()
tx = lims_x[0] + (lims_x[1]-lims_x[0])*tx_rel
ty = lims_y[0] + (lims_y[1]-lims_y[0])*ty_rel
ax1.text(tx, ty, r'(c)', color='k')

lims_x = ax2.get_xlim()
lims_y = ax2.get_ylim()
tx = lims_x[0] + (lims_x[1]-lims_x[0])*tx_rel
ty = lims_y[0] + (lims_y[1]-lims_y[0])*ty_rel
ax2.text(tx, ty, r'(d)', color='k')

tx_rel = -0.12
ty_rel =  0.985

lims_x = ax.get_xlim()
lims_y = ax.get_ylim()
tx = lims_x[0] + (lims_x[1]-lims_x[0])*tx_rel
ty = lims_y[0] + (lims_y[1]-lims_y[0])*ty_rel
ax.text(tx, ty, r'(b)', color='k')

lims_x = ax3.get_xlim()
lims_y = ax3.get_ylim()
tx = lims_x[0] + (lims_x[1]-lims_x[0])*tx_rel
ty = lims_y[0] + (lims_y[1]-lims_y[0])*ty_rel
ax3.text(tx, ty, r'(a)', color='k')

fig.savefig('fig-04.pdf', dpi=1000)
plt.show()