#%% VSM muestras C y F de Pablo Tancredi - Julio 2025
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import os
from sklearn.metrics import r2_score 
from mlognormfit import fit3
from mvshtools import mvshtools as mt
import re
from uncertainties import ufloat
#%%
def lineal(x,m,n):
    return m*x+n

def coercive_field(H, M):
    """
    Devuelve los valores de campo coercitivo (Hc) donde la magnetización M cruza por cero.
    
    Parámetros:
    - H: np.array, campo magnético (en A/m o kA/m)
    - M: np.array, magnetización (en emu/g)
    
    Retorna:
    - hc_values: list de valores Hc (puede haber más de uno si hay múltiples cruces por cero)
    """
    H = np.asarray(H)
    M = np.asarray(M)
    hc_values = []

    for i in range(len(M)-1):
        if M[i]*M[i+1] < 0:  # Cambio de signo indica cruce por cero
            # Interpolación lineal entre (H[i], M[i]) y (H[i+1], M[i+1])
            h1, h2 = H[i], H[i+1]
            m1, m2 = M[i], M[i+1]
            hc = h1 - m1 * (h2 - h1) / (m2 - m1)
            hc_values.append(hc)

    return hc_values
#%% Levanto Archivos
data_C1 = np.loadtxt(os.path.join('data','C1.txt'), skiprows=12)
H_C1 = data_C1[:, 0]  # Gauss
m_C1 = data_C1[:, 1]  # emu

data_C2 = np.loadtxt(os.path.join('data','C2.txt'), skiprows=12)
H_C2 = data_C2[:, 0]  # Gauss
m_C2 = data_C2[:, 1]  # emu

data_C3 = np.loadtxt(os.path.join('data','C3.txt'), skiprows=12)
H_C3 = data_C3[:, 0]  # Gauss
m_C3 = data_C3[:, 1]  # emu

data_C4 = np.loadtxt(os.path.join('data','C4.txt'), skiprows=12)
H_C4 = data_C4[:, 0]  # Gauss
m_C4 = data_C4[:, 1]  # emu

data_F1 = np.loadtxt(os.path.join('data','F1.txt'), skiprows=12)
H_F1 = data_F1[:, 0]  # Gauss
m_F1 = data_F1[:, 1]  # emu

data_F2 = np.loadtxt(os.path.join('data','F2.txt'), skiprows=12)
H_F2 = data_F2[:, 0]  # Gauss
m_F2 = data_F2[:, 1]  # emu

data_F3 = np.loadtxt(os.path.join('data','F3.txt'), skiprows=12)
H_F3 = data_F3[:, 0]  # Gauss
m_F3 = data_F3[:, 1]  # emu

data_F4 = np.loadtxt(os.path.join('data','F4.txt'), skiprows=12)
H_F4 = data_F4[:, 0]  # Gauss
m_F4 = data_F4[:, 1]  # emu


#%% PLOTEO ALL
fig, (a,b) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, sharey=True, constrained_layout=True)

# Arriba: muestras C
a.plot(H_C1, m_C1, '.-', label='C1 raw')
a.plot(H_C2, m_C2, '.-', label='C2 raw')
a.plot(H_C3, m_C3, '.-', label='C3 raw')
a.plot(H_C4, m_C4, '.-', label='C4 raw')
a.set_ylabel('m (emu)')
a.legend()
a.grid()
a.set_title('Muestras C')

# Abajo: muestras F
b.plot(H_F1, m_F1, '.-', label='F1 raw')
b.plot(H_F2, m_F2, '.-', label='F2 raw')
b.plot(H_F3, m_F3, '.-', label='F3 raw')
b.plot(H_F4, m_F4, '.-', label='F4 raw')
b.set_ylabel('m (emu)')
b.set_xlabel('H (G)')
b.legend()
b.grid()
b.set_title('Muestras F')

plt.show()
#%% Normalizo por masa de la muestra y ploteo
masa_C1 = 0.1121-0.0613  # g
masa_C2 = 0.1142-0.0618  # g
masa_C3 = 0.1061-0.0549  # g
masa_C4 = 0.1005-0.0500  # g
masa_F1 = 0.1082-0.0580  # g
masa_F2 = 0.1194-0.0585  # g
masa_F3 = 0.1010-0.0489  # g   
masa_F4 = 0.1192-0.0691  # g

Concentracion_mm=10/1000 #uso densidad del H2O 1000 g/L

m_C1_norm = m_C1 / masa_C1 / Concentracion_mm
m_C2_norm = m_C2 / masa_C2 / Concentracion_mm
m_C3_norm = m_C3 / masa_C3 / Concentracion_mm
m_C4_norm = m_C4 / masa_C4 / Concentracion_mm

m_F1_norm = m_F1 / masa_F1 / Concentracion_mm
m_F2_norm = m_F2 / masa_F2 / Concentracion_mm
m_F3_norm = m_F3 / masa_F3 / Concentracion_mm
m_F4_norm = m_F4 / masa_F4 / Concentracion_mm

fig, (a,b) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, sharey=True, constrained_layout=True)

# Arriba: muestras C
a.plot(H_C1, m_C1_norm, '.-', label='C1')
a.plot(H_C2, m_C2_norm, '.-', label='C2')
a.plot(H_C3, m_C3_norm, '.-', label='C3')
a.plot(H_C4, m_C4_norm, '.-', label='C4')
a.set_ylabel('m (emu/g)')
a.legend()
a.grid()
a.set_title('Muestras C')

# Abajo: muestras F
b.plot(H_F1, m_F1_norm, '.-', label='F1')
b.plot(H_F2, m_F2_norm, '.-', label='F2')
b.plot(H_F3, m_F3_norm, '.-', label='F3')
b.plot(H_F4, m_F4_norm, '.-', label='F4')
b.set_ylabel('m (emu/g)')
b.set_xlabel('H (G)')
b.legend()
b.grid()
b.set_title('Muestras F')
plt.savefig('VSM_muestras_C_F.png', dpi=300)
plt.show()
#%% Curvas Anhistéricas y fit para todas las muestras C y F
resultados_fit = {}
H_fit_arrays = {}
m_fit_arrays = {}

for nombre, H, m in [
    ('C1', H_C1, m_C1_norm), ('C2', H_C2, m_C2_norm),
    ('C3', H_C3, m_C3_norm), ('C4', H_C4, m_C4_norm),
    ('F1', H_F1, m_F1_norm), ('F2', H_F2, m_F2_norm),
    ('F3', H_F3, m_F3_norm), ('F4', H_F4, m_F4_norm),
]:
    H_anhist, m_anhist = mt.anhysteretic(H, m)
    fit = fit3.session(H_anhist, m_anhist, fname=nombre, divbymass=False)
    fit.fix('sig0')
    fit.fix('mu0')
    fit.free('dc')
    fit.fit()
    fit.update()
    fit.free('sig0')
    fit.free('mu0')
    fit.set_yE_as('sep')
    fit.fit()
    fit.update()
    fit.save()
    fit.print_pars()
    # Obtengo la contribución lineal usando los parámetros del fit
    C = fit.params['C'].value
    dc = fit.params['dc'].value
    linear_contrib = lineal(fit.X, C, dc)
    m_fit_sin_lineal = fit.Y - linear_contrib
    m_saturacion = ufloat(np.mean([max(m_fit_sin_lineal),-min(m_fit_sin_lineal)]), np.std([max(m_fit_sin_lineal),-min(m_fit_sin_lineal)]))
    resultados_fit[nombre]={'H_anhist': H_anhist,
                            'm_anhist': m_anhist,
                            'H_fit': fit.X,
                            'm_fit': fit.Y,
                            'm_fit_sin_lineal': m_fit_sin_lineal,
                            'linear_contrib': linear_contrib,
                            'Ms':m_saturacion,
                            'fit': fit}
    
    H_fit_arrays[nombre] = fit.X
    m_fit_arrays[nombre] = m_fit_sin_lineal

#%% Ploteo VSM normalizado y fitting para cada muestra individualmente
muestras = [
    ('C1', H_C1, m_C1_norm),
    ('C2', H_C2, m_C2_norm),
    ('C3', H_C3, m_C3_norm),
    ('C4', H_C4, m_C4_norm),
    ('F1', H_F1, m_F1_norm),
    ('F2', H_F2, m_F2_norm),
    ('F3', H_F3, m_F3_norm),
    ('F4', H_F4, m_F4_norm),]

for idx, (nombre, H, m_norm) in enumerate(muestras):
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    
    ax.plot(H, m_norm, '.-', label='VSM normalizado')
    ax.plot(H_fit_arrays[nombre], m_fit_arrays[nombre], '-', label='Fitting')
    
    # Ms con error a 2 cifras significativas
    Ms = resultados_fit[nombre]['Ms']
    # Formateo con 2 cifras significativas en el error
    Ms_str = f"Ms = {Ms:.1uS} emu/g"
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.75, 0.5, Ms_str, transform=ax.transAxes, fontsize=12,
            va='center',ha='center', bbox=props)
    
    ax.grid()
    ax.legend()
    ax.set_xlabel('H (G)')
    ax.set_ylabel('m (emu/g)')
    plt.suptitle('VSM normalizado y fitting '+nombre)
    plt.savefig(f'VSM_normalizado_vs_fit_{nombre}.png', dpi=300)
    plt.show()

#plt.savefig('VSM_normalizado_vs_fit_por_muestra.png', dpi=300)
#%% Ploteo fits 

muestras_C = [
    ('C1', H_C1, m_C1_norm),
    ('C2', H_C2, m_C2_norm),
    ('C3', H_C3, m_C3_norm),
    ('C4', H_C4, m_C4_norm),]
muestras_F = [
    ('F1', H_F1, m_F1_norm),
    ('F2', H_F2, m_F2_norm),
    ('F3', H_F3, m_F3_norm),
    ('F4', H_F4, m_F4_norm),]

fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True, constrained_layout=True)

# Izquierda: muestras C
for idx, (nombre, H, m_norm) in enumerate(muestras_C):
    a.plot(H, m_norm, '.', label=nombre )
    a.plot(H_fit_arrays[nombre], m_fit_arrays[nombre], '-', label=nombre+' Fitting')

a.set_ylabel('m (emu/g)')
a.legend(ncol=2)
a.grid()
a.set_title('Muestras C')
a.set_xlabel('H (G)')

# Derecha: muestras F
for idx, (nombre, H, m_norm) in enumerate(muestras_F):
    b.plot(H, m_norm, '.', label=nombre )
    b.plot(H_fit_arrays[nombre], m_fit_arrays[nombre], '-', label=nombre+' Fitting')

b.set_ylabel('m (emu/g)')
b.set_xlabel('H (G)')
b.legend(ncol=2)
b.grid()
b.set_title('Muestras F')

plt.savefig('VSM_fits_C_F.png', dpi=300)
plt.show()



# %%
