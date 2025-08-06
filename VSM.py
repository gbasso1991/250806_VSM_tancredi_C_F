#%% VSM muestras de Pablo Tancredi - Julio 2025
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
fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True, sharey=True, constrained_layout=True)

# Arriba: muestras C
axs[0].plot(H_C1, m_C1, '.-', label='C1')
axs[0].plot(H_C2, m_C2, '.-', label='C2')
axs[0].plot(H_C3, m_C3, '.-', label='C3')
axs[0].plot(H_C4, m_C4, '.-', label='C4')
axs[0].set_ylabel('m (emu)')
axs[0].legend()
axs[0].grid()
axs[0].set_title('Muestras C')

# Abajo: muestras F
axs[1].plot(H_F1, m_F1, '.-', label='F1')
axs[1].plot(H_F2, m_F2, '.-', label='F2')
axs[1].plot(H_F3, m_F3, '.-', label='F3')
axs[1].plot(H_F4, m_F4, '.-', label='F4')
axs[1].set_ylabel('m (emu)')
axs[1].set_xlabel('H (G)')
axs[1].legend()
axs[1].grid()
axs[1].set_title('Muestras F')

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

fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True, sharey=True, constrained_layout=True)

# Arriba: muestras C
axs[0].plot(H_C1, m_C1_norm, '.-', label='C1')
axs[0].plot(H_C2, m_C2_norm, '.-', label='C2')
axs[0].plot(H_C3, m_C3_norm, '.-', label='C3')
axs[0].plot(H_C4, m_C4_norm, '.-', label='C4')
axs[0].set_ylabel('m (emu)')
axs[0].legend()
axs[0].grid()
axs[0].set_title('Muestras C')

# Abajo: muestras F
axs[1].plot(H_F1, m_F1_norm, '.-', label='F1')
axs[1].plot(H_F2, m_F2_norm, '.-', label='F2')
axs[1].plot(H_F3, m_F3_norm, '.-', label='F3')
axs[1].plot(H_F4, m_F4_norm, '.-', label='F4')
axs[1].set_ylabel('m (emu/g)')
axs[1].set_xlabel('H (G)')
axs[1].legend()
axs[1].grid()
axs[1].set_title('Muestras F')
plt.savefig('VSM_muestras_C_F.png', dpi=300)
plt.show()

#%% Nuevo gráfico 2x2 comparando Cn vs Fn
fig2, axs2 = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True, constrained_layout=True)

# C1 vs F1
axs2[0, 0].plot(H_C1, m_C1_norm, '.-', label='C1')
axs2[0, 0].plot(H_F1, m_F1_norm, '.-', label='F1')
axs2[0, 0].set_title('C1 / F1',loc='left')
axs2[0, 0].legend()
axs2[0, 0].grid()

# C2 / F2
axs2[0, 1].plot(H_C2, m_C2_norm, '.-', label='C2')
axs2[0, 1].plot(H_F2, m_F2_norm, '.-', label='F2')
axs2[0, 1].set_title('C2 / F2',loc='left')
axs2[0, 1].legend()
axs2[0, 1].grid()

# C3 / F3
axs2[1, 0].plot(H_C3, m_C3_norm, '.-', label='C3')
axs2[1, 0].plot(H_F3, m_F3_norm, '.-', label='F3')
axs2[1, 0].set_title('C3 / F3',loc='left')
axs2[1, 0].legend()
axs2[1, 0].grid()

# C4 / F4
axs2[1, 1].plot(H_C4, m_C4_norm, '.-', label='C4')
axs2[1, 1].plot(H_F4, m_F4_norm, '.-', label='F4')
axs2[1, 1].set_title('C4 / F4',loc='left')
axs2[1, 1].legend()
axs2[1, 1].grid()
# Solo los ylabel en la columna izquierda, xlabel en la fila de abajo
for i in range(2):
    axs2[i, 0].set_ylabel('m (emu/g)')
for j in range(2):
    axs2[1, j].set_xlabel('H (G)')
plt.savefig('VSM_muestras_C_F_comparacion.png', dpi=300)
plt.show()

#%% Curvas Anhistéricas y fit para todas las muestras C y F

resultados_fit = {}
H_fit_arrays = {}
m_fit_arrays = {}

for nombre, H, m in [
    ('C1', H_C1, m_C1_norm),
    ('C2', H_C2, m_C2_norm),
    ('C3', H_C3, m_C3_norm),
    ('C4', H_C4, m_C4_norm),
    ('F1', H_F1, m_F1_norm),
    ('F2', H_F2, m_F2_norm),
    ('F3', H_F3, m_F3_norm),
    ('F4', H_F4, m_F4_norm),
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
    resultados_fit[nombre] = {
        'H_anhist': H_anhist,
        'm_anhist': m_anhist,
        'H_fit': fit.X,
        'm_fit': fit.Y,
        'fit': fit
    }
    H_fit_arrays[nombre] = fit.X
    m_fit_arrays[nombre] = fit.Y - lineal(H_anhist, fit.params['C'].value, fit.params['dc'].value)

    
    
#%% Ploteo fits 
# Plot 2x1: Todas las C arriba, todas las F abajo, con fits
fig, (a,b) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, sharey=True, constrained_layout=True)

# Arriba: muestras C
for nombre in ['C1', 'C2', 'C3', 'C4']:
    a.plot(resultados_fit[nombre]['H_anhist'], resultados_fit[nombre]['m_anhist'], '.', label=f'{nombre} data')
    a.plot(resultados_fit[nombre]['H_fit'], resultados_fit[nombre]['m_fit'], '-', label=f'{nombre} fit')
a.set_ylabel('m (emu/g)')
a.legend(ncol=2)
a.grid()
a.set_title('Muestras C (Anhisteréticas y fits)')

# Abajo: muestras F
for nombre in ['F1', 'F2', 'F3', 'F4']:
    b.plot(resultados_fit[nombre]['H_anhist'], resultados_fit[nombre]['m_anhist'], '.', label=f'{nombre} data')
    b.plot(resultados_fit[nombre]['H_fit'], resultados_fit[nombre]['m_fit'], '-', label=f'{nombre} fit')
b.set_ylabel('m (emu/g)')
b.set_xlabel('H (G)')
b.legend(ncol=2)
b.grid()
b.set_title('Muestras F (Anhisteréticas y fits)')

plt.savefig('VSM_fits_C_F.png', dpi=300)
plt.show()

# Plot 2x2: Comparación Cn vs Fn, con fits
fig2, axs2 = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True, constrained_layout=True)

pares = [('C1', 'F1'), ('C2', 'F2'), ('C3', 'F3'), ('C4', 'F4')]
for idx, (c, f) in enumerate(pares):
    i, j = divmod(idx, 2)
    axs2[i, j].plot(resultados_fit[c]['H_anhist'], resultados_fit[c]['m_anhist'], '.', label=f'{c} data')
    axs2[i, j].plot(resultados_fit[c]['H_fit'], resultados_fit[c]['m_fit'], '-', label=f'{c} fit')
    axs2[i, j].plot(resultados_fit[f]['H_anhist'], resultados_fit[f]['m_anhist'], '.', label=f'{f} data')
    axs2[i, j].plot(resultados_fit[f]['H_fit'], resultados_fit[f]['m_fit'], '-', label=f'{f} fit')
    axs2[i, j].set_title(f'{c} / {f}', loc='left')
    axs2[i, j].legend(ncol=2)
    axs2[i, j].grid()

for i in range(2):
    axs2[i, 0].set_ylabel('m (emu/g)')
for j in range(2):
    axs2[1, j].set_xlabel('H (G)')

plt.savefig('VSM_fits_C_F_comparacion.png', dpi=300)
plt.show()



# %%
