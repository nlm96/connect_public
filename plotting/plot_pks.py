import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from classy import Class

model = tf.keras.models.load_model('trained_models/Lya_output_2', compile=False)

#params = {'output': 'tCl,lCl,pCl,mPk', 'lensing': 'yes', 'k_pivot': 0.05, 'N_ur': 2.0328, 'N_ncdm': 1, 'm_ncdm': 0.06, 'T_ncdm': 0.71611, 'omega_b': 0.0223, 'omega_cdm': 0.119, 'H0': 67.7, 'ln10^{10}A_s': 3.04, 'n_s': 0.965, 'tau_reio': 0.054, 'P_k_max_1/Mpc': 506.67000000000024, 'z_max_pk': 3.0}

#params = {'k_pivot': 0.05, 'N_ur': 2.0328, 'N_ncdm': 1, 'T_ncdm': 0.71611, 'P_k_max_h/Mpc': 600, 'z_max_pk': 3.0, 'lensing': 'yes', 'output': 'tCl lCl pCl mPk', 'l_max_scalars': 2508, 'omega_b': 0.0223, 'omega_cdm': 0.119, 'H0': 67.8, 'n_s': 0.965, 'tau_reio': 0.05, 'm_ncdm': 0.06, 'A_s': 2.090524323509276e-09}
params =  {'k_pivot': 0.05, 'N_ur': 2.046, 'P_k_max_h/Mpc': 600, 'z_max_pk': 3.0, 'lensing': 'yes', 'output': 'tCl lCl pCl mPk', 'l_max_scalars': 2508, 'omega_b': 0.02230000532417142, 'omega_cdm': 0.11900002841149772, 'H0': 67.80000809369508, 'n_s': 0.965, 'tau_reio': 0.05, 'A_s': 2.090524323509276e-09}

pks = model.get_Pks([[0.02230000532417142,0.11900002841149772,67.80000809369508,3.04,0.965,0.05,0.06,2.046]])

k = pks['k_grid'].numpy()

print('lol')

cosmo = Class()
from copy import deepcopy
params2 = deepcopy(params)
params2.update({'N_ncdm':1, 'deg_ncdm':0, 'm_ncdm':0.0})
cosmo.set(params2)
cosmo.compute()

print('done computing')


pks = {'pk_tilt':{'0.0':[],'3.0':[]}, 'pk_lin':{'0.0':[],'3.0':[]}, 'pk':{'0.0':[],'3.0':[]}, 'k_grid':k}

for kk in k:
    pks['pk_tilt']['0.0'].append(cosmo.pk_tilt(kk, 0.0))
    pks['pk_tilt']['3.0'].append(cosmo.pk_tilt(kk, 3.0))
    pks['pk_lin']['0.0'].append(cosmo.pk_lin(kk, 0.0))
    pks['pk_lin']['3.0'].append(cosmo.pk_lin(kk, 3.0))
    pks['pk']['0.0'].append(cosmo.pk(kk, 0.0))
    pks['pk']['3.0'].append(cosmo.pk(kk, 3.0))


cosmo = Class()
cosmo.set(params)
cosmo.compute()

print('done computing')

k_class = np.logspace(np.log10(5e-5),np.log10(300),500)
pks_class = {'pk_tilt':{'0.0':[],'3.0':[]}, 'pk_lin':{'0.0':[],'3.0':[]}, 'pk':{'0.0':[],'3.0':[]}, 'k_grid':k_class}

for kk in k_class:
    pks_class['pk_tilt']['0.0'].append(cosmo.pk_tilt(kk, 0.0))
    pks_class['pk_tilt']['3.0'].append(cosmo.pk_tilt(kk, 3.0))
    pks_class['pk_lin']['0.0'].append(cosmo.pk_lin(kk, 0.0))
    pks_class['pk_lin']['3.0'].append(cosmo.pk_lin(kk, 3.0))
    pks_class['pk']['0.0'].append(cosmo.pk(kk, 0.0))
    pks_class['pk']['3.0'].append(cosmo.pk(kk, 3.0))

print('done extracting')
    
plt.figure(1)
plt.title('pk_tilt')
plt.plot(k_class, pks_class['pk_tilt']['0.0'], 'k-', label='class, z=0.0')
plt.plot(k_class, pks_class['pk_tilt']['3.0'], 'r-', label='class, z=3.0')
plt.plot(k, pks['pk_tilt']['0.0'],#[0,:].numpy(),
         'k*', label='connect, z=0.0')
plt.plot(k, pks['pk_tilt']['3.0'],#[0,:].numpy(),
         'r*', label='connect, z=3.0')
plt.legend()
plt.xscale('log')

plt.figure(2)
plt.title('pk_lin')
plt.plot(k_class, pks_class['pk_lin']['0.0'], 'k-', label='class, z=0.0')
plt.plot(k_class, pks_class['pk_lin']['3.0'], 'r-', label='class, z=3.0')
plt.plot(k, pks['pk_lin']['0.0'],#[0,:].numpy(),
         'k*', label='connect, z=0.0')
plt.plot(k, pks['pk_lin']['3.0'],#[0,:].numpy(),
         'r*', label='connect, z=3.0')
plt.legend()
plt.xscale('log')
plt.yscale('log')

plt.figure(3)
plt.title('pk')
plt.plot(k_class, pks_class['pk']['0.0'], 'k-', label='class, z=0.0')
plt.plot(k_class, pks_class['pk']['3.0'], 'r-', label='class, z=3.0')
plt.plot(k, pks['pk']['0.0'],#[0,:].numpy(),
         'k*', label='connect, z=0.0')
plt.plot(k, pks['pk']['3.0'],#[0,:].numpy(),
         'r*', label='connect, z=3.0')
plt.legend()
plt.xscale('log')
plt.yscale('log')

plt.show()
