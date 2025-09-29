import numpy as np
from scipy.special import i0, i1, k0, k1
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares
from numba import njit, prange

G =  4.302e-6 # gravitational constant in kpc (km/s)^2/M_sun
a0 = 1.2e-13 # MOND acceleration constant in km/s^2
Rd = 3.5 # disk scale length in kpc
Md = 1e11 # disk mass in M_sun
Rh = 16.59 # halo scale length in kpc
rhoh = 9.55e6 # halo density in M_sun/kpc^3
Rb = 0.7 # bulge scale length in kpc
Mb = 1e9 # bulge mass in M_sun

#loading density grid from density.py
rho = np.load('density.npy')

#----------use the same r-z grid as in density.py----------
r_eval_precision = 101
r_eval_bound = 100
r_eval = np.linspace(0, r_eval_bound, r_eval_precision)  

r_grid_precision = 201
r_grid_bound = 100 
r_grid = np.linspace(0, r_grid_bound, r_grid_precision) 

# z_grid_precision = 200
# z_grid_bound = 30 # 30
# z_grid = np.linspace(-z_grid_bound, z_grid_bound, z_grid_precision)
# mask = z_grid>0

z_fine_precision = 51
z_fine_bound = 5
z_fine = np.linspace(0, z_fine_bound, z_fine_precision)
#dz_fine = z_fine[1] - z_fine[0]

z_coarse_precision = 151
z_coarse_bound = 80
z_coarse = np.linspace(z_fine_bound, z_coarse_bound, z_coarse_precision)
#dz_coarse = z_coarse[1] - z_coarse[0]
z_grid = np.concatenate((z_fine, z_coarse[1:]))

dz_array = np.empty_like(z_grid)
dz_array[:-1] = z_grid[1:] - z_grid[:-1] 
dz_array[-1] = dz_array[-2]

phi_grid_precision = 1000
phi_grid_bound =  np.pi
phi_grid = np.linspace(0, phi_grid_bound, phi_grid_precision)
dphi = phi_grid[1] - phi_grid[0]

#---------- define functions for rotation curve calculation----------
@njit
def Fun(r, rp, zp, phip, rho):
    cos_phip = np.cos(phip)
    
    # Numerator multiplied by rp
    numerator = rp * r - rp**2 * cos_phip 
    
    # Denominator terms
    r_sq = r**2
    rp_sq = rp**2
    zp_sq = zp**2
    cross_term = 2 * rp * r * cos_phip
    
    # Denominator calculation
    denominator = (rp_sq + r_sq + zp_sq - cross_term)**1.5
    
    return numerator * rho / denominator

@njit(parallel=True)  # Parallelize over loops
def compute_vrot(r_eval, r_grid, z_grid, phi_grid, dphi, rho):
    vrot = np.zeros(len(r_eval))
    G = 4.302e-6
    for i in prange(len(r_eval)):          # Parallel loop over r_eval
        r = r_eval[i]
        integral = 0.0
        for ir in range(len(r_grid)):  # Nested loops over grid
            dr = r_grid[1] - r_grid[0]  # Default dr
            # dr = dr_fine if (ir < r_fine_precision) else dr_coarse  # Adjust dr based on index
            for iz in range(len(z_grid)):
                dz = dz_array[iz]  # Use precomputed dz array
               # dz = dz_fine if (iz < z_fine_precision) else dz_coarse  # Adjust dz based on index
                irho = rho[ir, iz]   # Lookup density
                for iphi in range(len(phi_grid)):
                    rp, zp, phip = r_grid[ir], z_grid[iz], phi_grid[iphi]
                    #integral += Fun(r, rp, zp, phip) * irho * dV
                    weight = dphi if (iphi != 0 and iphi != len(phi_grid)-1) else 0.5 * dphi
                    integral += Fun(r, rp, zp, phip, irho) * dr * dz * weight
        vrot[i] = integral
    vrot = ((4 * G * r_eval * vrot) + 1e-10)**0.5 # Calculate rotation velocity
    return vrot

#---------- calculate rotation curve----------
vrot_list = compute_vrot(r_eval, r_grid, z_grid[1:], phi_grid, dphi, rho)
np.save(f'rotation_curve.npy', vrot_list)