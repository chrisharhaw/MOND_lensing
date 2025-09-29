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

#loading density grid from density_grid.py
rho = np.load('density.npy')

#----------use the same r-z grid as in density_grid.py----------
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


#---------- DM fitting ----------

def gNr(r):
    """
    Calculate the Newtonian acceleration in the radial direction for a given radius r.
    """
    t1 = G * Md/(2 * Rd**3) * r *  (i0(r/(2*Rd)) * k0(r/(2*Rd)) - i1(r/(2*Rd)) * k1(r/(2*Rd))) / 3.086e16 # convert to km/s^2
    t2 = G * Mb * r * (r**2 + Rb**2)**(-3/2) / 3.086e16 # convert to km/s^2
    return t1 + t2

def vcNdisc(r):
    """
    Calculate the Newtonian circular velocity for a given radius r.
    """
    return np.sqrt( gNr(r) * r * 3.086e16 ) # convert to km/s

def NSIS_fit(r, r0, rho0):
    """
    Fit the NSIS halo model to the data.
    """
    # rho = rho0 / (1 + (r/r0)**2)
    # M = 4 * np.pi * rho * r**3 / 3
    M = 4 * np.pi * rho0 * ((r0**2 * r) - r0**3 * np.arctan(r/r0)) 
    return np.sqrt( G * M / (r + 1e-8)) 

steps = 101
radii = np.linspace(1e-3, 100, steps)


def residuals_NSIS(params, r, data):
    r0, rho0 = params
    v_model = np.sqrt(vcNdisc(r)**2 + NSIS_fit(r, r0, rho0)**2)
    return v_model - data

result_NSIS = least_squares(
    residuals_NSIS, x0=[5, 1e7], args=(radii, vrot_list), 
    bounds=([1, 1e6], [10, 1e8]), loss='soft_l1'
)
r0_fit, rho0_fit = result_NSIS.x
print(f"Best-fit parameters: r0 = {r0_fit:.3e}, rho0 = {rho0_fit:.3e}")

def vcN(r):
    """
    Calculate the total Newtonian circular velocity for a given radius r.
    """
    return np.sqrt(vcNdisc(r)**2 + NSIS_fit(r, r0_fit, rho0_fit)**2)


#---------- Plotting ----------

plt.figure(figsize=(10, 6))
plt.plot(radii, vrot_list, label='MOND', color='black')
plt.plot(radii, vcN(radii), label='DM best fit', color='magenta', linestyle='--')

plt.xlabel(r'$r$ [kpc]')
plt.ylabel(r'$v_{ROT}$ [km/s]')
plt.legend(fontsize = 14)
plt.tight_layout()
plt.savefig('rotation_curve_MOND_vs_NSIS.pdf')
plt.show()
plt.close()