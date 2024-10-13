import numpy as np
from astropy.cosmology import FlatLambdaCDM 
from scipy.integrate import quad, quad_vec
from tqdm import tqdm
from multiprocessing.pool import Pool
import time

class Density():
    def __init__(self, r, z, phi,
                 Md = 8.1e10,
                 rd = 2.1,
                 zd = 0.7,
                 Mb = 8e9,
                 Rb = 0.8,
                 sigv = 200
                 ):
        self.Md = Md # disc mass in Msolar
        self.rd = rd # disc radius in kpc
        self.zd = zd # disc hegight in kpc
        self.Mb = Mb # bulge mass in Msolar
        self.Rb = Rb # bulge scale length in Msolar
        self.sigv = sigv #km/s
        self.G = 4.302e-6 # kpc(km/s)^2 / Msolar or  6.67e-11  m^3 kg^-1 s^-2
        self.c = 299792.458 # km/s
        self.km2kpc = 3.24078e-17
        self.fudge = 1e-15


        dr = np.abs(r[1] - r[0])
        dz = np.abs(z[1] - z[0])
        dphi = np.abs(phi[1] - phi[0])
        dV = dr*dz*dphi 

        self.R, self.Z, self.Phi = np.meshgrid(r, z, phi, indexing = 'ij') #no point recalulating mesh
        

        ### FOR SPHERICAL COORDINATES ###
        # R_sph, Theta_sph, Phi_sph = np.meshgrid(r, theta, phi, indexing = 'ij')
        # self.r_sph = R_sph
        # self.theta_sph = Theta_sph
        # self.phi_sph = Phi_sph
        # dtheta = np.abs(theta[1] - theta[0])
        # dV = dr * dphi * dtheta 

    def densitiesBaryons(self):
        # Calculate the disk density (rhod)
        rhod = (self.Md / (4 * np.pi * self.rd**2 * self.zd)) * np.exp(-self.R / self.rd) * np.exp(-np.abs(self.Z) / self.zd)
        # Calculate the bulge density (rhob)
        rhob = (3 * self.rd**2 * self.Mb) / (4 * np.pi * (self.Rb**2 + self.R**2 + self.Z**2)**(5/2))
        # Return the combined density
        return (rhod + rhob)*self.R
    
    def densitiesSIS(self):
        rhohalo = self.sigv**2 / (2 * np.pi * self.G * (self.R**2 + self.Z**2) + self.fudge)  
        return rhohalo * self.R
    
    def compute_density_chunk(self, chunk_indices):
        """Compute density for a chunk of the full grid"""
        # Extract the chunk indices (start and end) for the R grid
        start_idx, end_idx = chunk_indices
        R_chunk = self.R[start_idx:end_idx, :, :]
        Z_chunk = self.Z[start_idx:end_idx, :, :]
        Phi_chunk = self.Phi[start_idx:end_idx, :, :]
        
        # SIS density calculation for the chunk
        rhohalo = self.sigv**2 / (2 * np.pi * self.G * (R_chunk**2 + Z_chunk**2) + self.fudge)
        return rhohalo * R_chunk

def split_indices(array_length, n_chunks):
    """Helper function to split array indices into n_chunks"""
    chunk_size = array_length // n_chunks
    indices = [(i, min(i + chunk_size, array_length)) for i in range(0, array_length, chunk_size)]
    return indices

def parallel_density_calculation(dn, n_processes=4):
    """Calculate density map in parallel"""
    # Split the indices of the R grid into chunks for parallel processing
    index_chunks = split_indices(dn.R.shape[0], n_processes)

    # Use multiprocessing pool to calculate densities in parallel
    results = []
    with Pool(processes=n_processes) as pool:
        # Use tqdm to track progress and specify a larger chunksize for better efficiency
        for result in tqdm(pool.imap(dn.compute_density_chunk, index_chunks, chunksize=2), total=len(index_chunks)):
            results.append(result)

    # Concatenate the results along the r axis
    return np.concatenate(results, axis=0)
    

if __name__ == '__main__':
    grid_precision_length = 2000
    grid_precision_phi = 2000
    grid_precision_theta = 2000
    grid_bound_length = 50
    r = np.linspace(0, grid_bound_length, grid_precision_length)
    z = np.linspace(-grid_bound_length, grid_bound_length, grid_precision_length)
    phi = np.linspace(0, 2*np.pi, grid_precision_phi)

    ### FOR SPHERICAL COORDINATES ###
    #theta = np.linspace(0, np.pi, grid_precision_theta)

    dn = Density(r, z, phi)
    n_processes = 20 #num of cores used
    
    rho_map = parallel_density_calculation(dn, n_processes)

    print("Density map calculated successfully.")


  
    