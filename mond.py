import numpy as np
from astropy.cosmology import FlatLambdaCDM 
from scipy.integrate import quad, quad_vec
from scipy.special import j0, j1
from multiprocessing import Pool
import time
from tqdm import tqdm
#import cupy as cp
#import blaze as bz


class MOND():

        def __init__(self):
            self.G = 4.302e-6 # kpc(km/s)^2 / Msolar or  6.67e-11  m^3 kg^-1 s^-2
            self.c = 299792.458 #km/s
            self.a0 = 1.2e-13  #MOND acceleration scale km/s^2
            self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
            self.zl = 0.2
            self.zs = 1.00
            self.Dl = self.cosmo.angular_diameter_distance(self.zl).value*1e3
            self.fudge = 1e-15

            self.Md = 8.1e10 # disc mass in Msolar
            self.rd = 2.1 # disc radius in kpc
            self.zd = 0.7 # disc hegight in kpc
            self.Mb = 8e9 # bulge mass in Msolar
            self.Rb = 0.8 # bulge scale length in Msolar
        
        def densitiesBaryons(self, r, z):
            rhod = (self.Md / (4 * np.pi * self.rd**2 * self.zd)) * np.exp(-np.sqrt(r**2)/self.rd) * np.exp(-np.abs(z)/self.zd)
            rhob = (3 * self.rd**2 * self.Mb) / (4 * np.pi * (self.Rb**2 + r**2 + z**2)**(5/2)) 
            return rhod + rhob
                
        def densitiesAll(self, r, z):
            rho_baryons = self.densitiesAll(r,z)
            rho_ph = self.rho_ph(r, z)
            return rho_baryons + rho_ph
        
        def grad_potential_disc(self, r, z):
            constant = (self.G * self.Md)
            grad_phi_r = constant * self.Ir(r, z)
            grad_phi_z = constant * self.Iz(r, z) 
            return grad_phi_r, grad_phi_z
        
        def grad_potential_bulge(self, r, z):
            const = (self.G * self.Mb/ (self.Rb**2 + r**2 + z**2)**(3/2))
            grad_phi_r = const * np.sqrt(r**2)
            grad_phi_z = const * z
            return grad_phi_r, grad_phi_z

        def grad_potential_Newton(self, r, z):
            grad_disc_r, grad_disc_z = self.grad_potential_disc(r, z)
            grad_bulge_r, grad_bulge_z = self.grad_potential_bulge(r, z)

            grad_tot_z = (grad_disc_z + grad_bulge_z) 
            grad_tot_r = (grad_disc_r + grad_bulge_r) 
            tot_acceleration = np.sqrt(grad_tot_z**2 + grad_tot_r**2)
            return tot_acceleration, grad_tot_r, grad_tot_z

        def ZZ(self, k, z):
            ZZ = (np.exp(-np.abs(z)*k)-self.zd*k*np.exp(-np.abs(z)/self.zd))/(1-(self.zd*k)**2)
            return ZZ
        
        def dZZdz(self, k, z):
            dZZdz = -np.sign(z)*k*(np.exp(-np.abs(z)*k)-np.exp(-np.abs(z)/self.zd))/(1-(self.zd*k)**2)
            return dZZdz
        
        def d2ZZdz(self, k, z):
            d2ZZdz = k*(k*np.exp(-np.abs(z)*k)-np.exp(-np.abs(z)/self.zd)/self.zd)/(1-(self.zd*k)**2)
            return d2ZZdz

        def Ir(self, r, z):
            def my_integrand(k):
                term1 = (j1(r*k) * k )/ ((1 + self.rd**2 * k**2)**(3/2))
                term2 = self.ZZ(k,z)
                return term1*term2

            Ir1, _ = quad_vec(my_integrand, 0, 1, limit=400)
            Ir2, _ = quad_vec(my_integrand, 1, np.inf, limit=1000)
            Ir = Ir1 + Ir2
            return Ir
        
        def Iz(self, r, z):
            def my_integrand(k):
                term1 =  -j0(r * k)  / ((1 + self.rd**2 * k**2)**(3/2))
                term2 =  self.dZZdz(k,z)
                return  term1*term2
            
            Iz1, _ = quad_vec(my_integrand, 0, 1, limit=400)
            Iz2, _ = quad_vec(my_integrand, 1, np.inf, limit=1000)
            Iz =  Iz1 + Iz2
            return Iz
        
        def dIrdz(self, r, z):
            def my_integrand(k):
                term1 = j1(r * k) * k / ((1 + self.rd**2 * k**2)**(3/2)) 
                term2 =  self.dZZdz(k,z)
                return term1*term2
            
            dIrdz1, _ = quad_vec(my_integrand, 0, 1, limit=400)
            dIrdz2, _ = quad_vec(my_integrand, 1, np.inf, limit=1000)
            dIrdz = dIrdz1 + dIrdz2
            return dIrdz
        
        def dIzdz (self, r, z):
            def my_integrand(k):
                term1 = - j0(r * k) / ((1 + self.rd**2 * k **2)**(3/2))
                term2 = self.d2ZZdz(k,z)
                return  term1*term2 
            
            dIzdz1, _ = quad_vec(my_integrand, 0, 1, limit = 400)
            dIzdz2, _ = quad_vec(my_integrand, 1, np.inf, limit = 1000)
            dIzdz = dIzdz1 + dIzdz2
            return dIzdz
        
        def dIrdr(self,r, z):
    
            def my_integrand(k):
                term1 = (1 + self.rd**2 * k**2)**(-3/2)
                term2 = (j0(r*k) - (j1(r*k)/(r * k + self.fudge)))*k**2 
                term3 = self.ZZ(k,z) 
                return term1*term2*term3
            
            dIrdr1, _ = quad_vec(my_integrand, 0, 1, limit = 400)
            dIrdr2, _ = quad_vec(my_integrand,1, np.inf, limit = 1000)
            dIrdr = dIrdr1 + dIrdr2
            return dIrdr
        
        def dIzdr(self, r, z):
            dIzdr = self.dIrdz(r,z)
            return dIzdr

        def nabla_norm_Phi(self, r, z):
            tot_acceleration, _ , _ = self.grad_potential_Newton(r, z)
            const = 1/ (2 * (tot_acceleration)+ self.fudge)

            t1a = 2 * self.Mb**2 * self.G**2 * r * (self.Rb**2 - 2 * (r**2 + z**2) ) / (self.Rb**2 + r**2 + z**2)**4
            t1b = 2 * self.Md**2 * self.G**2 * ((self.Ir(r, z) * self.dIrdr(r, z)) + (self.Iz(r, z) * self.dIzdr(r, z) ) )
            t1c = 2 * self.Mb * self.Md * self.G**2 * ( ( ( self.dIrdr(r, z)*r) + self.dIzdr(r, z) *z) / (self.Rb**2 + r**2 + z**2)**(3/2) ) 
            t1d = 2 * self.Mb * self.Md * self.G**2 * ( (z**2 - 2 * r**2 + self.Rb**2) * self.Ir(r, z) - (3 * r * z * self.Iz(r, z)) ) / (self.Rb**2 + r**2 + z**2)**(5/2)
            nabla_norm_Phi_r = const * (t1a + t1b + t1c + t1d) / 3.086e16

            t2a = 2 * self.Mb**2 * self.G**2 * z * (self.Rb**2 - 2*(r**2 + z**2)) / (self.Rb**2 + r**2 + z**2)**4
            t2b = 2 * self.Md**2 * self.G**2 * (self.dIzdz(r, z) * self.Iz(r, z) + self.Ir(r, z) * self.dIrdz(r, z))
            t2c = 2 * self.Mb * self.Md * self.G**2 * ( ((self.dIrdz(r, z)*r) ) + self.dIzdz(r, z) * z) / (self.Rb**2 + r**2 + z**2)**(3/2) 
            t2d = 2 * self.Mb * self.Md * self.G**2 * ( (r**2 - 2*z**2 + self.Rb**2) * self.Iz(r, z) - (3 * z * r * self.Ir(r, z)) ) / (self.Rb**2 + r**2 + z**2)**(5/2)
            nabla_norm_Phi_z = const * (t2a + t2b + t2c + t2d) / 3.086e16
            return nabla_norm_Phi_r, nabla_norm_Phi_z 

        def rho_ph(self, r, z):

            nabla_norm_Phi_r, nabla_norm_Phi_z = self.nabla_norm_Phi(r, z)
            tot_acceleration, grad_tot_r, grad_tot_z = self.grad_potential_Newton(r, z)
            const = -1 / (8 * np.pi * self.G)
            t1 = ((self.a0 / ((tot_acceleration + self.fudge)/ 3.086e16) )**2)  / np.sqrt(0.25 + (self.a0 / ((tot_acceleration + self.fudge)/3.086e16)))
            t2 = ((nabla_norm_Phi_r * (grad_tot_r)) + (nabla_norm_Phi_z * (grad_tot_z))) / self.a0 
            t3 = self.densitiesBaryons(r, z) / 2
            t4 = np.sqrt((0.25 + (self.a0 / ((tot_acceleration + self.fudge)/3.086e16)))) * self.densitiesBaryons(r, z)
            return (const * (t1 * t2) - t3 + t4)*r
        
        def plot_density_rz_contour(self,r, z):
            # Get the full 3D density values from the function
            densities = self.densitiesBaryons_map(r, z)
            
            # Create a contour plot for the rz plane
            plt.figure(figsize=(8, 6))
            contourf = plt.contourf(r, z, densities, levels=50, cmap='viridis')
            plt.colorbar(contourf, label='Density')  # Add a color bar to represent density values

            # Set labels and title
            plt.xlabel('x')
            plt.ylabel('z')
            plt.title('Baryonic Density Contour in the rz-plane (y = 0)')

            # Show the plot
            filename = 'CUNT.png'
            plt.savefig(filename)
            plt.show()

        # def rho_ph_parallel(self, x, y, z):
        #     coords = [(xi, yi, z) for xi in x for yi in y]
        #     with Pool(processes=20) as pool:
        #         results = pool.starmap(self.rho_ph, coords)
        #     return np.array(results).reshape(len(x), len(y))



if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import time

    ts = MOND()

    # x = bz.Array(np.linspace(1e-10, 100, 1000))  # x values
    # y = bz.Array(np.linspace(1e-10, 100, 1000))  # y values (y=0 plane is in the middle)
    # z = bz.Array(np.linspace(1e-10, 100, 1000))  # z values

    r = np.linspace(0, 100, 1000)  # x values
    z = np.linspace(0, 100, 1000) # np.linspace(1e-10, 100, 1000, dtype = np.float32)   # z values

    # rho = ts.rho_ph(r,0)

    # plt.figure(figsize=(8,6))
    # plt.plot(r,rho)
    # plt.show()
   
    R, Z = np.meshgrid(r, z, indexing = 'ij')
    Rho = ts.rho_ph(R,Z)

    # plt.figure(figsize=(8, 6))
    # plt.contourf(R, Z, Rho, levels=50, cmap='viridis')
    # plt.colorbar(label='rho_ph')
    # plt.xlabel("r [kpc]", fontsize=12)
    # plt.ylabel("z [kpc]", fontsize=12)
    # plt.tight_layout()
    # filename = 'rho_phantom_cycl.png'
    # plt.savefig(filename)
    # plt.show()

 

    # Save the resulting array to a file
    np.save('rho_ph_values.npy', Rho)

    # Optional: Plotting the results
    plt.figure(figsize=(8, 6))
    plt.contourf(R, Z, Rho, levels=50, cmap='inferno')
    plt.colorbar(label='rho_ph')
    plt.xlabel("r [kpc]", fontsize=12)
    plt.ylabel("z [kpc]", fontsize=12)
    plt.tight_layout()
    plt.savefig('rho_phantom_cycl.png')
    plt.show()

    




