import numpy as np
from scipy.integrate import quad_vec
from scipy.special import j0, j1
import matplotlib.pyplot as plt

plt.rc('font', family='serif')

fig_width_pt = 244.0  # Get this from LaTeX using \the\columnwidth
text_width_pt = 508.0 # Get this from LaTeX using \the\textwidth

inches_per_pt = 1.0/72.27               # Convert pt to inches
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt*1.5 # width in inches
fig_width_full = text_width_pt*inches_per_pt*1.5  # 17
fig_height =fig_width*golden_mean # height in inches
fig_size = [fig_width,fig_height] #(9,5.5) #(9, 4.5)
fig_height_full = fig_width_full*golden_mean

class MOND():
    def __init__(self, Md, rd, zd, Mb, Rb):
        self.G = 4.302e-6 # kpc(km/s)^2 / Msolar or  6.67e-11  m^3 kg^-1 s^-2
        self.c = 299792.458 #km/s
        self.a0 = 1.2e-13  #MOND acceleration scale km/s^2
        self.fudge = 1e-15

        self.Md = Md #8.1e10 # disc mass in Msolar
        self.rd = rd #2.1 # disc radius in kpc
        self.zd = zd #0.7 # disc hegight in kpc
        self.Mb = Mb #8e9 # bulge mass in Msolar
        self.Rb = Rb #0.8 # bulge scale length in Msolar
        self.rhoDM0 = 3.425 
        self.RDMO = 6.356e7

    def densitiesBaryons(self, r, z):
        rhod = (self.Md / (4 * np.pi * self.rd**2 * self.zd)) * np.exp(-r/self.rd) * np.exp(-np.abs(z)/self.zd)
        rhob = (3 * self.Rb**2 * self.Mb) / (4 * np.pi * (self.Rb**2 + r**2 + z**2)**(5/2)) 
        return (rhod + rhob)

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
        tot_acceleration_normalised = (self.a0 / ((tot_acceleration)/3.086e16))**(-1)
        const = 1 / (4 * np.pi * self.G)
        nu = -0.5 + (0.25+1/tot_acceleration_normalised)**(1/2)

        t1 = -0.5*((0.25+1/tot_acceleration_normalised)**(-1/2))*(tot_acceleration_normalised**(-2))
        t2  = ((nabla_norm_Phi_r * (grad_tot_r)) + (nabla_norm_Phi_z * (grad_tot_z))) / self.a0 
        t4 = nu * self.densitiesBaryons(r, z)
        return (const * (t1 * t2) + t4)
    
    def rho_ph_RAR(self, r, z):
        nabla_norm_Phi_r, nabla_norm_Phi_z = self.nabla_norm_Phi(r, z)
        tot_acceleration, grad_tot_r, grad_tot_z = self.grad_potential_Newton(r, z)
        tot_acceleration_normalised = (self.a0 / ((tot_acceleration)/3.086e16))**(-1)

        const = 1 / (4 * np.pi * self.G)
        nu = np.exp(-(tot_acceleration_normalised**(1/2)))/(1 - np.exp(-(tot_acceleration_normalised**(1/2))))
        t1 = -(1/(2*np.sqrt(tot_acceleration_normalised)))*((np.exp(-(tot_acceleration_normalised**(1/2)))/(1 - np.exp(-(tot_acceleration_normalised**(1/2)))))**2)
        t2  = ((nabla_norm_Phi_r * (grad_tot_r)) + (nabla_norm_Phi_z * (grad_tot_z))) / self.a0 
        t4 = nu * self.densitiesBaryons(r, z)

        return (const * (t1 * t2) + t4)

    def rho_DM(self, r, z):
            
        DM = self.rhoDM0/(1+(r**2+z**2)/(self.RDMO**2))

        return DM
    
    def rho_total(self,r,z):

        # PDM = self.rho_ph_RAR(r,z)*r
        baryons =  self.densitiesBaryons(r, z)*r
        PDM = self.rho_ph(r,z)*r
        # DM = self.rho_DM(r, z)*r

        return PDM + baryons
       

if __name__ == '__main__':
    # Define the grid parameters
    r_grid_precision_length = 500
    z_grid_precision_length = 500
    r_grid_bound_length = 40
    z_grid_bound_length = 20
    r = np.linspace(0, r_grid_bound_length, r_grid_precision_length, dtype=np.float64)
    r_plot = np.linspace(-r_grid_bound_length, r_grid_bound_length, 2*r_grid_precision_length, dtype=np.float64)
    z = np.linspace(-z_grid_bound_length, z_grid_bound_length, z_grid_precision_length, dtype=np.float64)

    dr = r_grid_bound_length/r_grid_precision_length
    dz = 2*z_grid_bound_length/z_grid_precision_length

    #take the positive z values only due to symmetry about the z=0 plane
    mask = z>0
    R, Z = np.meshgrid(r, z[mask], indexing = 'ij')
    
    R_plot, Z_plot = np.meshgrid(r_plot, z, indexing = 'ij')

    # Define the parameters for the disc model - [Md, rd, zd, Mb, Rb]
    parameters = [[1e11, 3.5, 0.035, 1e9, 0.35],[1e11, 3.5, 0.035, 1e9, 1.05],[1e11, 3.5, 0.1, 1e9, 0.35],
                  [1e11, 3.5, 0.1, 1e9, 0.7], [1e11, 3.5, 0.1, 1e9, 1.05]]


    def compute_density(param):
        Md, rd, zd, Mb, Rb = param
        ts = MOND(Md, rd, zd, Mb, Rb)

        # Compute the density
        Rho = ts.rho_total(R, Z)

        return Rho

    for param in parameters:
        Md, rd, zd, Mb, Rb = param
        my_rho = compute_density(param)

        # Create filename
        filename = f"full_rho_map_Md{Md:.2e}_rd{rd:.2f}_zd{zd:.2f}_Mb{Mb:.2e}_Rb{Rb:.2f}.npy"
        # Save to .npy
        np.save(filename, my_rho)
        print(f"Saved {filename}")

