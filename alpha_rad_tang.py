import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM 
from tqdm import tqdm
from multiprocessing.pool import Pool
from scipy.ndimage import gaussian_filter
import os

plt.rc('font', family='serif')

n_processes = 80 # number of processes to run in parallel

# Define the resolution and bounds of the 2D grid
xi_resolution = 100 
zeta_resolution = 100 
xi_upper_bound = 5
zeta_upper_bound = 5 
xi_lower_bound = xi_upper_bound / (2*xi_resolution +1)  #aka the centre of the grid
zeta_lower_bound = zeta_upper_bound / (2*zeta_resolution+1) #aka the centre of the grid

#grid defining 3d - this is the density grid and these parameters should match those of density_grid.py
grid_precision_length = 500
grid_precision_phi = 500
r_grid_bound_length = 40
z_grid_bound_length = 20
r = np.linspace(0, r_grid_bound_length, grid_precision_length, dtype=np.float64)
z = np.linspace(-z_grid_bound_length, z_grid_bound_length, grid_precision_length, dtype=np.float64)
phi = np.linspace(0, 2*np.pi, grid_precision_phi, dtype=np.float64)

dr = np.abs(r[1] - r[0])
dz = np.abs(z[1] - z[0])
dphi = np.abs(phi[1] - phi[0])
dV = dr*dz*dphi 

class Disc():
    def __init__(self,rho_flipped, inc = np.pi/2):
        self.G = 4.302e-6 # kpc(km/s)^2 / Msolar or  6.67e-11  m^3 kg^-1 s^-2
        self.c = 299792.458 # km/s
        self.conv = 180*3600/np.pi # radians to arcsec
        self.const = (2*self.G/(self.c**2))*self.conv # physical conversion and from radians to arcsec
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.zl = 0.5
        self.zs = 2.00
        self.Dl = self.cosmo.angular_diameter_distance(self.zl).value*1e3
        self.Dls = self.cosmo.angular_diameter_distance_z1z2(self.zl, self.zs).value*1e3
        self.Ds = self.cosmo.angular_diameter_distance(self.zs).value*1e3
        self.fudge_factor = 1e-20
        self.my_sin = np.sin(inc)
        self.my_cos = np.cos(inc)

        self.R, self.Z, self.Phi = np.meshgrid(r, z, phi, indexing = 'ij', sparse = True) #no point recalulating mesh
        self.coeff = dV * self.const * (self.Dls/self.Ds)
        self.X = self.R*np.cos(self.Phi)
        self.Y = self.R*np.sin(self.Phi)
        self.rho_flipped = rho_flipped

    def part_inclination_map(self, xi, zeta):
        xi = (xi*self.Dl)/self.conv
        zeta = (zeta*self.Dl)/self.conv
        t2 = (self.X*self.my_cos + self.Z*self.my_sin)
        my_a = (self.X)**2 + self.Z**2 + (self.Y - zeta)**2  + xi**2 -2*xi*t2
        my_Delta = -(2*(self.X*self.my_sin - self.Z*self.my_cos)) **2 + 4*my_a
        my_dem =np.where(np.abs(my_Delta) <= 1e-16, 0, 8 / my_Delta)

        part_inclination_xi = (-xi + t2)*my_dem
        part_inclination_zeta = (self.Y - zeta)*my_dem
        return part_inclination_xi, part_inclination_zeta
	
    def alpha(self,xi,zeta):
        xi_inclination, zeta_inclination = self.part_inclination_map(xi, zeta)

        alpha_xi_list = []
        alpha_zeta_list = []
        for rho_array in self.rho_flipped:
            alpha_xi = np.nansum(rho_array*xi_inclination) * self.coeff
            alpha_zeta = np.nansum(rho_array*zeta_inclination) * self.coeff

            # Append the components to the lists
            alpha_xi_list.append(alpha_xi)
            alpha_zeta_list.append(alpha_zeta)

        return alpha_xi_list, alpha_zeta_list
    
class ArrayExtractor:
    def __init__(self, rho_flipped):
        for i, array in enumerate(rho_flipped):
            setattr(self, f'M{i+1}', array)
       
def do_pixel(var):
    Xi = var[0]
    Zeta = var[1]
    alpha_x_list , alpha_y_list  = ts.alpha(Xi,Zeta)
    return alpha_x_list, alpha_y_list 

def inclination_map():
    xi = np.linspace(xi_lower_bound, xi_upper_bound, xi_resolution)
    zeta = np.linspace(zeta_lower_bound, zeta_upper_bound, zeta_resolution)
    XI, ZETA = np.meshgrid(xi, zeta)
    Xi = XI.flatten()
    Zeta = ZETA.flatten()

    full_stack = np.column_stack((Xi, Zeta))

    with Pool(processes =n_processes) as pool:
        results = list(tqdm(pool.imap(do_pixel, full_stack), total = len(Xi), desc = 'Processing:'))
    return results

def flipper_component(arr, type = 'x'):
    if type == 'x':
        grid2 = np.array(arr).reshape(xi_resolution, zeta_resolution)
        grid4 = np.flipud(grid2)
        grid1 = -np.fliplr(grid2)
        grid3 = np.flipud(grid1)

        top_row = np.column_stack((grid1, grid2))
        bottom_row = np.column_stack((grid3, grid4))
        final_grid = np.row_stack((bottom_row, top_row))
    elif type == 'y':
        grid2 = np.array(arr).reshape(xi_resolution, zeta_resolution)
        grid4 = -np.flipud(grid2)
        grid1 = np.fliplr(grid2)
        grid3 = -np.flipud(grid1)

        top_row = np.column_stack((grid1, grid2))
        bottom_row = np.column_stack((grid3, grid4))
        final_grid = np.row_stack((bottom_row, top_row))
    elif type == 'total':
        grid2 = np.array(arr).reshape(xi_resolution, zeta_resolution)
        grid4 = np.flipud(grid2)
        grid1 = np.fliplr(grid2)
        grid3 = np.flipud(grid1)

        top_row = np.column_stack((grid1, grid2))
        bottom_row = np.column_stack((grid3, grid4))
        final_grid = np.row_stack((bottom_row, top_row))
    else:
        raise ValueError('Invalid type. Must be x or y or total (not for components).')
    return final_grid

def hessian(alpha_x, alpha_y):
        #x and y are labelled confusingly, but this is the correct form.
        # Compute the Hessian using finite differences
        dx = (X[0, 1] - X[0, 0]) 
        dy = (Y[1, 0] - Y[0, 0]) 
        
        # Compute the Hessian using finite differences
        alpha_x_y = np.gradient(alpha_x, dx, axis=0) #/ dy
        alpha_y_x = np.gradient(alpha_y, dy, axis=1) #/ dx
        alpha_x_x = np.gradient(alpha_x, dy, axis=1) #/ dx
        alpha_y_y = np.gradient(alpha_y, dx, axis=0) #/ dy

        return alpha_x_x, alpha_y_y, alpha_x_y, alpha_y_x

def detA(alpha_x, alpha_y, plot = False, sigma = 1):
    def plot_and_save(data, title, filename, label, X, Y, cmap='viridis'):
        plt.figure(figsize=(8, 6))
        plt.contourf(X, Y, data, 100, cmap=cmap)
        plt.colorbar(label=label)
        plt.title(title)
        plt.savefig(f"{filename}.png")
        plt.show()
        plt.close()

    alpha_x = -np.array(alpha_x)
    alpha_y = -np.array(alpha_y)

    alpha_x_grid = flipper_component(alpha_x, type='x')
    alpha_y_grid = flipper_component(alpha_y, type='y')

    alpha_x_grid = gaussian_filter(alpha_x_grid, sigma=sigma)
    alpha_y_grid = gaussian_filter(alpha_y_grid, sigma=sigma)

    alpha_mag = np.sqrt(alpha_x_grid**2 + alpha_y_grid**2)

    alpha_x_x, alpha_y_y, alpha_x_y, alpha_y_x = hessian(alpha_x_grid, alpha_y_grid)
    alpha_x_y = -alpha_x_y
    alpha_y_x = -alpha_y_x
    A_00 =  1 - alpha_x_x
    A_01 = - alpha_x_y
    A_10 = - alpha_y_x
    A_11 =  1 - alpha_y_y
    det_A = A_00 * A_11 - A_01 * A_10

    tang = 0.5* (A_00 + A_11) - np.sqrt(A_01*A_10 + 0.25*(A_11 - A_00)**2)
    rad = 0.5* (A_00 + A_11) + np.sqrt(A_01*A_10 + 0.25*(A_11 - A_00)**2)

    mask = det_A > 0
    Z_masked = np.ma.masked_where(mask, det_A)

    mu = 1 / det_A
    mag_mask = (mu > 10) | (mu < -10)
    mu_masked = np.ma.masked_where(mag_mask, mu)

    if plot:
        ##deflection angle components
        plot_and_save(alpha_x_grid, 'Alpha X', f"alpha_x_{np.degrees(inc):.2f}", 'Alpha X', X, Y)
        plot_and_save(alpha_y_grid, 'Alpha Y', f"alpha_y_{np.degrees(inc):.2f}", 'Alpha Y', X, Y)

        # #deflection angle alpha
        plot_and_save(alpha_mag, 'Alpha Magnitude', f"alpha_mag_{np.degrees(inc):.2f}", 'Alpha Magnitude', X, Y)

        # gamma components and total magnitude
        plot_and_save(A_11 - A_00, r'$\gamma_1$', f"gam1_{np.degrees(inc):.2f}", r'$\gamma_1$', X, Y)
        plot_and_save(-A_01, r'$\gamma_2$', f"gam2_{np.degrees(inc):.2f}", r'$\gamma_2$', X, Y)
        plot_and_save(np.sqrt((A_11 - A_00)**2 + A_01**2), r'$\gamma$', f"gamma_{np.degrees(inc):.2f}", r'$\gamma$', X, Y)

        # Determinant of the Jacobian
        plot_and_save(det_A, 'Determinant of the Jacobian', f"det_A_{np.degrees(inc):.2f}", 'Determinant of the Jacobian', X, Y)
        # Masked Determinant of the Jacobian
        plot_and_save(Z_masked, 'Masked Determinant of the Jacobian', f"Z_masked_{np.degrees(inc):.2f}", 'Determinant of the Jacobian', X, Y)
        # # Magnification
        plot_and_save(mu_masked, 'Magnification', f"mu_{np.degrees(inc):.2f}", 'Magnification', X, Y)

        plot_and_save(tang, 'Tangential Critical Curve', f"tang_{np.degrees(inc):.2f}", 'Tangential Critical Curve', X, Y)

        ##deflection angle derivatives
        plot_and_save(alpha_x_x, r'$\alpha_{xx}$', f"alpha_x_x_{np.degrees(inc):.2f}", r'$\alpha_{xx}$', X, Y)
        plot_and_save(alpha_y_y, r'$\alpha_{yy}$', f"alpha_y_y_{np.degrees(inc):.2f}", r'$\alpha_{yy}$', X, Y)
        plot_and_save(alpha_x_y, r'$\alpha_{xy}$', f"alpha_x_y_{np.degrees(inc):.2f}", r'$\alpha_{xy}$', X, Y)
        plot_and_save(alpha_y_x, r'$\alpha_{yx}$', f"alpha_y_x_{np.degrees(inc):.2f}", r'$\alpha_{yx}$', X, Y)

    return alpha_x_grid, alpha_y_grid, det_A, Z_masked , tang, rad
    
def density_map(rho_map):
        for i, rho in enumerate(rho_map):
            rho = np.concatenate((rho[:, ::-1], rho), axis=1)
            rho = rho[:, :, np.newaxis]
            rho_map[i] = rho  # Update the rho_map list with the new rho
        return rho_map

def load_npy_files(folder_path):
    # List to store the loaded arrays
    arrays = []
    # Loop through all files in the directory
    for file_name in os.listdir(folder_path):
        # Check if the file is a .npy file
        if file_name.endswith('.npy'):
            # Construct the full path of the file
            file_path = os.path.join(folder_path, file_name)
            # Load the .npy file and append it to the list
            arrays.append(np.load(file_path))
    return arrays

if __name__ == '__main__':
    #load in the densities and flip them to make the full grid
    folder_path = 'densities/'
    rho_map = load_npy_files(folder_path)
    print(f"Loaded {len(rho_map)} arrays.")
    rho_flipped = density_map(rho_map)

    # Define the inclination angles you want to loop over (in radians)
    inc_start = 70
    inc_end = 90
    inclination_angles = np.linspace(np.deg2rad(inc_start), np.deg2rad(inc_end), (inc_end - inc_start) + 1)

    # Create a grid for finding deflection angles and derivatives
    plotx = np.linspace(xi_lower_bound, xi_upper_bound, xi_resolution)
    plotx = np.append(-np.flip(plotx), plotx)
    ploty = np.linspace(zeta_lower_bound, zeta_upper_bound, zeta_resolution)
    ploty = np.append(-np.flip(ploty), ploty)
    X, Y = np.meshgrid(plotx, ploty)
    
    for inc in inclination_angles:
        print(f"Calculating for inclination angle: {np.degrees(inc):.2f}")
        #initialize the Disc class with the flipped density map and inclination angle
        ts = Disc(rho_flipped, inc)
        dummy_area = []
        
        # The main calculation function that returns alpha_x and alpha_y for each pixel
        results = inclination_map()

        #Makes directories for each inclination angle to store results
        folder_ax = f'inc_{np.degrees(inc)}/alpha_x'
        folder_ay = f'inc_{np.degrees(inc)}/alpha_y'
        folder_tang = f'inc_{np.degrees(inc)}/tang'
        folder_rad = f'inc_{np.degrees(inc)}/rad'
        os.makedirs(folder_ax, exist_ok=True)
        os.makedirs(folder_ay, exist_ok=True)
        os.makedirs(folder_tang, exist_ok=True)
        os.makedirs(folder_rad, exist_ok=True)

        # Loop over each array in rho_flipped (i.e., each entry in alpha_x_list and alpha_y_list)
        for i in range(len(rho_flipped)):
            # Extract the i-th alpha_x and alpha_y for all pixels from the results
            alpha_x = np.array([alpha_x_list[i] for alpha_x_list, _ in results])
            alpha_y = np.array([alpha_y_list[i] for _, alpha_y_list in results])

            # Reshape alpha_x_grid and alpha_y_grid to match the grid shape
            alpha_x = alpha_x.reshape(xi_resolution, zeta_resolution)
            alpha_y = alpha_y.reshape(xi_resolution, zeta_resolution)

    # Now compute det_A, Z_masked, and caustic area for this specific rho_flipped array
            alpha_x_grid, alpha_y_grid, det_A, Z_masked, tang, rad = detA(alpha_x, alpha_y, plot=False)
            ax_path = os.path.join(folder_ax, f'alpha_x_{i}.npy')
            np.save(ax_path, alpha_x_grid)
            ay_path = os.path.join(folder_ay, f'alpha_y_{i}.npy')
            np.save(ay_path, alpha_y_grid)
            tang_path = os.path.join(folder_tang, f'tang_{i}.npy')
            np.save(tang_path, tang)
            rad_path = os.path.join(folder_rad, f'rad_{i}.npy')
            np.save(rad_path, rad)













