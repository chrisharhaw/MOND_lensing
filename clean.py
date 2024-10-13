import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM 
from scipy.integrate import quad, quad_vec
from tqdm import tqdm
from multiprocessing.pool import Pool
from matplotlib.path import Path
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


n_processes = 30 # number of processes to run in parallel

# Define the resolution and bounds of the grid
xi_resolution = 25
zeta_resolution = 25

xi_upper_bound = 4 #0.6552855003153877 
xi_lower_bound = xi_upper_bound / (2*xi_resolution +1) #0.012843595806181598 #aka the centre of the grid
zeta_upper_bound = 4 #0.6552855003153877 
zeta_lower_bound = zeta_upper_bound / (2*zeta_resolution+1) #0.012843595806181598 #aka the centre of the grid


#grid defining 3d
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
    def __init__(self,rho_map, inc = np.pi/2):
        self.G = 4.302e-6 # kpc(km/s)^2 / Msolar or  6.67e-11  m^3 kg^-1 s^-2
        self.c = 299792.458 # km/s
        self.conv = 180*3600/np.pi # radians to arcsec
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.zl = 0.5
        self.zs = 2.00
        self.Dl = self.cosmo.angular_diameter_distance(self.zl).value*1e3
        self.Dls = self.cosmo.angular_diameter_distance_z1z2(self.zl, self.zs).value*1e3
        self.Ds = self.cosmo.angular_diameter_distance(self.zs).value*1e3
        self.fudge_factor = 1e-20
        self.my_sin = np.sin(inc)
        self.my_cos = np.cos(inc)

        R, Z, Phi = np.meshgrid(r, z, phi, indexing = 'ij') #no point recalulating mesh
        self.R = R.flatten() # and for part_inclination_map
        self.Z = Z.flatten()
        self.Phi = Phi.flatten()
        self.X = self.R*np.cos(self.Phi)
        self.Y = self.R*np.sin(self.Phi)

        self.my_b = 2*(self.X*self.my_sin - self.Z*self.my_cos)  #no need to calculate on every loop as no dependence on xi or zeta
        self.rho_map = rho_map
        self.const = (2*self.G/(self.c**2))*self.conv # physical conversion and from radians to arcsec

    def part_inclination_map(self, xi, zeta):
        xi = (xi*self.Dl)/self.conv
        zeta = (zeta*self.Dl)/self.conv
        my_a = self.X**2 + (self.Y - zeta)**2 + self.Z**2 + xi**2 -2*xi*(self.X*self.my_cos + self.Z*self.my_sin)  
        my_Delta = -self.my_b**2 + 4*my_a

        my_dem =np.where(np.abs(my_Delta) <= 1e-16, 0, 8 / my_Delta)
        part_inclination_xi = ((-xi + self.X * self.my_cos + self.Z*self.my_sin)*my_dem).reshape(len(r), len(z), len(phi)) #varies with coordinates?
        part_inclination_zeta = ((self.Y - zeta)*my_dem).reshape(len(r), len(z), len(phi))

        return part_inclination_xi, part_inclination_zeta
	
    def alpha(self,xi,zeta):
        xi_inclinaton, zeta_inclination = self.part_inclination_map(xi, zeta)

        alpha_xi = np.nansum(self.rho_map*xi_inclinaton)*dV*self.const * (self.Dls/self.Ds)
        alpha_zeta = np.nansum(self.rho_map*zeta_inclination)*dV*self.const * (self.Dls/self.Ds)

        #alpha = np.sqrt(alpha_xi**2 + alpha_zeta**2) #dV changes with coordinates
        return alpha_xi, alpha_zeta
    
class Density():
    def __init__(self):
        self.G = 4.302e-6 # kpc(km/s)^2 / Msolar or  6.67e-11  m^3 kg^-1 s^-2
        self.c = 299792.458 # km/s
        self.Md = 8.1e11 # disc mass in Msolar
        self.rd = 2.1 # disc radius in kpc
        self.zd = 0.7 # disc hegight in kpc
        self.Mb = 8e9 # bulge mass in Msolar
        self.Rb = 0.8 # bulge scale length in Msolar
        self.sigv = 200 #km/s
        self.km2kpc = 3.24078e-17
        self.fudge = 1e-15

        R, Z, Phi = np.meshgrid(r, z, phi, indexing = 'ij') #no point recalulating mesh
        #R, Z = np.meshgrid(r, z, indexing = 'ij') #no point recalulating mesh
        self.rr = R #need mesh in different forms for density_map 
        self.zz = Z
        self.phiphi = Phi

    def densitiesBaryons(self):
        # Calculate the disk density (rhod)
        rhod = (self.Md / (4 * np.pi * self.rd**2 * self.zd)) * np.exp(-self.rr / self.rd) * np.exp(-np.abs(self.zz) / self.zd)
        # Calculate the bulge density (rhob)
        rhob = (3 * self.rd**2 * self.Mb) / (4 * np.pi * (self.Rb**2 + self.rr**2 + self.zz**2)**(5/2))
        # Return the combined density
        return (rhod + rhob)*self.rr
       
def do_pixel(var):
    Xi = var[0]
    Zeta = var[1]
    alpha_x , alpha_y  = ts.alpha(Xi,Zeta)
    return alpha_x, alpha_y

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

def detA(results, plot = True, sigma = 1):
    def plot_and_save(data, title, filename, label, X, Y, cmap='viridis'):
        plt.figure(figsize=(8, 6))
        plt.contourf(X, Y, data, 100, cmap=cmap)
        plt.colorbar(label=label)
        plt.title(title)
        plt.savefig(f"{filename}.png")
        plt.show()
        plt.close()
    
    alpha_x, alpha_y = zip(*results)
    alpha_x = np.array(alpha_x)
    alpha_y = np.array(alpha_y)

    alpha_x_grid = flipper_component(alpha_x, type='x')
    alpha_y_grid = flipper_component(alpha_y, type='y')

    alpha_x_grid = gaussian_filter(alpha_x_grid, sigma=sigma)
    alpha_y_grid = gaussian_filter(alpha_y_grid, sigma=sigma)

    alpha_mag = np.sqrt(alpha_x_grid**2 + alpha_y_grid**2)

    alpha_x_x, alpha_y_y, alpha_x_y, alpha_y_x = hessian(alpha_x_grid, alpha_y_grid)
    A_00 =  1 - alpha_x_x
    A_01 = - alpha_x_y
    A_10 = - alpha_y_x
    A_11 =  1 - alpha_y_y
    det_A = A_00 * A_11 - A_01 * A_10

    tang = 0.5* (A_00 + A_11) - np.sqrt(A_01*A_10 + 0.25*(A_11 - A_00)**2)

    mask = det_A > 0
    Z_masked = np.ma.masked_where(mask, det_A)

    mu = 1 / det_A
    mag_mask = mu > 10 
    mu_masked = np.ma.masked_where(mag_mask, mu)

    if plot:
        ##deflection angle components
        # plot_and_save(alpha_x_grid, 'Alpha X', f"alpha_x_{np.degrees(inc):.2f}", 'Alpha X', X, Y)
        # plot_and_save(alpha_y_grid, 'Alpha Y', f"alpha_y_{np.degrees(inc):.2f}", 'Alpha Y', X, Y)

        #deflection angle alpha
        plot_and_save(alpha_mag, 'Alpha Magnitude', f"alpha_mag_{np.degrees(inc):.2f}", 'Alpha Magnitude', X, Y)

        # gamma components and total magnitude
        plot_and_save(A_11 - A_00, r'$\gamma_1$', f"gam1_{np.degrees(inc):.2f}", r'$\gamma_1$', X, Y)
        plot_and_save(-A_01, r'$\gamma_2$', f"gam2_{np.degrees(inc):.2f}", r'$\gamma_2$', X, Y)
        plot_and_save(np.sqrt((A_11 - A_00)**2 + A_01**2), r'$\gamma$', f"gamma_{np.degrees(inc):.2f}", r'$\gamma$', X, Y)

        # Determinant of the Jacobian
        plot_and_save(det_A, 'Determinant of the Jacobian', f"det_A_{np.degrees(inc):.2f}", 'Determinant of the Jacobian', X, Y)
        # Masked Determinant of the Jacobian
        plot_and_save(Z_masked, 'Masked Determinant of the Jacobian', f"Z_masked_{np.degrees(inc):.2f}", 'Determinant of the Jacobian', X, Y)
        # Magnification
        plot_and_save(mu_masked, 'Magnification', f"mu_{np.degrees(inc):.2f}", 'Magnification', X, Y)

        plot_and_save(tang, 'Tangential Critical Curve', f"tang_{np.degrees(inc):.2f}", 'Tangential Critical Curve', X, Y)

        ##deflection angle derivatives
        # plot_and_save(alpha_x_x, r'$\alpha_{xx}$', f"alpha_x_x_{np.degrees(inc):.2f}", r'$\alpha_{xx}$', X, Y)
        # plot_and_save(alpha_y_y, r'$\alpha_{yy}$', f"alpha_y_y_{np.degrees(inc):.2f}", r'$\alpha_{yy}$', X, Y)
        # plot_and_save(alpha_x_y, r'$\alpha_{xy}$', f"alpha_x_y_{np.degrees(inc):.2f}", r'$\alpha_{xy}$', X, Y)
        # plot_and_save(alpha_y_x, r'$\alpha_{yx}$', f"alpha_y_x_{np.degrees(inc):.2f}", r'$\alpha_{yx}$', X, Y)

    return alpha_x_grid, alpha_y_grid, det_A, Z_masked, tang

if __name__ == '__main__':
    dn = Density()
    rho_map = dn.densitiesBaryons()
  
    inc = np.pi/2
    ts = Disc(rho_map, inc)
    results = inclination_map()

    plotx = np.linspace(xi_lower_bound, xi_upper_bound, xi_resolution)
    plotx = np.append(-np.flip(plotx), plotx)
    X, Y = np.meshgrid(plotx, plotx)
    
    alpha_x_grid, alpha_y_grid, det_A, Z_masked, tang = detA(results)
    
    # Find the contour where det_A = 0
    contour = plt.contour(X, Y, det_A, levels=[0], colors='red')
    plt.close()  # Close the plot as we only need the contour data

    # Extract the contour points
    contour_paths = contour.collections[0].get_paths()
    contour_points = contour_paths[0].vertices

    # Separate the x and y coordinates of the contour points
    X_crit = contour_points[:, 0]
    Y_crit = contour_points[:, 1]

    print(np.sqrt(X_crit**2 + Y_crit**2))

    # Interpolate alpha_x and alpha_y at the critical curve points
    alpha_x_crit = griddata((X.flatten(), Y.flatten()), alpha_x_grid.flatten(), (X_crit, Y_crit), method='linear')
    alpha_y_crit = griddata((X.flatten(), Y.flatten()), alpha_y_grid.flatten(), (X_crit, Y_crit), method='linear')


    # Check for NaNs in the interpolated values and handle them
    if np.any(np.isnan(alpha_x_crit)) or np.any(np.isnan(alpha_y_crit)):
        print("Warning: NaNs detected in interpolated alpha values. Using nearest method for interpolation.")
        alpha_x_crit = griddata((X.flatten(), Y.flatten()), alpha_x_grid.flatten(), (X_crit, Y_crit), method='nearest')
        alpha_y_crit = griddata((X.flatten(), Y.flatten()), alpha_y_grid.flatten(), (X_crit, Y_crit), method='nearest')

    # Project the critical curve points onto the source plane using the lens equation
    beta_x = X_crit -  alpha_x_crit
    beta_y = Y_crit -  alpha_y_crit

    # Sort the points to form a closed polygon
    points = np.column_stack((beta_x, beta_y))
    path = Path(points)
    sorted_points = path.vertices

    # Compute the area enclosed by the polygon using the shoelace formula
    x = sorted_points[:, 0]
    y = sorted_points[:, 1]
    caustic_area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    print(f"Area of the {inc} Caustic: {caustic_area}")

    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Z_masked, 100, cmap='viridis')
    plt.colorbar(label='Determinant of the Jacobian')
    plt.plot(X_crit, Y_crit, 'r-', label='Critical Curve')
    plt.plot(beta_x, beta_y, 'b-', label='Caustic Curve')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Critical Curve and Caustic Curve')
    plt.savefig(f"caustics_{np.degrees(inc):.2f}.png")
    plt.show()
   

