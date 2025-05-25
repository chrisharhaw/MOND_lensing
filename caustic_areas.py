import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.interpolate import griddata
import os

plt.rc('font', family='serif')

# Define the resolution and bounds of the grid
xi_resolution = 100 #100
zeta_resolution = 100 #100

xi_upper_bound = 5 
xi_lower_bound = xi_upper_bound / (2*xi_resolution +1)  #aka the centre of the grid
zeta_upper_bound = 5  
zeta_lower_bound = zeta_upper_bound / (2*zeta_resolution+1) #aka the centre of the grid

def caustics(X, Y, alpha_x_grid, alpha_y_grid, tang, rad, i):
    empty_array = np.array([])
    dummy_fig, dummy_ax = plt.subplots()
    
    # Initialize all outputs as empty
    beta_x_tang = empty_array
    beta_y_tang = empty_array
    beta_x_rad = empty_array
    beta_y_rad = empty_array
    X_crit_tang = empty_array
    Y_crit_tang = empty_array
    
    try:
        # Compute tangential contours (if they exist)
        contour_tang = dummy_ax.contour(X, Y, tang, levels=[0], colors='red')
        if (len(contour_tang.collections) > 0 and 
            len(contour_tang.collections[0].get_paths()) > 0):
            contour_tang_points = contour_tang.collections[0].get_paths()[0].vertices
            X_crit_tang = contour_tang_points[:, 0]
            Y_crit_tang = contour_tang_points[:, 1]
            alpha_x_crit_tang = griddata((X.flatten(), Y.flatten()), alpha_x_grid.flatten(), 
                                         (X_crit_tang, Y_crit_tang), method='linear')
            alpha_y_crit_tang = griddata((X.flatten(), Y.flatten()), alpha_y_grid.flatten(), 
                                        (X_crit_tang, Y_crit_tang), method='linear')
            beta_x_tang = X_crit_tang - alpha_x_crit_tang
            beta_y_tang = Y_crit_tang - alpha_y_crit_tang

        # Compute radial contours (if they exist)
        contour_rad = dummy_ax.contour(X, Y, rad, levels=[0], colors='red', linestyles='dashed')
        if (len(contour_rad.collections) > 0 and 
            len(contour_rad.collections[0].get_paths()) > 0):
            contour_rad_points = contour_rad.collections[0].get_paths()[0].vertices
            X_crit_rad = contour_rad_points[:, 0]
            Y_crit_rad = contour_rad_points[:, 1]
            alpha_x_crit_rad = griddata((X.flatten(), Y.flatten()), alpha_x_grid.flatten(), 
                                       (X_crit_rad, Y_crit_rad), method='linear')
            alpha_y_crit_rad = griddata((X.flatten(), Y.flatten()), alpha_y_grid.flatten(), 
                                       (X_crit_rad, Y_crit_rad), method='linear')
            beta_x_rad = X_crit_rad - alpha_x_crit_rad
            beta_y_rad = Y_crit_rad - alpha_y_crit_rad

        plt.close(dummy_fig)

        return beta_x_tang, beta_y_tang, beta_x_rad, beta_y_rad, X_crit_tang, Y_crit_tang

    except Exception as e:
        plt.close(dummy_fig)
        print(f"Error in caustics: {e}")
        return empty_array, empty_array, empty_array, empty_array, empty_array, empty_array

if __name__ == '__main__':

    # Define the inclination angles you want to loop over (in radians)
    inc_start = 70
    inc_end = 90
    inclination_angles = np.linspace(np.deg2rad(inc_start), np.deg2rad(inc_end), (inc_end - inc_start) + 1)


    plotx = np.linspace(xi_lower_bound, xi_upper_bound, xi_resolution)
    plotx = np.append(-np.flip(plotx), plotx)
    ploty = np.linspace(zeta_lower_bound, zeta_upper_bound, zeta_resolution)
    ploty = np.append(-np.flip(ploty), ploty)
    X, Y = np.meshgrid(plotx, ploty)
    casutic_area = []
    
    for inc in inclination_angles:
        print(f"Calculating for inclination angle: {np.degrees(inc):.2f}")
        dummy_area = []

        #Define the directories for the current inclination angle
        folder_ax = f'inc_{np.degrees(inc):.1f}/alpha_x'
        folder_ay = f'inc_{np.degrees(inc):.1f}/alpha_y'
        folder_tang = f'inc_{np.degrees(inc):.1f}/tang'
        folder_rad = f'inc_{np.degrees(inc):.1f}/rad'
        
        # Get the number of files to load (assuming same number of files in each folder)
        num_files = len(os.listdir(folder_ax))  # Number of files in one folder

        # Loop over each file inde
        for i in range(num_files):
            # Construct file paths for each data type
            ax_path = os.path.join(folder_ax, f'alpha_x_{i}.npy')
            ay_path = os.path.join(folder_ay, f'alpha_y_{i}.npy')
            tang_path = os.path.join(folder_tang, f'tang_{i}.npy')
            rad_path = os.path.join(folder_rad, f'rad_{i}.npy')

            # Load the data using np.load
            alpha_x_grid = np.load(ax_path)
            alpha_y_grid = np.load(ay_path)
            tang = np.load(tang_path)
            rad = np.load(rad_path)

            # Now you can process or analyze the loaded data
            print(f"Loaded data for inclination angle: {np.degrees(inc):.2f}, file index: {i}")
           
            beta_x_tang, beta_y_tang, beta_x_rad, beta_y_rad, X_crit_tang, Y_crit_tang = caustics( X, Y, alpha_x_grid, alpha_y_grid, tang, rad, i)
            
            # Sort the points to form a closed polygon
            points = np.column_stack((beta_x_tang, beta_y_tang))
            path = Path(points)
            sorted_points = path.vertices

            # Compute the area enclosed by the polygon using the shoelace formula
            x = sorted_points[:, 0]
            y = sorted_points[:, 1]
            caustic_area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            dummy_area.append(caustic_area)
        casutic_area.append(dummy_area)
    np.save('caustic_areas.npy', casutic_area)








