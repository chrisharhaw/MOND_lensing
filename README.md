# A Novel Test for MOND: Gravitational Lensing by Disc Galaxies 
Christopher Harvey-Hawes and Marco Galoppo
(May 2025)

https://arxiv.org/abs/2411.17888

## Introduction
The series of python scripts given in this repository generate a density distribution for a disc galaxy, for both the MOND and Dark Matter paradigm, calculates the corresponding deflection angle associated with the inclination of the disc galaxy, and finally calculates the area contained within the caustic lines produced. 

The density profile choosen to represent a disc galaxy is that of an exponential disc with a central spherical bulge modelled by a Plummer model. 

The scripts should be straightforward to read and relatively easily modified for an individual's specific user case. However, please contact chrish-h@hotmail.co.uk if you have any issues or further questions. 

## Usage
Whilst it would have been more than possible to combine the scripts used into one file, we found it convenient, particularly for checking errors in development, to separate the processes into individual scripts. The following describes how to use the code in this repository.

1) Generate the density distribution/s that you would like the investigate using the density_grid.py script. You will need to choose:

    - the grid scale and precision though for our work we take  r_grid_precision_length = 500, 
    z_grid_precision_length = 500
    r_grid_bound_length = 40, 
    z_grid_bound_length = 20, 
    as standard. 

    - the interpolating function you wish to investigate. Here both a simple MOND and RAR function are used, but this can be easily adapted if needed.

    - The disc galaxy parameters, namely: the disc mass, disc radius, bulge mass, bulge radius, and disc thickness.

    The script will output this array as a .npy file, taking only the $r>0$, and $z>0$ quadrant, making use of the symmetries of the system. 

2) Calculate the deflection angle components over a ($\xi_1$, $\xi_2$) grid on the sky for each density distribution at a particular inclination angle via the alpha_rad_tang.py script. From these, all the lensing properties can be later determined, i.e, deflection angle, magnification, shear, caustic curves, etc. 

    You will need to determine the precision of the 2D ($\xi_1$, $\xi_2$) grid, which will be the main hurdle in computation, as this script will require substantial computational power to run. For our purposes we ran a 5x5 arcsecond grid, with a resolution of 100 pixels each direction, parallelised over 80 cpus. 
    (Each cpu will handle the calculation for a different pixel independently, so the running speed scales well with increased available cpus. )

    You will also need to specify the range of inclination angles relative to the observer you wish to investigate. 

    The script will output a series of directories for each inclination specified, each containing the components of the deflection angle, and the radial and tangential eigenvalues of the inverse magnification matrix. These will be .npy files indexed alphabetically by the file names of the density arrays originally loaded. 

3) The caustic_areas.py script will then load in the numpy arrays from the previous script and calculate the corresponding caustic areas, which will be outputted as another numpy array. 



