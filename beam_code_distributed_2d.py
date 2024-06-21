import pdb
import numpy as np
from scipy.integrate import cumtrapz
import pandas as pd
from io import StringIO
from scipy.interpolate import interp1d
import os, glob

def read_struct_points(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            try:
                x, y, z = map(float, line.split())
                data.append([x, y, z])
            except ValueError:
                break
    return pd.DataFrame(data, columns=['X', 'Y', 'Z'])

def read_surfacesection_loads(file_path):
    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find the start of the table
    start_index = 0

    for i, line in enumerate(lines):
        if 'Offset, Chord, X_QC, Z_QC, Fx, Fz, Moment' in line:
            start_index = i + 1  # The table starts after this line
        if 'Force Units:' in line:
            end_index = i - 2
            break

    # Extract table data
    table_data = lines[start_index+1:end_index+1]

    # Use Pandas to read the table data
    data = pd.read_csv(StringIO('\n'.join(table_data)), sep=",", header=None)
    data.columns = ['Y', 'Chord', 'X', 'Z', 'Fx', 'Fz', 'My', 'empty']
    # Convert to NumPy array
    # numpy_array = data.to_numpy()

    return data

def beam_deflection(x, w, E, I):
    """
    Calculate beam deflection

    Parameters:
    x (array): Points along the beam where the moments are applied.
    w (array): Distributed forces along the points.
    E (float): Modulus of elasticity of the material.
    I (float): Polar moment of inertia of the beam's cross-section.
    """
    # Number of points
    n = len(x)
    
    # Initialize bending moment and deflection arrays
    M = np.zeros(n)
    delta = np.zeros(n)

    # Integrate to find bending moment for a cantilever beam
    # The moment needs to be accumulated from the free end
    for i in range(n-2, -1, -1):
        dx = x[i+1] - x[i]
        avg_w = (w[i] + w[i+1]) / 2
        M[i] = M[i+1] + avg_w * dx

    # Second integration for deflection
    # For a cantilever, integrate from the free end
    delta = cumtrapz(cumtrapz(M, x, initial=0), x, initial=0) / (E * I)

    return delta, M

def cantilever_torsion(x, m, G, J):
    """
    Calculate the angle of twist and torsional moment in a cantilever beam under a distributed moment.

    Parameters:
    x (array): Points along the beam where the moments are applied.
    m (array): Distributed torsional moments at the points.
    G (float): Modulus of rigidity of the material.
    J (float): Polar moment of inertia of the beam's cross-section.

    Returns:
    tuple: Angle of twist and torsional moment at each point.
    """
    n = len(x)
    
    # Initialize torsional moment and angle of twist arrays
    T = np.zeros(n)
    theta = np.zeros(n)

    # Calculate torsional moment
    for i in range(n-2, -1, -1):
        dx = x[i+1] - x[i]
        avg_m = (m[i] + m[i+1]) / 2
        T[i] = T[i+1] + avg_m * dx

    # Calculate angle of twist
    theta = cumtrapz(T / (G * J), x, initial=0)

    return theta, T

def interpolate_aero_to_structural(Xa, Fz, Xs):
    """
    Interpolates aerodynamic loads to structural nodes.

    Parameters:
    Xa (array): Aerodynamic node points.
    Fz (array): Loads at aerodynamic nodes.
    Xs (array): Structural node points.

    Returns:
    array: Interpolated loads at structural nodes.
    """
    # Create a linear interpolation function
    interp_func = interp1d(Xa, Fz, kind='linear', fill_value='extrapolate')

    # Use this function to find loads at structural node points
    Fz_interpolated = interp_func(Xs)

    return Fz_interpolated

def rotate_point_displacement(x1, y1, x2, y2, theta):

    # Translation to the origin
    x2_translated = x2 - x1
    y2_translated = y2 - y1

    # Rotation matrix
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    # Apply the rotation
    x2_rot, y2_rot = R @ np.array([x2_translated, y2_translated])

    # Calculate displacements
    dx = x2_rot - x2_translated
    dy = y2_rot - y2_translated
    
    return dx, dy

######
# Main start
# Read surface section loads output from FSI
# output columns: Offset, Chord, X_QC, Z_QC, Fx, Fz, Moment
load_df = read_surfacesection_loads('FS_SurfaceSection_Loads.txt') 
struct_df = read_struct_points('Structural_Nodes_2D.txt')
struct_lra = struct_df[0:21] # nodes on the load reference axis: HARDCODED

# Example usage
I = 1  # Moment of Inertia, m^4
J = 1 # torsion moment
E = 2.e4  # Modulus of Elasticity, Pa; replaced with E*I
G = 1.e4 # modulus of rigidity; replaced with G*J

# Example load distribution (x in meters, w in Newtons/meter)
aero_y = load_df.Y
aero_fz = load_df.Fz 
aero_my = load_df.My 

struct_fz = interpolate_aero_to_structural(aero_y, aero_fz, struct_lra.Y) 
struct_my = interpolate_aero_to_structural(aero_y, aero_my, struct_lra.Y)
struct_acx = interpolate_aero_to_structural(aero_y, load_df.X, struct_lra.Y) # get nearest aero X location to structural node
rad = struct_lra.X - struct_acx 
aero_moment = rad * struct_fz # aero moment caused by lift around elastic axis
struct_my = aero_moment + struct_my

# Calculate beam deflections
delta_z, M = beam_deflection(struct_lra.Y, struct_fz, E, I)
theta, T = cantilever_torsion(struct_lra.Y, struct_my, G, J)

# calculate beam deflection due to bending angle
psi = np.arcsin(delta_z / struct_lra.Y) 
psi = psi.fillna(0)
yp = struct_lra.Y * np.cos(psi) # projection of beam deflection on y-axis
delta_y = -1*(struct_lra.Y - yp ) # the difference should be subtracted from the final deflection
delta_y = delta_y.fillna(0)

# now go through ribs and calculate dz dz due to local twist angle
struct_fore = struct_df[len(struct_lra):len(struct_lra)*2].reset_index() # the nodes on the ribs
struct_aft = struct_df[len(struct_lra)*2:].reset_index() # the nodes on the ribs

delta_fore = np.array([[0., 0.]]*len(struct_fore))
delta_aft = np.array([[0., 0.]]*len(struct_aft))

for ii, node in struct_lra.iterrows():
    dx_fore, dz_fore = rotate_point_displacement(node.X, node.Z, struct_fore.X[ii], struct_fore.Z[ii],  -theta[ii])
    dx_aft, dz_aft   = rotate_point_displacement(node.X, node.Z, struct_aft.X[ii],  struct_aft.Z[ii],   -theta[ii])
    
    dz_fore += delta_z[ii]
    dz_aft += delta_z[ii]
    delta_fore[ii, :] = np.array([dx_fore, dz_fore]) # has same dy as the spar
    delta_aft[ii, :] = np.array([dx_aft, dz_aft])

# insert delta_y
delta_fore = np.insert(delta_fore, 1, delta_y, axis=1)
delta_aft = np.insert(delta_aft, 1, delta_y, axis=1)

# Write beam absolute displacements and dt values to output file
# these are not deltas off the previous deflected beam, but rather absolute values off of the baseline undeflected beam
with open('FSIDisp.txt', 'w') as file:
    for dy, dz in zip(delta_y, delta_z):
        file.write(f"0 {dy} {dz}\n")
    
    for dx, dy, dz in delta_fore:
        file.write(f"{dx} {dy} {dz} \n")

    for dx, dy, dz in delta_aft:
        file.write(f"{dx} {dy} {dz} \n")

##### Write out file history
# Count existing FSIDisp files
existing_files_count = len(glob.glob('FSIDisp_*.txt'))


# Create a copy with an index
new_filename = f'FSIDisp_{existing_files_count + 1}.txt'
os.system(f'copy FSIDisp.txt {new_filename}')


# Count existing AeroLoad files
existing_files_count = len(glob.glob('AeroLoad_*.txt'))

# Write Y and Fz columns to AeroLoad.txt
with open('AeroLoad.txt', 'w') as file:
    for y, fz in zip(load_df.Y, load_df.Fz):
        file.write(f"{y} {fz}\n")

# Create a copy with an index
new_filename = f'AeroLoad_{existing_files_count + 1}.txt'
os.system(f'copy AeroLoad.txt {new_filename}')
