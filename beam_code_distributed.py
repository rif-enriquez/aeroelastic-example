import pdb, argparse
import numpy as np
from scipy.integrate import cumtrapz
import pandas as pd
from io import StringIO
from scipy.interpolate import interp1d
import os, glob
from anastruct import SystemElements

def read_displacement_points(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            try:
                x, y, z = map(float, line.split())
                data.append([x, y, z])
            except ValueError:
                break
    return pd.DataFrame(data, columns=['dx', 'dy', 'dz'])
    
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

def cantilever_torsion(x, T, G, J):
    """
    Calculate the twist at each point along a beam given a vector of points, a vector of torques at each point,
    and the constants G (shear modulus) and J (polar moment of inertia).

    Parameters:
    x (np.array): The points along the beam (monotonically increasing).
    T (np.array): The applied torques at each point corresponding to x.
    G (float): Shear modulus of the material.
    J (float): Polar moment of inertia of the cross-section.

    Returns:
    np.array: The twist at each point along the beam.
    """
    # Initialize the twist array with zeros
    theta = np.zeros_like(x)
    
    # Calculate the differential twist at each segment using the trapezoidal rule
    for i in range(1, len(x)):
        # Calculate the average torque between two points
        average_torque = (T[i] + T[i-1]) / 2.0
        
        # Calculate the twist contribution from the current segment
        segment_twist = (average_torque / (G * J)) * (x[i] - x[i-1])
        
        # Add this segment's twist to all subsequent points
        theta[i:] += segment_twist

    return theta

def anastruct_cantilever_beam(x, y, w, E, I):
    """
    Analyze a cantilever beam given specific points, loads, and material properties.

    Parameters:
    x (list of floats): The x-coordinates of the points where loads are applied.
    w (list of floats): The load values at each point in x.
    E (float): Young's Modulus of the beam.
    I (float): Second moment of area of the beam.
    """
    # The number of elements is one less than the number of points
    num_elements = len(x) - 1
    
    # Create a structural system
    ss = SystemElements()
    

    # Add elements to the system
    for i in range(num_elements):
        # ss.add_element(location=[[x[i], 0], [x[i+1], 0]], E=E, I=I)
        ss.add_element(location=[[x[i], y[i]], [x[i+1], y[i+1]]], EI=E*I)

    # Assign a fixed support at the first node (start of the beam)
    ss.add_support_fixed(node_id=1)

    # Apply loads
    # Assumption: loads are applied directly at the points between elements
    dx = x.iloc[1] - x.iloc[0]
    for i in range(1, len(w)):
        load = w[i]
        # load = 10 * dx # 10 N/m; distributed load test
        # Apply point load at the i-th node (1-based index)
        ss.point_load(node_id=i+1, Fz=load)
    #ss.point_load(node_id=i+1, Fz=10) # point load test
    
    # Solve the system
    ss.solve()
    
    # Get displacements at each node
    displacements = ss.get_node_displacements()

    ux =  ss.get_node_result_range('ux')
    uy =  ss.get_node_result_range('uy')
    slope =  ss.get_node_result_range('phi_z')
    #bending_moments = ss.get_bending_moment()
    # Show results: structure, bending moment, shear force, and displacement
    # ss.show_structure()
    # ss.show_bending_moment()
    # ss.show_shear_force()
    # ss.show_reaction_force()
    # xp, yp = ss.show_displacement(factor=1, scale=1, values_only=False)
    
    # output uy, moment, slope, ux
    return np.array(uy), np.array(ux)
    
def calculate_spanwise(x, dy):
    dx = np.zeros_like(x)
    for i in range(1, len(x)):
        delta_x = x[i] - x[i-1]
        delta_dy = dy[i] - dy[i-1]
        delta_dx = np.sqrt(max(delta_x**2 - delta_dy**2, 0)) - delta_x
        dx[i] = dx[i-1] + delta_dx
    return dx
    
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

def parse_args():
    parser = argparse.ArgumentParser(description='Aeroelastic example runner')
    parser.add_argument('--structural_nodes_file', type=str, default='Structural_nodes.txt',
                        help='File with structural nodes information')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    ########################
    # Read surface section loads output from FSI
    # output columns: Offset, Chord, X_QC, Z_QC, Fx, Fz, Moment
    aero_df = read_surfacesection_loads('FS_SurfaceSection_Loads.txt') 
    structural_nodes_file = args.structural_nodes_file
    struct_df = read_struct_points(structural_nodes_file)
    
    # with multiple csys defined, the offset goes to 0, so I had to add this line to define Y
    if np.average(aero_df.Y) < 0.1:
        aero_df.Y = np.linspace(0, struct_df.Y.iloc[-1], len(aero_df.Y)) 

    # Read previous displacement and apply to the structural mesh
    # Count existing FSIDisp files
    existing_files_count = len(glob.glob('FSIDisp_*.txt'))
    if existing_files_count > 0:
        last_file = f'FSIDisp_{existing_files_count}.txt'
        struct_delta = read_displacement_points(last_file)

        if existing_files_count > 2:
            # averaging last 2 deflection files helps FSI convergence
            last_file = f'FSIDisp_{existing_files_count-1}.txt'
            struct_delta2 = read_displacement_points(last_file)
            struct_delta.dx = (struct_delta.dx + struct_delta2.dx) /2
            struct_delta.dy = (struct_delta.dy + struct_delta2.dy) /2
            struct_delta.dz = (struct_delta.dz + struct_delta2.dz) /2

        struct_df.X = struct_df.X + struct_delta.dx
        struct_df.Y = struct_df.Y + struct_delta.dy
        struct_df.Z = struct_df.Z + struct_delta.dz 


    # Example usage
    I = 1  # Moment of Inertia, m^4
    J = 1 # torsion moment
    E = 2.e4  # Modulus of Elasticity, Pa; replaced with E*I (bending rigidity - N*m2)
    G = 1.e4 # modulus of rigidity; replaced with G*J (torsional rigidity  - N*m2)
    E2 = 5.e6 # Edgewise bending rigidity
    I2 = 1.
    # Example load distribution (x in meters, w in Newtons/meter)
    aero_df.X.iloc[-1] = aero_df.X.iloc[-2] # hardcode the aero center of the last point

    struct_fz = interpolate_aero_to_structural( aero_df.Y, aero_df.Fz, struct_df.Y) 
    struct_fx = interpolate_aero_to_structural( aero_df.Y, aero_df.Fx, struct_df.Y)
    struct_my = interpolate_aero_to_structural( aero_df.Y, aero_df.My, struct_df.Y)
    struct_acx = interpolate_aero_to_structural(aero_df.Y, aero_df.X, struct_df.Y) # get nearest aero X location to structural node

    # aero moment caused by lift around elastic axis
    rad = struct_df.X - struct_acx 
    aero_moment = rad * struct_fz 

    # Calculate beam deflections
    delta_z, _  = anastruct_cantilever_beam(struct_df.Y,  struct_df.Z, struct_fz, E, I)
    delta_x, _ = anastruct_cantilever_beam(struct_df.Y,  struct_df.X, struct_fx, E2, I2) # edgewise bending
    if '2D' in structural_nodes_file:
        theta = cantilever_torsion(struct_df.Y, aero_moment + struct_my, G, J)

    # calculate spanwise beam deflection due to large bending angle
    # psi = np.arcsin(delta_z / struct_df.Y)
    # yp = struct_df.Y * np.cos(psi) # projection of beam deflection on y-axis
    # delta_y = -1*(struct_df.Y - yp ) # the difference should be subtracted from the final deflection
    # delta_y = delta_y.fillna(0)
    # alternate method that preservers arc length
    delta_y = calculate_spanwise(struct_df.Y, delta_z)

    # Write beam displacements and dt values to output file
    with open('FSIDisp.txt', 'w') as file:
        for dx, dy, dz in zip(delta_x, delta_y, delta_z):
            file.write(f"{dx} {dy} {dz}\n")

    ##### Write out file history
    # Create a copy with an index
    new_filename = f'FSIDisp_{existing_files_count + 1}.txt'
    os.system(f'copy FSIDisp.txt {new_filename}')


    # Count existing AeroLoad files
    existing_files_count = len(glob.glob('AeroLoad_*.txt'))

    # Write Y and Fz columns to AeroLoad.txt
    with open('AeroLoad.txt', 'w') as file:
        for y, fz in zip(struct_df.Y, struct_fz):
            file.write(f"{y} {fz}\n")

    # Create a copy with an index
    new_filename = f'AeroLoad_{existing_files_count + 1}.txt'
    os.system(f'copy AeroLoad.txt {new_filename}')
