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
    Calculate the twist along a cantilever beam with a fixed root and free tip.

    Parameters:
    x (np.array): The points along the beam (monotonically increasing).
    T (np.array): The applied torques at each point corresponding to x.
    G (float): Shear modulus of the material.
    J (float): Polar moment of inertia of the cross-section.

    Returns:
    np.array: The twist at each point along the beam.
    """
    N = len(x) - 1  # Number of elements

    # Length of each element
    dx = np.diff(x)

    # Element stiffness matrix
    Ke = (G * J / dx[0]) * np.array([[1, -1], [-1, 1]])

    # Global stiffness matrix
    K = np.zeros((N+1, N+1))

    # Assemble global stiffness matrix
    for i in range(N):
        K[i:i+2, i:i+2] += Ke

    # Force vector (applied torques)
    F = np.zeros(N+1)
    F[1:] = T[:-1] * dx  # Apply torque to each node except the first one

    # Apply boundary conditions: theta_0 = 0 (fixed root)
    K = K[1:, 1:]  # Remove first row and column
    F = F[1:]      # Remove first entry

    # Solve for nodal twists
    theta = np.linalg.solve(K, F)

    # Add boundary condition theta_0 = 0 back to the solution
    theta = np.insert(theta, 0, 0) 
    amplify = 1

    return theta * 180 / np.pi * amplify # convert to degrees

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
    dx = x[1] - x[0]
    for i in range(0, len(w)):
        load = w[i] * dx
        # load = 10 * dx # 10 N/m; distributed load test

        ss.point_load(node_id=i+1, Fz=load) # nodes are (1-based index)

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
    if False:
        ss.show_structure()
        ss.show_bending_moment()
        ss.show_shear_force()
        ss.show_reaction_force()
        xp, yp = ss.show_displacement(factor=1, scale=1, values_only=False)
    
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

    Fz_total = np.trapz(Fz, Xa)
    Fz_int_total = np.trapz(Fz_interpolated, Xs)
    dfz = Fz_int_total - Fz_total
    ddfz =  dfz / len(Xs) * np.ones(len(Xs))
    Fz_interpolated = Fz_interpolated - ddfz # knockdown to ensure conservation of force

    return Fz_interpolated

def parse_args():
    parser = argparse.ArgumentParser(description='Aeroelastic example runner')
    parser.add_argument('-n', type=str, default='Structural_nodes.txt',
                        help='File with structural nodes information')
    return parser.parse_args()

def rotate_point_displacement(x0, z0, x1, z1, theta):
    """
    Apply a rotation to the point (x1, z1) around the point (x0, z0).

    Parameters:
    x0, z0 (float): Coordinates of the center of rotation.
    x1, z1 (float): Coordinates of the point to be rotated.
    theta (float): Rotation angle in degrees (counter-clockwise positive).

    Returns:
    dx, dz (float): Horizontal and vertical displacements of the point (x1, z1).
    """
    # Translate point (x1, z1) to the origin
    x1_translated = x1 - x0
    z1_translated = z1 - z0

    # Apply rotation
    theta = theta * np.pi / 180 # convert to radians
    x1_rotated = x1_translated * np.cos(theta) - z1_translated * np.sin(theta)
    z1_rotated = x1_translated * np.sin(theta) + z1_translated * np.cos(theta)

    # Translate back to the original position
    x1_new = x1_rotated + x0
    z1_new = z1_rotated + z0

    # Calculate displacements
    dx =-x1_new + x1
    dz = z1_new - z1

    return dx, dz

def calculate_weights(L, mu, N):
    # Length, mass/length, nodes
    g = 9.81  # acceleration due to gravity in m/s^2
    delta_L = L / (N - 1)  # length of each segment
    weight_per_segment = mu * delta_L * g  # weight of each segment

    weights = []
    for i in range(N):
        if i == 0 or i == N-1:  # boundary nodes
            weights.append(0.5 * weight_per_segment)
        else:  # internal nodes
            weights.append(weight_per_segment)

    return weights

if __name__ == "__main__":
    args = parse_args()

    # Beam Properties
    I = 1  # Moment of Inertia, m^4
    J = 1 # torsion moment
    E = 2.e4  # Modulus of Elasticity, Pa; replaced with E*I (bending rigidity - N*m2)
    G = 1.e4 # modulus of rigidity; replaced with G*J (torsional rigidity  - N*m2)
    E2 = 5.e6 # Edgewise bending rigidity
    I2 = 1.
    ml = 0.75 # kg/m mass/unit length

    ########################
    # Read surface section loads output from FSI
    # output columns: Offset, Chord, X_QC, Z_QC, Fx, Fz, Moment
    aero_df = read_surfacesection_loads('FS_SurfaceSection_Loads.txt') 
    structural_nodes_file = args.n
    struct_df = read_struct_points(structural_nodes_file)

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

        relax = 1.0 # relaxation factor
        struct_df.X = struct_df.X + relax*struct_delta.dx
        struct_df.Y = struct_df.Y + relax*struct_delta.dy
        struct_df.Z = struct_df.Z + relax*struct_delta.dz 

    # define the loard-reference axis
    struct_lra = struct_df # default for 1D beam case
    nodes = len(struct_lra.Y)

    # alternate defintion for 2D plate case
    if structural_nodes_file == 'Structural_nodes_2D.txt': 
        nodes = int(len(struct_df)/3) # N nodes on the load reference axis
        struct_lra = struct_df[0:nodes] 
 
    # undeflected beam propeties
    span = np.max(struct_lra.Y) 

    #  load distribution (x in meters, w in Newtons/meter)
    aero_df.X.iloc[-1] = aero_df.X.iloc[-2]
    struct_fz  = interpolate_aero_to_structural(aero_df.Y.values, aero_df.Fz.values, struct_lra.Y.values) # interp aero forces to structural
    struct_fx  = interpolate_aero_to_structural(aero_df.Y.values, aero_df.Fx.values, struct_lra.Y.values)
    struct_my  = interpolate_aero_to_structural(aero_df.Y.values, aero_df.My.values, struct_lra.Y.values)
    struct_acx = interpolate_aero_to_structural(aero_df.Y.values,  aero_df.X.values, struct_lra.Y.values) # interp aero center to structural

    # apply the inertial relief
    weights = calculate_weights(span, ml, nodes)
    # struct_fz = struct_fz - ml * 9.81 * np.ones(nodes)
    
    # Calculate beam deflections
    delta_z, _  = anastruct_cantilever_beam(struct_lra.Y.values,  struct_lra.Z.values, struct_fz, E, I)
    delta_x, _  = anastruct_cantilever_beam(struct_lra.Y.values,  struct_lra.X.values, struct_fx, E2, I2) # edgewise bending
    
    # calculate spanwise beam deflection due to large bending angle
    delta_y = calculate_spanwise(struct_lra.Y, delta_z)

    if structural_nodes_file == 'Structural_nodes_2D.txt': 
        # aero moment caused by lift around elastic axis
        # i dont understand what is the location of the struct_moment. 
        # i think it is about the reference axis
        rad = struct_lra.X - 0 * struct_acx 
        aero_moment = rad * struct_fz 

        theta = cantilever_torsion(struct_lra.Y, aero_moment + struct_my, G, J)

        # apply the twist to the forward and aft beam nodes
        # now go through ribs and calculate dz dz due to local twist angle
        struct_fore = struct_df[nodes:nodes*2].reset_index() # the nodes on the ribs
        struct_aft = struct_df[nodes*2:].reset_index() # the nodes on the ribs

        delta_fore = np.zeros([nodes, 2]) # init arrays
        delta_aft = np.zeros([nodes, 2])

        for ii, node in struct_lra.iterrows():
            dx_fore, dz_fore = rotate_point_displacement(node.X, node.Z, struct_fore.X[ii], struct_fore.Z[ii],  -theta[ii])
            dx_aft, dz_aft   = rotate_point_displacement(node.X, node.Z, struct_aft.X[ii],  struct_aft.Z[ii],   -theta[ii])

            dz_fore += delta_z[ii]
            dz_aft += delta_z[ii]
            delta_fore[ii, :] = np.array([dx_fore, dz_fore]) # has same dy as the center spar
            delta_aft[ii, :] = np.array([dx_aft, dz_aft])

        # insert delta_y
        delta_fore = np.insert(delta_fore, 1, delta_y, axis=1)
        delta_aft = np.insert(delta_aft, 1, delta_y, axis=1)
    
    # Write beam displacements and dt values to output file
    with open('FSIDisp.txt', 'w') as file:
        for dx, dy, dz in zip(delta_x, delta_y, delta_z):
            file.write(f"{dx} {dy} {dz}\n")

        if structural_nodes_file == 'Structural_nodes_2D.txt': 
            for dx, dy, dz in delta_fore:
                file.write(f"{dx} {dy} {dz} \n")

            for dx, dy, dz in delta_aft:
                file.write(f"{dx} {dy} {dz} \n")

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

    # write the theta vector to a file
    if structural_nodes_file == 'Structural_nodes_2D.txt':
        with open('theta.txt', 'w') as file:
            for t in theta:
                file.write(f"{t}\n")
