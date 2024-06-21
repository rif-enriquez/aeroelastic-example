import numpy as np
import pdb

# Global variables and Beam Object definition
pi = 3.14159265358793
g = 9.807
class BeamObject:
    def __init__(self):
        self.X = self.Y = self.Z = 0.0
        self.Fx = self.Fy = self.Fz = 0.0
        self.dx = self.dy = self.dz = 0.0

def read_flightstream_data(filename):
    data = []
    start_reading = False

    with open(filename, 'r') as file:
        for line in file:
            
            # Check if we've reached the table header
            if 'Offset, Chord, X_QC, Z_QC, Fx, Fz, Moment' in line:
                start_reading = True
                continue  # Skip the header line

            # Read data after the header
            if start_reading:
                # Check for the end of the table
                if '-----' not in line:
                   try:
                      # Split the line into individual values and convert to float
                      
                      line_data = line.strip().split(',')
                      line_data.pop() # remove last element
                      if line_data and line_data[0]:  # check if line is not empty
                          data.append([float(value) for value in line_data])
                   except:
                      continue
    data = np.array(data)
    
    return data

# Main program
def Beam1D():
    # Material properties
    L = 3.0
    density = 1000.0
    E = 60.0 * 1e7
    second_moment_of_area = 3.9325e-7
    Area = 0.0016
    max_deflection = 0.5
    Fo = max_deflection * 30.0 * E * second_moment_of_area / (L**4)

    # Count number of vertices
    ipts = 0
    with open('Structural_Nodes.txt', 'r') as file:
        for line in file:
            try:
                x, y, z = map(float, line.split())
                ipts += 1
            except ValueError:
                break

    # Load sectional data
    # data format: Offset, Chord, X_QC, Z_QC, Fx, Fz, Moment
    data = read_flightstream_data('FS_SurfaceSection_Loads.txt')
    print('data read:\n', data)
    Fo = data[0,5]
    # Beam analysis
    if ipts > 0 :
        # Allocate beam memory
        Beam = [BeamObject() for _ in range(ipts)]

        # Beam unperturbed finite element discretization
        for i in range(ipts):
            Beam[i].Y = L * i / (ipts - 1)

        # Compute deflections for all beam nodes
        for i, beam in enumerate(Beam):
            y = beam.Y
            dz = Fo * (y**2) * (10*(L**3) - 10*(L**2)*y + 5*L*(y**2) - y**3) / (120 * L * E * second_moment_of_area)
            beam.dz = dz

        # Write beam displacements to output file
        with open('FSIDisp.txt', 'w') as file:
            for beam in Beam:
                file.write(f"{beam.dx} {beam.dy} {beam.dz}\n")

# Run the program
if __name__ == "__main__":
    Beam1D()
