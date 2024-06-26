# Aeroelastic Example

## Description

This folder contains files needed to run an aeroelastic validation case coupling a beam problem with FlightStream. The beam problem is a cantilevered beam with distribution of aerodynamic forces.

The problem set up is taken from the included pdf [CFD-BASED ANALYSIS OF NONLINEAR AEROELASTIC BEHAVIOR OF HIGH-ASPECT RATIO WINGS](./aeroelastic-smith2001.pdfAeroelastic-smith2001.pdf). The expected results can be found in [Aeroelastic-Results.xlsx](./Aeroelastic-Results.xlsx)

Other files:
cleanup_files.bat - a simple Windows batch script for deleting all results files and resetting the directory to a clean state.
wing.csv - Component Cross Sections file containing the cross sections and mesh settings used for the wing.

## Environment Setup

1. Ensure you have the latest version of FlightStream installed.
2. Install the requirements.txt file in your python environment: `pip install requirements.txt`
3. Open the Aeroelastic toolbox in FlightStream and set the runtime folder to this environment.
4. Set the runtime command to: `python beam_code_distributed.py` add the optional argument to the structural nodes file. e.g. `Structural_nodes.txt` or `Structural_nodes.2D.txt`

## Running

To run the aeroelastic problem, open the Aeroelastic toolbox in FlightStream.

1. Ensure all boundaries are selected.
2. Import the structural nodes file either `Structural_nodes.txt` (1D Beam) or `Structural_nodes_2D.txt` (2D Plate).
3. Initialize the solver with Mirror boundary conditions.
4. Run the aeroelastic problem.

## Authors

Daniel Enriquez - denriquez@altair.com - Altair

Should you have any problems do not hesitate to reach out.
