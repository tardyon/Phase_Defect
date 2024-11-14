from parameters import Parameters
from PD_Sim import PDSim
from stats import Stats
from plots import Plots

# Initialize parameters
params = Parameters().get_parameters()  # Retrieve simulation parameters

# Run simulation
simulation = PDSim(params)  # Instantiate the simulation class with parameters
results = simulation.get_results()  # Obtain simulation results

# Placeholder for statistics
stats = Stats(params, results)  # Instantiate the statistics class with parameters and results
stats.calculate_statistics()  # Perform statistical analysis (currently a placeholder)

# Plot results
plots = Plots(params, results)  # Instantiate the plotting class with parameters and results
plots.plot_results()  # Generate and display plots