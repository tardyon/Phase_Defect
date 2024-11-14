import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import erf

# Parameters
a = 0.400  # Semi-major axis in millimeters
b = 0.400  # Semi-minor axis in millimeters
defect_edge_smoothing = 0.03  # Width of defect edge transition in millimeters
N = 1000  # Grid dimension in x direction (pixels)
M = 1000  # Grid dimension in y direction (pixels)
L_x = 1000 / 100  # Field of view in x (millimeters)
L_y = 1000 / 100  # Field of view in y (millimeters)
max_thickness = 0.01  # Maximum thickness of the defect in millimeters

# Create grid
x = np.linspace(-L_x / 2, L_x / 2, N)
y = np.linspace(-L_y / 2, L_y / 2, M)
x, y = np.meshgrid(x, y)

# Calculate defect profile
r_defect = np.sqrt((x / a)**2 + (y / b)**2)
distance_from_defect_edge = (1 - r_defect) / defect_edge_smoothing
defect_transition = 0.5 * (1 + erf(distance_from_defect_edge))
defect_transition[r_defect >= 1] = 0  # Outside the defect
t_xy = defect_transition * max_thickness  # Thickness profile

# Plot the 3D shape of the defect
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, t_xy, cmap='viridis')
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Thickness (mm)')
ax.set_title('3D Shape of the Ellipsoidal Defect')
plt.show()