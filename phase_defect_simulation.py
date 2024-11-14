import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# === PARAMETERS ===
# Simulation parameters with units that can be easily modified

# Wavelength in nanometers
wavelength_nm = 1000

# Pixel density in pixels per millimeter (sets pixels per unit length)
pixels_per_mm = 1000.0 / 10

# Grid dimensions (number of pixels along x and y)
N = 1000  # Pixels in x direction
M = 1000  # Pixels in y direction

# Physical size of the ellipsoidal defect in millimeters
object_thickness_mm = 0.01     # Thickness of the object (mm)
a_mm = 0.400                   # Semi-major axis in x direction (mm)
b_mm = 0.400                   # Semi-minor axis in y direction (mm)

# Opacity parameter for the defect (0 = fully transparent, 1 = fully opaque)
opacity = 0.0

# Error function roll-off for smooth transition within defect in millimeters
defect_edge_smoothing_mm = 0.03  # Width of the defect edge transition (mm)

# Propagation distance in millimeters
propagation_distance_mm = 0  # Set propagation distance (e.g., 1 mm)

# Spatial filter cutoff frequency in cycles per millimeter
spatial_filter_cutoff_cpm = 2  # Set spatial filter cutoff (cycles/mm)

# === UNIT CONVERSIONS ===
# Convert all units to meters for consistent calculations
wavelength = wavelength_nm * 1e-9
object_thickness = object_thickness_mm * 1e-3
a = a_mm * 1e-3
b = b_mm * 1e-3
defect_edge_smoothing = defect_edge_smoothing_mm * 1e-3
propagation_distance = propagation_distance_mm * 1e-3
spatial_filter_cutoff = spatial_filter_cutoff_cpm * 1e3  # cycles per meter

# Calculate the physical field of view based on pixel density and grid dimensions
L = params['canvas_size_mm'] * 1e-3  # Field of view in meters (square)

# === GRID SETUP ===
# Define x and y coordinates for the simulation grid (in physical units)
x = np.linspace(-L / 2, L / 2, N)  # N points from -L_x/2 to L_x/2
y = np.linspace(-L / 2, L / 2, M)  # M points from -L_y/2 to L_y/2

# Vectorized operations using broadcasting
X2 = x[:, np.newaxis]**2  # Shape (N, 1)
Y2 = y[np.newaxis, :]**2  # Shape (1, M)
t_xy = object_thickness * (1 - (X2 / a**2) - (Y2 / b**2))  # Thickness profile
t_xy[t_xy < 0] = 0  # Ensure thickness is non-negative

# Vectorized error function roll-off for the defect edge
r_defect = np.sqrt((x[:, np.newaxis] / a)**2 + (y[np.newaxis, :] / b)**2)
distance_from_defect_edge = (1 - r_defect) / defect_edge_smoothing
defect_transition = 0.5 * (1 + erf(distance_from_defect_edge))
defect_transition[r_defect >= 1] = 0  # Outside the defect
t_xy *= defect_transition

# === AMPLITUDE AND PHASE MODULATION ===
# Define the amplitude mask based on opacity and the defect profile
amplitude_mask = 1 - opacity * (t_xy > 0).astype(float)

# Calculate the phase shift based on the thickness profile
delta_n = 1.5 - 1.0  # Difference in refractive index
phase_shift_xy = (2 * np.pi / wavelength) * delta_n * t_xy

# Initial plane wave and apply combined amplitude-phase modulation
initial_wave = np.ones((N, M), dtype=complex)
modulated_wave = initial_wave * amplitude_mask * np.exp(1j * phase_shift_xy)

# === DEFINE THE FRESNEL TRANSFER FUNCTION, APPLY SPATIAL FILTER, AND PROPAGATE ===
# Define spatial frequency arrays
d_x = L_x / N  # Pixel spacing in x direction (meters per pixel)
d_y = L_y / M  # Pixel spacing in y direction (meters per pixel)
fx = np.fft.fftfreq(N, d=d_x)
fy = np.fft.fftfreq(M, d=d_y)

# Fresnel transfer function for the specified propagation distance
H_x = np.exp(-1j * np.pi * wavelength * propagation_distance * fx**2)  # 1D x-component
H_y = np.exp(-1j * np.pi * wavelength * propagation_distance * fy**2)  # 1D y-component
H = H_x[:, np.newaxis] * H_y  # 2D transfer function

# Apply FFT for propagation
modulated_wave_fft = np.fft.fft2(modulated_wave)  # Forward FFT

# Define the spatial filter mask in the frequency domain
filter_mask_x = np.abs(fx) <= spatial_filter_cutoff  # 1D filter along x-axis
filter_mask_y = np.abs(fy) <= spatial_filter_cutoff  # 1D filter along y-axis
filter_mask = filter_mask_x[:, np.newaxis] * filter_mask_y  # 2D filter

# Apply the spatial filter by masking high frequencies
filtered_wave_fft = modulated_wave_fft * filter_mask  # Apply spatial filter

# Apply the transfer function and perform the inverse FFT
U_z = np.fft.ifft2(filtered_wave_fft * H)  # Apply transfer function and inverse FFT
intensity = np.abs(U_z)**2  # Compute intensity

# === PLOTTING 2x2 GRID OF RESULTS ===
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Top-left: Intensity profile
extent = (-L / 2 * 1e3, L / 2 * 1e3, -L / 2 * 1e3, L / 2 * 1e3)
intensity_img = axs[0, 0].imshow(intensity, cmap="gray", extent=extent, origin='lower')
axs[0, 0].set_title("Intensity Profile")
axs[0, 0].set_xlabel("x (mm)")
axs[0, 0].set_ylabel("y (mm)")
fig.colorbar(intensity_img, ax=axs[0, 0], label="Intensity")

# Bottom-left: Horizontal cross-section
center_y_idx = M // 2  # Center index along y-axis
axs[1, 0].plot(x * 1e3, intensity[:, center_y_idx])  # Plot intensity vs. x
axs[1, 0].set_xlabel("x (mm)")
axs[1, 0].set_ylabel("Intensity")
axs[1, 0].set_title("Horizontal Cross-section")

# Top-right: Vertical cross-section
center_x_idx = N // 2  # Center index along x-axis
axs[0, 1].plot(y * 1e3, intensity[center_x_idx, :])  # Plot intensity vs. y
axs[0, 1].set_xlabel("y (mm)")
axs[0, 1].set_ylabel("Intensity")
axs[0, 1].set_title("Vertical Cross-section")

# Bottom-right: Physical amplitude profile of defect
axs[1, 1].plot(x * 1e3, amplitude_mask[:, center_y_idx])  # Plot amplitude vs. x
axs[1, 1].set_xlabel("x (mm)")
axs[1, 1].set_ylabel("Amplitude")
axs[1, 1].set_title("Amplitude Profile")

plt.tight_layout()
plt.show()
