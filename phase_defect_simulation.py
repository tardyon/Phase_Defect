import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# === PARAMETERS ===
# Simulation parameters with units that can be easily modified

# Wavelength in nanometers
wavelength_nm = 1000

# Pixel density in pixels per millimeter (sets pixels per unit length)
pixels_per_mm = 50

# Grid dimensions (number of pixels along x and y)
N = 1000   # Pixels in x direction
M = 1000  # Pixels in y direction

# Physical size of the ellipsoidal defect in millimeters
object_thickness_mm = 0.05     # Thickness of the object (mm)
a_mm = 0.100                     # Semi-major axis in x direction (mm)
b_mm = 0.200                     # Semi-minor axis in y direction (mm)

# Opacity parameter for the defect (0 = fully transparent, 1 = fully opaque)
opacity = 0.1

# Error function roll-off for smooth transition within defect in millimeters
defect_edge_smoothing_mm = 0.003  # Width of the defect edge transition (mm)

# Propagation distance in millimeters
propagation_distance_mm = 0  # Set propagation distance (e.g., 1 mm)

# Spatial filter cutoff frequency in cycles per millimeter
spatial_filter_cutoff_cpm = 2  # Set spatial filter cutoff (cycles/mm)

# Error function roll-off for the buffer zone in the padded area, in meters
buffer_edge_smoothing_mm = 0.5  # Width of the buffer edge roll-off (mm)

# Minimum padding width to add on each side before rounding up to the next
# power of two
min_padding = 320  # Minimum padding in pixels

# === UNIT CONVERSIONS ===
# Convert all units to meters for consistent calculations
wavelength = wavelength_nm * 1e-9
object_thickness = object_thickness_mm * 1e-3
a = a_mm * 1e-3
b = b_mm * 1e-3
defect_edge_smoothing = defect_edge_smoothing_mm * 1e-3
buffer_edge_smoothing = buffer_edge_smoothing_mm * 1e-3
propagation_distance = propagation_distance_mm * 1e-3
spatial_filter_cutoff = spatial_filter_cutoff_cpm * 1e3

# Calculate the physical field of view based on pixel
# density and grid dimensions
L_x = N / pixels_per_mm * 1e-3  # Field of view in x direction (meters)
L_y = M / pixels_per_mm * 1e-3  # Field of view in y direction (meters)


# === FUNCTION TO CALCULATE PADDED SIZE WITH MINIMUM BUFFER ===
def calculate_padded_size(size, min_padding=32):
    padded_size = size + 2 * min_padding
    return 2**int(np.ceil(np.log2(padded_size)))

# Calculate padded sizes for efficient FFT, adding minimum padding buffer
N_padded = calculate_padded_size(N, min_padding=min_padding)
M_padded = calculate_padded_size(M, min_padding=min_padding)

# Define the spatial frequency arrays based on pixel spacing in meters
d_x = L_x / N_padded  # Pixel spacing in x direction (meters per pixel)
d_y = L_y / M_padded  # Pixel spacing in y direction (meters per pixel)

# Define Fourier frequency arrays for each axis without using meshgrid
fx = np.fft.fftfreq(N_padded, d=d_x)
fy = np.fft.fftfreq(M_padded, d=d_y)

# === GRID SETUP ===
# Define x and y coordinates for the simulation grid (in physical units)
x = np.linspace(-L_x / 2, L_x / 2, N)  # Match N (x-direction)
y = np.linspace(-L_y / 2, L_y / 2, M)  # Match M (y-direction)

# Vectorized operations using broadcasting
X2 = x[:, np.newaxis]**2  # Shape (N, 1)
Y2 = y[np.newaxis, :]**2  # Shape (1, M)
t_xy = object_thickness * (1 - (X2 / a**2) - (Y2 / b**2))  # Broadcasting to shape (N, M)
t_xy[t_xy < 0] = 0

# Vectorized error function roll-off for the defect edge
r2 = X2 + Y2  # Broadcasting
distance_from_defect_edge = (a - np.sqrt(r2)) / defect_edge_smoothing
defect_transition = 0.5 * (1 + erf(distance_from_defect_edge))
t_xy *= defect_transition

# === AMPLITUDE AND PHASE MODULATION ===
# Define the amplitude mask based on opacity and the defect profile
amplitude_mask = 1 - opacity * (t_xy > 0).astype(float)

# Calculate the phase shift based on the thickness profile
delta_n = 1.5 - 1.0
phase_shift_xy = (2 * np.pi / wavelength) * delta_n * t_xy

# Initial plane wave and apply combined amplitude-phase modulation
initial_wave = np.ones((N, M), dtype=complex)
modulated_wave = initial_wave * amplitude_mask * np.exp(1j * phase_shift_xy)

# === BUFFER EDGE ROLL-OFF IN PADDING ZONE ===
# Create coordinate arrays for the padded grid using broadcasting
x_padded = np.linspace(-L_x / 2, L_x / 2, N_padded)[:, np.newaxis]  # Shape (N_padded, 1)
y_padded = np.linspace(-L_y / 2, L_y / 2, M_padded)[np.newaxis, :]  # Shape (1, M_padded)

# Calculate distance from edges using broadcasting
distance_from_edge_x = np.minimum(
    np.abs(x_padded - x_padded.min()),
    np.abs(x_padded - x_padded.max())
)  # Shape (N_padded, 1)

distance_from_edge_y = np.minimum(
    np.abs(y_padded - y_padded.min()),
    np.abs(y_padded - y_padded.max())
)  # Shape (1, M_padded)

# Calculate roll-off using broadcasting
buffer_rolloff = (0.5 * (1 + erf(3 * distance_from_edge_x / buffer_edge_smoothing)) * 
                 0.5 * (1 + erf(3 * distance_from_edge_y / buffer_edge_smoothing)))

# Pad the modulated wave
pad_x = ((N_padded - N) // 2, (N_padded - N) // 2)
pad_y = ((M_padded - M) // 2, (M_padded - M) // 2)
modulated_wave_padded = np.pad(modulated_wave, (pad_x, pad_y), mode='constant')

# Apply buffer roll-off
modulated_wave_padded *= buffer_rolloff

# === DEFINE THE FRESNEL TRANSFER FUNCTION, APPLY SPATIAL FILTER, AND PROPAGATE ===
# Fresnel transfer function for the specified propagation distance
H_x = np.exp(-1j * np.pi * wavelength * propagation_distance * fx**2)  # 1D x-component
H_y = np.exp(-1j * np.pi * wavelength * propagation_distance * fy**2)  # 1D y-component
H = H_x[:, np.newaxis] * H_y  # 2D transfer function with broadcasting

# Apply FFT for propagation
modulated_wave_fft = np.fft.fft2(modulated_wave_padded, s=(N_padded, M_padded))  # Forward FFT with padding

# Define the spatial filter mask in the frequency domain
filter_mask_x = np.abs(fx) <= spatial_filter_cutoff  # 1D filter along x-axis
filter_mask_y = np.abs(fy) <= spatial_filter_cutoff  # 1D filter along y-axis
filter_mask = filter_mask_x[:, np.newaxis] * filter_mask_y  # 2D filter with broadcasting

# Apply the spatial filter by masking high frequencies
filtered_wave_fft = modulated_wave_fft * filter_mask  # Apply the spatial filter

# Apply the transfer function and perform the inverse FFT
U_z = np.fft.ifft2(filtered_wave_fft * H)  # Apply transfer function and inverse FFT
intensity = np.abs(U_z)**2  # Full intensity without cropping

# Crop back to original grid size
start_x = (N_padded - N) // 2
end_x = start_x + N
start_y = (M_padded - M) // 2
end_y = start_y + M
intensity_cropped = intensity[start_x:end_x, start_y:end_y]

# === PLOTTING 2x2 GRID OF RESULTS ===
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Top-left: Intensity profile
ax = axs[0, 0]
intensity_img = ax.imshow(intensity_cropped, cmap="gray", extent=(-L_x/2*1e3, L_x/2*1e3, -L_y/2*1e3, L_y/2*1e3))
ax.set_title("Intensity Profile")
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
fig.colorbar(intensity_img, ax=ax, label="Intensity")

# Bottom-left: Horizontal cross-section
ax = axs[1, 0]
center_y_idx = M // 2  # Use M for the y-axis
ax.plot(x * 1e3, intensity_cropped[:, center_y_idx])  # Plot intensity vs. x
ax.set_xlabel("x (mm)")
ax.set_ylabel("Intensity")
ax.set_title("Horizontal Cross-section")

# Top-right: Vertical cross-section
ax = axs[0, 1]
center_x_idx = N // 2  # Use N for the x-axis
ax.plot(y * 1e3, intensity_cropped[center_x_idx, :])  # Plot intensity vs. y
ax.set_xlabel("y (mm)")
ax.set_ylabel("Intensity")
ax.set_title("Vertical Cross-section")

# Bottom-right: Physical amplitude profile of defect
ax = axs[1, 1]
ax.plot(x * 1e3, amplitude_mask[:, center_y_idx])  # Plot amplitude vs. x
ax.set_xlabel("x (mm)")
ax.set_ylabel("Amplitude")
ax.set_title("Amplitude Profile")

plt.tight_layout()
plt.show()

