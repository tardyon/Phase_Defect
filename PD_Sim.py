import numpy as np
from scipy.special import erf

class PDSim:
    def __init__(self, params):
        self.params = params
        self.unit_conversions()  # Convert input parameters to SI units
        self.grid_setup()        # Setup the simulation grid
        self.simulate()          # Perform the simulation

    def unit_conversions(self):
        # Convert all units to meters for consistent calculations
        self.wavelength = self.params['wavelength_nm'] * 1e-9  # Wavelength in meters
        self.object_thickness = self.params['object_thickness_mm'] * 1e-3  # Object thickness in meters
        self.a = self.params['a_mm'] * 1e-3  # Semi-major axis in meters
        self.b = self.params['b_mm'] * 1e-3  # Semi-minor axis in meters
        self.defect_edge_smoothing = self.params['defect_edge_smoothing_mm'] * 1e-3  # Smoothing width in meters
        self.propagation_distance = self.params['propagation_distance_mm'] * 1e-3  # Propagation distance in meters
        self.spatial_filter_cutoff = self.params['spatial_filter_cutoff_cpm'] * 1e3  # Spatial filter cutoff in cycles per meter

    def grid_setup(self):
        # Calculate the physical field of view based on pixel density and grid dimensions
        self.L_x = self.params['N'] / self.params['pixels_per_mm'] * 1e-3  # Field of view in x direction (meters)
        self.L_y = self.params['M'] / self.params['pixels_per_mm'] * 1e-3  # Field of view in y direction (meters)
        # Define x and y coordinates for the simulation grid (in physical units)
        self.x = np.linspace(-self.L_x / 2, self.L_x / 2, self.params['N'])  # x-coordinates
        self.y = np.linspace(-self.L_y / 2, self.L_y / 2, self.params['M'])  # y-coordinates

    def simulate(self):
        # Vectorized operations using broadcasting
        X2 = self.x[:, np.newaxis]**2  # Square of x-coordinates
        Y2 = self.y[np.newaxis, :]**2  # Square of y-coordinates
        t_xy = self.object_thickness * (1 - (X2 / self.a**2) - (Y2 / self.b**2))  # Thickness profile
        t_xy[t_xy < 0] = 0  # Ensure thickness is non-negative

        # Vectorized error function roll-off for the defect edge
        r_defect = np.sqrt((self.x[:, np.newaxis] / self.a)**2 + (self.y[np.newaxis, :] / self.b)**2)
        distance_from_defect_edge = (1 - r_defect) / self.defect_edge_smoothing
        defect_transition = 0.5 * (1 + erf(distance_from_defect_edge))
        defect_transition[r_defect >= 1] = 0  # Outside the defect
        t_xy *= defect_transition  # Apply transition

        # Define the amplitude mask based on opacity and the defect profile
        self.amplitude_mask = 1 - self.params['opacity'] * (t_xy > 0).astype(float)

        # Calculate the phase shift based on the thickness profile
        delta_n = 1.5 - 1.0  # Difference in refractive index
        self.phase_shift_xy = (2 * np.pi / self.wavelength) * delta_n * t_xy  # Phase shift

        # Initial plane wave and apply combined amplitude-phase modulation
        initial_wave = np.ones((self.params['N'], self.params['M']), dtype=complex)  # Initial wave
        self.modulated_wave = initial_wave * self.amplitude_mask * np.exp(1j * self.phase_shift_xy)  # Modulated wave

        # Define spatial frequency arrays
        d_x = self.L_x / self.params['N']  # Pixel spacing in x direction
        d_y = self.L_y / self.params['M']  # Pixel spacing in y direction
        fx = np.fft.fftfreq(self.params['N'], d=d_x)  # Spatial frequencies in x
        fy = np.fft.fftfreq(self.params['M'], d=d_y)  # Spatial frequencies in y

        # Fresnel transfer function for the specified propagation distance
        H_x = np.exp(-1j * np.pi * self.wavelength * self.propagation_distance * fx**2)  # Transfer function x-component
        H_y = np.exp(-1j * np.pi * self.wavelength * self.propagation_distance * fy**2)  # Transfer function y-component
        H = H_x[:, np.newaxis] * H_y  # Combined transfer function

        # Apply FFT for propagation
        modulated_wave_fft = np.fft.fft2(self.modulated_wave)  # Forward FFT

        # Define the spatial filter mask in the frequency domain
        filter_mask_x = np.abs(fx) <= self.spatial_filter_cutoff  # Filter along x-axis
        filter_mask_y = np.abs(fy) <= self.spatial_filter_cutoff  # Filter along y-axis
        filter_mask = filter_mask_x[:, np.newaxis] * filter_mask_y  # Combined spatial filter

        # Apply the spatial filter by masking high frequencies
        filtered_wave_fft = modulated_wave_fft * filter_mask  # Apply filter

        # Apply the transfer function and perform the inverse FFT
        U_z = np.fft.ifft2(filtered_wave_fft * H)  # Inverse FFT after transfer
        self.intensity = np.abs(U_z)**2  # Compute intensity

    def get_results(self):
        # Return simulation results
        return {
            'intensity': self.intensity,
            'modulated_wave': self.modulated_wave,
            'amplitude_mask': self.amplitude_mask,
            'phase_shift_xy': self.phase_shift_xy
        }