import matplotlib.pyplot as plt
import numpy as np

class Plots:
    def __init__(self, params, results):
        # Initialize parameters and results for plotting
        self.params = params
        self.results = results

    def plot_results(self):
        # Create a 2x2 grid of subplots
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        L = self.params['canvas_size_mm'] * 1e-3  # Field of view in meters (square)
        x = np.linspace(-L / 2, L / 2, self.params['N'])  # x-coordinates in meters
        y = np.linspace(-L / 2, L / 2, self.params['M'])  # y-coordinates in meters

        # Top-left: Intensity profile
        extent = (-L / 2 * 1e3, L / 2 * 1e3, -L / 2 * 1e3, L / 2 * 1e3)  # Extent in millimeters
        intensity_img = axs[0, 0].imshow(self.results['intensity'], cmap="gray", extent=extent, origin='lower')
        axs[0, 0].set_title("Intensity Profile")
        axs[0, 0].set_xlabel("x (mm)")
        axs[0, 0].set_ylabel("y (mm)")
        fig.colorbar(intensity_img, ax=axs[0, 0], label="Intensity")

        # Bottom-left: Horizontal cross-section
        center_y_idx = self.params['M'] // 2  # Center index along y-axis
        axs[1, 0].plot(x * 1e3, self.results['intensity'][:, center_y_idx])  # Plot intensity vs. x
        axs[1, 0].set_xlabel("x (mm)")
        axs[1, 0].set_ylabel("Intensity")
        axs[1, 0].set_title("Horizontal Cross-section")

        # Top-right: Vertical cross-section
        center_x_idx = self.params['N'] // 2  # Center index along x-axis
        axs[0, 1].plot(y * 1e3, self.results['intensity'][center_x_idx, :])  # Plot intensity vs. y
        axs[0, 1].set_xlabel("y (mm)")
        axs[0, 1].set_ylabel("Intensity")
        axs[0, 1].set_title("Vertical Cross-section")

        # Bottom-right: Physical amplitude profile of defect
        axs[1, 1].plot(x * 1e3, self.results['amplitude_mask'][:, center_y_idx])  # Plot amplitude vs. x
        axs[1, 1].set_xlabel("x (mm)")
        axs[1, 1].set_ylabel("Amplitude")
        axs[1, 1].set_title("Amplitude Profile")

        # Adjust layout and display the plots
        plt.tight_layout()
        plt.show()