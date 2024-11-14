class Parameters:
    """
    A class to hold and manage simulation parameters.
    """

    def __init__(self):
        # Simulation parameters with units that can be easily modified
        self.wavelength_nm = 1000  # Wavelength in nanometers
        self.pixels_per_mm = 30  # Pixel density in pixels per millimeter
        self.canvas_size_mm = 10.0  # Canvas size in millimeters (square)
        self.object_thickness_mm = 0.01  # Physical thickness of the object in millimeters
        self.a_mm = 0.400  # Semi-major axis of the ellipsoidal defect in millimeters (x-direction)
        self.b_mm = 0.400  # Semi-minor axis of the ellipsoidal defect in millimeters (y-direction)
        self.opacity = 0.0  # Opacity parameter for the defect (0 = transparent, 1 = opaque)
        self.defect_edge_smoothing_mm = 0.03  # Width of defect edge transition in millimeters
        self.propagation_distance_mm = 0  # Propagation distance in millimeters
        self.spatial_filter_cutoff_cpm = 2  # Spatial filter cutoff frequency in cycles per millimeter

    def get_parameters(self):
        """
        Return parameters as a dictionary.
        """
        return self.__dict__