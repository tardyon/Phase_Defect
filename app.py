import streamlit as st
from parameters import Parameters
from PD_Sim import PDSim
import matplotlib.pyplot as plt
import numpy as np

# Initialize parameters
params = Parameters()

# Sidebar - Simulation Parameters
with st.sidebar.expander("Simulation Parameters", expanded=False):
    col1, col2 = st.columns([4, 1])
    with col1:
        params.wavelength_nm = st.slider("Wavelength (nm)", 400, 1500, int(params.wavelength_nm), key="wavelength", step=1)
    with col2:
        wavelength_input = st.text_input("Wavelength Input", value=str(params.wavelength_nm), key="wavelength_input", label_visibility="collapsed")
        if wavelength_input:
            params.wavelength_nm = int(wavelength_input)
    col1, col2 = st.columns([4, 1])
    with col1:
        params.pixels_per_mm = st.slider("Pixels per mm", 0.0, 500.0, float(params.pixels_per_mm), key="pixels_per_mm", step=0.01)
    with col2:
        pixels_input = st.text_input("Pixels per mm Input", value=str(params.pixels_per_mm), key="pixels_input", label_visibility="collapsed")
        if pixels_input:
            params.pixels_per_mm = float(pixels_input)
    col1, col2 = st.columns([4, 1])
    with col1:
        params.canvas_size_mm = st.slider("Canvas Size (mm)", 1.0, 100.0, float(params.canvas_size_mm), key="canvas_size", step=0.01)
    with col2:
        canvas_size_input = st.text_input("Canvas Size Input", value=str(params.canvas_size_mm), key="canvas_size_input", label_visibility="collapsed")
        if canvas_size_input:
            params.canvas_size_mm = float(canvas_size_input)
    col1, col2 = st.columns([4, 1])
    with col1:
        params.propagation_distance_mm = st.slider("Propagation Distance (mm)", 0.0, 500.0, float(params.propagation_distance_mm), key="propagation", step=0.1)
    with col2:
        propagation_input = st.text_input("Propagation Distance Input", value=str(params.propagation_distance_mm), key="propagation_input", label_visibility="collapsed")
        if propagation_input:
            params.propagation_distance_mm = float(propagation_input)
    col1, col2 = st.columns([4, 1])
    with col1:
        params.spatial_filter_cutoff_cpm = st.slider("Spatial Filter Cutoff (cpm)", 0.001, 10.0, float(params.spatial_filter_cutoff_cpm), key="filter_cutoff", step=0.001)
    with col2:
        filter_input = st.text_input("Spatial Filter Cutoff Input", value=str(params.spatial_filter_cutoff_cpm), key="filter_input", label_visibility="collapsed")
        if filter_input:
            params.spatial_filter_cutoff_cpm = float(filter_input)

# Sidebar - Defect Parameters
with st.sidebar.expander("Defect Parameters", expanded=True):
    col1, col2 = st.columns([4, 1])
    with col1:
        params.object_thickness_mm = st.slider("Object Thickness (mm)", 0.0, 0.25, float(params.object_thickness_mm), key="thickness", step=0.001)
    with col2:
        thickness_input = st.text_input("Object Thickness Input", value=str(params.object_thickness_mm), key="thickness_input", label_visibility="collapsed")
        if thickness_input:
            params.object_thickness_mm = float(thickness_input)
    col1, col2 = st.columns([4, 1])
    with col1:
        params.a_mm = st.slider("Semi-Major Axis X (mm)", 0.0, 3.0, float(params.a_mm), key="a_mm", step=0.001)
    with col2:
        a_mm_input = st.text_input("Semi-Major Axis X Input", value=str(params.a_mm), key="a_mm_input", label_visibility="collapsed")
        if a_mm_input:
            params.a_mm = float(a_mm_input)
    col1, col2 = st.columns([4, 1])
    with col1:
        params.b_mm = st.slider("Semi-Major Axis Y (mm)", 0.0, 3.0, float(params.b_mm), key="b_mm", step=0.001)
    with col2:
        b_mm_input = st.text_input("Semi-Major Axis Y Input", value=str(params.b_mm), key="b_mm_input", label_visibility="collapsed")
        if b_mm_input:
            params.b_mm = float(b_mm_input)
    col1, col2 = st.columns([4, 1])
    with col1:
        params.opacity = st.slider("Opacity", 0.0, 1.0, float(params.opacity), key="opacity", step=0.001)
    with col2:
        opacity_input = st.text_input("Opacity Input", value=str(params.opacity), key="opacity_input", label_visibility="collapsed")
        if opacity_input:
            params.opacity = float(opacity_input)
    col1, col2 = st.columns([4, 1])
    with col1:
        params.defect_edge_smoothing_mm = st.slider("Defect Edge Smoothing (mm)", 0.0001, 0.4, float(params.defect_edge_smoothing_mm), key="smoothing", step=0.001)
    with col2:
        smoothing_input = st.text_input("Defect Edge Smoothing Input", value=str(params.defect_edge_smoothing_mm), key="smoothing_input", label_visibility="collapsed")
        if smoothing_input:
            params.defect_edge_smoothing_mm = float(smoothing_input)

# Main Panel: Cross-sections
st.title("Nearfield Defect Simulator", anchor="center")

# Progress bar
progress_bar = st.progress(0)

# Run simulation
simulation = PDSim(params.__dict__)
results = simulation.get_results()

# Update progress bar
progress_bar.progress(100)

# Cross-section X
st.subheader("Cross-Section X")
fig, ax = plt.subplots()
center_y_idx = simulation.M // 2
ax.plot(np.linspace(-simulation.L / 2, simulation.L / 2, simulation.N), results['intensity'][:, center_y_idx])
ax.set_xlabel("x (mm)")
ax.set_ylabel("Intensity")
ax.set_ylim(bottom=0)  # Set y-axis minimum to zero
ax.set_aspect(aspect='auto')
st.pyplot(fig)

# Cross-section Y
st.subheader("Cross-Section Y")
fig, ax = plt.subplots()
center_x_idx = simulation.N // 2
ax.plot(np.linspace(-simulation.L / 2, simulation.L / 2, simulation.M), results['intensity'][center_x_idx, :])
ax.set_xlabel("y (mm)")
ax.set_ylabel("Intensity")
ax.set_ylim(bottom=0)  # Set y-axis minimum to zero
ax.set_aspect(aspect='auto')
st.pyplot(fig)

# Sidebar - Intensity Profile
with st.sidebar.expander("Intensity Profile", expanded=True):
    fig, ax = plt.subplots()
    extent = (-simulation.L / 2, simulation.L / 2, -simulation.L / 2, simulation.L / 2)
    intensity_img = ax.imshow(results['intensity'], cmap="gray", extent=extent, origin='lower')
    ax.set_title("Intensity Profile")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect(aspect='auto')
    fig.colorbar(intensity_img, ax=ax, label="Intensity")
    st.pyplot(fig)