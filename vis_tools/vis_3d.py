import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def slice_viewer(scan, cmap='inferno'):
    '''
    Parameters:
    - scan: np.ndarray
        3D array of intensity values (shape: [Z, Y, X])
    - cmap: str
        Colormap for visualization

    Returns:
    - fig: plotly.graph_objects.Figure
        Visualizes one slice of data in x-y plane at a time (z-index slider)
    '''
    if scan.ndim != 3:
        raise ValueError("Input scan must be a 3D numpy array")

    z_slices = scan.shape[0]
    
    # Create initial image
    fig = go.Figure()

    # Add all slices as separate frames
    for z in range(z_slices):
        fig.add_trace(go.Heatmap(z=scan[z], visible=(z == 0), colorscale=cmap))

    # Add slider steps to toggle visibility
    steps = []
    for i in range(z_slices):
        step = dict(
            method="update",
            args=[{"visible": [j == i for j in range(z_slices)]}],
            label=f"Z={i}"
        )
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Slice: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        title="3D Scan Slice Viewer",
        sliders=sliders,
        margin=dict(l=20, r=20, t=40, b=20),
        width=600,
        height=600
    )

    return fig

def plot_cube(scan, intensity_threshold=0.1, alpha_range=(0.1, 0.8), cmap='inferno'):
    """
    Visualize a 3D scan as a cube of voxels, with color and transparency
    determined by intensity values.

    Parameters:
    - scan: np.ndarray
        3D array of intensity values.
    - intensity_threshold: float
        Minimum normalized intensity (0–1) to display a voxel (default: 0.1).
    - alpha_range: tuple
        (min_alpha, max_alpha) range for transparency mapping (default: (0.1, 0.8)).
    - cmap: str
        Colormap for voxel intensities (default: 'inferno').

    Returns:
    - fig: matplotlib.figure.Figure
        The figure object, for saving or further use.
    """
    # normalizing scan
    scan_norm = (scan - scan.min()) / np.ptp(scan)
    
    # thresholding voxels
    mask = scan_norm > intensity_threshold

    # color mapping for visible voxels
    from matplotlib import cm
    cmap_func = cm.get_cmap(cmap)
    colors = cmap_func(scan_norm)

    # adjust transparency
    min_alpha, max_alpha = alpha_range
    alphas = min_alpha + (max_alpha - min_alpha) * scan_norm
    colors[..., 3] = alphas  # set alpha channel

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.voxels(mask, facecolors=colors, edgecolors=None)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Voxel Visualization')
    plt.tight_layout()
    return fig