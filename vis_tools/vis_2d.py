import numpy as np
import matplotlib.pyplot as plt

def project_scan(scan): # 3D np array --> 2D np array
    return np.sum(scan, axis=0)

def plot_projection(projected_scan, marks=(10, 90), cmap='inferno'):
    """
    Plot a 2D projection of a scan with percentile-based intensity clipping.

    Parameters:
    - projected_scan: np.ndarray
        2D array representing the projected scan image.
    - marks: tuple of (low, high)
        Percentiles to clip the color scaling (default: (10, 90)).
    - cmap: str
        Colormap to use for visualization (default: 'inferno').

    Returns:
    - fig: matplotlib.figure.Figure
        The matplotlib figure object for further use (e.g., saving).
    """
    vmin = np.percentile(projected_scan, marks[0])
    vmax = np.percentile(projected_scan, marks[1])

    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.imshow(projected_scan, cmap=cmap, origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label(f'Intensity (clipped between {marks[0]}th and {marks[1]}th percentiles)')
    
    ax.set_title("Projection of LiF Scan")
    ax.set_xlabel("X pixels")
    ax.set_ylabel("Y pixels")
    
    fig.tight_layout()

    return fig

def intensity_hist(scan, bins=100, marks=(10, 90)):
    """
    Generate a histogram of voxel intensities from a 3D scan with optional percentile markers.

    Parameters:
    - scan: np.ndarray
        3D array of intensity values.
    - bins: int
        Number of histogram bins (default: 100).
    - marks: tuple
        Percentiles to mark with vertical lines (default: (10, 90)).

    Returns:
    - fig: matplotlib.figure.Figure
        The matplotlib figure object for further use (e.g., saving).
    """
    flat_scan = scan.flatten()
    min_val = flat_scan.min()
    max_val = flat_scan.max()
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    counts = np.histogram(flat_scan, bins=bin_edges)[0]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(bin_edges[:-1], np.log1p(counts), width=np.diff(bin_edges),
           align='edge', edgecolor='black', color='steelblue')
    ax.set_xlabel("Voxel Intensity")
    ax.set_ylabel("log(1 + Count)")
    ax.set_title("Histogram of Voxel Intensities")

    handles = []
    labels = []
    for mark in marks:
        mark_val = np.percentile(flat_scan, mark)
        line = ax.axvline(mark_val, color='red', linestyle='--', linewidth=1.5)
        handles.append(line)
        labels.append(f'{mark}th percentile')

    if handles:
        ax.legend(handles, labels, loc='upper right')

    fig.tight_layout()
    return fig
