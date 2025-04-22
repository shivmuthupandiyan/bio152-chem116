#Graphing Functions
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import numpy as np
from scipy.signal import find_peaks

def plot_mean_phenotypes(xbarH, xbarP, tt, figlabel, mark_times=[]):
    plt.figure(figsize=(8, 6))

    # Plot the trajectories
    plt.plot(xbarH[:, 0], xbarH[:, 1], 'r-', label='Virus mean phenotype')
    plt.plot(xbarP[:, 0], xbarP[:, 1], 'b--', label='DIP mean phenotype')

    # Mark the initial points
    plt.scatter(xbarP[0, 0], xbarP[0, 1], color='b', s=100, zorder=5, marker='o', label='DIP start')
    plt.scatter(xbarH[0, 0], xbarH[0, 1], color='r', s=100, zorder=5, marker='o', label='Virus start')

    # Optionally mark crosses at specified times
    for t in mark_times:
        idx = (np.abs(tt - t)).argmin()  # Find the index of the closest time
        plt.scatter(xbarH[idx, 0], xbarH[idx, 1], color='r', s=50, marker='x', zorder=10, label='_nolegend_')  # No legend for these markers
        plt.scatter(xbarP[idx, 0], xbarP[idx, 1], color='b', s=50, marker='x', zorder=10, label='_nolegend_')  # No legend for these markers

    plt.xlabel('$x_1, y_1$', fontsize=16)
    plt.ylabel('$x_2, y_2$', fontsize=16)
    plt.legend(fontsize=11)
    plt.axis('equal')
    plt.grid(True)

    # Save the figure

    plt.show()

def plot_mean_phenotypes_ax(xbarH, xbarP, tt, figlabel, ax, mark_times=[]):
    # Remove plt.figure() since we're plotting on an existing axes
    
    # Plot the trajectories
    ax.plot(xbarH[:, 0], xbarH[:, 1], 'r-', label='Virus mean phenotype')
    ax.plot(xbarP[:, 0], xbarP[:, 1], 'b--', label='DIP mean phenotype')

    # Mark the initial points
    ax.scatter(xbarP[0, 0], xbarP[0, 1], color='b', s=100, zorder=5, marker='o', label='DIP start')
    ax.scatter(xbarH[0, 0], xbarH[0, 1], color='r', s=100, zorder=5, marker='o', label='Virus start')

    # Optionally mark crosses at specified times
    for t in mark_times:
        idx = (np.abs(tt - t)).argmin()  # Find the index of the closest time
        ax.scatter(xbarH[idx, 0], xbarH[idx, 1], color='r', s=50, marker='x', zorder=10, label='_nolegend_')
        ax.scatter(xbarP[idx, 0], xbarP[idx, 1], color='b', s=50, marker='x', zorder=10, label='_nolegend_')

    ax.set_xlabel('$x_1, y_1$', fontsize=16)
    ax.set_ylabel('$x_2, y_2$', fontsize=16)
    ax.legend(fontsize=11)
    ax.axis('equal')
    ax.grid(True)
    ax.set_title(f'Mean phenotypes')
    

def plot_mean_distance(xbarH, xbarP, tt, T):
   
    idx = (np.abs(tt - T)).argmin()

    if len(xbarH) != len(xbarP):
        raise ValueError("Both dataframes must have the same number of rows")

    distance = np.zeros(len(range(idx)))

    for i in range(idx):
    
        point_h = xbarH[i, :]
        point_p = xbarP[i, :]
        
        # Calculate Euclidean distance
        distance[i] = euclidean(point_h, point_p)

    plt.figure()
    plt.xlabel("Time")
    plt.ylabel("Distance between Means")
    plt.title("Distance between Mean Phenotype over Time")
    plt.grid(True)
    plt.plot(tt[:idx], distance)
    plt.show()



def plot_densities_at_time_T(tt, UHM3, UPM3, xx, yy, T):
    # Find the index of the time step closest to the desired time T
    idx = (np.abs(tt - T)).argmin()

    # Select the density matrices for Viruss and DIPs at time T
    UHM_at_T = UHM3[:, :, idx]
    UPM_at_T = UPM3[:, :, idx]

    # Create the plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    # Plot Virus density
    c1 = axs[0].pcolor(xx, yy, UHM_at_T, shading='auto', cmap='Reds')
    fig.colorbar(c1, ax=axs[0], label='Virus Density')
    axs[0].set_title(f'Virus Density at T={T}')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')

    # Plot DIP density
    c2 = axs[1].pcolor(xx, yy, UPM_at_T, shading='auto', cmap='Blues')
    fig.colorbar(c2, ax=axs[1], label='DIP Density')
    axs[1].set_title(f'DIP Density at T={T}')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')

    plt.suptitle('Densities at Time T')
    plt.show()

def plot_combined_heatmaps_at_time_T(tt, UHM3, UPM3, nx, bord, T, figlabel):
    # Find the index of the closest time step to the desired time T
    idx = (np.abs(tt - T)).argmin()

    # Recreate xx yy grids
    ny = nx
    dx = dy = (2 * bord) / nx
    xmin, xmax, ymin, ymax = -bord, bord, -bord, bord
    xx, yy = np.linspace(xmin + dx, xmax, nx), np.linspace(ymin + dy, ymax, ny)

    # Select the density matrices for Viruss and DIPs at time T
    UHM_at_T = UHM3[:, :, idx]
    UPM_at_T = UPM3[:, :, idx]

    plt.figure(figsize=(10, 7))

    # Plot Virus density with heatmap
    Virus = plt.pcolor(xx, yy, UHM_at_T, shading='auto', cmap='Reds', alpha=0.7)
    Virus_cb = plt.colorbar(Virus, label='Virus Density')
    Virus_cb.ax.tick_params(labelsize=14)  # Set font size of color bar ticks
    Virus_cb.set_label('Virus Density', fontsize=11)  # Set font size of color bar label


    # Overlay DIP density with another heatmap, adjusting transparency
    DIP = plt.pcolor(xx, yy, UPM_at_T, shading='auto', cmap='Blues', alpha=0.5)
    DIP_cb = plt.colorbar(DIP, label='DIP Density')
    DIP_cb.ax.tick_params(labelsize=14)  # Set font size of color bar ticks
    DIP_cb.set_label('DIP Density', fontsize=11)  # Set font size of color bar label


    plt.title(f't={T}',fontsize=16)
    plt.xlabel('$x_1, y_1$',fontsize=16)
    plt.ylabel('$x_2, y_2$',fontsize=16)

    # Increase font size of tick labels
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Set x and y axis limits
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)

    # Save the figure

    plt.show()

def plot_phenotypes_vs_time(xbarH, xbarP, tt, figlabel, mark_times=[], legend_pos='TR'):
    plt.figure(figsize=(12, 8))

    # Plot x1 and x2 for Viruss vs time
    plt.subplot(2, 1, 1)
    plt.plot(tt, xbarH[:, 0], 'r-', label='$x_1$ (Virus)')
    plt.plot(tt, xbarH[:, 1], 'r--', label='$x_2$ (Virus)')

    # Optionally mark points at specified times
    for t in mark_times:
        idx = (np.abs(tt - t)).argmin()  # Find the index of the closest time
        plt.scatter(tt[idx], xbarH[idx, 0], color='r', s=50, marker='o', zorder=10, label='_nolegend_')
        plt.scatter(tt[idx], xbarH[idx, 1], color='r', s=50, marker='o', zorder=10, label='_nolegend_')

    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Virus Phenotypes', fontsize=16)
    plt.grid(True)
    # Determine legend position
    if legend_pos == 'TL':
        loc = 'upper left'
    elif legend_pos == 'TR':
        loc = 'upper right'
    else:
        loc = 'best'  # Default to the best position if not specified
    plt.legend(fontsize=12, loc=loc)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Plot y1 and y2 for pathogens vs time
    plt.subplot(2, 1, 2)
    plt.plot(tt, xbarP[:, 0], 'b-', label='$y_1$ (Pathogen)')
    plt.plot(tt, xbarP[:, 1], 'b--', label='$y_2$ (Pathogen)')

    # Optionally mark points at specified times
    for t in mark_times:
        idx = (np.abs(tt - t)).argmin()  # Find the index of the closest time
        plt.scatter(tt[idx], xbarP[idx, 0], color='b', s=50, marker='o', zorder=10, label='_nolegend_')
        plt.scatter(tt[idx], xbarP[idx, 1], color='b', s=50, marker='o', zorder=10, label='_nolegend_')

    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Pathogen Phenotypes', fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12, loc=loc)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Adjust layout to prevent overlaps
    plt.tight_layout()

    plt.show()

def plot_populations_vs_time(NH, NP, tt):
    plt.figure()
    plt.plot(tt, NH, label='Virus Population')
    plt.plot(tt, NP, label='Pathogen Population')
    plt.legend()
    plt.show()

def compute_period(tt, signal):
    # Find peaks in the signal
    peaks, _ = find_peaks(signal)

    # Compute the differences between consecutive peaks
    if len(peaks) > 1:
        periods = np.diff(tt[peaks])
        return np.mean(periods), periods
    else:
        return None, []
    
def create_heatmap_gif(tt, UHM3, UPM3, xx, yy, 
                      filename='heatmap_animation.gif', 
                      num_frames=20, 
                      frame_duration=0.2,
                      time_range=None):

    import os
    import tempfile
    import numpy as np
    import matplotlib.pyplot as plt
    import imageio
    
    # Create a temporary directory to store frames
    temp_dir = tempfile.mkdtemp()
    
    # Determine time range to use
    if time_range is None:
        t_min = tt.min()
        t_max = tt.max()
    else:
        t_min, t_max = time_range
    
    # Generate time points for the animation
    animation_times = np.linspace(t_min, t_max, num_frames)
    
    # List to store the generated frame filenames
    frame_filenames = []
    
    # Generate each frame
    for i, T in enumerate(animation_times):
        # Create the plot for this time point
        plt.figure(figsize=(10, 7))
        
        # Find the index of the closest time step to the desired time T
        idx = (np.abs(tt - T)).argmin()
        
        # Get the actual time value (might be slightly different from T due to discretization)
        actual_T = tt[idx]
        
        # Select the density matrices for Viruss and DIPs at time T
        UHM_at_T = UHM3[:, :, idx]
        UPM_at_T = UPM3[:, :, idx]
        
        # Plot Virus density
        Virus = plt.pcolor(xx, yy, UHM_at_T, shading='auto', cmap='Reds', alpha=0.7)
        Virus_cb = plt.colorbar(Virus, label='Virus Density')
        Virus_cb.ax.tick_params(labelsize=14)
        Virus_cb.set_label('Virus Density', fontsize=11)
        
        # Overlay DIP density
        DIP = plt.pcolor(xx, yy, UPM_at_T, shading='auto', cmap='Blues', alpha=0.5)
        DIP_cb = plt.colorbar(DIP, label='DIP Density')
        DIP_cb.ax.tick_params(labelsize=14)
        DIP_cb.set_label('DIP Density', fontsize=11)
        
        # Set title and labels with time indicator
        plt.title(f'T={actual_T:.2f}', fontsize=16)
        plt.xlabel('$x_1, y_1$', fontsize=16)
        plt.ylabel('$x_2, y_2$', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        
        # Save the frame
        frame_filename = os.path.join(temp_dir, f'frame_{i:04d}.png')
        plt.savefig(frame_filename, bbox_inches='tight')
        frame_filenames.append(frame_filename)
        plt.close()
    
    # Combine frames into a GIF
    with imageio.get_writer(filename, mode='I', duration=frame_duration) as writer:
        for frame_filename in frame_filenames:
            image = imageio.imread(frame_filename)
            writer.append_data(image)
    
    # Clean up temporary files
    for frame_filename in frame_filenames:
        os.remove(frame_filename)
    os.rmdir(temp_dir)
    
    print(f"Animation saved to {filename}")
    return filename
