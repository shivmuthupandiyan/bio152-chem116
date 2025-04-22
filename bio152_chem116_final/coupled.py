import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags, eye, kron, bmat, csc_matrix
from scipy.sparse import vstack as sp_vstack
from numpy.fft import fft2, ifft2
import time
import matplotlib.pyplot as plt

# Graph functions

def plot_mean_phenotypes(results, title="Mean Phenotype Trajectories", mark_times=[]):
    tt = results['t']
    xbarH = results['xbarV_t']
    xbarP = results['xbarD_t']

    plt.figure(figsize=(8, 6))
    plt.plot(xbarH[:, 0], xbarH[:, 1], 'r-', label='Virus mean phenotype (V)')
    plt.plot(xbarP[:, 0], xbarP[:, 1], 'b--', label='DIP mean phenotype (D)')
    plt.scatter(xbarH[0, 0], xbarH[0, 1], color='r', s=100, zorder=5, marker='o', label='Virus start')
    plt.scatter(xbarP[0, 0], xbarP[0, 1], color='b', s=100, zorder=5, marker='o', label='DIP start')

    if mark_times:
        first_marker = True
        for t in mark_times:
            idx = (np.abs(tt - t)).argmin()
            actual_t = tt[idx]
            v_label = f'V @ t≈{actual_t:.1f}' if first_marker else "_nolegend_"
            d_label = f'D @ t≈{actual_t:.1f}' if first_marker else "_nolegend_"
            plt.scatter(xbarH[idx, 0], xbarH[idx, 1], color='r', s=50, marker='x', zorder=10, label=v_label)
            plt.scatter(xbarP[idx, 0], xbarP[idx, 1], color='b', s=50, marker='x', zorder=10, label=d_label)
            first_marker = False

    plt.xlabel('$x_1, y_1$', fontsize=14)
    plt.ylabel('$x_2, y_2$', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=11)
    plt.axis('equal')
    plt.grid(True)
    plt.show()

def plot_cell_virus_pops(results, T_range=None, main_title="Population Dynamics", use_log=True):
    tt = results['t']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(main_title, fontsize=16)

    total_V, total_D = None, None
    cell_pops = {}
    pop_keys = ['C_t', 'CV_t', 'CD_t', 'CDV_t']
    pop_labels = {'C_t': 'Uninfected (C)', 'CV_t': 'V-Infected (CV)',
                  'CD_t': 'D-Infected (CD)', 'CDV_t': 'Co-Infected (CDV)'}

    if 'UH_t' in results and results['UH_t'] is not None:
        total_V = np.sum(results['UH_t'], axis=(0, 1)) * results['params']['dx'] * results['params']['dy']
    if 'UP_t' in results and results['UP_t'] is not None:
        total_D = np.sum(results['UP_t'], axis=(0, 1)) * results['params']['dx'] * results['params']['dy']

    for key in pop_keys:
        if key in results and results[key] is not None:
            cell_pops[key] = results[key]

    ax1.set_title("Total Virus (V) and DIP (D)")
    has_VD = False
    if total_V is not None:
        ax1.plot(tt, total_V, label='Total Virus (V)', linestyle='--', color='red')
        has_VD = True
    if total_D is not None:
        ax1.plot(tt, total_D, label='Total DIP (D)', linestyle=':', color='blue')
        has_VD = True

    if has_VD:
        ax1.set_ylabel("Total Amount")
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax1.legend()
        if use_log:
            max_VD = 0
            if total_V is not None: max_VD = max(max_VD, np.max(total_V))
            if total_D is not None: max_VD = max(max_VD, np.max(total_D))
            if max_VD > 0:
                 current_ylim = ax1.get_ylim()
                 ax1.set_ylim(bottom=max(1e-9, current_ylim[0]*0.1))
                 ax1.set_yscale('log')
            else:
                print("Warning: V/D data is zero or missing, cannot use log scale for subplot 1.")
    else:
        ax1.text(0.5, 0.5, 'No V/D total data (UH_t/UP_t)\nfound in results.',
                 horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)

    ax2.set_title("Cell Populations")
    has_cells = False
    cell_colors = {'C_t': 'green', 'CV_t': 'orange', 'CD_t': 'purple', 'CDV_t': 'brown'}
    for key, data in cell_pops.items():
        ax2.plot(tt, data, label=pop_labels[key], color=cell_colors.get(key, 'black'))
        has_cells = True

    if has_cells:
        ax2.set_ylabel("Population Size")
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax2.legend()
        if use_log:
             max_cell = 0
             for data in cell_pops.values(): max_cell = max(max_cell, np.max(data))
             if max_cell > 0:
                 current_ylim = ax2.get_ylim()
                 ax2.set_ylim(bottom=max(1e-9, current_ylim[0]*0.1))
                 ax2.set_yscale('log')
             else:
                 print("Warning: Cell data is zero or missing, cannot use log scale for subplot 2.")

    else:
        ax2.text(0.5, 0.5, 'No cell population data (C_t, etc.)\nfound in results.',
                 horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)

    ax2.set_xlabel("Time")
    if T_range:
        ax2.set_xlim(T_range)
    else:
        ax2.set_xlim(tt.min(), tt.max())

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def quick_check(results, T_range, n_heatmaps, nx, bord, relative_heatmap_scaling=False):
    print(f"--- Running Quick Check for T = {T_range} ---")
    tt = results.get('t')
    if tt is None:
        print("Error: 't' (time array) not found in results. Aborting quick_check.")
        return

    print("\n[1/4] Plotting Mean Phenotype Trajectories...")
    plot_mean_phenotypes(results,
                         title=f"Mean Phenotypes (T={T_range})",
                         mark_times=[T_range[0], T_range[1]])

    print("\n[2/4] Plotting Mean Euclidean Distance...")
    xbarH = results.get('xbarV_t')
    xbarP = results.get('xbarD_t')

    if xbarH is not None and xbarP is not None:
        start_idx = np.abs(tt - T_range[0]).argmin()
        end_idx = np.abs(tt - T_range[1]).argmin()
        end_idx = max(start_idx, end_idx)

        times_in_range = tt[start_idx:end_idx+1]
        xbarH_in_range = xbarH[start_idx:end_idx+1]
        xbarP_in_range = xbarP[start_idx:end_idx+1]

        if len(times_in_range) > 0:
            mean_dist = np.linalg.norm(xbarH_in_range - xbarP_in_range, axis=1)
            plt.figure(figsize=(8, 5))
            plt.plot(times_in_range, mean_dist)
            plt.xlabel("Time")
            plt.ylabel("Euclidean Distance between Means")
            plt.title(f"Mean Phenotype Distance (T={T_range})")
            plt.grid(True)
            plt.xlim(T_range)
            if len(times_in_range) > 1: plt.ylim(bottom=0)
            plt.show()
        else:
            print("  -> No time points found within the specified T_range for distance plot.")
    else:
        print("  -> Skipping Mean Distance plot (missing 'xbarV_t' or 'xbarD_t' in results).")


    print(f"\n[3/4] Plotting {n_heatmaps} Combined Heatmaps (Scaling: {'Relative' if relative_heatmap_scaling else 'Absolute'})...")
    UHM3 = results.get('UH_t')
    UPM3 = results.get('UP_t')

    if n_heatmaps > 0 and UHM3 is not None and UPM3 is not None:
        if UHM3.ndim != 3 or UPM3.ndim != 3:
             print(f"  -> Error: Density arrays UH_t/UP_t must be 3D. Found shapes: {UHM3.shape}, {UPM3.shape}")
             return

        heatmap_times = np.linspace(T_range[0], T_range[1], n_heatmaps)
        ny = nx

        nrows = n_heatmaps
        ncols = 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(8, nrows * 4.5), squeeze=False)
        axes_flat = axes.flatten()

        global_vmax_V = 1e-9
        global_vmax_D = 1e-9
        indices_to_plot = []
        valid_times_found = False
        for t_target in heatmap_times:
            idx = (np.abs(tt - t_target)).argmin()
            if idx < UHM3.shape[2] and idx < UPM3.shape[2]:
                 indices_to_plot.append(idx)
                 if not relative_heatmap_scaling:
                      global_vmax_V = max(global_vmax_V, np.max(UHM3[:, :, idx]))
                      global_vmax_D = max(global_vmax_D, np.max(UPM3[:, :, idx]))
                 valid_times_found = True
            else:
                 indices_to_plot.append(None)
                 print(f"  -> Warning: Time t={t_target:.2f} maps to index {idx}, out of bounds for density arrays.")

        if not valid_times_found:
             print("  -> Error: No valid time points found for heatmaps within density array bounds.")
             plt.close(fig)
             return

        x_edges = np.linspace(-bord, bord, nx + 1)
        y_edges = np.linspace(-bord, bord, ny + 1)

        imV_handles = []
        imD_handles = []

        for i in range(n_heatmaps):
            ax = axes_flat[i]
            idx = indices_to_plot[i]

            if idx is None:
                 ax.set_title(f'Invalid time index for t ≈ {heatmap_times[i]:.2f}')
                 ax.axis('off')
                 continue

            actual_T = tt[idx]
            UHM_at_T = UHM3[:, :, idx]
            UPM_at_T = UPM3[:, :, idx]

            if relative_heatmap_scaling:
                vmax_V = max(np.max(UHM_at_T), 1e-9)
                vmax_D = max(np.max(UPM_at_T), 1e-9)
            else:
                vmax_V = global_vmax_V
                vmax_D = global_vmax_D

            imV = ax.pcolormesh(x_edges, y_edges, UHM_at_T, shading='flat', cmap='Reds', alpha=0.7, vmin=0, vmax=vmax_V)
            imD = ax.pcolormesh(x_edges, y_edges, UPM_at_T, shading='flat', cmap='Blues', alpha=0.6, vmin=0, vmax=vmax_D)
            imV_handles.append(imV)
            imD_handles.append(imD)

            ax.set_title(f't ≈ {actual_T:.2f}', fontsize=12)
            ax.set_aspect('equal', adjustable='box')
            ax.set_ylabel('$x_2$', fontsize=12)

            cbarV = fig.colorbar(imV, ax=ax, fraction=0.046, pad=0.04, label='V Dens')
            cbarD = fig.colorbar(imD, ax=ax, fraction=0.046, pad=0.1, label='D Dens')
            cbarV.ax.tick_params(labelsize=10)
            cbarD.ax.tick_params(labelsize=10)


        valid_axes = [axes_flat[j] for j, idx in enumerate(indices_to_plot) if idx is not None]
        if valid_axes:
            valid_axes[-1].set_xlabel('$x_1$', fontsize=12)


        fig.suptitle(f'Phenotype Density Snapshots (T={T_range})', fontsize=16, y=0.99)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()

    elif n_heatmaps > 0:
         print("  -> Skipping Heatmaps (missing 'UH_t' or 'UP_t' in results).")
    else:
        print("  -> Skipping heatmaps (n_heatmaps=0).")


    print("\n[4/4] Plotting Population Dynamics...")
    plot_cell_virus_pops(results, T_range=T_range, main_title=f"Populations (T={T_range})")

    print("\n--- Quick Check Complete ---")

# FFT Helper

def create_fft_kernel_padded(params, scale_param_name, padded_nx, padded_ny):
    dx = (2.0 * params['bord']) / params['nx']
    dy = dx
    scale = params.get(scale_param_name, 0.0)

    #print(f"Creating FFT kernel for '{scale_param_name}' with scale={scale:.4f}")

    if scale <= 1e-9:
        #print(f"  Scale is near zero. Using delta kernel (FFT=1).")
        return np.ones((padded_ny, padded_nx), dtype=complex)

    scale_sq = scale**2
    ix = np.arange(padded_nx)
    iy = np.arange(padded_ny)
    ixx, iyy = np.meshgrid(ix, iy)
    dx_sq = np.minimum(ixx, padded_nx - ixx)**2 * dx**2
    dy_sq = np.minimum(iyy, padded_ny - iyy)**2 * dy**2
    dist_sq_kernel = dx_sq + dy_sq
    kernel = np.exp(-dist_sq_kernel / (2 * scale_sq))

    kernel_integral = np.sum(kernel) * dx * dy
    if kernel_integral > 1e-12:
        kernel /= kernel_integral
        #print(f"  Kernel normalized. Integral = {np.sum(kernel)*dx*dy:.4f}")
    else:
        #print(f"  Warning: Kernel integral near zero for {scale_param_name}. Check scale.")
        kernel = np.zeros((padded_ny, padded_nx))
        if dx > 0 and dy > 0:
             kernel[0, 0] = 1.0 / (dx * dy)
        else:
             kernel[0, 0] = 1.0

    fft_K = fft2(kernel)
    #print(f"  FFT of PADDED kernel computed ({padded_ny}x{padded_nx}).")
    return fft_K

# Initialize model w/o specific dynamics

class BaseModel:
    def __init__(self, params):
        self.params = params.copy()
        self.nx = self.params.get('nx', 30)
        self.params['ny'] = self.nx
        self.ny = self.nx
        self.bord = self.params.get('bord', 2)

        if self.nx <= 1 or self.ny <= 1:
            raise ValueError("Grid dimensions nx and ny must be greater than 1.")

        self.n_grid = self.nx * self.ny

        self._setup_grid()
        self._setup_diffusion_matrix_sparse()
        self._setup_initial_conditions()

        self.sol = None
        self.results = None

        #print(f"Base Model Initialized. Grid: {self.nx}x{self.ny}, dx={self.dx:.3f}, dy={self.dy:.3f}")
        #print(f"Phenotype space: [-{self.bord}, {self.bord}] x [-{self.bord}, {self.bord}]")

    def _setup_grid(self):
        self.dx = (2 * self.bord) / self.nx
        self.dy = (2 * self.bord) / self.ny
        xmin, xmax = -self.bord + self.dx/2, self.bord - self.dx/2
        ymin, ymax = -self.bord + self.dy/2, self.bord - self.dy/2
        xx = np.linspace(xmin, xmax, self.nx)
        yy = np.linspace(ymin, ymax, self.ny)
        self.PHENOx, self.PHENOy = np.meshgrid(xx, yy)
        #print(f"Grid setup: nx={self.nx}, ny={self.ny}, dx={self.dx:.4f}, dy={self.dy:.4f}")

    def _setup_diffusion_matrix_sparse(self):
        A_diag = np.ones(self.nx - 1)
        A_sparse = diags([A_diag, -2 * np.ones(self.nx), A_diag], [-1, 0, 1], format='csr')
        A_sparse[0, 0] = -1
        A_sparse[-1, -1] = -1
        A_sparse = A_sparse / (self.dx**2)

        B_diag = np.ones(self.ny - 1)
        B_sparse = diags([B_diag, -2 * np.ones(self.ny), B_diag], [-1, 0, 1], format='csr')
        B_sparse[0, 0] = -1
        B_sparse[-1, -1] = -1
        B_sparse = B_sparse / (self.dy**2)

        eye_nx = eye(self.nx)
        eye_ny = eye(self.ny)

        self.Laplacian = kron(eye_ny, A_sparse) + kron(B_sparse, eye_nx)
        #print(f"Sparse Laplacian matrix calculated (shape: {self.Laplacian.shape}, nnz: {self.Laplacian.nnz}).")

    def _gaussian_2d(self, x, y, mean, std_dev, total_pop, dx, dy):
        var = std_dev**2
        if var < 1e-12:
            density = np.zeros_like(x)
            center_ix = np.argmin(np.abs(x[0,:] - mean[0]))
            center_iy = np.argmin(np.abs(y[:,0] - mean[1]))
            cell_area = dx * dy
            if cell_area > 0:
                density[center_iy, center_ix] = total_pop / cell_area
            elif total_pop > 0:
                 density[center_iy, center_ix] = total_pop
                 #print("Warning: Cell area is zero during Gaussian initialization.")
            else:
                 density[center_iy, center_ix] = 0
            #print(f"Gaussian std dev near zero. Placing {total_pop:.2e} at index ({center_iy}, {center_ix}).")
            return density

        dist_sq = (x - mean[0])**2 + (y - mean[1])**2
        relative_density = np.exp(-dist_sq / (2 * var))

        current_integral = np.sum(relative_density) * dx * dy
        if current_integral > 1e-12:
            norm_factor = total_pop / current_integral
            density = norm_factor * relative_density
        else:
            density = np.zeros_like(x)
            if total_pop > 0:
                pass #print(f"Warning: Initial density numerical normalization failed for pop {total_pop:.1e}, stddev {std_dev:.1e}. Possible reasons: std dev too small for grid resolution, or mean outside boundary. Setting density to zero.")
        return density

    def _setup_initial_conditions(self):
        p = self.params

        p['V_pheno_mean_init'] = np.array(p.get('V_pheno_mean_init', [0.0, 0.0]))
        p['D_pheno_mean_init'] = np.array(p.get('D_pheno_mean_init', [0.0, 0.0]))

        #print("Setting up initial conditions...")
        UH0 = self._gaussian_2d(self.PHENOx, self.PHENOy, p['V_pheno_mean_init'],
                                p['V_pheno_std_init'], p['V_total_init'], self.dx, self.dy)
        UP0 = self._gaussian_2d(self.PHENOx, self.PHENOy, p['D_pheno_mean_init'],
                                p['D_pheno_std_init'], p['D_total_init'], self.dx, self.dy)

        init_V_check = np.sum(UH0) * self.dx * self.dy
        init_D_check = np.sum(UP0) * self.dx * self.dy
        #print(f"Initial V total check: {init_V_check:.3e} (target: {p['V_total_init']:.3e})")
        #print(f"Initial D total check: {init_D_check:.3e} (target: {p['D_total_init']:.3e})")

        self.initial_state = {
            'UH0': UH0, 'UP0': UP0,
            'C0': p['C0'], 'CV0': p['CV0'], 'CD0': p['CD0'], 'CDV0': p['CDV0'],
        }
        self.Q0 = np.concatenate([
            UH0.flatten(),
            UP0.flatten(),
            np.array([p['C0'], p['CV0'], p['CD0'], p['CDV0']])
        ])
        #print(f"Initial state vector Q0 created with length {len(self.Q0)}.")

    def _system_dynamics(self, t, Q):
        raise NotImplementedError("Subclasses must implement the `_system_dynamics` method.")

    def solve(self, t_span, t_eval=None, method='LSODA', rtol=1e-6, atol=1e-9, use_jacobian=False, **kwargs):
        print(f"\n--- Starting Solver ({method}) ---")
        print(f"Time span: {t_span}, t_eval provided: {t_eval is not None}")
        print(f"Tolerances: rtol={rtol}, atol={atol}")
        start_time = time.time()

        jacobian_func = self._jacobian if use_jacobian else None
        if use_jacobian:
            print("Using analytical Jacobian.")
        else:
            print("Using finite difference Jacobian (solver default).")

        sol = solve_ivp(
            fun=lambda t, Q: self._system_dynamics(t, Q),
            t_span=t_span,
            y0=self.Q0,
            method=method,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            jac = jacobian_func,
            **kwargs
        )
        end_time = time.time()
        print(f"Solver finished in {end_time - start_time:.2f} seconds.")
        print(f"Success: {sol.success}")
        if not sol.success:
            print(f"Solver message: {sol.message}")

        self.sol = sol
        self._process_results()
        return self.sol

    def _process_results(self):
        if self.sol is None:
             print("Error: Cannot process results. Run solve() first.")
             return
        if not self.sol.success:
             print("Warning: Processing results from a failed simulation.")

        #print("Processing results...")
        sol = self.sol
        t = sol.t
        Y = sol.y
        idx_CV = self.n_grid * 2

        UH_t_flat = np.maximum(0, Y[:self.n_grid, :])
        UP_t_flat = np.maximum(0, Y[self.n_grid:idx_CV, :])
        UH_t = UH_t_flat.reshape(self.ny, self.nx, len(t))
        UP_t = UP_t_flat.reshape(self.ny, self.nx, len(t))

        C_t = np.maximum(0, Y[idx_CV, :])
        CV_t = np.maximum(0, Y[idx_CV + 1, :])
        CD_t = np.maximum(0, Y[idx_CV + 2, :])
        CDV_t = np.maximum(0, Y[idx_CV + 3, :])

        cell_area = self.dx * self.dy
        V_total_t = np.sum(UH_t, axis=(0, 1)) * cell_area
        D_total_t = np.sum(UP_t, axis=(0, 1)) * cell_area

        xbarV_t = np.full((len(t), 2), np.nan)
        xbarD_t = np.full((len(t), 2), np.nan)

        safe_V_total_t = np.maximum(V_total_t, 1e-12)
        safe_D_total_t = np.maximum(D_total_t, 1e-12)

        phenoX_grid = self.PHENOx[np.newaxis, :, :]
        phenoY_grid = self.PHENOy[np.newaxis, :, :]
        UH_t_res = UH_t.transpose(2, 0, 1)
        UP_t_res = UP_t.transpose(2, 0, 1)

        xbarV_t[:, 0] = np.sum(phenoX_grid * UH_t_res, axis=(1, 2)) * cell_area / safe_V_total_t
        xbarV_t[:, 1] = np.sum(phenoY_grid * UH_t_res, axis=(1, 2)) * cell_area / safe_V_total_t
        xbarD_t[:, 0] = np.sum(phenoX_grid * UP_t_res, axis=(1, 2)) * cell_area / safe_D_total_t
        xbarD_t[:, 1] = np.sum(phenoY_grid * UP_t_res, axis=(1, 2)) * cell_area / safe_D_total_t

        xbarV_t[V_total_t < 1e-12, :] = np.nan
        xbarD_t[D_total_t < 1e-12, :] = np.nan

        self.results = {
            't': t,
            'UH_t': UH_t,
            'UP_t': UP_t,
            'C_t': C_t,
            'CV_t': CV_t,
            'CD_t': CD_t,
            'CDV_t': CDV_t,
            'V_total_t': V_total_t,
            'D_total_t': D_total_t,
            'xbarV_t': xbarV_t,
            'xbarD_t': xbarD_t,
            'params': self.params.copy(),
            'initial_state': self.initial_state.copy()
        }
        #print("Results processed and stored in self.results.")

# Normal model dynamics

class CenterFitness(BaseModel):
    def __init__(self, params):
        super().__init__(params)
        self.penalty_a = self.params.get('penalty_a', 0.0)
        self.penalty_b = self.params.get('penalty_b', 0.0)

        # Pre-calculate squared penalties for efficiency
        self.a_sq_half = (self.penalty_a**2) / 2.0
        self.b_sq_half = (self.penalty_b**2) / 2.0

        #print(f"--> Initialized 'CenterFitness' model.")

    def _system_dynamics(self, t, Q):
        params = self.params
        Laplacian = self.Laplacian
        PHENOx, PHENOy = self.PHENOx, self.PHENOy
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy
        n_grid = self.n_grid
        idx_CV = n_grid * 2

        a_sq_half = self.a_sq_half
        b_sq_half = self.b_sq_half

        UH_flat = Q[:n_grid]
        UP_flat = Q[n_grid:idx_CV]
        C, CV, CD, CDV = Q[idx_CV:]

        UH = np.maximum(0, UH_flat).reshape(ny, nx)
        UP = np.maximum(0, UP_flat).reshape(ny, nx)
        C = max(0, C); CV = max(0, CV); CD = max(0, CD); CDV = max(0, CDV)

        cell_area = dx * dy
        V_total = np.sum(UH) * cell_area
        D_total = np.sum(UP) * cell_area
        safe_V_total = max(V_total, 1e-12)
        safe_D_total = max(D_total, 1e-12)

        xbarV = np.array([0.0, 0.0])
        xbarD = np.array([0.0, 0.0])
        if V_total > 1e-12:
            xbarV[0] = np.sum(PHENOx * UH) * cell_area / safe_V_total
            xbarV[1] = np.sum(PHENOy * UH) * cell_area / safe_V_total

        if D_total > 1e-12:
            xbarD[0] = np.sum(PHENOx * UP) * cell_area / safe_D_total
            xbarD[1] = np.sum(PHENOy * UP) * cell_area / safe_D_total

        beta = params['beta']
        DELTA_V_matrix = np.zeros_like(PHENOx)
        if D_total > 1e-12:
            delta_V_max = params['delta_V_max']; delta_V_scale = params['delta_V_scale']
            dist_V_scale_sq = delta_V_scale**2
            if dist_V_scale_sq > 1e-12:
                dist_sq_from_xbarD = (PHENOx - xbarD[0])**2 + (PHENOy - xbarD[1])**2
                DELTA_V_matrix = delta_V_max * np.exp(-dist_sq_from_xbarD / (2 * dist_V_scale_sq))
            elif delta_V_max > 0:
                center_ix = np.argmin(np.abs(PHENOx[0,:] - xbarD[0]))
                center_iy = np.argmin(np.abs(PHENOy[:,0] - xbarD[1]))
                DELTA_V_matrix[center_iy, center_ix] = delta_V_max

        KAPPA_V_matrix = 1.0 + DELTA_V_matrix / (1.0 + beta)
        safe_KAPPA_V_matrix = np.maximum(KAPPA_V_matrix, 1e-12)

        DELTA_D_matrix = np.zeros_like(PHENOx)
        if V_total > 1e-12:
            delta_D_max = params['delta_D_max']; delta_D_scale = params['delta_D_scale']
            dist_D_scale_sq = delta_D_scale**2
            if dist_D_scale_sq > 1e-12:
                dist_sq_from_xbarV = (PHENOx - xbarV[0])**2 + (PHENOy - xbarV[1])**2
                DELTA_D_matrix = delta_D_max * np.exp(-dist_sq_from_xbarV / (2 * dist_D_scale_sq))
            elif delta_D_max > 0:
                center_ix = np.argmin(np.abs(PHENOx[0,:] - xbarV[0]))
                center_iy = np.argmin(np.abs(PHENOy[:,0] - xbarV[1]))
                DELTA_D_matrix[center_iy, center_ix] = delta_D_max

        Fitness_Factor_Origin = np.ones_like(PHENOx)
        if a_sq_half > 1e-12:
             dist_sq_origin = PHENOx**2 + PHENOy**2
             Fitness_Factor_Origin = np.exp(-a_sq_half * dist_sq_origin)

        Fitness_Factor_Center = np.ones_like(PHENOx)
        if b_sq_half > 1e-12:
             dist_sq_center = (PHENOx - xbarV[0])**2 + (PHENOy - xbarV[1])**2
             Fitness_Factor_Center = np.exp(-b_sq_half * dist_sq_center)

        Total_Fitness_Factor = Fitness_Factor_Origin * Fitness_Factor_Center

        eta = params['eta']
        iota = params['iota']
        alpha = params['alpha']
        R_V_CV = alpha * eta * CV
        R_V_CDV = alpha * eta * CDV
        R_D_CV_beta = alpha * eta * beta * CV
        R_D_CDV_beta = alpha * eta * beta * CDV
        R_D_CDV_delta = alpha * eta * CDV

        dC_dt = -iota * C * (V_total + D_total)
        dCV_dt = iota * C * V_total - CV * (iota * D_total + alpha)
        dCD_dt = iota * C * D_total - CD * (iota * V_total + alpha)
        dCDV_dt = iota * (CD * V_total + CV * D_total) - alpha * CDV

        Source_V_matrix = np.zeros_like(UH)
        Source_D_matrix = np.zeros_like(UP)

        if V_total > 1e-12:
            norm_UH_density_over_V = UH / safe_V_total
            Source_V_from_CV = R_V_CV * norm_UH_density_over_V * Total_Fitness_Factor
            Source_V_from_CDV = R_V_CDV * (1.0 / safe_KAPPA_V_matrix) * norm_UH_density_over_V * Total_Fitness_Factor
            Source_D_from_CV_beta = R_D_CV_beta * norm_UH_density_over_V * Total_Fitness_Factor
            Source_D_from_CDV_beta = R_D_CDV_beta * (1.0 / safe_KAPPA_V_matrix) * norm_UH_density_over_V * Total_Fitness_Factor

            Source_V_matrix += Source_V_from_CV + Source_V_from_CDV
            Source_D_matrix += Source_D_from_CV_beta + Source_D_from_CDV_beta

        if D_total > 1e-12:
            norm_UP_density_over_D = UP / safe_D_total
            Source_D_from_CDV_delta = R_D_CDV_delta * (DELTA_D_matrix / safe_KAPPA_V_matrix) * norm_UP_density_over_D
            Source_D_matrix += Source_D_from_CDV_delta

        diff_UH_flat = params['mu_V'] * (Laplacian @ UH_flat)
        diff_UP_flat = params['mu_D'] * (Laplacian @ UP_flat)

        sink_UH = iota * (C + CD) * UH
        decay_UH = params['gamma_V'] * UH
        sink_UP = iota * (C + CV) * UP
        decay_UP = params['gamma_D'] * UP

        dUHM_dt_flat = (diff_UH_flat
                        + Source_V_matrix.flatten()
                        - sink_UH.flatten()
                        - decay_UH.flatten()
                       )
        dUPM_dt_flat = (diff_UP_flat
                        + Source_D_matrix.flatten()
                        - sink_UP.flatten()
                        - decay_UP.flatten())

        dQ_dt = np.concatenate([
            dUHM_dt_flat, dUPM_dt_flat,
            np.array([dC_dt, dCV_dt, dCD_dt, dCDV_dt])
        ])

        return dQ_dt

    def _jacobian(self, t, Q):
        params = self.params
        Laplacian = self.Laplacian
        PHENOx, PHENOy = self.PHENOx, self.PHENOy
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy
        n_grid = self.n_grid
        idx_CV = n_grid * 2
        a_sq_half = self.a_sq_half
        b_sq_half = self.b_sq_half

        UH_flat = Q[:n_grid]
        UP_flat = Q[n_grid:idx_CV]
        C, CV, CD, CDV = Q[idx_CV:]

        UH = np.maximum(0, UH_flat).reshape(ny, nx)
        UP = np.maximum(0, UP_flat).reshape(ny, nx)
        UH_flat_nonneg = UH.flatten()
        UP_flat_nonneg = UP.flatten()
        C = max(0, C); CV = max(0, CV); CD = max(0, CD); CDV = max(0, CDV)

        cell_area = dx * dy
        V_total = np.sum(UH) * cell_area
        D_total = np.sum(UP) * cell_area
        safe_V_total = max(V_total, 1e-12)
        safe_D_total = max(D_total, 1e-12)

        xbarV = np.array([0.0, 0.0])
        xbarD = np.array([0.0, 0.0])
        if V_total > 1e-12:
            xbarV[0] = np.sum(PHENOx * UH) * cell_area / safe_V_total
            xbarV[1] = np.sum(PHENOy * UH) * cell_area / safe_V_total

        if D_total > 1e-12:
            xbarD[0] = np.sum(PHENOx * UP) * cell_area / safe_D_total
            xbarD[1] = np.sum(PHENOy * UP) * cell_area / safe_D_total

        beta = params['beta']
        DELTA_V_matrix = np.zeros_like(PHENOx)
        if D_total > 1e-12:
            delta_V_max = params['delta_V_max']; delta_V_scale = params['delta_V_scale']
            dist_V_scale_sq = delta_V_scale**2
            if dist_V_scale_sq > 1e-12:
                dist_sq_from_xbarD = (PHENOx - xbarD[0])**2 + (PHENOy - xbarD[1])**2
                DELTA_V_matrix = delta_V_max * np.exp(-dist_sq_from_xbarD / (2 * dist_V_scale_sq))
            elif delta_V_max > 0:
                center_ix = np.argmin(np.abs(PHENOx[0,:] - xbarD[0]))
                center_iy = np.argmin(np.abs(PHENOy[:,0] - xbarD[1]))
                DELTA_V_matrix[center_iy, center_ix] = delta_V_max

        KAPPA_V_matrix = 1.0 + DELTA_V_matrix / (1.0 + beta)
        safe_KAPPA_V_matrix = np.maximum(KAPPA_V_matrix, 1e-12)
        inv_safe_KAPPA_V_flat = (1.0 / safe_KAPPA_V_matrix).flatten()

        DELTA_D_matrix = np.zeros_like(PHENOx)
        if V_total > 1e-12:
            delta_D_max = params['delta_D_max']; delta_D_scale = params['delta_D_scale']
            dist_D_scale_sq = delta_D_scale**2
            if dist_D_scale_sq > 1e-12:
                dist_sq_from_xbarV = (PHENOx - xbarV[0])**2 + (PHENOy - xbarV[1])**2
                DELTA_D_matrix = delta_D_max * np.exp(-dist_sq_from_xbarV / (2 * dist_D_scale_sq))
            elif delta_D_max > 0:
                center_ix = np.argmin(np.abs(PHENOx[0,:] - xbarV[0]))
                center_iy = np.argmin(np.abs(PHENOy[:,0] - xbarV[1]))
                DELTA_D_matrix[center_iy, center_ix] = delta_D_max
        DELTA_D_flat = DELTA_D_matrix.flatten()

        Fitness_Factor_Origin_flat = np.ones(n_grid)
        if a_sq_half > 1e-12:
             dist_sq_origin = PHENOx**2 + PHENOy**2
             Fitness_Factor_Origin_flat = np.exp(-a_sq_half * dist_sq_origin).flatten()

        Fitness_Factor_Center_flat = np.ones(n_grid)
        if b_sq_half > 1e-12:
             dist_sq_center = (PHENOx - xbarV[0])**2 + (PHENOy - xbarV[1])**2
             Fitness_Factor_Center_flat = np.exp(-b_sq_half * dist_sq_center).flatten()

        Total_Fitness_Factor_flat = Fitness_Factor_Origin_flat * Fitness_Factor_Center_flat

        eta = params['eta']
        iota = params['iota']
        alpha = params['alpha']
        mu_V = params['mu_V']
        mu_D = params['mu_D']
        gamma_V = params['gamma_V']
        gamma_D = params['gamma_D']

        diag_J_UU = -iota * (C + CD) - gamma_V

        if V_total > 1e-12:
            diag_J_UU += (alpha * eta * CV / safe_V_total) * Total_Fitness_Factor_flat
            diag_J_UU += (alpha * eta * CDV / safe_V_total) * inv_safe_KAPPA_V_flat * Total_Fitness_Factor_flat

        J_UU = mu_V * Laplacian + diags(diag_J_UU, 0, shape=(n_grid, n_grid), format='csc')

        diag_J_PP = -iota * (C + CV) - gamma_D
        if D_total > 1e-12:
            diag_J_PP += (alpha * eta * CDV / safe_D_total) * (DELTA_D_flat * inv_safe_KAPPA_V_flat)
        J_PP = mu_D * Laplacian + diags(diag_J_PP, 0, shape=(n_grid, n_grid), format='csc')

        diag_J_PU = np.zeros(n_grid)
        if V_total > 1e-12:
            diag_J_PU += (alpha * eta * beta * CV / safe_V_total) * Total_Fitness_Factor_flat
            diag_J_PU += (alpha * eta * beta * CDV / safe_V_total) * inv_safe_KAPPA_V_flat * Total_Fitness_Factor_flat
        J_PU = diags(diag_J_PU, 0, shape=(n_grid, n_grid), format='csc')

        J_UD = csc_matrix((n_grid, n_grid))

        dUdC = -iota * UH_flat_nonneg
        dUdCV = np.zeros(n_grid)
        dUdCD = -iota * UH_flat_nonneg
        dUdCDV = np.zeros(n_grid)
        if V_total > 1e-12:
             dUdCV += (alpha * eta / safe_V_total) * UH_flat_nonneg * Total_Fitness_Factor_flat
             dUdCDV += (alpha * eta / safe_V_total) * UH_flat_nonneg * inv_safe_KAPPA_V_flat * Total_Fitness_Factor_flat

        J_U_ode = csc_matrix(np.vstack([dUdC, dUdCV, dUdCD, dUdCDV]).T)

        dPdC = -iota * UP_flat_nonneg
        dPdCV = -iota * UP_flat_nonneg
        dPdCD = np.zeros(n_grid)
        dPdCDV = np.zeros(n_grid)
        if V_total > 1e-12:
             dPdCV += (alpha * eta * beta / safe_V_total) * UH_flat_nonneg * Total_Fitness_Factor_flat
             dPdCDV += (alpha * eta * beta / safe_V_total) * UH_flat_nonneg * inv_safe_KAPPA_V_flat * Total_Fitness_Factor_flat
        if D_total > 1e-12:
             dPdCDV += (alpha * eta / safe_D_total) * UP_flat_nonneg * DELTA_D_flat * inv_safe_KAPPA_V_flat

        J_P_ode = csc_matrix(np.vstack([dPdC, dPdCV, dPdCD, dPdCDV]).T)

        dC_dUHj = -iota * C * cell_area; dC_dUPj = -iota * C * cell_area
        dCV_dUHj = iota * C * cell_area; dCV_dUPj = -iota * CV * cell_area
        dCD_dUHj = -iota * CD * cell_area; dCD_dUPj = iota * C * cell_area
        dCDV_dUHj = iota * CD * cell_area; dCDV_dUPj = iota * CV * cell_area

        row_CU = csc_matrix(np.full((1, n_grid), dC_dUHj)); row_CP = csc_matrix(np.full((1, n_grid), dC_dUPj))
        row_VU = csc_matrix(np.full((1, n_grid), dCV_dUHj)); row_VP = csc_matrix(np.full((1, n_grid), dCV_dUPj))
        row_DU = csc_matrix(np.full((1, n_grid), dCD_dUHj)); row_DP = csc_matrix(np.full((1, n_grid), dCD_dUPj))
        row_dVU = csc_matrix(np.full((1, n_grid), dCDV_dUHj)); row_dVP = csc_matrix(np.full((1, n_grid), dCDV_dUPj))
        J_ode_PDE_top = bmat([[row_CU, row_CP]]); J_ode_PDE_mid1 = bmat([[row_VU, row_VP]])
        J_ode_PDE_mid2 = bmat([[row_DU, row_DP]]); J_ode_PDE_bot = bmat([[row_dVU, row_dVP]])
        J_ode_PDE = sp_vstack([J_ode_PDE_top, J_ode_PDE_mid1, J_ode_PDE_mid2, J_ode_PDE_bot], format='csc')

        J_ode_ode = np.zeros((4, 4))
        J_ode_ode[0, 0] = -iota * (V_total + D_total)
        J_ode_ode[1, 0] = iota * V_total; J_ode_ode[1, 1] = -iota * D_total - alpha
        J_ode_ode[2, 0] = iota * D_total; J_ode_ode[2, 2] = -iota * V_total - alpha
        J_ode_ode[3, 1] = iota * D_total; J_ode_ode[3, 2] = iota * V_total; J_ode_ode[3, 3] = -alpha
        J_ode_ode_sparse = csc_matrix(J_ode_ode)

        J_ode_U = J_ode_PDE[:, :n_grid]; J_ode_P = J_ode_PDE[:, n_grid:]
        J = bmat([
            [J_UU,    J_UD,              J_U_ode],
            [J_PU,    J_PP,              J_P_ode],
            [J_ode_U, J_ode_P, J_ode_ode_sparse]
        ], format='csc')

        return J

# Fast fourier transform model dynamics

class CenterFitnessFFT(BaseModel):
    def __init__(self, params):
        super().__init__(params)

        self.fft_K_V = None
        self.fft_K_D = None

        self.ny = self.nx
        self.padded_nx = 2 * self.nx
        self.padded_ny = 2 * self.ny

        self._setup_fft_kernels()

        #print(f"--> Initialized 'CenterFitness_FFT' (FFT Interactions)")


    def _setup_fft_kernels(self):
        if 'interaction_scale_V' in self.params:
            self.fft_K_V = create_fft_kernel_padded(self.params, 'interaction_scale_V',
                                                    self.padded_nx, self.padded_ny)
        else:
            pass #print("  WARNING: 'interaction_scale_V' not defined in params. FFT interaction V <- D will be zero.")

        if 'interaction_scale_D' in self.params:
            self.fft_K_D = create_fft_kernel_padded(self.params, 'interaction_scale_D',
                                                    self.padded_nx, self.padded_ny)
        else:
            pass #print("  WARNING: 'interaction_scale_D' not defined in params. FFT interaction D <- V will be zero.")


    def _convolve_fft(self, Density_2D, fft_Kernel):
        if fft_Kernel is None:
            return np.zeros((self.ny, self.nx))

        Density_padded = np.zeros((self.padded_ny, self.padded_nx))
        Density_padded[:self.ny, :self.nx] = Density_2D
        fft_Density_padded = fft2(Density_padded)
        fft_Result_padded = fft_Kernel * fft_Density_padded
        Result_padded = np.real(ifft2(fft_Result_padded))
        Convolution_Matrix = Result_padded[:self.ny, :self.nx]
        return Convolution_Matrix

    def _system_dynamics(self, t, Q):
        params = self.params; Laplacian = self.Laplacian
        PHENOx, PHENOy = self.PHENOx, self.PHENOy
        nx, ny = self.nx, self.ny; dx, dy = self.dx, self.dy
        n_grid = self.n_grid; idx_CV = n_grid * 2

        # Get penalty strength directly from params (BaseModel stores a copy)
        a_sq_half = (params.get('penalty_a', 0.0)**2) / 2.0
        # Note: penalty_b is not used in the current FFT version's dynamics equations

        UH_flat = Q[:n_grid]; UP_flat = Q[n_grid:idx_CV]
        C, CV, CD, CDV = Q[idx_CV:]
        UH = np.maximum(0, UH_flat).reshape(ny, nx)
        UP = np.maximum(0, UP_flat).reshape(ny, nx)
        C=max(0, C); CV=max(0, CV); CD=max(0, CD); CDV=max(0, CDV)

        cell_area = dx * dy
        V_total = np.sum(UH) * cell_area; D_total = np.sum(UP) * cell_area
        safe_V_total = max(V_total, 1e-12); safe_D_total = max(D_total, 1e-12)

        xbarV = np.array([0.0, 0.0])
        if V_total > 1e-12:
            xbarV[0] = np.sum(PHENOx * UH) * cell_area / safe_V_total
            xbarV[1] = np.sum(PHENOy * UH) * cell_area / safe_V_total

        beta = params['beta']
        delta_V_max = params['delta_V_max']
        delta_D_max = params['delta_D_max']

        if self.fft_K_V is not None and delta_V_max > 1e-12:
            V_Interaction_Matrix = self._convolve_fft(UP, self.fft_K_V)
            DELTA_V_matrix = delta_V_max * V_Interaction_Matrix
        else:
            DELTA_V_matrix = np.zeros_like(PHENOx)

        if self.fft_K_D is not None and delta_D_max > 1e-12:
            D_Interaction_Matrix = self._convolve_fft(UH, self.fft_K_D)
            DELTA_D_matrix = delta_D_max * D_Interaction_Matrix
        else:
            DELTA_D_matrix = np.zeros_like(PHENOx)


        KAPPA_V_matrix = 1.0 + DELTA_V_matrix / (1.0 + beta)
        safe_KAPPA_V_matrix = np.maximum(KAPPA_V_matrix, 1e-12)

        Fitness_Factor_Origin = np.ones_like(PHENOx)
        if a_sq_half > 1e-12:
             dist_sq_origin = PHENOx**2 + PHENOy**2
             Fitness_Factor_Origin = np.exp(-a_sq_half * dist_sq_origin)

        eta=params['eta']; iota=params['iota']; alpha=params['alpha']
        R_V_CV=alpha*eta*CV; R_V_CDV=alpha*eta*CDV
        R_D_CV_beta=alpha*eta*beta*CV; R_D_CDV_beta=alpha*eta*beta*CDV
        R_D_CDV_delta=alpha*eta*CDV

        dC_dt = -iota * C * (V_total + D_total)
        dCV_dt = iota*C*V_total - CV*(iota*D_total + alpha)
        dCD_dt = iota*C*D_total - CD*(iota*V_total + alpha)
        dCDV_dt = iota*(CD*V_total + CV*D_total) - alpha*CDV

        Source_V_matrix = np.zeros_like(UH); Source_D_matrix = np.zeros_like(UP)
        if V_total > 1e-12:
            norm_UH_density_over_V = UH / safe_V_total
            Source_V_from_CV = R_V_CV * norm_UH_density_over_V * Fitness_Factor_Origin
            Source_V_from_CDV = R_V_CDV * (1.0/safe_KAPPA_V_matrix) * norm_UH_density_over_V * Fitness_Factor_Origin
            Source_D_from_CV_beta = R_D_CV_beta * norm_UH_density_over_V * Fitness_Factor_Origin
            Source_D_from_CDV_beta = R_D_CDV_beta * (1.0/safe_KAPPA_V_matrix) * norm_UH_density_over_V * Fitness_Factor_Origin
            Source_V_matrix += Source_V_from_CV + Source_V_from_CDV
            Source_D_matrix += Source_D_from_CV_beta + Source_D_from_CDV_beta
        if D_total > 1e-12:
            norm_UP_density_over_D = UP / safe_D_total
            Source_D_from_CDV_delta = R_D_CDV_delta * (DELTA_D_matrix / safe_KAPPA_V_matrix) * norm_UP_density_over_D
            Source_D_matrix += Source_D_from_CDV_delta

        diff_UH_flat = params['mu_V'] * (Laplacian @ UH_flat)
        diff_UP_flat = params['mu_D'] * (Laplacian @ UP_flat)
        sink_UH = iota*(C + CD)*UH; decay_UH = params['gamma_V']*UH
        sink_UP = iota*(C + CV)*UP; decay_UP = params['gamma_D']*UP

        dUHM_dt_flat = diff_UH_flat + Source_V_matrix.flatten() - sink_UH.flatten() - decay_UH.flatten()
        dUPM_dt_flat = diff_UP_flat + Source_D_matrix.flatten() - sink_UP.flatten() - decay_UP.flatten()

        dQ_dt = np.concatenate([dUHM_dt_flat, dUPM_dt_flat, np.array([dC_dt, dCV_dt, dCD_dt, dCDV_dt])])
        return dQ_dt