import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.linalg import eigh

def parse_xyz_backbone_trajectory(xyz_path: str):
    """Parses XYZ trajectory and extracts coordinates for ONLY 'B' atoms per frame."""
    frames_bb = []
    steps = []
    
    with open(xyz_path, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            
            try:
                num_atoms = int(line)
            except ValueError:
                continue
            
            header = f.readline().strip()
            step_val = len(frames_bb)
            if "Step" in header:
                parts = header.split()
                for i, p in enumerate(parts):
                    if p == "Step" and i + 1 < len(parts):
                        try:
                            step_val = int(parts[i+1])
                        except ValueError:
                            pass
            
            frame_coords = []
            for _ in range(num_atoms):
                atom_line = f.readline().split()
                if len(atom_line) >= 4:
                    name = atom_line[0].strip()
                    if name == 'B':
                        x, y, z = float(atom_line[1]), float(atom_line[2]), float(atom_line[3])
                        frame_coords.append([x, y, z])
            
            steps.append(step_val)
            frames_bb.append(np.array(frame_coords))
            
    return np.array(steps), np.array(frames_bb)

def fit_transform_tica(features: np.ndarray, lag: int = 10, n_components: int = 2):
    """
    Performs Time-lagged Independent Component Analysis (TICA).
    Solves the generalized eigenvalue problem: C(tau) v = lambda C(0) v
    """
    N = len(features)
    if N <= lag:
        raise ValueError(f"Trajectory length ({N}) must be greater than lag time ({lag}).")

    # 1. Mean-center features
    mean_feat = np.mean(features, axis=0)
    X = features - mean_feat

    # 2. Time-lagged data splits
    X_0 = X[:-lag]
    X_tau = X[lag:]

    # 3. Calculate zero-lag covariance matrix C(0)
    C_0 = (X_0.T @ X_0 + X_tau.T @ X_tau) / (2.0 * (N - lag))

    # 4. Calculate time-lagged covariance matrix C(tau)
    C_tau = (X_0.T @ X_tau) / (N - lag)
    C_tau = (C_tau + C_tau.T) / 2.0  # Symmetrize

    # 5. Solve generalized eigenvalue problem: C_tau * v = lambda * C_0 * v
    # Add small regularization for numerical stability
    C_0 += 1e-6 * np.eye(C_0.shape[0])
    eigvals, eigvecs = eigh(C_tau, C_0)

    # 6. Sort eigenvalues & eigenvectors in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Select top components
    top_vecs = eigvecs[:, :n_components]

    # 7. Project full features onto TICs
    tic_projection = X @ top_vecs

    return tic_projection, eigvals[:n_components]

def main():
    parser = argparse.ArgumentParser(description="Compute TICA and Free Energy Landscape for CG backbone trajectory.")
    parser.add_argument("--xyz", required=True, help="Path to trajectory XYZ file")
    parser.add_argument("--lag", type=int, default=10, help="TICA lag time in frames (default: 10)")
    parser.add_argument("--bins", type=int, default=50, help="Number of bins for 2D FEL histogram (default: 50)")
    parser.add_argument("--temp", type=float, default=300.0, help="Temperature in Kelvin (default: 300.0)")
    parser.add_argument("--eq_steps", type=int, default=0, help="Number of initial simulation steps to discard as equilibration (default: 0)")
    parser.add_argument("--out", default="tica_fel.png", help="Output plot filename (default: tica_fel.png)")
    args = parser.parse_args()

    # 1. Load Backbone Trajectory
    print(f"Loading trajectory from: {args.xyz}")
    steps, bb_coords = parse_xyz_backbone_trajectory(args.xyz)
    total_frames = len(steps)
    
    # Filter out equilibration steps
    valid_indices = steps >= args.eq_steps
    steps = steps[valid_indices]
    bb_coords = bb_coords[valid_indices]
    
    n_frames, n_beads, _ = bb_coords.shape
    discarded = total_frames - n_frames
    
    print(f"Discarded {discarded} equilibration frames (steps < {args.eq_steps}).")
    print(f"Processing {n_frames} production frames with {n_beads} backbone 'B' beads each.")

    if n_frames <= args.lag:
        raise ValueError(f"Not enough frames left after discarding equilibration ({n_frames}) to support lag time ({args.lag}).")

    # 2. Compute Internal Pairwise Distance Features (Rotationally/Translationally Invariant)
    print("Computing backbone pairwise distance features...")
    feature_matrix = np.zeros((n_frames, int(n_beads * (n_beads - 1) / 2)))
    for i in range(n_frames):
        feature_matrix[i] = pdist(bb_coords[i], metric='euclidean')

    # 3. Perform TICA
    print(f"Fitting TICA with lag time = {args.lag} frames...")
    tics, eigenvalues = fit_transform_tica(feature_matrix, lag=args.lag, n_components=2)
    print(f"Top 2 TICA Eigenvalues (Autocorrelations): TIC1 = {eigenvalues[0]:.4f}, TIC2 = {eigenvalues[1]:.4f}")

    # 4. Compute 2D Free Energy Landscape (FEL)
    # F = -kB * T * ln(P), normalized to min(F) = 0
    kB = 0.0019872042  # kcal/(mol*K)
    kT = kB * args.temp

    counts, xedges, yedges = np.histogram2d(tics[:, 0], tics[:, 1], bins=args.bins, density=True)
    
    # Avoid log(0)
    counts = np.maximum(counts, 1e-10)
    free_energy = -kT * np.log(counts)
    free_energy -= np.min(free_energy)  # Set global minimum to 0 kcal/mol

    # Mask unvisited regions for plotting
    free_energy = np.where(counts > 1e-9, free_energy, np.nan)

    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])

    # 5. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Trajectory in TIC space colored by simulation step
    sc = ax1.scatter(tics[:, 0], tics[:, 1], c=steps, cmap='plasma', s=1, alpha=0.7)
    ax1.set_xlabel('TIC 1', fontsize=12)
    ax1.set_ylabel('TIC 2', fontsize=12)
    ax1.set_title('Production Trajectory Projection on TIC Space', fontsize=14)
    cbar1 = fig.colorbar(sc, ax=ax1)
    cbar1.set_label('Simulation Step', fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.5)

    # Subplot 2: Free Energy Landscape Contours
    X_grid, Y_grid = np.meshgrid(xcenters, ycenters)
    contour = ax2.contourf(X_grid, Y_grid, free_energy.T, levels=20, cmap='jet_r')
    ax2.set_xlabel('TIC 1', fontsize=12)
    ax2.set_ylabel('TIC 2', fontsize=12)
    ax2.set_title(f'Free Energy Landscape (kcal/mol at {args.temp}K)', fontsize=14)
    cbar2 = fig.colorbar(contour, ax=ax2)
    cbar2.set_label('Free Energy (kcal/mol)', fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"TICA Free Energy Landscape saved to: {args.out}\n")

if __name__ == "__main__":
    main()
