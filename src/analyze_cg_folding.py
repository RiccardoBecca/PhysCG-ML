import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

def parse_pdb_backbone(pdb_path: str) -> np.ndarray:
    """Extract coordinates of backbone 'B' atoms from reference PDB."""
    coords = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                if line[12:16].strip() == 'B':
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
    return np.array(coords)

def parse_xyz_trajectory(xyz_path: str):
    """Parse XYZ trajectory frames and step numbers."""
    steps = []
    frames = []
    
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
            step_val = len(frames)
            if "Step" in header:
                parts = header.split()
                for i, p in enumerate(parts):
                    if p == "Step" and i + 1 < len(parts):
                        try:
                            step_val = int(parts[i+1])
                        except ValueError:
                            pass
            
            frame_names = []
            frame_coords = []
            for _ in range(num_atoms):
                atom_line = f.readline().split()
                if len(atom_line) >= 4:
                    name = atom_line[0].strip()
                    x, y, z = float(atom_line[1]), float(atom_line[2]), float(atom_line[3])
                    frame_names.append(name)
                    frame_coords.append([x, y, z])
            
            steps.append(step_val)
            frames.append((frame_names, np.array(frame_coords)))
            
    return steps, frames

def compute_rmsd_kabsch(P: np.ndarray, Q: np.ndarray) -> float:
    """Calculates minimal RMSD after optimal rotation using Kabsch algorithm."""
    P_centered = P - np.mean(P, axis=0)
    Q_centered = Q - np.mean(Q, axis=0)
    H = P_centered.T @ Q_centered
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    E = np.eye(3)
    if d < 0:
        E[2, 2] = -1
    R = Vt.T @ E @ U.T
    P_rotated = P_centered @ R.T
    diff = P_rotated - Q_centered
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))

def main():
    parser = argparse.ArgumentParser(description="Analyze End-to-End distance and Contact Maps for CG trajectory.")
    parser.add_argument("--xyz", required=True, help="Path to trajectory XYZ file")
    parser.add_argument("--pdb", required=True, help="Path to reference folded PDB file")
    parser.add_argument("--out", default="folding_analysis.png", help="Output plot filename")
    args = parser.parse_args()

    # 1. Load Reference PDB Backbone
    ref_bb = parse_pdb_backbone(args.pdb)
    num_residues = len(ref_bb)
    ref_e2e = np.linalg.norm(ref_bb[0] - ref_bb[-1])
    
    # 2. Load Trajectory
    steps, frames = parse_xyz_trajectory(args.xyz)
    
    rmsds = []
    e2e_distances = []
    bb_trajectory = [] # Store just the backbone coordinates
    
    for step, (names, coords) in zip(steps, frames):
        bb_indices = [i for i, name in enumerate(names) if name == 'B']
        frame_bb = coords[bb_indices]
        bb_trajectory.append(frame_bb)
        
        # Compute RMSD to find the best frame
        rmsd = compute_rmsd_kabsch(frame_bb, ref_bb)
        rmsds.append(rmsd)
        
        # Compute End-to-End Distance (Distance between first and last 'B' bead)
        e2e = np.linalg.norm(frame_bb[0] - frame_bb[-1])
        e2e_distances.append(e2e)

    # 3. Identify Minimum RMSD Frame
    min_idx = np.argmin(rmsds)
    min_step = steps[min_idx]
    min_rmsd = rmsds[min_idx]
    best_frame_bb = bb_trajectory[min_idx]
    best_frame_e2e = e2e_distances[min_idx]

    # Print summary to terminal
    print("\n--- Trajectory Analysis Summary ---")
    print(f"Total Frames Processed : {len(frames)}")
    print(f"Min RMSD Frame Found   : Step {min_step} (RMSD: {min_rmsd:.4f} Å)")
    print(f"Crystal End-to-End Dist: {ref_e2e:.2f} Å")
    print(f"Min RMSD End-to-End    : {best_frame_e2e:.2f} Å")
    print("-----------------------------------\n")

    # 4. Compute Distance Maps
    # Using scipy.spatial.distance.cdist to create NxN distance matrices
    dist_map_ref = distance.cdist(ref_bb, ref_bb, 'euclidean')
    dist_map_min = distance.cdist(best_frame_bb, best_frame_bb, 'euclidean')

    # 5. Plotting
    fig = plt.figure(figsize=(15, 5))

    # Plot A: End-to-End Distance
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(steps, e2e_distances, color='#2ca02c', label='Trajectory E2E')
    ax1.axhline(y=ref_e2e, color='r', linestyle='--', label=f'Crystal E2E ({ref_e2e:.1f} Å)')
    ax1.axvline(x=min_step, color='gray', linestyle=':', label=f'Min RMSD (Step {min_step})')
    ax1.set_xlabel('Simulation Step')
    ax1.set_ylabel('End-to-End Distance (Å)')
    ax1.set_title('B1 to B10 Distance')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Plot B: Reference Distance Map
    ax2 = plt.subplot(1, 3, 2)
    im2 = ax2.imshow(dist_map_ref, cmap='viridis_r', origin='lower')
    ax2.set_title('Crystal Structure Distance Map')
    ax2.set_xlabel('Residue Index')
    ax2.set_ylabel('Residue Index')
    ax2.set_xticks(range(num_residues))
    ax2.set_yticks(range(num_residues))
    fig.colorbar(im2, ax=ax2, label='Distance (Å)', fraction=0.046, pad=0.04)

    # Plot C: Min RMSD Distance Map
    ax3 = plt.subplot(1, 3, 3)
    im3 = ax3.imshow(dist_map_min, cmap='viridis_r', origin='lower', vmin=np.min(dist_map_ref), vmax=np.max(dist_map_ref))
    ax3.set_title(f'Min RMSD Structure Map\n(Step {min_step})')
    ax3.set_xlabel('Residue Index')
    ax3.set_ylabel('Residue Index')
    ax3.set_xticks(range(num_residues))
    ax3.set_yticks(range(num_residues))
    fig.colorbar(im3, ax=ax3, label='Distance (Å)', fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Analysis plots saved to: {args.out}")

if __name__ == "__main__":
    main()
