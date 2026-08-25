import argparse
import numpy as np
import matplotlib.pyplot as plt

def parse_pdb_backbone(pdb_path: str) -> np.ndarray:
    """Extract coordinates of backbone 'B' atoms from reference PDB."""
    coords = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_name = line[12:16].strip()
                if atom_name == 'B':
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
    
    # Covariance matrix
    H = P_centered.T @ Q_centered
    
    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)
    
    # Handle reflection
    d = np.linalg.det(Vt.T @ U.T)
    E = np.eye(3)
    if d < 0:
        E[2, 2] = -1
        
    R = Vt.T @ E @ U.T
    P_rotated = P_centered @ R.T
    
    diff = P_rotated - Q_centered
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    return rmsd

def main():
    parser = argparse.ArgumentParser(description="Compute backbone RMSD from trajectory relative to folded PDB.")
    parser.add_argument("--xyz", required=True, help="Path to trajectory XYZ file")
    parser.add_argument("--pdb", required=True, help="Path to reference folded PDB file")
    parser.add_argument("--out", default="rmsd_plot.png", help="Output plot filename (default: rmsd_plot.png)")
    args = parser.parse_args()

    # Load reference backbone coordinates
    ref_bb = parse_pdb_backbone(args.pdb)
    if len(ref_bb) == 0:
        raise ValueError("No backbone 'B' atoms found in reference PDB.")

    # Load trajectory
    steps, frames = parse_xyz_trajectory(args.xyz)
    if not frames:
        raise ValueError("No frames parsed from XYZ trajectory file.")

    rmsds = []
    for step, (names, coords) in zip(steps, frames):
        # Select ONLY 'B' backbone atoms
        bb_indices = [i for i, name in enumerate(names) if name == 'B']
        frame_bb = coords[bb_indices]
        
        if len(frame_bb) != len(ref_bb):
            raise ValueError(f"Step {step}: mismatch in 'B' atom count (frame: {len(frame_bb)}, ref: {len(ref_bb)}).")

        rmsd = compute_rmsd_kabsch(frame_bb, ref_bb)
        rmsds.append(rmsd)

    min_rmsd = min(rmsds)
    min_step = steps[rmsds.index(min_rmsd)]

    # Print results to terminal
    print("\n--- Backbone RMSD Analysis ---")
    print(f"Total Frames Processed : {len(rmsds)}")
    print(f"Minimum Backbone RMSD  : {min_rmsd:.4f} Å (at Step {min_step})")
    print(f"Final Backbone RMSD    : {rmsds[-1]:.4f} Å")
    print("------------------------------")

    # Generate and save plot
    plt.figure(figsize=(8, 5))
    plt.plot(steps, rmsds, label='Backbone (B) RMSD', color='#1f77b4', lw=1.5)
    plt.axhline(y=min_rmsd, color='r', linestyle='--', label=f'Min RMSD: {min_rmsd:.2f} Å')
    plt.xlabel('Simulation Step', fontsize=12)
    plt.ylabel('Backbone RMSD (Å)', fontsize=12)
    plt.title('Backbone RMSD vs Folded Reference (1uao_CG)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    plt.savefig(args.out, dpi=300)
    print(f"RMSD plot saved to: {args.out}\n")

if __name__ == "__main__":
    main()
