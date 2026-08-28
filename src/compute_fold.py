import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

def analyze_distances(filepath, lag=1):
    times = []
    dist_1_N = []
    dist_2_N1 = []

    # 1. Calculate total number of frames
    with open(filepath, 'r') as f:
        first_line = f.readline()
        if not first_line:
            return times, dist_1_N, dist_2_N1
        num_atoms = int(first_line.strip())
        lines_per_frame = num_atoms + 2  # atoms + header + step line

    with open(filepath, 'r') as f:
        total_lines = sum(1 for _ in f)
    total_frames = total_lines // lines_per_frame

    # 2. Parse trajectory and compute Euclidean distances
    with open(filepath, 'r') as f, tqdm(total=total_frames, desc="Analyzing Distances", unit="frames") as pbar:
        for frame_idx in range(total_frames):
            if frame_idx % lag == 0:
                line = f.readline()
                if not line:
                    break

                step_line = f.readline().strip()
                step = int(step_line.split()[1])
                times.append(step)

                # Collect backbone coordinates only
                b_coords = []
                for _ in range(num_atoms):
                    atom_line = f.readline().split()
                    if atom_line[0] == 'B':
                        b_coords.append([float(atom_line[1]), float(atom_line[2]), float(atom_line[3])])

                b_coords = np.array(b_coords)
                n_residues = len(b_coords)

                if n_residues >= 4:
                    # B_1 to B_N distance (first to last)
                    d1 = np.linalg.norm(b_coords[0] - b_coords[-1])
                    
                    # B_2 to B_(N-1) distance (second to second-to-last)
                    d2 = np.linalg.norm(b_coords[1] - b_coords[-2])

                    dist_1_N.append(d1)
                    dist_2_N1.append(d2)
                else:
                    dist_1_N.append(0.0)
                    dist_2_N1.append(0.0)
            else:
                # Skip non-sampled frames
                for _ in range(lines_per_frame):
                    f.readline()

            pbar.update(1)

    return times, dist_1_N, dist_2_N1


def plot_distances(times, dist_1_N, dist_2_N1, outfile):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Subplot 1: B_1 - B_N
    ax1.plot(times, dist_1_N, color='crimson', label=r'$B_1 - B_N$ (First to Last)', linewidth=1.2, alpha=0.8)
    ax1.set_ylim(0, 20)
    ax1.set_ylabel(r'Distance ($\AA$)')
    ax1.set_title('Backbone Distance Analysis over Time')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Subplot 2: B_2 - B_{N-1}
    ax2.plot(times, dist_2_N1, color='royalblue', label=r'$B_2 - B_{N-1}$ (2nd to 2nd-to-last)', linewidth=1.2, alpha=0.8)
    ax2.set_ylim(0, 20)
    ax2.set_xlabel('Simulation Step')
    ax2.set_ylabel(r'Distance ($\AA$)')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute backbone terminal-pair distances from CG MD trajectory.")
    parser.add_argument('--xyz', required=True, type=str, help="trajectory xyz file")
    parser.add_argument('--out', required=True, type=str, help="output plot image filename")
    parser.add_argument('--lag', required=False, type=int, default=1, help="Stride interval for frame sampling")
    
    args = parser.parse_args()
    
    t, d_1_N, d_2_N1 = analyze_distances(args.xyz, lag=args.lag)
    plot_distances(t, d_1_N, d_2_N1, args.out)
