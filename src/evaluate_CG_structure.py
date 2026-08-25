import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Constants
KB_T = 0.008314462 * 300.0  # k_B * T in kJ/mol at 300 K

def parse_cg_pdb(pdb_file):
    """Extracts only the backbone 'B' beads from the CG PDB."""
    beads = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                if atom_name == 'B':
                    res_name = line[17:20].strip()
                    res_id = int(line[22:26].strip())
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    beads.append({
                        'res_name': res_name,
                        'res_id': res_id,
                        'coord': np.array([x, y, z])
                    })
    return beads

def calc_angle(p1, p2, p3):
    """Calculates the angle between three points in radians."""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    # Clip to handle potential floating point errors outside [-1, 1]
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))

def calc_dihedral(p1, p2, p3, p4):
    """Calculates the dihedral angle between four points in radians."""
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2)
    
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    return np.arctan2(y, x)

def boltzmann_inversion(data, bins, domain_range):
    """Computes V(x) = -kBT*ln(P(x)) normalized to min(V) = 0."""
    counts, bin_edges = np.histogram(data, bins=bins, range=domain_range, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    valid_mask = counts > 0
    counts = counts[valid_mask]
    bin_centers = bin_centers[valid_mask]
    
    V = -KB_T * np.log(counts)
    V -= np.min(V)
    
    return bin_centers, V

def plot_potential(key, raw_data, obs_val_rad, is_angle, output_path):
    """Plots the potential curve and adds a red vertical line for the observed value."""
    # Convert data and observed value to degrees
    data_deg = np.degrees(raw_data)
    obs_val_deg = np.degrees(obs_val_rad)
    
    if is_angle:
        domain = (0, 180)
        xlabel = "Theta (degrees)"
        title = f"Angle Potential: {key}"
    else:
        domain = (-180, 180)
        xlabel = "Phi (degrees)"
        title = f"Dihedral Potential: {key}"
        
    x_vals, V_vals = boltzmann_inversion(data_deg, bins=60, domain_range=domain)
    
    plt.figure(figsize=(6, 4))
    plt.plot(x_vals, V_vals, color='navy', lw=2, label='Statistical Potential')
    
    # Add vertical line for the PDB structure's value
    plt.axvline(x=obs_val_deg, color='red', linestyle='--', lw=2, 
                label=f'PDB Value ({obs_val_deg:.1f}°)')
    
    plt.title(title, fontweight='bold')
    plt.xlabel(xlabel)
    plt.ylabel("V (kJ/mol)")
    plt.ylim(-0.5, 20)  # Cap at 20 kJ/mol to easily see the minimum
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=200)
    plt.close()

def main(args):
    json_file = args.json_file
    pdb_file = args.pdb  # Change this to your PDB filename if different
    out_dir = args.outdir
    
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Loading potentials from {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    print(f"Parsing backbone beads from {pdb_file}...")
    beads = parse_cg_pdb(pdb_file)
    print(f"Found {len(beads)} backbone (B) beads.")
    
    # 1. Evaluate Angles
    print("\n--- Processing Angles ---")
    for i in range(1, len(beads) - 1):
        r_prev, r_curr, r_next = beads[i-1], beads[i], beads[i+1]
        key = f"{r_prev['res_name']}-{r_curr['res_name']}-{r_next['res_name']}"
        
        obs_rad = calc_angle(r_prev['coord'], r_curr['coord'], r_next['coord'])
        
        if key in data['angles']:
            filename = os.path.join(out_dir, f"Angle_{i:02d}_{key}.png")
            plot_potential(key, data['angles'][key], obs_rad, is_angle=True, output_path=filename)
            print(f"Plotted {key:^15} (Idx {i}) -> {np.degrees(obs_rad):.1f}°")
        else:
            print(f"WARNING: Key {key} not found in JSON data. Skipping.")

    # 2. Evaluate Dihedrals
    print("\n--- Processing Dihedrals ---")
    for i in range(1, len(beads) - 2):
        r_m1, r_0, r_1, r_2 = beads[i-1], beads[i], beads[i+1], beads[i+2]
        key = f"{r_0['res_name']}-{r_1['res_name']}"
        
        obs_rad = calc_dihedral(r_m1['coord'], r_0['coord'], r_1['coord'], r_2['coord'])
        
        if key in data['dihedrals']:
            filename = os.path.join(out_dir, f"Dihedral_{i:02d}_{key}.png")
            plot_potential(key, data['dihedrals'][key], obs_rad, is_angle=False, output_path=filename)
            print(f"Plotted {key:^15} (Idx {i}) -> {np.degrees(obs_rad):.1f}°")
        else:
            print(f"WARNING: Key {key} not found in JSON data. Skipping.")
            
    print(f"\nAll plots saved successfully in the '{out_dir}' directory.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a pdb structure in the intermolecular potential")
    parser.add_argument('--json_file', type=str, default="cg_statistics.json", help="JSON file with intermolecular potential")
    parser.add_argument('--outdir', type=str, default="data/structure_plots", help="Directory to save png files.")
    parser.add_argument('--pdb', type=str, required=True, help="PDB file with structure to evaluate")
    args = parser.parse_args()

    main(args)
