import argparse
import numpy as np

def unfold_cg_pdb(input_pdb: str, output_pdb: str):
    """
    Unfolds a CG PDB by placing backbone beads in a 3D zig-zag pattern.
    This avoids exactly 180-degree angles and undefined dihedrals which 
    cause infinite forces/NaNs during energy minimization.
    """
    with open(input_pdb, 'r') as f:
        lines = f.readlines()

    # Group atoms by residue index
    residues = []
    current_res_id = None
    current_res_atoms = []

    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            res_id = line[22:26].strip()
            if res_id != current_res_id:
                if current_res_atoms:
                    residues.append(current_res_atoms)
                current_res_atoms = []
                current_res_id = res_id
            
            atom_name = line[12:16].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            
            current_res_atoms.append({
                'line': line,
                'name': atom_name,
                'coord': np.array([x, y, z])
            })
    if current_res_atoms:
        residues.append(current_res_atoms)

    new_lines = []

    # Place each residue in a zig-zag to avoid collinearity
    # Target bond length ~ 3.8 A.
    # We use delta_x = 3.2 A, delta_y = alternating +/- 1.9 A, delta_z = slight twist
    for idx, res in enumerate(residues):
        dx = idx * 3.2
        dy = 1.9 if (idx % 2 == 0) else -1.9
        dz = 0.5 if (idx % 3 == 0) else -0.5
        
        new_bb_coord = np.array([dx, dy, dz])
        
        # Locate the backbone bead 'B' to use as local origin
        bb_atom = next((atom for atom in res if atom['name'] == 'B'), None)
        orig_ref = bb_atom['coord'] if bb_atom is not None else res[0]['coord']

        for atom in res:
            # Preserve internal sidechain-to-backbone displacement vector
            offset = atom['coord'] - orig_ref
            new_coord = new_bb_coord + offset
            
            # Format standard PDB line with updated X, Y, Z coordinates
            line = atom['line']
            updated_line = (
                line[:30] +
                f"{new_coord[0]:8.3f}{new_coord[1]:8.3f}{new_coord[2]:8.3f}" +
                line[54:]
            )
            new_lines.append(updated_line)

    with open(output_pdb, 'w') as f:
        f.writelines(new_lines)

    print(f"Unfolded zig-zag structure saved to: {output_pdb}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unfold a CG PDB into a zig-zag extended chain.")
    parser.add_argument("--input", required=True, help="Path to input folded PDB file")
    parser.add_argument("--output", required=True, help="Path to output unfolded PDB file")
    args = parser.parse_args()

    unfold_cg_pdb(args.input, args.output)
