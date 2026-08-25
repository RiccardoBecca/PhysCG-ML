
import argparse
from Bio import PDB
import numpy as np

coarse_grain_map = {
    "ACE": {"N": ["C"]}, #C=O-CH3
    "NME": {"C": ["N"]}, #NH-CH3
    "ALA": {"B": ["CA"], "S1": ["CB"]},
    "GLY": {"B": ["CA"]},
    "ILE": {"B": ["CA"], "S2": ["CB", "CG2", "CG1"]},
    "LEU": {"B": ["CA"], "S4": ["CG"]},
    "PRO": {"B": ["CA"], "S5": ["CA", "CB", "CG", "CD", "N"]},
    "VAL": {"B": ["CA"], "S6": ["CB"]},
    "PHE": {"B": ["CA"], "S7": ["CG", "CD2", "CE2", "CZ", "CE1", "CD1"]},
    "TRP": {"B": ["CA"], "S8": ["CG", "CD1", "NE1", "CE2", "CD2"], "S9": ["CD2", "CE2", "CZ2", "CH2", "CZ3", "CE3"]},
    "TYR": {"B": ["CA"], "S10": ["CG", "CD2", "CE2", "CZ", "CE1", "CD1"]},
    "ASP": {"B": ["CA"], "S11": ["CG"]},
    "ASH": {"B": ["CA"], "S12": ["CG"]},
    "GLU": {"B": ["CA"], "S13": ["CD"]},
    "GLH": {"B": ["CA"], "S14": ["CD"]},
    "ARG": {"B": ["CA"], "S15": ["CB", "CG", "CD"], "S16": ["CZ"]},
    "HID": {"B": ["CA"], "S17": ["CG", "ND1", "CE1", "NE2", "CD2"]},
    "HIE": {"B": ["CA"], "S18": ["CG", "ND1", "CE1", "NE2", "CD2"]},
    "HIP": {"B": ["CA"], "S19": ["CG", "ND1", "CE1", "NE2", "CD2"]},
    "LYN": {"B": ["CA"], "S20": ["CB", "CG", "CD"], "S21": ["CE", "NZ"]},
    "LYS": {"B": ["CA"], "S22": ["CB", "CG", "CD"], "S23": ["CE", "NZ"]},
    "SER": {"B": ["CA"], "S24": ["CB", "OG"]},
    "THR": {"B": ["CA"], "S25": ["CB"]},
    "CYS": {"B": ["CA"], "S26": ["CB", "SG"]},
    "MET": {"B": ["CA"], "S27": ["CB", "CG"], "S28": ["SD", "CE"]},
    "ASN": {"B": ["CA"], "S29": ["CG"]},
    "GLN": {"B": ["CA"], "S30": ["CB", "CG"], "S31": ["CD"]},
}


def coarse_grain_pdb(input_pdb, output_pdb):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", input_pdb)
    
    with open(output_pdb, "w") as out_file:
        atom_index = 1
        for model in structure:
            for chain in model:
                for residue in chain:
                    resname = residue.get_resname()
                    if resname in coarse_grain_map:

                        #iterate over bead types
                        for bead in coarse_grain_map[resname].keys():
                            #iterate over atom which defin bead COM
                            bead_cog = np.zeros((3))
                            for atom in residue:
                                atom_name = atom.get_name()
                                if atom_name in coarse_grain_map[resname][bead]:
                                    coord = atom.get_coord()
                                    bead_cog += coord
                            bead_cog /= len(coarse_grain_map[resname][bead])
                            out_file.write("ATOM  {:5d}  {:<4}{:<3} {}{:4d}    {:8.3f}{:8.3f}{:8.3f}\n".format(
                                atom_index, bead, resname, chain.get_id(), residue.get_id()[1], *bead_cog
                            ))
                            atom_index += 1

def main(args):

    coarse_grain_pdb(args.input, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help="pdb file to coarse grain", required=True)
    parser.add_argument('--output', type=str, help="name output pdb", required=True)
    args = parser.parse_args()
    main(args) 
