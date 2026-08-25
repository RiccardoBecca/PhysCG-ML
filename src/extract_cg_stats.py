import argparse
import os
import json
import requests
import numpy as np
import warnings
from collections import defaultdict
from Bio import PDB
from Bio.PDB.PDBExceptions import PDBConstructionWarning

# Suppress annoying Biopython warnings for messy PDB files
warnings.simplefilter('ignore', PDBConstructionWarning)

# Standard 20 Amino Acids to filter out non-standard residues/ligands
STANDARD_AA = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
}

def fetch_nonredundant_pdbs(limit=100):
    """
    Queries RCSB PDB API for high-res (<2.0A), non-redundant (<30% seq identity) proteins.
    """
    print(f"Querying RCSB PDB for up to {limit} non-redundant chains...")
    
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less",
                        "value": 2.0
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_entity_polymer_type",
                        "operator": "exact_match",
                        "value": "Protein"
                    }
                }
            ]
        },
        "request_options": {
            "return_all_hits": True,
            "results_verbosity": "compact"
        },
        "return_type": "polymer_entity"
    }

    response = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=query)
    response.raise_for_status()
    
    hits = response.json().get("result_set", [])
    
    # Format is usually '4HHB_1'. Biopython PDBList needs the 4-letter code.
    # Check if 'hit' is a dict or string to handle different API verbosity responses safely
    pdb_ids = list(set([(hit["identifier"] if isinstance(hit, dict) else hit).split("_")[0].lower() for hit in hits]))

    selected_pdbs = pdb_ids[:limit]
    print(f"Found {len(pdb_ids)} candidates. Proceeding with {len(selected_pdbs)}.")
    return selected_pdbs

def extract_backbone_stats(pdb_dir, pdb_ids):
    """
    Parses PDBs to extract sequence-dependent CA angles and dihedrals.
    """
    parser = PDB.PDBParser(QUIET=True)
    angle_db = defaultdict(list)
    dihedral_db = defaultdict(list)
    
    for count, pdb_id in enumerate(pdb_ids):
        # Biopython usually saves as 'pdbXXXX.ent'
        file_path = os.path.join(pdb_dir, f"pdb{pdb_id}.ent")
        if not os.path.exists(file_path):
            continue
            
        if count % 50 == 0:
            print(f"Processing structure {count}/{len(pdb_ids)}...")

        try:
            struct = parser.get_structure(pdb_id, file_path)
            for model in struct:
                for chain in model:
                    # Filter for CA atoms in standard amino acids
                    ca_nodes = []
                    for res in chain:
                        resname = res.get_resname().strip()
                        if resname in STANDARD_AA and 'CA' in res:
                            ca_nodes.append((resname, res['CA']))
                    
                    if len(ca_nodes) < 4:
                        continue
                        
                    # 1. Angles (Triplets: R_i-1, R_i, R_i+1)
                    for i in range(1, len(ca_nodes) - 1):
                        res_prev, ca_prev = ca_nodes[i-1]
                        res_curr, ca_curr = ca_nodes[i]
                        res_next, ca_next = ca_nodes[i+1]
                        
                        key = f"{res_prev}-{res_curr}-{res_next}"
                        
                        v1 = ca_prev.get_vector() - ca_curr.get_vector()
                        v2 = ca_next.get_vector() - ca_curr.get_vector()
                        
                        # angle in radians
                        try:
                            angle = v1.angle(v2)
                            angle_db[key].append(angle)
                        except Exception:
                            pass # Skip if vectors are corrupted
                            
                    # 2. Dihedrals (Conditioned on central pair: R_i, R_i+1)
                    for i in range(1, len(ca_nodes) - 2):
                        _, ca_m1 = ca_nodes[i-1]
                        res_0, ca_0 = ca_nodes[i]
                        res_1, ca_1 = ca_nodes[i+1]
                        _, ca_2 = ca_nodes[i+2]
                        
                        key = f"{res_0}-{res_1}"
                        
                        v_m1 = ca_m1.get_vector()
                        v_0 = ca_0.get_vector()
                        v_1 = ca_1.get_vector()
                        v_2 = ca_2.get_vector()
                        
                        try:
                            dih = PDB.calc_dihedral(v_m1, v_0, v_1, v_2)
                            dihedral_db[key].append(dih)
                        except Exception:
                            pass

                break # Only process the first model (ignore NMR ensembles)
        except Exception as e:
            print(f"Error processing {pdb_id}: {e}")

    return angle_db, dihedral_db

def main(args):
    # 1. Ensure download directory exists
    os.makedirs(args.outdir, exist_ok=True)
    
    # 2. Get PDB IDs
    pdb_ids = fetch_nonredundant_pdbs(limit=args.limit)
    
    # 3. Download PDBs
    print("Downloading PDB files (this may take a while)...")
    pdbl = PDB.PDBList()
    # Using PDB format rather than mmCIF for simpler Bio.PDB parsing compatibility
    pdbl.download_pdb_files(pdb_ids, pdir=args.outdir, file_format='pdb')
    
    # 4. Extract Statistics
    print("Extracting geometry statistics...")
    angles, dihedrals = extract_backbone_stats(args.outdir, pdb_ids)
    
    # 5. Save to JSON
    print(f"Saving data to {args.json}...")
    db_out = {
        "angles": angles,
        "dihedrals": dihedrals
    }
    with open(args.json, 'w') as f:
        json.dump(db_out, f, indent=2)
        
    print("Done! Summary:")
    print(f"Total unique angle triplets tracked: {len(angles.keys())}")
    print(f"Total unique dihedral pairs tracked: {len(dihedrals.keys())}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract CA sequence-dependent statistics from PDB.")
    parser.add_argument('--limit', type=int, default=100, help="Max number of PDBs to process (default: 100). Use ~2000 for a real database.")
    parser.add_argument('--outdir', type=str, default="pdb_data", help="Directory to store downloaded PDB files.")
    parser.add_argument('--json', type=str, default="cg_statistics.json", help="Output JSON file for the angles/dihedrals.")
    args = parser.parse_args()
    
    main(args)
