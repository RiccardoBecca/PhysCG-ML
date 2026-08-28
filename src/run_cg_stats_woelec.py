import argparse
import ast
import json
import math
import sys
import numpy as np
import pandas as pd
import scipy.ndimage
import openmm as mm
from openmm import app
from openmm import unit

KB_T = 0.008314462 * 300.0  # kJ/mol at 300 K

class XYZReporter(object):
    def __init__(self, file, reportInterval):
        self._out = open(file, 'w')
        self._reportInterval = reportInterval

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, True, False, False, False, None)

    def report(self, simulation, state):
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstroms)
        topology = simulation.topology
        self._out.write(f"{len(positions)}\n")
        self._out.write(f"Step {simulation.currentStep}\n")
        
        for atom, pos in zip(topology.atoms(), positions):
            element = atom.name.strip() 
            self._out.write(f"{element} {pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}\n")

def calc_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def water_permittivity(T_celsius: float) -> float:
    return (
        87.9144
        - 0.4044 * T_celsius
        + 9.587e-4 * T_celsius**2
        - 1.328e-6 * T_celsius**3
    )

def inv_debye_length_nm(ionic_strength: float, T_celsius: float) -> float:
    N_A = 6.02214076e23
    e = 1.602176634e-19
    epsilon_0 = 8.854187817e-12
    k_B = 1.380649e-23
    T_kelvin = T_celsius + 273.15
    epsilon_r = water_permittivity(T_celsius)
    kappa_m = np.sqrt(e**2 / (epsilon_r * epsilon_0 * k_B * T_kelvin) * 2 * N_A * ionic_strength)
    return kappa_m * 1e-9  # 1/nm

def load_params(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    params = {}
    for _, row in df.iterrows():
        bead = str(row['bead']).strip()
        params[bead] = {
            'C12': np.array(ast.literal_eval(str(row['C12']))),
            'b':     np.array(ast.literal_eval(str(row['b']))),
            'C6': np.array(ast.literal_eval(str(row['C6']))),
        }
    return params

def process_angle_potential(raw_angles, num_bins=100, sigma=1.5):
    domain = (0.0, np.pi)
    counts, _ = np.histogram(raw_angles, bins=num_bins, range=domain, density=True)
    counts = np.maximum(counts, 1e-6)
    counts = scipy.ndimage.gaussian_filter1d(counts, sigma=sigma, mode='nearest')
    V = -KB_T * np.log(counts)
    V -= np.min(V) 
    return V.tolist()

def process_dihedral_potential(raw_dihedrals, num_bins=100, sigma=2.0):
    domain = (-np.pi, np.pi)
    counts, _ = np.histogram(raw_dihedrals, bins=num_bins, range=domain, density=True)
    counts = np.maximum(counts, 1e-6)
    counts_padded = np.tile(counts, 3)
    counts_smoothed = scipy.ndimage.gaussian_filter1d(counts_padded, sigma=sigma, mode='wrap')
    counts = counts_smoothed[num_bins:2*num_bins]
    V = -KB_T * np.log(counts)
    V -= np.min(V)
    return V.tolist()

def run_cg_simulation(folded_pdb_file, unfolded_pdb_file, json_file, csv_file, 
                      output_xyz="trajectory.xyz", steps=20000, 
                      temp_celsius=10.0, ionic_strength=0.150):
    
    # 1. Load the folded PDB to get equilibrium parameters (r_0)
    print(f"Loading folded CG structure for equilibrium bond parameters: {folded_pdb_file}")
    folded_pdb = app.PDBFile(folded_pdb_file)
    folded_coords = folded_pdb.positions.value_in_unit(unit.nanometers)
    
    # 2. Load the unfolded PDB for topology and starting positions
    print(f"Loading unfolded CG structure for starting simulation: {unfolded_pdb_file}")
    unfolded_pdb = app.PDBFile(unfolded_pdb_file)
    
    if folded_pdb.topology.getNumAtoms() != unfolded_pdb.topology.getNumAtoms():
        raise ValueError("Folded and unfolded PDB files must have the same number of atoms!")
    
    print(f"Loading statistical potential dataset: {json_file}")
    with open(json_file, 'r') as f:
        stat_db = json.load(f)

    print(f"Loading non-bonded parameter CSV: {csv_file}")
    nb_params = load_params(csv_file)


    res_beads = []
    particle_res_map = {}
    particle_bead_names = []
    
    # Use the unfolded topology as our main reference for building the system
    for res in unfolded_pdb.topology.residues():
        b_idx = -1
        s_indices = []
        for atom in res.atoms():
            particle_res_map[atom.index] = res.index
            atom_name = atom.name.strip()
            particle_bead_names.append(atom_name)
            
            if atom_name == 'B':
                b_idx = atom.index
            elif atom_name.startswith('S'):
                s_indices.append(atom.index)
        
        res_beads.append({'res_name': res.name.strip(), 'b_idx': b_idx, 's_indices': s_indices})

    system = mm.System()
    for _ in unfolded_pdb.topology.atoms():
        system.addParticle(100.0 * unit.amu)

    # 1. BONDED FORCES (2-body only)
    bond_force = mm.HarmonicBondForce()
    k_bond = 20000.0 * unit.kilojoule_per_mole / unit.nanometer**2

    for i in range(len(res_beads)):
        b_i = res_beads[i]['b_idx']
        s_i = res_beads[i]['s_indices']

        # Backbone bonds (B_i to B_{i+1})
        if i < len(res_beads) - 1:
            b_next = res_beads[i+1]['b_idx']
            if b_i != -1 and b_next != -1:
                bond_force.addBond(b_i, b_next, calc_distance(folded_coords[b_i], folded_coords[b_next]), k_bond)

        # Sidechain bonds (B_i to S_i[0], and S_i[0] to S_i[1] etc.)
        if b_i != -1 and len(s_i) > 0:
            bond_force.addBond(b_i, s_i[0], calc_distance(folded_coords[b_i], folded_coords[s_i[0]]), k_bond)
            if len(s_i) > 1:
                bond_force.addBond(s_i[0], s_i[1], calc_distance(folded_coords[s_i[0]], folded_coords[s_i[1]]), k_bond)

    system.addForce(bond_force)

    # 2. STATISTICAL BACKBONE ANGLES
    angle_forces = {}
    for i in range(1, len(res_beads) - 1):
        b_prev, b_curr, b_next = res_beads[i-1]['b_idx'], res_beads[i]['b_idx'], res_beads[i+1]['b_idx']
        key = f"{res_beads[i-1]['res_name']}-{res_beads[i]['res_name']}-{res_beads[i+1]['res_name']}"

        if key in stat_db['angles']:
            if key not in angle_forces:
                V_table = process_angle_potential(stat_db['angles'][key])
                tab_func = mm.Continuous1DFunction(V_table, 0.0, np.pi)
                force = mm.CustomCompoundBondForce(3, "V_stat(angle(p1, p2, p3))")
                force.addTabulatedFunction("V_stat", tab_func)
                system.addForce(force)
                angle_forces[key] = force
            angle_forces[key].addBond([b_prev, b_curr, b_next], [])

    # 3. STATISTICAL BACKBONE DIHEDRALS
    dih_forces = {}
    for i in range(1, len(res_beads) - 2):
        b_m1, b_0, b_1, b_2 = res_beads[i-1]['b_idx'], res_beads[i]['b_idx'], res_beads[i+1]['b_idx'], res_beads[i+2]['b_idx']
        key = f"{res_beads[i]['res_name']}-{res_beads[i+1]['res_name']}"

        if key in stat_db['dihedrals']:
            if key not in dih_forces:
                V_table = process_dihedral_potential(stat_db['dihedrals'][key])
                tab_func = mm.Continuous1DFunction(V_table, -np.pi, np.pi)
                force = mm.CustomCompoundBondForce(4, "V_stat(dihedral(p1, p2, p3, p4))")
                force.addTabulatedFunction("V_stat", tab_func)
                system.addForce(force)
                dih_forces[key] = force
            dih_forces[key].addBond([b_m1, b_0, b_1, b_2], [])

    # 4. TANG-TOENNIES + DEBYE-HUCKEL NON-BONDED POTENTIAL
    eps_r = water_permittivity(temp_celsius)
    kappa_nm = inv_debye_length_nm(ionic_strength, temp_celsius)
    
    energy_expr = (
        "u_tt;"
        "u_tt = (C12/r)^10 - f6 * (C6/r)^6;"
        "f6 = 1.0 - exp(-x) * (1.0 + x + 0.5*x^2 + x^3/6.0 + x^4/24.0 + x^5/120.0 + x^6/720.0);"
        "x = b * r;"
        "C12 = sqrt(max(1e-12, (C12_01*C12_02 + C12_11*C12_12 + C12_21*C12_22)/3.0));"
        "b = sqrt(max(1e-12, (b_01*b_02 + b_11*b_12 + b_21*b_22)/3.0));"
        "C6 = sqrt(max(1e-12, (C6_01*C6_02 + C6_11*C6_12 + C6_21*C6_22)/3.0));"
    )

    custom_nb = mm.CustomNonbondedForce(energy_expr)
    
    for param_name in ["C12_0", "C12_1", "C12_2", 
                        "b_0", "b_1", "b_2", 
                        "C6_0", "C6_1", "C6_2"]:
        custom_nb.addPerParticleParameter(param_name)


    c_rep_scale = (4.184)**(0.1) / 10.0
    c_att_scale = (4.184)**(1.0/6.0) / 10.0
    
    for atom in unfolded_pdb.topology.atoms():
        bead_name = particle_bead_names[atom.index]
        p = nb_params[bead_name]
        
        c_rep_nm = (p['C12'] * c_rep_scale).tolist()
        b_nm = (p['b'] * 10.0).tolist()
        c_att_nm = (p['C6'] * c_att_scale).tolist()

        particle_params = c_rep_nm + b_nm + c_att_nm
        custom_nb.addParticle(particle_params)

    custom_nb.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic)
    custom_nb.setCutoffDistance(1.8 * unit.nanometer)

    # IMPOSING SEQUENCE LAG:
    num_particles = system.getNumParticles()
    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            if abs(particle_res_map[i] - particle_res_map[j]) < 4:
                custom_nb.addExclusion(i, j)

    system.addForce(custom_nb)

    # 5. SIMULATION EXECUTION
    integrator = mm.LangevinMiddleIntegrator((temp_celsius + 273.15)*unit.kelvin, 1.0/unit.picosecond, 0.002*unit.picoseconds)
    
    # Use the unfolded topology for the Simulation object
    simulation = app.Simulation(unfolded_pdb.topology, system, integrator)
    
    # Set the initial positions to the unfolded structure
    simulation.context.setPositions(unfolded_pdb.positions)
    simulation.minimizeEnergy()
    
    simulation.context.setVelocitiesToTemperature((temp_celsius + 273.15)*unit.kelvin)
    simulation.reporters.append(XYZReporter(output_xyz, 100))
    simulation.reporters.append(app.StateDataReporter(sys.stdout, 1000, step=True, potentialEnergy=True, temperature=True))

    simulation.step(steps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Transferable CG MD with Tang-Toennies + Debye-Huckel potentials.")
    parser.add_argument('--folded_pdb', type=str, required=True, help="Input folded CG PDB file (for equilibrium bonded parameters)")
    parser.add_argument('--unfolded_pdb', type=str, required=True, help="Input unfolded CG PDB file (for starting coordinates)")
    parser.add_argument('--json', type=str, default="cg_statistics.json", help="Input JSON file with statistical angle/dihedral distributions")
    parser.add_argument('--csv', type=str, required=True, help="Path to non-bonded parameters CSV file")
    parser.add_argument('--out', type=str, default="trajectory.xyz", help="Output XYZ trajectory file")
    parser.add_argument('--steps', type=int, default=20000, help="Number of MD steps")
    parser.add_argument('--temp', type=float, default=10.0, help="Temperature in Celsius")
    parser.add_argument('--ionic', type=float, default=0.150, help="Ionic strength in Molar")
    args = parser.parse_args()
    
    run_cg_simulation(args.folded_pdb, args.unfolded_pdb, args.json, args.csv,  args.out, args.steps, args.temp, args.ionic)
