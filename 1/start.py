# this script starts a new AIMS calculation for Ethylene at SA-3-CAS(2,2)SCF/6-31G*.
import numpy as np
import pyspawn
import pyspawn.general
import pyspawn.process_geometry as pg
import pyspawn.dictionaries as dicts
import sys

# Processing geometry.xyz file (positions are in hessian.hdf5 so we don't need them)
natoms, atoms, _, comment = pg.process_geometry('geometry.xyz')

# Getting atomic masses from the dictionary and converting to atomic units
# If specific isotopes are needed masses array can be set manually
mass_dict = dicts.get_atomic_masses()
masses = np.asarray([mass_dict[atom]*1822.0 for atom in atoms for i in range(3)])

widths_dict = {'C': 22.7, 'H': 4.7} # Obtained from A L Thompson et al Chemical Physics 370 (2010) 70-77.
widths = np.asarray([widths_dict[atom] for atom in atoms for i in range(3)])

# finite wigner temperature
wigner_temp = 0

# random number seed
seed=69884

# Velocity Verlet classical propagator
clas_prop = "vv"

# adapative 2nd-order Runge-Kutta quantum propagator
qm_prop = "fulldiag"

# adiabtic NPI quantum Hamiltonian
qm_ham = "adiabatic"

# use TeraChem CASSCF or CASCI to compute potentials
potential = "molcas_cas"

# initial time
t0 = 0.0

# time step
ts = 10.0

# final simulation time
tfinal = 3500.0 

# number of dimensions                                                                                           
numdims = natoms*3

# number of electronic states                                                                                                                    
numstates = 3

# OpenMolcas job options set method to 'casscf' or 'caspt2'                                                                                    
molcas_options = {
    "method":       'casscf',
    "pt2":          'xms',
    "basis":        '6-31G*',
    "atoms":        atoms,
    "charge":       0,
    "spinmult":     1,
    "nactel":       2,
    "actorb":       2,
    "inactive":     7,
    "ipea":	        0.0,
    "imaginary":    0.2,
    "cassinglets":  numstates,
    "castargetmult": 1,
    "cas_energy_states": [0,1,1],
    "cas_energy_mults": [1,1,1],
    "python3" : '/cvmfs/soft.ccr.buffalo.edu/versions/2023.01/compat/usr/bin/python', 
    "project": 'ICX'
    }

# trajectory parameters
traj_params = {
    # initial time
    "time": t0,
    # time step
    "timestep": ts,
    # final simulation time
    "maxtime": tfinal,
    # coupling threshhold
    "spawnthresh": (0.5 * np.pi) / ts / 20.0,
    # initial electronic state (indexed such that 0 is the ground state)
    "istate": 1,
    # Gaussian widths
    "widths": widths,
    # atom labels
    "atoms": molcas_options["atoms"],
    # nuclear masses (in a.u)    
    "masses": masses,
    # molcas options (above)
    "molcas_options": molcas_options
    }

sim_params = {
    # initial time   
    "quantum_time": traj_params["time"],
    # time step
    "timestep": traj_params["timestep"],
    # final simulation time
    "max_quantum_time": traj_params["maxtime"],
    # initial qm amplitudes
    "qm_amplitudes": np.ones(1,dtype=np.complex128),
    # energy shift used in quantum propagation
    "qm_energy_shift": 0.000000,
}

# import routines needed for propagation
exec("pyspawn.import_methods.into_simulation(pyspawn.qm_integrator." + qm_prop + ")")
exec("pyspawn.import_methods.into_simulation(pyspawn.qm_hamiltonian." + qm_ham + ")")
exec("pyspawn.import_methods.into_traj(pyspawn.potential." + potential + ")")
exec("pyspawn.import_methods.into_traj(pyspawn.classical_integrator." + clas_prop + ")")

# check for the existence of files from a past run
pyspawn.general.check_files()    

# set up first trajectory
traj1 = pyspawn.traj(numdims, numstates)
traj1.set_numstates(numstates)
traj1.set_numdims(numdims)
traj1.set_parameters(traj_params)

# set momentum by reading the file velocities.xyz
traj1.initial_wigner(seed)

## set up simulation 
sim = pyspawn.simulation()
sim.add_traj(traj1)
sim.set_parameters(sim_params)

#SSAIMS control
sim.enable_ssaims(
   epsilon=1e-10,             # tune for your system
   ss_seed=527516,            # optional
   suspend_during_spawn=True, # suspend SSAIMS during spawn process
   spawn_delay_steps=10,      # wait N time steps after spawn to avoide premature killing
   min_tbf_to_start=2,        # require at least M TBFs to begin selection
   verbose=True               # detailed output for SSAIM. Good for debugging
)

# if you want to turn it off:
#sim.disable_ssaims()
## begin propagation
sim.propagate()

