# this script restarts the simulation using data from sim.json (which 
# contains the entire current state of the simulation) and sim.hdf5 (which 
# contains a selected history of the simulation
import numpy as np
import pyspawn         

pyspawn.import_methods.into_simulation(pyspawn.qm_integrator.fulldiag)
pyspawn.import_methods.into_simulation(pyspawn.qm_hamiltonian.adiabatic)
pyspawn.import_methods.into_traj(pyspawn.potential.molcas_cas)
pyspawn.import_methods.into_traj(pyspawn.classical_integrator.vv)
    
tfinal = 3500.0

sim = pyspawn.simulation()

sim.restart_from_file("sim.2.json","sim.2.hdf5")

sim.set_maxtime_all(tfinal)
sim.enable_ssaims(
   epsilon=1e-10,            
   ss_seed=527516,           
   suspend_during_spawn=True,
   spawn_delay_steps=10,     
   min_tbf_to_start=2,       
   verbose=True              
)
sim.propagate()







