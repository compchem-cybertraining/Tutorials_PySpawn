<div align="center">

# 🌱 PySpawn Tutorial — Ab Initio Multiple Spawning (AIMS)

### Modeling Quantum Dynamics of Excited States with PySpawn + OpenMolcas

[![Workshop](https://img.shields.io/badge/CyberTraining-2026-1f6feb?style=flat-square)](https://compchem-cybertraining.github.io/Cyber_Training_Workshop_2026/)
[![PySpawn](https://img.shields.io/badge/PySpawn-pySpawn17-2ea44f?style=flat-square)](https://github.com/blevine37/pySpawn17)
[![Python](https://img.shields.io/badge/Python-2.7-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![OpenMolcas](https://img.shields.io/badge/Electronic%20Structure-OpenMolcas-orange?style=flat-square)](https://gitlab.com/Molcas/OpenMolcas)

*Hands-on session material for **Multiple Spawning with PySpawn** — CyberTraining Summer School & Workshop, University at Buffalo, July 5–11, 2026.*
</div>

## 📖 Overview

This tutorial walks you through running a complete **Ab Initio Multiple Spawning (AIMS)** nonadiabatic dynamics simulation from start to finish, using **PySpawn** coupled with **OpenMolcas** for on-the-fly electronic structure.

The worked example is the photodynamics of **ethylene (C₂H₄)** at the **SA-3-CAS(2,2)SCF/6-31G\*** level, launched from the Franck–Condon geometry on the S₁ state. The PySpawn–OpenMolcas interface used throughout this tutorial is described in:

> **L. M. Ibele, A. Mehmood, B. G. Levine, D. Avagliano**, *J. Chem. Theory Comput.* (2024). [doi:10.1021/acs.jctc.4c00855](https://doi.org/10.1021/acs.jctc.4c00855)

You will learn how to:

- Install PySpawn and its dependencies in a clean Conda environment
- Convert a Hessian from common electronic structure codes into PySpawn's format
- Generate an ensemble of Wigner-sampled initial conditions (ICs)
- Launch, restart, and analyze AIMS trajectories on an HPC cluster (SLURM)

> 💡 **New to AIMS?** AIMS describes nuclear wavepacket dynamics with a basis of traveling Gaussian *trajectory basis functions* (TBFs) that **spawn** new basis functions near regions of strong nonadiabatic coupling — capturing population transfer through conical intersections without preselecting a reaction coordinate.
>
> **Background reading:**
>
> - Ben-Nun, Quenneville & Martínez, *J. Phys. Chem. A* **104**, 5161 (2000) — [doi:10.1021/jp994174i](https://doi.org/10.1021/jp994174i)
> - Ben-Nun & Martínez, *J. Chem. Phys.* **108**, 7244 (1998) — [doi:10.1063/1.476142](https://doi.org/10.1063/1.476142)
> - Curchod & Martínez, *Chem. Rev.* **118**, 3305 (2018) — [doi:10.1021/acs.chemrev.7b00423](https://doi.org/10.1021/acs.chemrev.7b00423)
> - Fedorov, Seritan, Fales, Martínez & Levine,  *J. Chem. Theory Comput.* **16**, 5485 (2020) — [doi:10.1021/acs.jctc.0c00575](https://doi.org/10.1021/acs.jctc.0c00575)

---

## 🗂️ Repository Layout

```
├── setup_pyspawn.sh                    # One-shot installer (Conda env + pySpawn17)
├── Generate_ICs.sh                     # Builds & submits ICs 2–50 from folder 1/
│
├── Molcas_2_pySpawn_hessian.py         # Hessian converters → pySpawn hessian.hdf5
├── Gaussian_2_pySpawn_hessian.py
├── Orca_2_pySpawn_hessian.py
│
├── Ethylene_S1_Population_Get_Data.py  # Post-processing: extract S1 population
├── Ethylene_S1_Population_Plot_Data.py # Post-processing: plot S1 population
├── Ethylene_AIMS_Data.tar.gz           # Reference results for comparison
│
└── 1/                                  # Template for the first initial condition
    ├── geometry.xyz                    # Franck–Condon geometry
    ├── hessian.hdf5                    # Hessian in pySpawn format (for Wigner sampling)
    ├── INPORB                          # CASSCF guess orbitals at the FC geometry
    ├── start.py                        # Starts a new AIMS run
    ├── restart.py                      # Restarts a run after failure/timeout
    ├── analysis.py                     # On-the-fly analysis (energies, geometry, populations)
    └── Slurm.job                       # SLURM submission script
```

---

## ⚙️ 1. Installation

PySpawn (the `pySpawn17` code) runs on **Python 2.7**. The provided script creates a dedicated Conda environment and installs everything:

```bash
bash setup_pyspawn.sh
```

This will:
1. Source Miniforge3 and create a Python 2.7 environment at `$HOME/pyspawn`
2. Install dependencies (`numpy`, `h5py`, `matplotlib`, `typing`)
3. Clone and install [`pySpawn17`](https://github.com/blevine37/pySpawn17)

Activate the environment whenever you work with the tutorial:

```bash
source /projects/academic/cyberwksp21/SOFTWARE_2026/miniforge3/etc/profile.d/conda.sh
conda activate $HOME/pyspawn
```

---

## 🧮 2. Preparing the Hessian

PySpawn samples initial conditions from a Wigner distribution built around the Franck–Condon point, which requires a **Hessian in PySpawn's native `hessian.hdf5` format**. You can compute the Hessian in your favorite code and convert it with the matching script:

| Source program | Converter script |
|---|---|
| OpenMolcas | `Molcas_2_pySpawn_hessian.py` |
| ORCA | `Orca_2_pySpawn_hessian.py` |
| Gaussian | `Gaussian_2_pySpawn_hessian.py` |

> A Hessian generated through the **TeraChem** interface is also supported by PySpawn directly.

The converted `hessian.hdf5` lives alongside `geometry.xyz` inside the IC template folder `1/`.

---

## 🌀 3. Generating Initial Conditions

Each initial condition is an independent AIMS trajectory differing only by its **random seed** (which sets the Wigner-sampled positions and momenta). Folder `1/` is the fully prepared template; the rest are cloned from it.

```bash
bash Generate_ICs.sh
```

This script will, for ICs **2 → 50**:
- Draw a unique random seed and log it to `IC_List.txt`
- Create the IC folder and copy `geometry.xyz`, `hessian.hdf5`, `INPORB`, `start.py`, `restart.py`, and `Slurm.job` from `1/`
- Patch the seed in `start.py` and the job name in `Slurm.job`
- Submit the job with `sbatch`

The key files inside each IC folder:

| File | Role |
|---|---|
| `geometry.xyz` | Franck–Condon geometry used to build the IC via the Hessian |
| `hessian.hdf5` | Hessian used for Wigner sampling of positions/momenta |
| `INPORB` | CASSCF orbitals at the FC geometry — guess orbitals for every IC |
| `start.py` | Starts a fresh AIMS run (change **`seed`** per IC) |
| `restart.py` | Restarts the run from `sim.json` + `sim.hdf5` after a failure or timeout |
| `Slurm.job` | SLURM script that sets up OpenMolcas and launches `start.py` |
| `analysis.py` | Run at any time to analyze a trajectory (see below) |

---

## 🚀 4. Running & Restarting

The simulation is launched automatically by `Generate_ICs.sh`. To run a single IC manually:

```bash
cd 1/
sbatch Slurm.job          # starts start.py under the pyspawn env + OpenMolcas
```

If a job dies (timeout, node failure, etc.), restart it from the saved state — PySpawn checkpoints the full simulation to `sim.json` and a trajectory history to `sim.hdf5`:

```bash
python restart.py         # resumes from sim.*.json + sim.*.hdf5
```

---

## 📊 5. Analysis

`analysis.py` processes the `sim.hdf5` file into human-readable output and figures — and **can be run while a simulation is still in progress**.

> ⚠️ **Run `analysis.py` from inside a dedicated `analysis/` subdirectory.** The script reads `sim.hdf5` from the parent folder (`../sim.hdf5`) and writes all of its `.dat` files, `.png` figures, and per-trajectory `.xyz` files into the current directory — keep it separate so it does not clutter the run folder.

From inside an IC folder:

```bash
mkdir analysis && cd analysis && mv ../analysis.py .
python analysis.py
```

It produces, among others:

- **Total electronic populations** per state → `Total_El_pop.png`
- **Potential & total energies** (energy conservation check) → `Energies.png`, `Total_Energies.png`
- **Nuclear basis function populations** → `Nuc_pop.png`
- **Geometric observables** — bonds, angles, dihedrals, pyramidalizations, twists → `*.dat`
- Per-trajectory `.xyz` files

### Ensemble S₁ population

For the ensemble-level ethylene S₁ population, use the dedicated post-processing scripts:

```bash
python Ethylene_S1_Population_Get_Data.py     # collect S1 populations across ICs
python Ethylene_S1_Population_Plot_Data.py    # plot the averaged decay
```

> 📦 **Short on time?** `Ethylene_AIMS_Data.tar.gz` contains pre-computed reference AIMS data. If your trajectories have not finished during the session, extract this archive and feed it to the scripts above to obtain (or plot) the S₁ population directly:
> ```bash
> tar -xzvf Ethylene_AIMS_Data.tar.gz
> ```

---

## ✅ Quick Start (TL;DR)

```bash
# 1. Install
bash setup_pyspawn.sh
conda activate $HOME/pyspawn

# 2. (Hessian already provided in 1/hessian.hdf5)

# 3. Generate + submit ICs 2–50
bash Generate_ICs.sh

# 4. Analyze any IC at any time
cd 1 && mkdir analysis && cd analysis && mv ../analysis.py . && python analysis.py
```

---

## 🔗 Resources

- **Workshop page:** [CyberTraining 2026](https://compchem-cybertraining.github.io/Cyber_Training_Workshop_2026/)
- **PySpawn source:** [github.com/blevine37/pySpawn17](https://github.com/blevine37/pySpawn17)
- **OpenMolcas:** [gitlab.com/Molcas/OpenMolcas](https://gitlab.com/Molcas/OpenMolcas)
- **Gaussian width parameters:** A. L. Thompson *et al.*, *Chemical Physics* **370** (2010) 70–77. Widths can be optimized with [**optimwidths**](https://github.com/ispg-group/optimwidths) (B. F. E. Curchod, ISPG group), which implements the method developed by Prof. Benjamin G. Levine.

---

## 🙏 Acknowledgement

Material prepared by **Arshad Mehmood** (Institute for Advanced Computational Science, Stony Brook University) for the **PySpawn / OpenMolcas** session led with **Professor Benjamin G. Levine**. This work is supported by the **NSF-OAC CyberTraining** program.

📧 **Post-workshop questions?** Email **[arshad.mehmood@stonybrook.edu](mailto:arshad.mehmood@stonybrook.edu)**.

<div align="center">

*Happy spawning! 🌱*
