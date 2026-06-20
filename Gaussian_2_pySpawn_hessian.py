# Script to convert hessian from Gaussian to pySapwn hessian.hdf5 
# By A. Mehmood 08/23/2025
import numpy as np
import h5py
import os

hfile = "Cd33Format_S0_Freq.fchk"

PTABLE = [
  "X",  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
  "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc",
  "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
  "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc",
  "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe",
  "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
  "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os",
  "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr",
  "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf",
  "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt",
  "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",]

def store_any_file(fname):
    """ 
    Reading Any file 
    """
    with open(fname, 'r') as f:
        all_lines = []
        for line in f:
            all_lines.append(line.strip())

    return all_lines

import numpy as np
from typing import Sequence, Optional

def write_xyz(symbols, coords, precision: int = 8):
    xyz = np.asarray(coords, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"`coords` must be an array of shape (N, 3), got {xyz.shape}.")
    n = xyz.shape[0]
    if len(symbols) != n:
        raise ValueError(f"Length of `symbols` ({len(symbols)}) != number of rows in coords ({n}).")
    lines = [str(n), "Geometry from Gaussian .fck file"]
    fmt = f"{{:<2s}} {{:>15.{precision}f}} {{:>15.{precision}f}} {{:>15.{precision}f}}"
    for s, (x, y, z) in zip(symbols, xyz):
        if not isinstance(s, str) or len(s) == 0:
            raise ValueError(f"Invalid atomic symbol: {s!r}")
        lines.append(fmt.format(s, x, y, z))
    with open("geometry.xyz", "w") as f:
        f.write("\n".join(lines) + "\n")


def get_atomic_numbers(all_lines):
    """
    Parse the 'Atomic numbers' block from a Gaussian .fchk file.
    """
    numbers: List[int] = []
    target_n = None
    capture = False    
    for line in all_lines:
        if not capture:
            if line.strip().startswith("Atomic numbers"):
                parts = line.split()
                target_n = int(parts[-1])
                capture = True
            continue
        parts = line.split()
        if not parts:
            continue
        if not all(p.lstrip("+-").isdigit() for p in parts):
            break
        numbers.extend(int(p) for p in parts)
        if len(numbers) >= target_n:
            numbers = numbers[:target_n]
            break
    if target_n is None:
        raise ValueError("Could not find 'Atomic numbers' header in file lines.")
    if len(numbers) != target_n:
        raise ValueError(f"Expected {target_n} atomic numbers, got {len(numbers)}.")
    return numbers

def Z_to_AtmSmbol(atomic_numbers):
    """
    Convert atomic numbers to element symbols.
    """
    symbols = []
    for Z in atomic_numbers:
        if Z < 1 or Z >= len(PTABLE) or PTABLE[Z] is None:
            raise ValueError(f"Atomic number out of range or unknown: {Z}")
        symbols.append(PTABLE[Z])
    return symbols

def range2(start, end):
     return range(start, end+1)

def flat_list(lis):
    flatList = []
    # Iterate with outer list
    for element in lis:
        if type(element) is list:
            # Check if type is list than iterate through the sublist
            for item in element:
                flatList.append(item)
        else:
            flatList.append(element)
    return flatList
 
def read_XYZ(all_lines):
    """ 
    Reading CC XYZ from fchk file 
    """
    #all_lines = all_lines.readlines()
    for s in range(len(all_lines)):                          # Get no of Atoms
        if 'Number of atoms' in all_lines[s]:
            Natom = int(all_lines[s][-10:])  
        
    No_cc = 3* Natom
    nlines = int(No_cc/5.0) + 1
    cc_xyz_list = []
    for s in range(len(all_lines)):                          #  Get CC Coordinates
        if 'Current cartesian coordinates' in all_lines[s]:                              #Reads the Input orientation information
            start = s
            for e in range2(start + 1, start + nlines):
                cc_xyz_list.append(all_lines[e].split() )
    
    cc_xyz_flat = flat_list(cc_xyz_list)

    if len(cc_xyz_flat) != No_cc:
        diff = abs(len(cc_xyz_flat) - No_cc)
        cc_xyz_flat_mod = cc_xyz_flat[:-diff]
        cc_xyz_1D = np.array(cc_xyz_flat_mod, float)
    else:
        cc_xyz_1D = np.array(cc_xyz_flat, float)

    cc_xyz_arr = np.reshape(cc_xyz_1D, (Natom,3))
    cc_xyz_arr = cc_xyz_arr*0.529177                          # Convert from Bohr to Ang.
    
    return Natom, cc_xyz_arr

def symmetricize(arr1D):
    ID = np.arange(arr1D.size)
    return arr1D[np.abs(ID - ID[:,None])]


def fill_lower_diag(a):
    n = int(np.sqrt(len(a)*2))+1
    mask = np.tri(n,dtype=bool, k=-1) # or np.arange(n)[:,None] > np.arange(n)
    out = np.zeros((n,n),dtype=int)
    out[mask] = a
    return out

def read_HessXYZ(all_lines, N_atom):
    """ 
    Reading XYZ Hessian from fchk file:
    all_lines : text file
    """

    hess_list = []
    for s in range(len(all_lines)):                          # Get no of Atoms
        if 'Cartesian Force Constants' in all_lines[s]:
            N_hess = int(all_lines[s][-10:])
            nlines = int(N_hess/5.0) + 1
            start = s
            for e in range2(start + 1, start + nlines):
                hess_list.append(all_lines[e].split() )
    
    hess_flat = flat_list(hess_list)

    # Take out Garbage Strings
    if len(hess_flat) != N_hess:
        diff = abs(len(hess_flat)-N_hess)
        # print('diff = ', diff)
        hess_flat_mod = hess_flat[:-diff]
        hess_1D = np.array(hess_flat_mod, float)
    else:
        hess_1D = np.array(hess_flat,float)

    #hess_1D = hess_1D * ((627.509391)/(0.529117*0.529117))   # From Hartree/bohr to kcal/mol / angstrom
    # hess_1D_mod = np.append(hess_1D, [0])
    # print(hess_1D_mod)

    # print(len(hess_1D))
    len_hess = 3*N_atom
    hess_XYZ = np.zeros((len_hess, len_hess))
    # print(hess_arr.shape, hess_arr.size)
    for i in range(len_hess + 1):
        # i_low = int( 0.5 * i * (i - 1) + -1 )        # Adding n(n+1)/2 elements in up/low triangular matrix
        i_low = int( 0.5 * i * (i - 1)  )              # Adding n(n+1)/2 elements in up/low triangular matrix
        # i_up = int( 0.5 * i *  (i + 1) + 1 )
        i_up = int( 0.5 * i *  (i + 1)   )
        # print(i, i_low, i_up, hess_1D[i_low:i_up])
        hess_XYZ[i-1,0:i] =  hess_1D[i_low: i_up]
        hess_XYZ[0:i,i-1] =  hess_1D[i_low: i_up]

    return hess_XYZ


all_lines = store_any_file("Cd33Format_S0_Freq.fchk")
atm_num = get_atomic_numbers(all_lines)
elements = Z_to_AtmSmbol(atm_num)
Natms, xyzs = read_XYZ(all_lines) 
xyz = np.array(xyzs).flatten().reshape(1, -1)
hess = read_HessXYZ(all_lines, Natms)

write_xyz(elements, xyzs)
xyz *= 1.8897161646321e0
h5out   = h5py.File("hessian.hdf5", "w")
h5out.create_dataset(str('geometry'),data=xyz)
h5out.create_dataset(str('hessian'),data=hess)
    

   

