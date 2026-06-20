# Script converts *.hess from Orca frequencies calculations to pySapwn hessian.hdf5 
# By A. Mehmood 02/12/2025
import numpy as np
import h5py

hfile = open("PTF_Cation_S0_Geom.hess", 'r')


def read_HessOrca(all_lines): 
    all_lines.seek(0)
    for line in all_lines:
        if not "$hessian" in line:
            continue
        else:
            for data in all_lines:
                ndim = int(data.strip())
                nblocks = ndim // 5
                rest = ndim % 5
                if rest == 0:
                    nblocks -= 1
                    rest = 5
                hess = np.zeros((ndim, ndim))
                next(all_lines)
                break
            count = 0
            while count <= nblocks:
                count2 = 0
                if count != nblocks:
                    for data in all_lines:
                        columns = [ int(num) for num in range(count*5,(count)*5+5) ]
                        data = data.split()
                        n_line = int(data[0])
                        if  count2 >= ndim:
                            break
                        for i in range(5):
                            j = columns[i]
                            hess[n_line][j] = float(data[i+1])
                        count2 += 1
                else:
                    for data in all_lines:
                        columns = [ int(num) for num in range(count*5,(count)*5+rest) ]
                        if data.strip() == '':
                            break
                        data = data.split()
                        n_line = int(data[0])
                        for i in range(rest):
                            j = columns[i]
                            hess[n_line][j] = float(data[i+1])
                        count2 += 1                        
                count += 1
    return hess

def get_xyz(all_lines):
    all_lines.seek(0)
    xyz_data = []
    inside_atoms_section = False
    for line in all_lines:
        line =line.strip()
        if line.startswith("$atoms"):
            inside_atoms_section = True
            continue        
        if inside_atoms_section:
            if line.isdigit():
                continue       
            parts = line.split()
            if len(parts) < 5:
                break
            xyz_data.append([float(parts[2]), float(parts[3]), float(parts[4])])
    xyz = np.array(xyz_data).flatten().reshape(1, -1)
    return xyz


hess = read_HessOrca(hfile) 
pos  = get_xyz(hfile)
 
h5out   = h5py.File("hessian.hdf5", "w")
h5out.create_dataset(str('geometry'),data=pos)
h5out.create_dataset(str('hessian'),data=hess)
