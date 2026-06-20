# Script converts an OpenMolcas SLAPAF *.h5 file to a pySpawn hessian.hdf5
# By A. Mehmood, adapted for OpenMolcas 06/08/2026
# 06/13/2026: project out translations/rotations so the 6 zero modes come out
# exactly real & zero (avoids complex Wigner displacements in pySpawn)
import sys
import numpy as np
import h5py

# atomic masses keyed by element symbol. Copied from pySpawn.
masses_dict = {'H' : 1.008,'HE' : 4.003, 'LI' : 6.941, 'BE' : 9.012,\
               'B' : 10.811, 'C' : 12.011, 'N' : 14.007, 'O' : 15.999,\
               'F' : 18.998, 'NE' : 20.180, 'NA' : 22.990, 'MG' : 24.305,\
               'AL' : 26.982, 'SI' : 28.086, 'P' : 30.974, 'S' : 32.066,\
               'CL' : 35.453, 'AR' : 39.948, 'K' : 39.098, 'CA' : 40.078,\
               'SC' : 44.956, 'TI' : 47.867, 'V' : 50.942, 'CR' : 51.996,\
               'MN' : 54.938, 'FE' : 55.845, 'CO' : 58.933, 'NI' : 58.693,\
               'CU' : 63.546, 'ZN' : 65.38, 'GA' : 69.723, 'GE' : 72.631,\
               'AS' : 74.922, 'SE' : 78.971, 'BR' : 79.904, 'KR' : 84.798,\
               'RB' : 84.468, 'SR' : 87.62, 'Y' : 88.906, 'ZR' : 91.224,\
               'NB' : 92.906, 'MO' : 95.95, 'TC' : 98.907, 'RU' : 101.07,\
               'RH' : 102.906, 'PD' : 106.42, 'AG' : 107.868, 'CD' : 112.414,\
               'IN' : 114.818, 'SN' : 118.711, 'SB' : 121.760, 'TE' : 126.7,\
               'I' : 126.904, 'XE' : 131.294, 'CS' : 132.905, 'BA' : 137.328,\
               'LA' : 138.905, 'CE' : 140.116, 'PR' : 140.908, 'ND' : 144.243,\
               'PM' : 144.913, 'SM' : 150.36, 'EU' : 151.964, 'GD' : 157.25,\
               'TB' : 158.925, 'DY': 162.500, 'HO' : 164.930, 'ER' : 167.259,\
               'TM' : 168.934, 'YB' : 173.055, 'LU' : 174.967, 'HF' : 178.49,\
               'TA' : 180.948, 'W' : 183.84, 'RE' : 186.207, 'OS' : 190.23,\
               'IR' : 192.217, 'PT' : 195.085, 'AU' : 196.967, 'HG' : 200.592,\
               'TL' : 204.383, 'PB' : 207.2, 'BI' : 208.980, 'PO' : 208.982,\
               'AT' : 209.987, 'RN' : 222.081, 'FR' : 223.020, 'RA' : 226.025,\
               'AC' : 227.028, 'TH' : 232.038, 'PA' : 231.036, 'U' : 238.029,\
               'NP' : 237, 'PU' : 244, 'AM' : 243, 'CM' : 247, 'BK' : 247,\
               'CT' : 251, 'ES' : 252, 'FM' : 257, 'MD' : 258, 'NO' : 259,\
               'LR' : 262, 'RF' : 261, 'DB' : 262, 'SG' : 266, 'BH' : 264,\
               'HS' : 269, 'MT' : 268, 'DS' : 271, 'RG' : 272, 'CN' : 285,\
               'NH' : 284, 'FL' : 289, 'MC' : 288, 'LV' : 292, 'TS' : 294,\
               'OG' : 294}

ATOMIC_MASSES = {k: v * 1822.0 for k, v in masses_dict.items()}

def read_molcas_h5(fname):
    """Read geometry, Hessian, and atom symbols from an OpenMolcas SLAPAF .h5 file.

    Returns
    -------
    pos    : (1, 3N) ndarray  -- flattened Cartesian geometry in bohr
    hess   : (3N, 3N) ndarray -- Cartesian Hessian in Hartree/bohr**2
    symbols: list of N element symbols
    """
    with h5py.File(fname, 'r') as f:
        ndof = int(f.attrs['DOF'])                 # 3N degrees of freedom

        # Geometry: CENTER_COORDINATES is the final optimized geometry (bohr),
        # shape (N, 3). Flatten to (1, 3N) to match the pySpawn/ORCA layout.
        coords = f['CENTER_COORDINATES'][:]        # (N, 3), bohr
        pos = coords.flatten().reshape(1, -1)

        # Atom labels: CENTER_LABELS is a (N,) array of fixed-width byte strings
        # like b'C       ', b'H       '. Strip whitespace and digits to get the
        # bare element symbol.
        raw_labels = f['CENTER_LABELS'][:]
        symbols = []
        for lab in raw_labels:
            s = lab.decode() if isinstance(lab, bytes) else str(lab)
            s = ''.join(ch for ch in s if ch.isalpha()).strip()
            # keep only the leading element part (e.g. 'C1' -> 'C')
            sym = s[0].upper() + s[1:2].lower() if len(s) > 1 else s.upper()
            # collapse to a known symbol by trying 2-char then 1-char
            if sym in ATOMIC_MASSES:
                symbols.append(sym)
            elif sym[0] in ATOMIC_MASSES:
                symbols.append(sym[0])
            else:
                symbols.append(sym)

        # Hessian: stored as the packed lower triangle (row-major, i >= j),
        # length 3N*(3N+1)/2. Unpack into the full symmetric (3N, 3N) matrix.
        tri = f['HESSIAN'][:]
        expected = ndof * (ndof + 1) // 2
        if tri.shape[0] != expected:
            raise ValueError(
                f"HESSIAN length {tri.shape[0]} != expected {expected} "
                f"for DOF={ndof}")

        hess = np.zeros((ndof, ndof))
        k = 0
        for i in range(ndof):
            for j in range(i + 1):
                hess[i, j] = tri[k]
                hess[j, i] = tri[k]
                k += 1

    return pos, hess, symbols


def build_trans_rot_basis(pos, masses):
    """Build an orthonormal basis (in mass-weighted Cartesians) spanning the
    6 (or 5 for linear) translational and rotational degrees of freedom.

    Parameters
    ----------
    pos    : (3N,) geometry in bohr
    masses : (3N,) per-coordinate masses (each atom mass repeated x3)

    Returns
    -------
    B : (3N, k) ndarray, orthonormal columns spanning trans/rot in the
        mass-weighted frame (q = sqrt(m) * x).
    """
    n3 = pos.shape[0]
    nat = n3 // 3
    x = pos.reshape(nat, 3)
    m_at = masses.reshape(nat, 3)[:, 0]          # one mass per atom
    sqrt_m = np.sqrt(m_at)

    # shift to center of mass
    com = (m_at[:, None] * x).sum(0) / m_at.sum()
    xc = x - com

    D = np.zeros((n3, 6))
    # translations: sqrt(m_a) along each Cartesian axis
    for a in range(nat):
        for ax in range(3):
            D[3 * a + ax, ax] = sqrt_m[a]
    # rotations: sqrt(m_a) * (e_axis x r_a)
    for a in range(nat):
        rx, ry, rz = xc[a]
        s = sqrt_m[a]
        # Rx
        D[3 * a + 0, 3] += 0.0
        D[3 * a + 1, 3] += -s * rz
        D[3 * a + 2, 3] += s * ry
        # Ry
        D[3 * a + 0, 4] += s * rz
        D[3 * a + 1, 4] += 0.0
        D[3 * a + 2, 4] += -s * rx
        # Rz
        D[3 * a + 0, 5] += -s * ry
        D[3 * a + 1, 5] += s * rx
        D[3 * a + 2, 5] += 0.0

    # orthonormalize columns, dropping any null columns (linear molecules
    # give only 5 independent rotations -> 1 column drops out).
    Q, R = np.linalg.qr(D)
    keep = np.abs(np.diag(R)) > 1e-8
    return Q[:, keep]


def project_out_trans_rot(hess, pos, masses):
    """Remove translational/rotational contamination from the Cartesian Hessian
    using an Eckart-style projection, so its null space comes out exactly zero.

    P = I - B B^T  (in mass-weighted coords); project, then transform back to
    plain Cartesian Hessian.
    """
    n3 = hess.shape[0]
    sqrt_m = np.sqrt(masses)

    # mass-weighted Hessian:  Hmw = M^{-1/2} H M^{-1/2}
    inv_sqrt_m = 1.0 / sqrt_m
    Hmw = (inv_sqrt_m[:, None] * hess) * inv_sqrt_m[None, :]
    Hmw = 0.5 * (Hmw + Hmw.T)

    B = build_trans_rot_basis(pos, masses)       # (3N, k) orthonormal
    P = np.eye(n3) - B @ B.T                      # projector onto internal space
    Hmw_proj = P @ Hmw @ P
    Hmw_proj = 0.5 * (Hmw_proj + Hmw_proj.T)

    # back-transform to plain Cartesian Hessian:  H = M^{1/2} Hmw M^{1/2}
    hess_proj = (sqrt_m[:, None] * Hmw_proj) * sqrt_m[None, :]
    hess_proj = 0.5 * (hess_proj + hess_proj.T)
    return hess_proj


if __name__ == "__main__":
    infile = sys.argv[1] if len(sys.argv) > 1 else "Ethylene.slapaf.h5"
    outfile = sys.argv[2] if len(sys.argv) > 2 else "hessian.hdf5"

    pos, hess, symbols = read_molcas_h5(infile)

    # build per-coordinate mass vector (each atom mass repeated 3x)
    try:
        m_at = np.array([ATOMIC_MASSES[s] for s in symbols], dtype=np.float64)
    except KeyError as e:
        raise KeyError(
            f"No mass tabulated for element {e}; add it to ATOMIC_MASSES.")
    masses = np.repeat(m_at, 3)

    # Clean the Hessian so pySpawn's np.linalg.eig returns REAL eigenvectors.
    # Symptoms: 
    # pySpawn re-diagonalizes this Hessian with np.linalg.eig (not eigh). If ANY
    # eigenvalue is complex, even the ~1e-20j numerical noise on the 6 trans/rot
    # zero modes, eig returns a fully complex eigenvector matrix, which makes
    # deltaq complex and crashes `pos += deltaq`.
    #     
    # Solutions (Thanks to Calude):
    # Eckart-project to decouple trans/rot, then (2) floor the 6 null
    # modes to a small POSITIVE value. A positive floor (not zero) stays positive
    # under pySpawn's re-diagonalization round-off, so every eigenvalue eig sees
    # is real -> eigenvectors are real -> deltaq is real. Those 6 modes are
    # sliced off inside initial_wigner anyway (evals[0:ndims-6]), so the floor
    # value never enters the simulations.
    hess = project_out_trans_rot(hess, pos.flatten(), masses)

    sqrt_m = np.sqrt(masses)
    inv = 1.0 / sqrt_m
    Hmw = (inv[:, None] * hess) * inv[None, :]
    Hmw = 0.5 * (Hmw + Hmw.T)
    w, V = np.linalg.eigh(Hmw)
    w = np.real(w)
    # Floor the 6 null modes to small, DISTINCT positive values. 
    # Well-separated values keep every mode non-degenerate -> eig stays real.
    order = np.argsort(w)
    floors = np.array([1e-7, 2e-7, 3e-7, 4e-7, 5e-7, 6e-7])
    nbad = int(np.sum(w < 1e-6))
    for k in range(min(nbad, 6)):
        w[order[k]] = floors[k]
    Hmw = (V * w) @ V.T
    Hmw = 0.5 * (Hmw + Hmw.T)
    hess = (sqrt_m[:, None] * Hmw) * sqrt_m[None, :]
    hess = 0.5 * (hess + hess.T)

    # Force real double precision
    pos = np.real_if_close(pos, tol=1000)
    hess = np.real_if_close(hess, tol=1000)
    pos = np.asarray(pos, dtype=np.float64)
    hess = np.asarray(hess, dtype=np.float64)
    # Enforce exact symmetry
    hess = 0.5 * (hess + hess.T)

    with h5py.File(outfile, "w") as h5out:
        h5out.create_dataset('geometry', data=pos)
        h5out.create_dataset('hessian', data=hess)

    print(f"Wrote {outfile}")
    print(f"  geometry shape: {pos.shape}")
    print(f"  hessian  shape: {hess.shape}")

    # Diagnostic: reproduce EXACTLY what pySpawn does (np.linalg.eig, not eigh)
    # and confirm the eigenvectors come back real. If they do, deltaq will be
    # real and `pos += deltaq` will not crash.
    sqrt_m = np.sqrt(masses)
    inv = 1.0 / sqrt_m
    Hmw = (inv[:, None] * hess) * inv[None, :]
    Hmw = 0.5 * (Hmw + Hmw.T)
    evals, modes = np.linalg.eig(Hmw)
    max_imag_val = np.abs(np.imag(evals)).max()
    max_imag_vec = np.abs(np.imag(modes)).max()
    print(f"  max |Im(eigenvalue)|  = {max_imag_val:.3e}  (must be 0)")
    print(f"  max |Im(eigenvector)| = {max_imag_vec:.3e}  (must be 0)")
    if max_imag_vec == 0.0:
        print("  OK: eig returns real eigenvectors -> pySpawn deltaq will be real.")
    else:
        print("  WARNING: complex eigenvectors remain -> pySpawn will still crash.")