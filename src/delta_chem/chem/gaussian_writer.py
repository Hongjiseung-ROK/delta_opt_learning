"""
Optimized XYZ → Gaussian 09 .com input file
"""
from pathlib import Path


def xyz_to_gaussian_com(
    xyz_path: str,
    com_path: str,
    route: str = "#p opt freq B3LYP/6-31G(d)",
    title: str = "Optimization",
    charge: int = 0,
    multiplicity: int = 1,
    nproc: int = 4,
    mem: str = "4GB",
) -> None:
    """
    Convert an XYZ file to a Gaussian 09 input (.com) file.

    Args:
        xyz_path:     Path to input XYZ file (optimized by xtb)
        com_path:     Output .com file path
        route:        Gaussian route section (keywords line)
        title:        Job title for the comment block
        charge:       Molecular charge
        multiplicity: Spin multiplicity
        nproc:        Number of CPU cores (%NProcShared)
        mem:          Memory allocation (%Mem)
    """
    atoms = _parse_xyz(xyz_path)

    com_path = Path(com_path)
    chk_name = com_path.stem + ".chk"

    with open(com_path, "w") as f:
        f.write(f"%NProcShared={nproc}\n")
        f.write(f"%Mem={mem}\n")
        f.write(f"%Chk={chk_name}\n")
        f.write(f"{route}\n")
        f.write("\n")
        f.write(f"{title}\n")
        f.write("\n")
        f.write(f"{charge} {multiplicity}\n")
        for symbol, x, y, z in atoms:
            f.write(f"{symbol:2s}  {x:14.8f}  {y:14.8f}  {z:14.8f}\n")
        f.write("\n")


def _parse_xyz(xyz_path: str) -> list[tuple[str, float, float, float]]:
    """Parse an XYZ file and return list of (symbol, x, y, z)."""
    atoms = []
    with open(xyz_path) as f:
        lines = f.readlines()

    n_atoms = int(lines[0].strip())
    # lines[1] is the comment line
    for line in lines[2 : 2 + n_atoms]:
        parts = line.split()
        if len(parts) < 4:
            continue
        symbol = parts[0]
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        atoms.append((symbol, x, y, z))

    return atoms
