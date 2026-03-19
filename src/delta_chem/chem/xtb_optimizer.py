"""
XYZ file → xtb optimization → optimized XYZ
"""
import subprocess
import shutil
import os
from pathlib import Path


def optimize_with_xtb(
    input_xyz: str,
    output_xyz: str,
    charge: int = 0,
    multiplicity: int = 1,
    xtb_exe: str = "xtb",
) -> None:
    """
    Run xtb geometry optimization on input_xyz.
    Writes the optimized geometry to output_xyz.

    Requires xtb to be installed and on PATH (or provide full path via xtb_exe).
    """
    input_path = Path(input_xyz).resolve()
    work_dir = input_path.parent

    uhf = multiplicity - 1  # xtb uses --uhf (number of unpaired electrons)

    cmd = [
        xtb_exe,
        str(input_path),
        "--opt",
        "--chrg", str(charge),
        "--uhf", str(uhf),
    ]

    result = subprocess.run(
        cmd,
        cwd=str(work_dir),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"xtb failed (exit code {result.returncode}):\n{result.stderr}"
        )

    # xtb writes optimized geometry to xtbopt.xyz in the working directory
    xtbopt_path = work_dir / "xtbopt.xyz"
    if not xtbopt_path.exists():
        raise FileNotFoundError(
            f"xtb ran successfully but xtbopt.xyz not found in {work_dir}"
        )

    shutil.move(str(xtbopt_path), output_xyz)

    # Clean up xtb scratch files
    for fname in ["xtbrestart", "xtbtopo.mol", "charges", "wbo", ".xtboptok"]:
        p = work_dir / fname
        if p.exists():
            p.unlink()
