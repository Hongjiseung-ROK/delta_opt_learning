"""Gaussian 09 실행 모듈."""
import subprocess
import os
from pathlib import Path

from delta_chem.config import G09_EXE, GAUSS_EXEDIR


def run_gaussian(com_path: str) -> str:
    """
    .com 파일을 받아 Gaussian 09를 실행하고 .out 경로를 반환한다.

    Gaussian 09 on Windows는 stdout이 아닌 <name>.out 파일에 직접 쓴다.
    cwd를 .com 디렉토리로 설정하고 파일명만 넘겨야 경로 중복 버그가 없다.
    """
    com_path = Path(com_path).resolve()
    env = os.environ.copy()
    env["GAUSS_EXEDIR"] = GAUSS_EXEDIR

    subprocess.run(
        [G09_EXE, com_path.name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        cwd=str(com_path.parent),
    )

    out_path = com_path.with_suffix(".out")
    if not out_path.exists():
        raise RuntimeError(f"Gaussian .out 파일이 없습니다: {out_path}")

    return str(out_path)
