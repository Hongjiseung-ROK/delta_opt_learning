"""Gaussian 16 실행 모듈."""
import subprocess
import os
from pathlib import Path

from delta_chem.config import G16_EXE, GAUSS_EXEDIR


def run_gaussian(com_path: str) -> str:
    """
    .com 파일을 받아 Gaussian 16를 실행하고 출력 파일 경로를 반환한다.

    Gaussian 16 on Linux는 <name>.log 파일에 직접 쓴다 (stdout은 비어 있음).
    cwd를 .com 디렉토리로 설정하고 파일명만 넘겨야 경로 중복 버그가 없다.
    반환 우선순위: .log (내용 있음) → .out (내용 있음) → RuntimeError
    """
    com_path = Path(com_path).resolve()
    env = os.environ.copy()
    env["GAUSS_EXEDIR"] = GAUSS_EXEDIR

    subprocess.run(
        [G16_EXE, com_path.name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        cwd=str(com_path.parent),
    )

    # Gaussian 16 on Linux → .log, Windows g09 → .out
    for suffix in (".log", ".out"):
        candidate = com_path.with_suffix(suffix)
        if candidate.exists() and candidate.stat().st_size > 0:
            return str(candidate)

    raise RuntimeError(f"Gaussian 출력 파일이 없거나 비어 있습니다: {com_path.stem}")
