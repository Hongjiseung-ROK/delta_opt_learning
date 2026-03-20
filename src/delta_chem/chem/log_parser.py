"""
Gaussian .out 파일 파서.
- 최적화 스텝 수, CPU 시간, 정상 종료 여부
- 최종 최적화 좌표 (Standard orientation) → ML 학습 타겟
"""
import re
from dataclasses import dataclass
from pathlib import Path

# 원자번호 → 원소기호 (1~36)
_ATOMIC_NUMBER = {
    1:"H", 2:"He", 3:"Li", 4:"Be", 5:"B", 6:"C", 7:"N", 8:"O", 9:"F", 10:"Ne",
    11:"Na",12:"Mg",13:"Al",14:"Si",15:"P", 16:"S", 17:"Cl",18:"Ar",
    19:"K", 20:"Ca",26:"Fe",28:"Ni",29:"Cu",30:"Zn",35:"Br",36:"Kr",
}


@dataclass
class GaussianResult:
    log_path: str
    normal_termination: bool
    opt_steps: int
    cpu_seconds: float


def parse_log(log_path: str) -> GaussianResult:
    """Gaussian .out 파일에서 결과 요약을 파싱한다."""
    text = _read(log_path)
    normal_termination = "Normal termination of Gaussian" in text
    step_matches = re.findall(r"Step number\s+(\d+)\s+out of", text)
    # opt freq 계산 시 freq 이후 추가 Berny 루프("Step number 1 out of 2")가
    # 붙기 때문에 last 값이 아닌 max 값을 사용해야 실제 opt 스텝 수를 얻는다.
    opt_steps = max(int(m) for m in step_matches) if step_matches else 0
    cpu_seconds = _parse_time(text, "Job cpu time:")
    return GaussianResult(
        log_path=log_path,
        normal_termination=normal_termination,
        opt_steps=opt_steps,
        cpu_seconds=cpu_seconds,
    )


def parse_final_geometry(
    log_path: str,
) -> list[tuple[str, float, float, float]]:
    """
    Gaussian .out에서 마지막 'Standard orientation:' 블록을 파싱하여
    [(원소기호, x, y, z), ...] 리스트를 반환한다 (단위: Angstrom).

    원자 순서는 .com 파일에 쓴 순서와 동일하다.
    """
    text = _read(log_path)

    # 모든 Standard orientation 블록의 시작 위치를 찾아 마지막 것만 사용
    header = "Standard orientation:"
    idx = text.rfind(header)
    if idx == -1:
        return []

    # "Standard orientation:" 이후 라인을 순서대로 읽어
    # 두 번째 구분선(---) 이후~세 번째 구분선 사이의 원자 행만 파싱
    lines = text[idx:].splitlines()
    sep_count = 0
    in_data = False
    row_pattern = re.compile(
        r"^\s+\d+\s+(\d+)\s+\d+\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
    )
    atoms = []
    for line in lines:
        if "---" in line and line.strip().startswith("-"):
            sep_count += 1
            if sep_count == 3:   # 데이터 끝 구분선
                break
            if sep_count == 2:   # 데이터 시작
                in_data = True
            continue
        if in_data:
            m = row_pattern.match(line)
            if m:
                atomic_num = int(m.group(1))
                symbol = _ATOMIC_NUMBER.get(atomic_num, f"X{atomic_num}")
                x, y, z = float(m.group(2)), float(m.group(3)), float(m.group(4))
                atoms.append((symbol, x, y, z))

    return atoms


def parse_input_geometry(
    log_path: str,
) -> list[tuple[str, float, float, float]]:
    """
    Gaussian .out에서 첫 번째 'Input orientation:' 블록을 파싱한다.

    이 블록은 Gaussian에 제출된 초기 좌표(MMFF 구조)를 나타낸다.
    Standard orientation과 동일한 테이블 형식이며, 원자 순서가 동일하다.
    feature_extractor에서 ETKDGv3를 재실행하지 않고 이 값을 MMFF 좌표로 사용한다.
    """
    text = _read(log_path)
    idx = text.find("Input orientation:")
    if idx == -1:
        return []

    lines = text[idx:].splitlines()
    sep_count = 0
    in_data = False
    row_pattern = re.compile(
        r"^\s+\d+\s+(\d+)\s+\d+\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
    )
    atoms = []
    for line in lines:
        if "---" in line and line.strip().startswith("-"):
            sep_count += 1
            if sep_count == 3:
                break
            if sep_count == 2:
                in_data = True
            continue
        if in_data:
            m = row_pattern.match(line)
            if m:
                atomic_num = int(m.group(1))
                symbol = _ATOMIC_NUMBER.get(atomic_num, f"X{atomic_num}")
                x, y, z = float(m.group(2)), float(m.group(3)), float(m.group(4))
                atoms.append((symbol, x, y, z))
    return atoms


def _read(log_path: str) -> str:
    path = Path(log_path)
    # .log가 비어 있으면 .out을 시도 (Gaussian 09 on Windows)
    if path.suffix == ".log" and path.stat().st_size == 0:
        out = path.with_suffix(".out")
        if out.exists():
            path = out
    return path.read_text(encoding="utf-8", errors="replace")


def _parse_time(text: str, label: str) -> float:
    pattern = (
        rf"{re.escape(label)}\s+"
        r"(\d+)\s+days\s+(\d+)\s+hours\s+(\d+)\s+minutes\s+([\d.]+)\s+seconds"
    )
    m = re.search(pattern, text)
    if not m:
        return 0.0
    d, h, mn, s = m.groups()
    return int(d)*86400 + int(h)*3600 + int(mn)*60 + float(s)
