"""
중앙 경로 설정. 환경변수로 오버라이드 가능.
  DELTA_G09_EXE, DELTA_XTB_EXE, GAUSS_EXEDIR
"""
import os

G09_EXE     = os.environ.get("DELTA_G09_EXE",  r"C:\G09W\g09.exe")
GAUSS_EXEDIR = os.environ.get("GAUSS_EXEDIR",  r"C:\G09W")
XTB_EXE     = os.environ.get(
    "DELTA_XTB_EXE",
    r"C:\Users\1172\miniconda3\envs\delta_chem\Library\bin\xtb.exe",
)
