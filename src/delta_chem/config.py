"""
중앙 경로 설정. 환경변수로 오버라이드 가능.
  DELTA_G16_EXE, DELTA_XTB_EXE, GAUSS_EXEDIR
"""
import os

G16_EXE      = os.environ.get("DELTA_G16_EXE", "/opt/gaussian/g16/g16")
GAUSS_EXEDIR = os.environ.get("GAUSS_EXEDIR",  "/opt/gaussian/g16/bsd:/opt/gaussian/g16")
XTB_EXE      = os.environ.get("DELTA_XTB_EXE", "xtb")
