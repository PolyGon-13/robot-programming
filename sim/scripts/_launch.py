import importlib
import os
import runpy
import sys

SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISAACLAB_RSL_RL = os.path.expanduser("~/IsaacLab/scripts/reinforcement_learning/rsl_rl")


def run(script: str) -> None:
    sys.path.insert(0, SIM_DIR)
    sys.path.insert(0, ISAACLAB_RSL_RL)
    importlib.import_module("twolink_rl")

    if "--task" not in sys.argv:
        sys.argv += ["--task", "Twolink-MarkerTrack-v0"]
    os.chdir(SIM_DIR)
    sys.argv[0] = os.path.join(ISAACLAB_RSL_RL, script)
    runpy.run_path(sys.argv[0], run_name="__main__")
