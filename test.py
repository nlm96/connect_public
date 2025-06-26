import os
import sys
import time
import itertools
import signal
import traceback
import copy

import subprocess
import pickle
import struct
import psutil
import io
from mpi4py import MPI

print("[PARENT] Starting parent process")

os.environ["UCX_LOG_LEVEL"] = "error"
param_file = "/home/maanson/Speciale/connectv2/data/debug/log_connect.param"
CONNECT_PATH = "/home/maanson/Speciale/connectv2"
sys.path.insert(0, CONNECT_PATH)

# python /home/maanson/Speciale/connectv2/source/calc_models_mpi_child.py --child-mode 1 /home/maanson/Speciale/connectv2/data/debug/log_connect.param /home/maanson/Speciale/connectv2 1

import numpy as np
import classy
from scipy.interpolate import CubicSpline


def spawn_child(rank):
    """
    Launch the child process that runs calc_models_mpi_child.py --child-mode
    Returns the Popen object
    """

    cmd = [
        sys.executable,
        os.path.join(CONNECT_PATH, "source", "calc_models_mpi_child.py"),
        "--child-mode",
        str(1),
        param_file,
        CONNECT_PATH,
        str(rank),
    ]

    print("[PARENT] Spawning child Popen object now")
    proc = subprocess.Popen(
        cmd,
        env=os.environ,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,  # Child writes errors directly to jobs .err file, not through the pipe
        bufsize=0,  # unbuffered
    )

    return proc


print("[PARENT] Spawning child process")

try:
    child_proc = spawn_child(1)
    print("[PARENT] Child process spawned")
except Exception as e:
    print("[PARENT] Error spawning child process")
    print(e)
    traceback.print_exc()
    sys.exit(1)
