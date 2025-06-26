#!/usr/bin/env python
"""
calc_models_mpi_child.py

A *single* long-lived child process that:
  - Sets resource limits (memory).
  - Initializes MontePython once (if needed).
  - Then loops: read a pickled "model" from stdin, do the CLASS + lkl computation,
    write pickled results to stdout. Repeat.
  - Exits only if it receives 'DONE' or is killed by the OS (OOM).

Usage:
   python calc_models_mpi_child.py --child-mode
(Or adapt as needed.)
"""
import sys
import os
import pickle
import resource
import signal
import traceback
import contextlib
import numpy as np
import time
import itertools
import copy
import psutil
import io

os.environ["UCX_LOG_LEVEL"] = "error"

param_file = sys.argv[2]
CONNECT_PATH = sys.argv[3]
rank = int(sys.argv[4])

prefix = f"[CHILD][Rank {rank}] |"  # Print prefix from all messages from this child


sys.path.insert(0, CONNECT_PATH)

import numpy as np
import classy
from scipy.interpolate import CubicSpline

from source.default_module import Parameters
from source.tools import get_computed_cls, get_z_idx, get_covmat

param_file = os.path.join(CONNECT_PATH, param_file)
param = Parameters(param_file)
param_names = list(param.parameters.keys())


if (
    param.mcmc_sampler == "montepython"
    and param.use_likelihood_filter
    and param.sampling == "iterative"
):
    from Speciale.connectv2.source.lkl_filter_module.likelihood_calc_montepython_debug import (
        MontePythonLikelihoodCalculator as likelihood_calculator,
    )
# elif param.mcmc_sampler == 'cobaya' and param.use_likelihood_filter and param.sampling == 'iterative':
# from source.lkl_filter_module.likelihood_calc_cobaya import CobayaLikelihoodCalculator as likelihood_calculator
#    pass


hard_limit_bytes = int(os.environ.get("SLURM_MEM_PER_CPU", "7000")) * 1024**2


def main_child():
    """
    Main loop for the child process.
    """

    try:
        if param.use_likelihood_filter and param.sampling == "iterative":
            likelihood_calc = likelihood_calculator(param, rank + 200)
            # print(f"cosmo_arguments: {initial_cosmo_arguments}", flush=True)

    except Exception as e:
        print(
            f"{prefix} [ERROR] Failed to initialize the likelihood calculator: {repr(e)}",
            file=sys.stderr,
            flush=True,
        )
        print(
            f"{prefix} traceback: {traceback.format_exc()}", file=sys.stderr, flush=True
        )
        sys.exit(1)

    # We'll define a loop that repeatedly reads a "command" from stdin.
    while True:
        # 3) Wait for one pickled command from the parent

        command, command_data = read_one_message(sys.stdin.buffer)
        if command and command_data is None:
            # parent closed pipe or something unexpected
            print(
                f"{prefix} Parent closed pipe or something unexpected",
                file=sys.stderr,
                flush=True,
            )
            print(f"{prefix} Received None from parent", file=sys.stderr, flush=True)
            break

        # The command could be e.g. "DONE" or "RUN_MODEL" or "RUN_LIKELIHOOD"
        if command == "DONE":
            # Normal shutdown
            break

        soft_limit_bytes = int(
            command_data.get("soft_limit_bytes", 0.8 * hard_limit_bytes)
        )

        resource.setrlimit(resource.RLIMIT_AS, (soft_limit_bytes, hard_limit_bytes))

        # We got a model to compute
        error_msg = None
        result = {
            "success": False,
            "error_msg": None,
            "cls": None,
            "ell": None,
            "pks": None,
            "der": None,
            "bg": None,
            "bg_idx": None,
            "z_bg": None,
            "th": None,
            "th_idx": None,
            "z_th": None,
            "extra_output": None,
            "model": None,
            "loglkl_true": None,
            "finished_cosmo": False,
            "finished_lkl": False,
        }

        # 4) Extract the relevant fields
        #    e.g. param, model, param_names, do_lkl, etc.
        params = command_data["params"]  # The dictionary to pass to CLASS
        model = command_data["model"]  # The param values
        global_ell = command_data["global_ell"]

        result["model"] = model

        # 5) We'll enforce a 200-second time limit for each model
        def timeout_handler(sig, frame):
            raise RuntimeError("timeout")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(200)

        success = False
        success_cosmo = False
        success_lkl = False
        finished_cosmo = False
        finished_lkl = False

        # ------------------------ COMPUTE COSMO ------------------------

        if command == "RUN_MODEL":

            # Compute cosmo and collect results
            try:

                import psutil
                import time
                import threading
                import classy

                """
                # Function to monitor memory usage
                def monitor_memory(pid, interval=0.1, monitor_name="CLASS computation"):
                    process = psutil.Process(pid)
                    peak_memory = 0
                    while not stop_monitoring.is_set():
                        mem_usage = process.memory_info().rss / 1e6  # Convert to MB
                        peak_memory = max(peak_memory, mem_usage)
                        time.sleep(interval)
                    print(
                        f"[DEBUG] {prefix} Peak memory usage during {monitor_name}: {peak_memory:.2f} MB",
                        file=sys.stderr,
                        flush=True,
                    )
                """

                cosmo = classy.Class()
                cosmo.set(params)
                cosmo.compute()

                if len(param.output_bg) > 0:
                    bg = cosmo.get_background()
                    result["bg"] = bg
                    if len(param.z_bg_list) > 0:
                        z_bg = param.z_bg_list
                        result["z_bg"] = z_bg
                    else:
                        bg_idx = get_z_idx(bg["z"])
                        result["bg_idx"] = bg_idx
                        z_bg = bg["z"][bg_idx]
                        result["z_bg"] = z_bg
                if len(param.output_th) > 0:
                    th = cosmo.get_thermodynamics()
                    result["th"] = th
                    if len(param.z_th_list) > 0:
                        z_th = param.z_th_list
                        result["z_th"] = z_th
                    else:
                        th_idx = get_z_idx(th["z"])
                        result["th_idx"] = th_idx
                        z_th = th["z"][th_idx]
                        result["z_th"] = z_th
                if len(param.output_derived) > 0:
                    der = cosmo.get_current_derived_parameters(param.output_derived)
                    result["der"] = der
                if len(param.output_Cl) > 0:
                    cls = get_computed_cls(cosmo, ell_array=global_ell)
                    result["cls"] = cls
                    if any(np.isnan(cls[key]).any() for key in cls):
                        raise classy.CosmoComputationError(
                            "Class computation completed with NaN values in CMB power spectra."
                        )
                        result["error_msg"] = (
                            f"{prefix} Class computation completed with NaN values in CMB power spectra."
                        )
                    ell = cls["ell"][2:]
                    result["ell"] = ell
                if len(param.output_Pk) > 0:
                    pks = {}
                    for pk in param.output_Pk:
                        pks[pk] = {}
                        for z in param.z_Pk_list:
                            pks[pk][z] = []
                            for k in param.k_grid:
                                pks[pk][z].append(eval(f"cosmo.{pk}(k,z)"))
                    result["pks"] = pks

                extra_output = {}
                for output in param.extra_output:
                    extra_output[output] = eval(param.extra_output[output])
                result["extra_output"] = extra_output

                success_cosmo = True
            except classy.CosmoComputationError as e:
                success_cosmo = False
                tb = traceback.format_exc()
                error_msg = f"{prefix} The following model failed in CLASS: \n"
                error_msg += f"params: {params}\n"
                error_msg += f"error: {repr(e)}\n"
                result["error_msg"] = error_msg
                error_msg += f"traceback: {tb}\n"
                print(error_msg, file=sys.stderr, flush=True)
            except classy.CosmoSevereError as e:
                success_cosmo = False
                tb = traceback.format_exc()
                error_msg = f"{prefix} The following model failed in CLASS: \n"
                error_msg += f"params: {params}\n"
                error_msg += f"error: {repr(e)}\n"
                result["error_msg"] = error_msg
                error_msg += f"traceback: {tb}\n"
                print(error_msg, file=sys.stderr, flush=True)
            except RuntimeError as e:
                success_cosmo = False
                # check if it's the "timeout"
                if str(e) == "timeout":
                    error_msg = (
                        f"{prefix} The following model took too long to complete: \n"
                    )
                    error_msg += f"params: {params}\n"
                    result["error_msg"] = error_msg
                    print(error_msg, file=sys.stderr, flush=True)
                else:
                    tb = traceback.format_exc()
                    error_msg = f"{prefix} The following model failed in CLASS: \n"
                    error_msg += f"params: {params}\n"
                    error_msg += f"error: {repr(e)}\n"
                    result["error_msg"] = error_msg
                    error_msg += f"traceback: {tb}\n"
                    print(error_msg, file=sys.stderr, flush=True)
            except MemoryError:
                # Might happen in Python space, but if in C it might OOM kill the process
                success_cosmo = False
                tb = traceback.format_exc()
                error_msg = f"{prefix} The following model failed in CLASS: \n"
                error_msg += f"params: {params}\n"
                error_msg += f"error: MemoryError in child process\n"
                result["error_msg"] = error_msg
                error_msg += f"traceback: {tb}\n"
                print(error_msg, file=sys.stderr, flush=True)
            except Exception as e:
                success_cosmo = False
                tb = traceback.format_exc()
                error_msg = f"{prefix} The following model failed in CLASS: \n"
                error_msg += f"params: {params}\n"
                success_cosmo = False
                error_msg += f"error: {repr(e)}\n"
                result["error_msg"] = error_msg
                error_msg += f"traceback: {tb}\n"
                print(error_msg, file=sys.stderr, flush=True)

            finished_cosmo = True  # Doesn't mean it was successful, just that the child hasn't been killed
            result["finished_cosmo"] = finished_cosmo

            if not param.use_likelihood_filter:
                signal.alarm(0)
                if success_cosmo:
                    result["success"] = True

        write_one_message(
            command="MODEL_DONE", output_stream=sys.stdout.buffer, obj=result
        )

        # ------------------------ COMPUTE LIKELIHOOD ------------------------

        if (
            param.use_likelihood_filter
            and param.sampling == "iterative"
            and success_cosmo
        ):

            command, command_data = read_one_message(sys.stdin.buffer)
            if command_data is None:
                # parent closed pipe or something unexpected
                break

            soft_limit_bytes = int(
                command_data.get("soft_limit_bytes", 0.8 * hard_limit_bytes)
            )
            resource.setrlimit(resource.RLIMIT_AS, (soft_limit_bytes, hard_limit_bytes))

            if command == "RUN_LIKELIHOOD":
                # Compute likelihood
                try:
                    position = {}
                    for i, par_name in enumerate(param_names):
                        position[par_name] = [model[i]]
                    # set nuisance parameters to fixed values based the input parameter nuisance_params_lkl
                    for (
                        nuisance_name,
                        nuisance_value,
                    ) in param.nuisance_params_lkl.items():
                        position[nuisance_name] = [nuisance_value]
                    # compute likelihood
                    loglkl_true = likelihood_calc.loglkl(position, cosmo)
                    result["loglkl_true"] = loglkl_true
                    success_lkl = True

                except classy.CosmoComputationError as e:
                    success_lkl = False
                    tb = traceback.format_exc()
                    error_msg = f"The following model failed likelihood computation: \n"
                    error_msg += f"params: {params}\n"
                    error_msg += f"error: {repr(e)}\n"
                    result["error_msg"] = error_msg
                    error_msg += f"traceback: {tb}\n"
                    print(error_msg, file=sys.stderr, flush=True)
                except classy.CosmoSevereError as e:
                    success_lkl = False
                    tb = traceback.format_exc()
                    error_msg = f"The following model failed likelihood computation: \n"
                    error_msg += f"params: {params}\n"
                    error_msg += f"error: {repr(e)}\n"
                    result["error_msg"] = error_msg
                    error_msg += f"traceback: {tb}\n"
                    print(error_msg, file=sys.stderr, flush=True)
                except RuntimeError as e:
                    success_lkl = False
                    # check if it's the "timeout"
                    if str(e) == "timeout":
                        error_msg = f"The following model took too long to complete: \n"
                        error_msg += f"params: {params}\n"
                        result["error_msg"] = error_msg
                        print(error_msg, file=sys.stderr, flush=True)
                    else:
                        tb = traceback.format_exc()
                        error_msg = (
                            f"The following model failed likelihood computation: \n"
                        )
                        error_msg += f"params: {params}\n"
                        error_msg += f"error: {repr(e)}\n"
                        result["error_msg"] = error_msg
                        error_msg += f"traceback: {tb}\n"
                        print(error_msg, file=sys.stderr, flush=True)
                except MemoryError:
                    # Might happen in Python space, but if in C it might OOM kill the process
                    success_lkl = False
                    tb = traceback.format_exc()
                    error_msg = f"The following model failed likelihood computation: \n"
                    error_msg += f"params: {params}\n"
                    error_msg += f"error: MemoryError in child process\n"
                    result["error_msg"] = error_msg
                    error_msg += f"traceback: {tb}\n"
                    print(error_msg, file=sys.stderr, flush=True)
                except Exception as e:
                    success_lkl = False
                    tb = traceback.format_exc()
                    error_msg = f"The following model failed likelihood computation: \n"
                    error_msg += f"params: {params}\n"
                    error_msg += f"error: {repr(e)}\n"
                    result["error_msg"] = error_msg
                    error_msg += f"traceback: {tb}\n"
                    print(error_msg, file=sys.stderr, flush=True)

                finished_lkl = True
                result["finished_lkl"] = finished_lkl

                if loglkl_true is not None and np.isnan(loglkl_true):
                    success_lkl = False
                    error_msg += (
                        f"The following model failed likelihood computation: \n"
                    )
                    error_msg += f"params: {params}\n"
                    error_msg += f"error: Likelihood computation completed with NaN values in loglkl.\n"
                    result["error_msg"] = error_msg
                    print(error_msg, file=sys.stderr, flush=True)

                if loglkl_true is None:
                    success_lkl = False
                    error_msg += (
                        f"The following model failed likelihood computation: \n"
                    )
                    error_msg += f"params: {params}\n"
                    error_msg += f"error: Likelihood computation returned None.\n"
                    result["error_msg"] = error_msg
                    print(error_msg, file=sys.stderr, flush=True)

            signal.alarm(0)

            if success_cosmo and success_lkl:
                result["success"] = True
            else:
                result["success"] = False

            write_one_message(
                command="LIKELIHOOD_DONE", output_stream=sys.stdout.buffer, obj=result
            )

        cosmo.struct_cleanup()

    # End of main child loop
    # If we get here, presumably we got "DONE" or the parent pipe closed
    print(
        f"{prefix} [DEBUG] Exiting main_child loop. Killing child process.",
        flush=True,
        file=sys.stderr,
    )
    sys.exit(0)


import pickle
import struct

import pickle
import tempfile
import os

# Find the correct scratch directory
SCRATCH_DIR = os.getenv("SLURM_JOB_ID")
if SCRATCH_DIR:
    SCRATCH_DIR = f"/scratch/{SCRATCH_DIR}/"
    os.makedirs(SCRATCH_DIR, exist_ok=True)  # Ensure the directory exists
else:
    raise RuntimeError(
        "SLURM_JOB_ID not found! This script must be run in a SLURM job."
    )


def write_one_message(command, output_stream, obj):
    """
    Saves `obj` as a temporary pickle file in SCRATCH, then writes `command` and file path to `output_stream`.

    Parameters:
        command (str): The command to send through the pipe.
        output_stream (stream): The output pipe to the other process.
        obj (any picklable object): The data to send.

    The function:
    - Saves `obj` as a pickle file.
    - Writes `command` + file path through the pipe.
    """
    # Create a temporary file in SCRATCH for the pickled data
    with tempfile.NamedTemporaryFile(
        dir=SCRATCH_DIR, delete=False, suffix=".pkl"
    ) as temp_file:
        pickle.dump(obj, temp_file, protocol=pickle.HIGHEST_PROTOCOL)
        file_path = temp_file.name  # Store the file path

    # Send the command and file path as a single message over the pipe
    message = f"{command} {file_path}\n".encode("utf-8")  # Convert string to bytes
    output_stream.write(message)
    output_stream.flush()


def read_one_message(input_stream):
    """
    Reads a command and file path from `input_stream`, loads the pickled data from the file, and deletes it.

    Parameters:
        input_stream (stream): The input pipe to read from.

    Returns:
        tuple: (command, data)
    """
    # Read the command and file path from the pipe
    line = input_stream.readline().strip()

    if not line:
        return None, None  # Handle EOF

    # Decode the binary input into a string (Important Fix!)
    if isinstance(line, bytes):
        line = line.decode("utf-8")  # Convert bytes to string

    # Extract command and file path
    parts = line.split(" ", 1)
    if len(parts) != 2:
        raise ValueError(
            f"[ERROR] Malformed message received: {line}"
        )  # Does ValueError work with flush and file or should it be print?

    command, file_path = parts

    # Read the pickle file
    with open(file_path, "rb") as file:
        data = pickle.load(file)

    # Delete the temporary pickle file
    os.remove(file_path)

    return command, data


if __name__ == "__main__":
    # We could parse sys.argv. If we see "--child-mode", run main_child.
    # E.g.:
    if "--child-mode" in sys.argv:

        main_child()
    else:
        print("This script is meant to run in --child-mode.")
        sys.exit(1)
