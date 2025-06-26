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
import threading
import psutil
import psutil
import time
import threading
import classy
import mpi4py.MPI as MPI

# Establish connection to parent
child_parent = MPI.Comm.Get_parent()  # Communicator to parent
child_parent.Set_errhandler(MPI.ERRORS_RETURN)  # Avoid crashing on errors
child_rank = child_parent.Get_rank()  # Child rank within this communicator


os.environ["UCX_LOG_LEVEL"] = "error"
os.environ["UCX_HANDLE_ERRORS"] = "no"
os.environ["UCX_ERROR_SIGNALS"] = "no"

param_file = sys.argv[2]
CONNECT_PATH = sys.argv[3]
parent_rank = int(sys.argv[4])


prefix = f"[CHILD][Parent Rank {parent_rank}] |"  # Print prefix from all messages from this child


sys.path.insert(0, CONNECT_PATH)

print(f"[DEBUG] Importing modules", file=sys.stderr, flush=True)
import numpy as np

print(f"[DEBUG] {prefix}: Imported numpy", file=sys.stderr, flush=True)
import classy

print(f"[DEBUG] {prefix}: Imported classy", file=sys.stderr, flush=True)
from scipy.interpolate import CubicSpline

print(f"[DEBUG] {prefix}: Imported CubicSpline", file=sys.stderr, flush=True)

print(f"[DEBUG] {prefix}: Importing modules", file=sys.stderr, flush=True)
from source.default_module import Parameters

print(f"[DEBUG] {prefix}: Imported Parameters", file=sys.stderr, flush=True)
from source.tools import get_computed_cls, get_z_idx, get_covmat

print(f"[DEBUG] {prefix}: Imported tools", file=sys.stderr, flush=True)

param_file = os.path.join(CONNECT_PATH, param_file)
param = Parameters(param_file)
param_names = list(param.parameters.keys())
print(f"[DEBUG] {prefix}: Loaded parameters", file=sys.stderr, flush=True)

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


hard_limit_bytes = (
    int(os.environ.get("SLURM_MEM_PER_CPU", "7000")) * 1024**2 * 0.9
)  # 90% of the memory limit
hard_limit_bytes = int(hard_limit_bytes)


def main_child():
    """
    Main loop for the child process.
    """
    print(
        f"[DEBUG] {prefix}: Location: top of main_child(): | Starting child process",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"[DEBUG] {prefix}: if you see this, the child is alive",
        file=sys.stderr,
        flush=True,
    )

    print(
        f"{prefix} [DEBUG] Python executable:",
        sys.executable,
        flush=True,
        file=sys.stderr,
    )
    print(f"{prefix} [DEBUG] Python path:", sys.path, flush=True, file=sys.stderr)
    print(
        f"{prefix} [DEBUG] PYTHONPATH:",
        os.environ.get("PYTHONPATH"),
        flush=True,
        file=sys.stderr,
    )

    # Create a stop event for the memory monitor thread
    stop_memory_monitor = threading.Event()
    # Initialize the memory limit as a global variable
    global MEMORY_LIMIT_BYTES  # Memory limit for the child process
    MEMORY_LIMIT_BYTES = hard_limit_bytes  # Initial memory limit
    memory_lock = threading.Lock()  # Lock for thread-safe updates

    def get_total_memory_usage():
        """Get total memory usage of the child process and its subprocesses."""
        process = psutil.Process(os.getpid())
        mem_usage = process.memory_info().rss  # Memory usage of the current process
        for child in process.children(recursive=True):
            mem_usage += child.memory_info().rss  # Memory use of any external processes
        return mem_usage

    def memory_monitor():
        """Monitor memory usage and exit safely if it exceeds the threshold."""
        global MEMORY_LIMIT_BYTES
        while not stop_memory_monitor.is_set():  # Exit if stop event is set
            mem_usage = get_total_memory_usage()

            with memory_lock:
                current_limit = MEMORY_LIMIT_BYTES  # Safely read the limit

            if mem_usage > current_limit:
                print(
                    f"{prefix} WARNING: Memory exceeded {mem_usage / (1024**3):.2f} GB! Shutting down safely.",
                    file=sys.stderr,
                    flush=True,
                )

                # Notify parent before exiting
                try:
                    child_parent.send(
                        ("MEMORY_EXCEEDED", {"placeholder": "None"}), dest=0, tag=0
                    )
                except MPI.Exception:
                    pass  # Parent might have already shut down

                # Exit safely to avoid MPI abort
                sys.exit(0)

            time.sleep(0.5)  # Check memory every 0.5s

        print(
            "{prefix} [DEBUG] Memory monitor thread exiting.",
            file=sys.stderr,
            flush=True,
        )

    # Start memory monitor in a separate thread
    monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
    monitor_thread.start()

    try:
        if param.use_likelihood_filter and param.sampling == "iterative":
            likelihood_calc = likelihood_calculator(param, f"{parent_rank}_child")
        # print(f"cosmo_arguments: {initial_cosmo_arguments}", flush=True)
        print(
            f"[DEBUG] {prefix}: Finished initializing the likelihood calculator",
            file=sys.stderr,
            flush=True,
        )
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
        print(
            f"[DEBUG] {prefix}: Waiting for command from parent",
            file=sys.stderr,
            flush=True,
        )
        try:
            sleep_initial = 0.0001  # Start with a reasonable value
            sleep_max = 0.1  # Avoid wasting CPU if the child takes a long time
            sleep_current = sleep_initial  # Start at 0.01s

            while not child_parent.Iprobe(source=0, tag=0):
                time.sleep(sleep_current)  # Wait before checking again
                sleep_current = min(sleep_current * 2, sleep_max)  # Increase wait time

            command, command_data = child_parent.recv(
                source=0, tag=0
            )  # Receive from parent
        except MPI.Exception as e:
            print(
                f"{prefix} ERROR: Failed to receive message from parent: {repr(e)}",
                file=sys.stderr,
                flush=True,
            )
            print(f"The child process will now exit.", file=sys.stderr, flush=True)
            print(
                f"{prefix} traceback: {traceback.format_exc()}",
                file=sys.stderr,
                flush=True,
            )

            sys.exit(0)  # Terminate child safely

        if command and command_data is None:
            # parent closed pipe or something unexpected
            print(
                f"{prefix} Parent closed pipe or something unexpected",
                file=sys.stderr,
                flush=True,
            )
            print(f"{prefix} Received None from parent", file=sys.stderr, flush=True)
            sys.exit(0)

        print(
            f"[DEBUG] {prefix}: Received command: {command}"
            f" with data: {command_data}",
            file=sys.stderr,
            flush=True,
        )

        # The command could be e.g. "DONE" or "RUN_MODEL" or "RUN_LIKELIHOOD"
        if command == "DONE":
            print(
                f"{prefix} [DEBUG] Received 'DONE' from parent. Shutting down safely...",
                file=sys.stderr,
                flush=True,
            )

            # 1. Acknowledge shutdown to the parent
            try:
                child_parent.send(("DONE_ACK", None), dest=0, tag=0)
                print(
                    f"{prefix} [DEBUG] Sent 'DONE_ACK' to parent.",
                    file=sys.stderr,
                    flush=True,
                )
            except MPI.Exception as e:
                print(
                    f"{prefix} WARNING: Failed to send 'DONE_ACK'. Parent may have already exited. Error: {repr(e)}",
                    file=sys.stderr,
                    flush=True,
                )

            # 2. Clean up resources
            try:
                if "cosmo" in locals():  # Only cleanup if cosmo was created
                    cosmo.struct_cleanup()
                    print(
                        f"{prefix} [DEBUG] Cleaned up CLASS resources.",
                        file=sys.stderr,
                        flush=True,
                    )
            except Exception as e:
                print(
                    f"{prefix} WARNING: Failed to clean up CLASS. Error: {repr(e)}",
                    file=sys.stderr,
                    flush=True,
                )

            print(
                f"{prefix} [DEBUG] Stopping memory monitor...",
                file=sys.stderr,
                flush=True,
            )
            # Stop monitoring memory
            stop_memory_monitor.set()
            # Close monitor thread
            monitor_thread.join()

            print(
                f"{prefix} [DEBUG] finished stopping memory monitor.",
                file=sys.stderr,
                flush=True,
            )

            # 3. Disconnect from the parent MPI communicator
            try:
                print(
                    f"{prefix} [DEBUG] Disconnecting child MPI communicator...",
                    file=sys.stderr,
                    flush=True,
                )
                child_parent.Disconnect()
                print(
                    f"{prefix} [DEBUG] Successfully disconnected MPI communicator.",
                    file=sys.stderr,
                    flush=True,
                )
            except MPI.Exception as e:
                print(
                    f"{prefix} WARNING: MPI communicator already cleaned up. Error: {repr(e)}",
                    file=sys.stderr,
                    flush=True,
                )

            # 4. Exit cleanly
            print(
                f"{prefix} [DEBUG] Child process exiting normally.",
                file=sys.stderr,
                flush=True,
            )
            sys.exit(0)

        soft_limit_bytes = int(command_data.get("soft_limit_bytes", hard_limit_bytes))
        with memory_lock:
            MEMORY_LIMIT_BYTES = int(soft_limit_bytes * 0.95)  # Update memory limit
        resource.setrlimit(resource.RLIMIT_AS, (soft_limit_bytes, hard_limit_bytes))

        # We got a model to compute
        error_msg = None
        result = {
            "success": False,
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
            print(
                f"[DEBUG] {prefix}: Computing model: {model}",
                file=sys.stderr,
                flush=True,
            )
            # Compute cosmo and collect results
            try:

                cosmo = classy.Class()
                cosmo.set(params)
                cosmo.compute()

                if parent_rank == 10:
                    # Trigger a memory error for testing
                    print(
                        f"{prefix} Triggering memory error for testing",
                        file=sys.stderr,
                        flush=True,
                    )

                    # Create a large array to trigger a memory error
                    large_array = np.zeros((1000000, 1000000), dtype=np.float64)
                    # This is approximately 7.6 GB of memory

                print(
                    f"[DEBUG] {prefix}: Finished computing model: {model}",
                    file=sys.stderr,
                    flush=True,
                )

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

                print(
                    f"[DEBUG] {prefix}: Finished extracting cosmo results",
                    file=sys.stderr,
                    flush=True,
                )

                success_cosmo = True
            except classy.CosmoComputationError as e:
                success_cosmo = False
                tb = traceback.format_exc()
                error_msg = f"{prefix} The following model failed in CLASS: \n"
                error_msg += f"params: {params}\n"
                error_msg += f"error: {repr(e)}\n"
                print(f"{error_msg}", file=sys.stdout, flush=True)
                error_msg += f"traceback: {tb}\n"
                print(error_msg, file=sys.stderr, flush=True)
            except classy.CosmoSevereError as e:
                success_cosmo = False
                tb = traceback.format_exc()
                error_msg = f"{prefix} The following model failed in CLASS: \n"
                error_msg += f"params: {params}\n"
                error_msg += f"error: {repr(e)}\n"
                print(f"{error_msg}", file=sys.stdout, flush=True)
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
                    print(error_msg, file=sys.stdout, flush=True)
                    print(error_msg, file=sys.stderr, flush=True)
                else:
                    tb = traceback.format_exc()
                    error_msg = f"{prefix} The following model failed in CLASS: \n"
                    error_msg += f"params: {params}\n"
                    error_msg += f"error: {repr(e)}\n"
                    print(error_msg, file=sys.stdout, flush=True)
                    error_msg += f"traceback: {tb}\n"
                    print(error_msg, file=sys.stderr, flush=True)
            except MemoryError:
                # Might happen in Python space, but if in C it might OOM kill the process
                success_cosmo = False
                tb = traceback.format_exc()
                error_msg = f"{prefix} The following model failed in CLASS: \n"
                error_msg += f"params: {params}\n"
                error_msg += f"error: MemoryError in child process\n"
                print(error_msg, file=sys.stdout, flush=True)
                error_msg += f"traceback: {tb}\n"
                print(error_msg, file=sys.stderr, flush=True)
            except Exception as e:
                success_cosmo = False
                tb = traceback.format_exc()
                error_msg = f"{prefix} The following model failed in CLASS: \n"
                error_msg += f"params: {params}\n"
                success_cosmo = False
                error_msg += f"error: {repr(e)}\n"
                print(error_msg, file=sys.stdout, flush=True)
                error_msg += f"traceback: {tb}\n"
                print(error_msg, file=sys.stderr, flush=True)

            finished_cosmo = True  # Doesn't mean it was successful, just that the child hasn't been killed
            result["finished_cosmo"] = finished_cosmo

            if not param.use_likelihood_filter:
                signal.alarm(0)
                if success_cosmo:
                    result["success"] = True

        print(f"[DEBUG] {prefix}: Finished cosmo loop", file=sys.stderr, flush=True)
        print(
            f"[DEBUG] {prefix}: sending results to parent.", file=sys.stderr, flush=True
        )

        # Send the results back to the parent through MPI
        try:
            pickled_result = pickle.dumps(result)
            child_parent.send(("MODEL_DONE", pickled_result), dest=0, tag=0)
        except MPI.Exception as e:
            print(
                f"{prefix} ERROR: Failed to send message to parent: {repr(e)}",
                file=sys.stderr,
                flush=True,
            )
            print(
                f"This happened after the model computation for model: {model}",
                file=sys.stderr,
                flush=True,
            )
            print(f"The child process will now exit.", file=sys.stderr, flush=True)
            print(
                f"{prefix} traceback: {traceback.format_exc()}",
                file=sys.stderr,
                flush=True,
            )
            sys.exit(0)

        # ------------------------ COMPUTE LIKELIHOOD ------------------------

        if (
            param.use_likelihood_filter
            and param.sampling == "iterative"
            and success_cosmo
        ):

            # Receive command from parent to compute likelihood

            try:
                sleep_initial = 0.0001  # Start with a reasonable value
                sleep_max = 0.1  # Avoid wasting CPU if the child takes a long time
                sleep_current = sleep_initial  # Start at 0.01s

                while not child_parent.Iprobe(source=0, tag=0):
                    time.sleep(sleep_current)  # Wait before checking again
                    sleep_current = min(
                        sleep_current * 2, sleep_max
                    )  # Increase wait time
                command, command_data = child_parent.recv(
                    source=0, tag=0
                )  # Receive from parent
            except MPI.Exception as e:
                print(
                    f"{prefix} ERROR: Failed to receive message from parent: {repr(e)}",
                    file=sys.stderr,
                    flush=True,
                )
                print(
                    f"This happened waiting for command to compute likelihood.",
                    file=sys.stderr,
                    flush=True,
                )
                print(f"The child process will now exit.", file=sys.stderr, flush=True)
                print(
                    f"{prefix} traceback: {traceback.format_exc()}",
                    file=sys.stderr,
                    flush=True,
                )

                sys.exit(0)

            if command and command_data is None:
                # parent closed pipe or something unexpected
                print(
                    f"{prefix} Parent closed pipe or something unexpected",
                    file=sys.stderr,
                    flush=True,
                )
                print(
                    f"{prefix} Received None from parent", file=sys.stderr, flush=True
                )
                sys.exit(0)

            soft_limit_bytes = int(
                command_data.get("soft_limit_bytes", hard_limit_bytes)
            )
            with memory_lock:
                MEMORY_LIMIT_BYTES = int(soft_limit_bytes * 0.95)
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
                    error_msg = f"{prefix} The following model failed likelihood computation: \n"
                    error_msg += f"params: {params}\n"
                    error_msg += f"error: {repr(e)}\n"
                    print(error_msg, file=sys.stdout, flush=True)
                    error_msg += f"traceback: {tb}\n"
                    print(error_msg, file=sys.stderr, flush=True)
                except classy.CosmoSevereError as e:
                    success_lkl = False
                    tb = traceback.format_exc()
                    error_msg = f"{prefix} The following model failed likelihood computation: \n"
                    error_msg += f"params: {params}\n"
                    error_msg += f"error: {repr(e)}\n"
                    print(error_msg, file=sys.stdout, flush=True)
                    error_msg += f"traceback: {tb}\n"
                    print(error_msg, file=sys.stderr, flush=True)
                except RuntimeError as e:
                    success_lkl = False
                    # check if it's the "timeout"
                    if str(e) == "timeout":
                        error_msg = f"{prefix} The following model took too long to complete: \n"
                        error_msg += f"params: {params}\n"
                        print(error_msg, file=sys.stdout, flush=True)
                        print(error_msg, file=sys.stderr, flush=True)
                    else:
                        tb = traceback.format_exc()
                        error_msg = f"{prefix} The following model failed likelihood computation: \n"
                        error_msg += f"params: {params}\n"
                        error_msg += f"error: {repr(e)}\n"
                        print(error_msg, file=sys.stdout, flush=True)
                        error_msg += f"traceback: {tb}\n"
                        print(error_msg, file=sys.stderr, flush=True)
                except MemoryError:
                    # Might happen in Python space, but if in C it might OOM kill the process
                    success_lkl = False
                    tb = traceback.format_exc()
                    error_msg = f"{prefix} The following model failed likelihood computation: \n"
                    error_msg += f"params: {params}\n"
                    error_msg += f"error: MemoryError in child process\n"
                    error_msg += f"The child were only allocated {soft_limit_bytes / (1000*1024**2):.2f} GB of memory.\n"
                    print(error_msg, file=sys.stdout, flush=True)
                    error_msg += f"traceback: {tb}\n"
                    print(error_msg, file=sys.stderr, flush=True)
                except Exception as e:
                    success_lkl = False
                    tb = traceback.format_exc()
                    error_msg = f"{prefix} The following model failed likelihood computation: \n"
                    error_msg += f"params: {params}\n"
                    error_msg += f"error: {repr(e)}\n"
                    print(error_msg, file=sys.stdout, flush=True)
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
                    print(error_msg, file=sys.stdout, flush=True)
                    print(error_msg, file=sys.stderr, flush=True)

                if loglkl_true is None:
                    success_lkl = False
                    error_msg += (
                        f"The following model failed likelihood computation: \n"
                    )
                    error_msg += f"params: {params}\n"
                    error_msg += f"error: Likelihood computation returned None.\n"
                    print(error_msg, file=sys.stdout, flush=True)
                    print(error_msg, file=sys.stderr, flush=True)

            signal.alarm(0)

            if success_cosmo and success_lkl:
                result["success"] = True
            else:
                result["success"] = False

            # Send the results back to the parent through MPI
            try:
                pickled_result = pickle.dumps(result)
                child_parent.send(("LIKELIHOOD_DONE", pickled_result), dest=0, tag=0)
            except MPI.Exception as e:
                print(
                    f"{prefix} ERROR: Failed to send message to parent: {repr(e)}",
                    file=sys.stderr,
                    flush=True,
                )
                print(
                    f"This happened after the likelihood computation for model: {model}",
                    file=sys.stderr,
                    flush=True,
                )
                print(f"The child process will now exit.", file=sys.stderr, flush=True)
                print(
                    f"{prefix} traceback: {traceback.format_exc()}",
                    file=sys.stderr,
                    flush=True,
                )
                sys.exit(0)

        cosmo.struct_cleanup()

        # Stop monitoring memory
    stop_memory_monitor.set()
    # Close monitor thread
    monitor_thread.join()

    # End of main child loop
    # If we get here, presumably we got "DONE" or the parent pipe closed
    print(
        f"{prefix} [DEBUG] Exiting main_child loop. Killing child process safely.",
        flush=True,
        file=sys.stderr,
    )
    sys.exit(0)


if __name__ == "__main__":
    # We could parse sys.argv. If we see "--child-mode", run main_child.
    # E.g.:
    if "--child-mode" in sys.argv:
        print(f"[DEBUG] {prefix}: Running main_child", file=sys.stderr, flush=True)
        main_child()
    else:
        print("This script is meant to run in --child-mode.")
        sys.exit(1)
