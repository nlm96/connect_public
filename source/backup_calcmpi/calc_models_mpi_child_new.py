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
child_id = int(sys.argv[5])


prefix = f"[CHILD][Parent Rank {parent_rank}] |"  # Print prefix from all messages from this child


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
    from source.lkl_filter_module.likelihood_calc_montepython import (
        MontePythonLikelihoodCalculator as likelihood_calculator,
    )
# elif param.mcmc_sampler == 'cobaya' and param.use_likelihood_filter and param.sampling == 'iterative':
# from source.lkl_filter_module.likelihood_calc_cobaya import CobayaLikelihoodCalculator as likelihood_calculator
#    pass

hard_limit_bytes = (
    int(os.environ.get("SLURM_MEM_PER_CPU", "7000")) * 1024**2 * 0.9
)  # 90% of the memory limit
hard_limit_bytes = int(hard_limit_bytes)


import signal

cosmo = None


def handle_signal(signum, frame):
    """Log unexpected termination signals and exit safely."""
    error_messages = {
        signal.SIGSEGV: "SIGSEGV (Segmentation Fault - Invalid Memory Access)",
        signal.SIGTERM: "SIGTERM (Job Cancelled or Terminated)",
        signal.SIGUSR1: "SIGUSR1 (Memory Limit Reached - SLURM OOM?)",
        signal.SIGABRT: "SIGABRT (Abort - Possibly Out of Memory?)",
        signal.SIGBUS: "SIGBUS (Bus Error - Bad Memory Access?)",
        signal.SIGILL: "SIGILL (Illegal Instruction - Corrupt Binary?)",
        signal.SIGPIPE: "SIGPIPE (Broken Pipe - Parent Process Closed?)",
    }
    error_msg = error_messages.get(signum, f"Signal {signum}")

    print(
        f"{prefix} CRITICAL ERROR: Process killed by {error_msg}",
        file=sys.stderr,
        flush=True,
    )

    sys.exit(0)  # Exit safely with code 0 to prevent MPI abort


# Register signals that can be handled
signal.signal(signal.SIGSEGV, handle_signal)  # Segmentation fault
signal.signal(signal.SIGTERM, handle_signal)  # Job termination (scancel, kill)
signal.signal(signal.SIGUSR1, handle_signal)  # SLURM OOM warnings
signal.signal(signal.SIGABRT, handle_signal)  # Abort signal
signal.signal(signal.SIGBUS, handle_signal)  # Bus error (bad memory access)
signal.signal(signal.SIGILL, handle_signal)  # Illegal CPU instruction
signal.signal(signal.SIGPIPE, handle_signal)  # Broken pipe (parent process closed)


def timeout_handler(sig, frame):
    raise RuntimeError("timeout")


signal.signal(signal.SIGALRM, timeout_handler)

DISCONNECTED = False


def main_child():
    """
    Main loop for the child process.
    """

    # Create a stop event for the memory monitor thread
    stop_memory_monitor = threading.Event()
    # Initialize the memory limit as a global variable
    global MEMORY_LIMIT_BYTES  # Memory limit for the child process
    MEMORY_LIMIT_BYTES = hard_limit_bytes  # Initial memory limit
    memory_lock = threading.Lock()  # Lock for thread-safe updates

    def get_total_memory_usage():
        """Get total memory usage of the child process and its subprocesses."""
        process = psutil.Process()
        mem_usage = process.memory_info().rss  # Memory usage of the current process
        for child in process.children(recursive=True):
            try:
                mem_usage += child.memory_info().rss  # Memory usage of all subprocesses
            except psutil.NoSuchProcess:
                pass  # Ignore terminated processes
        return mem_usage

    peak_memory_list = [0]  # Mutable object to store peak memory usage
    peak_memory_list[0] = 0  # Initialize peak memory usage

    def track_peak_memory(stop_event, peak_memory_list, memory_lock):
        """
        # Monitors memory usage and records the peak memory while stop_event is not set.
        # Stores the result in peak_memory_list[0] (a mutable list).
        """
        peak_memory = 0
        while not stop_event.is_set():  # Keep tracking while stop_event is not set
            current_memory = get_total_memory_usage()

            with memory_lock:  # Ensure thread safety when modifying shared variable
                peak_memory = max(peak_memory, current_memory)
                peak_memory_list[0] = peak_memory  # Update the shared list safely

            time.sleep(
                0.001
            )  # Adjust sampling rate (smaller value = more accurate tracking)

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

                    req = mpi_isend(
                        comm=child_parent,
                        data=("MEMORY_EXCEEDED", {"placeholder": "None"}),
                        dest=0,
                        tag=0,
                        sleep=0.1,
                    )

                except MPI.Exception:
                    pass  # Parent might have already shut down

                # Exit safely to avoid MPI abort
                sys.exit(0)

            time.sleep(0.1)  # Check memory every 0.1s

    # Start memory monitor in a separate thread
    # monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
    # monitor_thread.start()

    try:
        if param.use_likelihood_filter and param.sampling == "iterative":
            likelihood_calc = likelihood_calculator(
                param, f"{parent_rank}_child_{child_id}"
            )
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
        sys.exit(0)

    # We'll define a loop that repeatedly reads a "command" from stdin.
    sample_number = 0
    time_computation = False  # Whether to measure time of computation
    time_cosmo = None
    time_lkl = None
    if time_computation:
        time_file_path = f"{CONNECT_PATH}/data/{param.jobname}/time_computation.txt"
        if parent_rank == 1:
            with open(time_file_path, "a") as time_file:
                time_file.write("# Cosmo time (s)\tLkl time (s)\n")

    while True:
        sample_number += 1

        signal.alarm(0)

        signal.alarm(100)

        # Check if the parent is still alive and connected
        try:

            command = mpi_irecv(
                comm=child_parent, source=0, tag=99, sleep=0.1, timeout=95
            )

            if command != "PING":
                raise RuntimeError(f"Unexpected command from parent: {command}")

            # 2) Respond with "PONG"
            mpi_isend(comm=child_parent, data="PONG", dest=0, tag=99, sleep=0.01)

        except RuntimeError as e:
            # **Explicitly check if the error is a timeout**
            if str(e) == "timeout":
                print(
                    f"{prefix} ERROR: Timeout while waiting for parent PING. Exiting safely.",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(
                    f"{prefix} ERROR: Unexpected response from parent: {repr(e)}. Exiting safely.",
                    file=sys.stderr,
                    flush=True,
                )

            # Ensure cleanup doesn't get stuck
            signal.alarm(10)  # Set a timeout of 10 seconds for cleanup
            try:
                if cosmo is not None:
                    cosmo.struct_cleanup()
            except Exception as e:
                print(
                    f"{prefix} WARNING: Failed to clean up CLASS before exit. Error: {repr(e)}",
                    file=sys.stderr,
                    flush=True,
                )
            finally:
                signal.alarm(0)  # Disable cleanup timeout to avoid unintended triggers

            sys.exit(0)

        except MPI.Exception as e:
            print(
                f"{prefix} ERROR: MPI communication failed while checking parent connection: {repr(e)}. Exiting safely.",
                file=sys.stderr,
                flush=True,
            )

            # Ensure cleanup doesn't get stuck
            signal.alarm(15)
            try:
                if cosmo is not None:
                    cosmo.struct_cleanup()
            except Exception as e:
                print(
                    f"{prefix} WARNING: Failed to clean up CLASS before exit. Error: {repr(e)}",
                    file=sys.stderr,
                    flush=True,
                )
            finally:
                signal.alarm(0)

            sys.exit(0)

        except Exception as e:
            print(
                f"{prefix} ERROR: Unexpected failure while checking parent: {repr(e)}. Exiting safely.",
                file=sys.stderr,
                flush=True,
            )

            # Ensure cleanup doesn't get stuck
            signal.alarm(20)
            try:
                if cosmo is not None:
                    cosmo.struct_cleanup()
            except Exception as e:
                print(
                    f"{prefix} WARNING: Failed to clean up CLASS before exit. Error: {repr(e)}",
                    file=sys.stderr,
                    flush=True,
                )
            finally:
                signal.alarm(0)

            sys.exit(0)

        finally:
            signal.alarm(0)  # Disable timeout alarm

        while True:

            try:
                signal.alarm(40)
                # 3) Wait for one pickled command from the parent
                try:
                    command, command_data = mpi_irecv(
                        comm=child_parent, source=0, tag=0, sleep=0.01
                    )
                except MPI.Exception as e:
                    print(
                        f"{prefix} ERROR: Failed to receive message from parent: {repr(e)}",
                        file=sys.stderr,
                        flush=True,
                    )
                    print(
                        f"The child process will now exit.", file=sys.stderr, flush=True
                    )
                    print(
                        f"{prefix} traceback: {traceback.format_exc()}",
                        file=sys.stderr,
                        flush=True,
                    )

                    sys.exit(0)  # Terminate child safely
                except RuntimeError as e:
                    if str(e) == "timeout":
                        print(
                            f"{prefix} ERROR: Timeout while waiting for parent command. Exiting safely.",
                            file=sys.stderr,
                            flush=True,
                        )
                    sys.exit(0)
                except Exception as e:
                    print(
                        f"{prefix} ERROR: Unexpected failure while receiving command: {repr(e)}. Exiting safely.",
                        file=sys.stderr,
                        flush=True,
                    )
                    print(
                        f"{prefix} traceback: {traceback.format_exc()}",
                        file=sys.stderr,
                        flush=True,
                    )

                    sys.exit(0)

                finally:
                    signal.alarm(0)

                if command and command_data is None:
                    # parent closed pipe or something unexpected
                    print(
                        f"{prefix} Parent closed pipe or something unexpected",
                        file=sys.stderr,
                        flush=True,
                    )
                    print(
                        f"{prefix} Received None from parent",
                        file=sys.stderr,
                        flush=True,
                    )
                    sys.exit(0)

                # The command could be e.g. "DONE" or "RUN_MODEL" or "RUN_LIKELIHOOD"
                if command == "DONE":
                    # 1. Acknowledge shutdown to the parent
                    signal.alarm(100)  # Set a timeout of 15 seconds for shutdown

                    try:
                        try:
                            req = mpi_isend(
                                comm=child_parent,
                                data=("DONE_ACK", None),
                                dest=0,
                                tag=0,
                                sleep=0.01,
                            )
                        except MPI.Exception as e:
                            print(
                                f"{prefix} WARNING: Failed to send 'DONE_ACK'. Parent may have already exited. Error: {repr(e)}",
                                file=sys.stderr,
                                flush=True,
                            )

                        signal.alarm(100)

                        # 2. Clean up resources
                        try:
                            if cosmo is not None:
                                cosmo.struct_cleanup()
                        except RuntimeError as e:
                            if str(e) == "timeout":
                                print(
                                    f"{prefix} WARNING: Timeout while waiting for CLASS cleanup. Exiting safely.",
                                    file=sys.stderr,
                                    flush=True,
                                )
                            else:
                                print(
                                    f"{prefix} WARNING: Failed to clean up CLASS. Error: {repr(e)}",
                                    file=sys.stderr,
                                    flush=True,
                                )

                        # try:
                        #     # Stop monitoring memory
                        #     stop_memory_monitor.set()
                        #     # Close monitor thread

                        #     monitor_thread.join()

                        # except RuntimeError as e:
                        #     if str(e) == "timeout":
                        #         print(
                        #             f"{prefix} WARNING: Timeout while waiting for memory monitor to stop. Exiting safely.",
                        #             file=sys.stderr,
                        #             flush=True,
                        #         )
                        #     else:
                        #         print(
                        #             f"{prefix} WARNING: Failed to stop memory monitor. Error: {repr(e)}",
                        #             file=sys.stderr,
                        #             flush=True,
                        #         )

                        signal.alarm(100)  # Disable shutdown timeout
                        # 3. Disconnect from the parent MPI communicator
                        try:
                            print(
                                f"{prefix} [INFO] Disconnecting from parent... this is PID: {os.getpid()}",
                                flush=True,
                                file=sys.stderr,
                            )
                            child_parent.Disconnect()
                            DISCONNECTED = True
                        except MPI.Exception as e:
                            print(
                                f"{prefix} WARNING: MPI communicator already cleaned up. Error: {repr(e)}",
                                file=sys.stderr,
                                flush=True,
                            )

                        print(
                            f"{prefix} [INFO] Exiting child process normally.",
                            flush=True,
                            file=sys.stderr,
                        )

                        # 4. Exit cleanly
                        sys.exit(0)
                    except RuntimeError as e:
                        if str(e) == "timeout":
                            print(
                                f"{prefix} ERROR: Timeout while waiting for parent shutting down. Exiting safely.",
                                file=sys.stderr,
                                flush=True,
                            )
                        else:
                            print(
                                f"{prefix} ERROR: Unexpected failure during shutdown: {repr(e)}. Exiting safely.",
                                file=sys.stderr,
                                flush=True,
                            )
                        sys.exit(0)
                    except Exception as e:
                        print(
                            f"{prefix} ERROR: Unexpected failure during shutdown: {repr(e)}. Exiting safely.",
                            file=sys.stderr,
                            flush=True,
                        )
                        sys.exit(0)

                soft_limit_bytes = int(
                    command_data.get("soft_limit_bytes", hard_limit_bytes)
                )
                with memory_lock:
                    MEMORY_LIMIT_BYTES = int(
                        soft_limit_bytes * 0.95
                    )  # Update memory limit
                resource.setrlimit(
                    resource.RLIMIT_AS, (soft_limit_bytes, hard_limit_bytes)
                )

                # We got a model to compute
                error_msg = None
                result = {
                    "success": False,
                    "success_cosmo": False,
                    "success_lkl": False,
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
                }

                # 4) Extract the relevant fields
                #    e.g. param, model, param_names, do_lkl, etc.
                params = command_data["params"]  # The dictionary to pass to CLASS
                model = command_data["model"]  # The param values
                global_ell = command_data["global_ell"]

                result["model"] = model

                # 5) We'll enforce a 200-second time limit for each model

                signal.alarm(300)

                success = False
                success_cosmo = False
                success_lkl = False

                track_memory = False
                measure_cpu = False

                # ------------------------ COMPUTE COSMO ------------------------

                if command == "RUN_MODEL":
                    # Compute cosmo and collect results
                    try:

                        if time_computation:
                            start_time = time.time()

                        if track_memory:
                            stop_event = threading.Event()
                            memory_thread = threading.Thread(
                                target=track_peak_memory,
                                args=(stop_event, peak_memory_list, memory_lock),
                            )
                            memory_thread.start()

                        if measure_cpu:
                            monitor_duration = 10.0
                            child_proc = psutil.Process()

                            # Prime CPU measurement for the child process
                            child_proc.cpu_percent(interval=None)

                            # Prime CPU measurement for all subprocesses
                            subproc_list = child_proc.children(recursive=True)
                            for subproc in subproc_list:
                                subproc.cpu_percent(interval=None)

                            def get_total_thread_cpu_time(process):
                                """Measure total CPU time used by all threads inside a process."""
                                total_time = sum(
                                    thread.system_time + thread.user_time
                                    for thread in process.threads()
                                )
                                return total_time

                            def get_total_cpu_usage():
                                """Measure total CPU usage of child process + all its threads and subprocesses."""
                                total_cpu = child_proc.cpu_percent(
                                    interval=None
                                )  # Main process
                                for subproc in child_proc.children(
                                    recursive=True
                                ):  # Include all subprocesses
                                    total_cpu += subproc.cpu_percent(interval=None)
                                return total_cpu

                            cpu_time_threads_before = get_total_thread_cpu_time(
                                child_proc
                            )

                            # Also track system-wide CPU usage before CLASS starts
                            system_cpu_before = psutil.cpu_percent(interval=None)

                            # Measure CPU time before CLASS computation
                            cpu_time_before = child_proc.cpu_times()

                        cosmo = classy.Class()
                        cosmo.set(params)

                        start_time2 = time.time()
                        cosmo.compute()
                        end_time2 = time.time()

                        # Simulate timeout
                        # if parent_rank == 5:
                        #     time.sleep(400)

                        # Simulate memory error
                        # if parent_rank == 8:
                        #     # Do a sigkill
                        #     os.kill(os.getpid(), signal.SIGKILL)

                        #     # Allocate a large amount of memory
                        #     memory_hog = np.ones((1000000, 1000000), dtype=np.float64)

                        if measure_cpu:
                            # Measure CPU time after CLASS computation
                            cpu_time_after = child_proc.cpu_times()

                            # Get CPU usage **immediately after** CLASS computation (reflects what happened during)
                            child_cpu_usage = child_proc.cpu_percent(interval=None)

                            # Capture total CPU usage including all subprocesses
                            total_child_cpu_usage = get_total_cpu_usage()

                            # Compute system-wide CPU usage difference
                            system_cpu_after = psutil.cpu_percent(interval=None)
                            system_cpu_usage = system_cpu_after - system_cpu_before

                            # Compute CPU time breakdown
                            cpu_time_user = cpu_time_after.user - cpu_time_before.user
                            cpu_time_system = (
                                cpu_time_after.system - cpu_time_before.system
                            )
                            cpu_time = cpu_time_user + cpu_time_system
                            elapsed_time = end_time2 - start_time2

                            cpu_time_threads_after = get_total_thread_cpu_time(
                                child_proc
                            )
                            thread_cpu_time_used = (
                                cpu_time_threads_after - cpu_time_threads_before
                            )

                            # Print results
                            print(
                                f"{prefix} CPU time used during CLASS computation: {cpu_time:.2f} s for sample {sample_number}",
                                file=sys.stdout,
                                flush=True,
                            )
                            print(
                                f"{prefix} CPU user time used during CLASS computation: {cpu_time_user:.2f} s for sample {sample_number}",
                                file=sys.stdout,
                                flush=True,
                            )
                            print(
                                f"{prefix} CPU system time used during CLASS computation: {cpu_time_system:.2f} s for sample {sample_number}",
                                file=sys.stdout,
                                flush=True,
                            )
                            print(
                                f"{prefix} CPU percent used by child process: {child_cpu_usage:.2f} % for sample {sample_number}",
                                file=sys.stdout,
                                flush=True,
                            )
                            print(
                                f"{prefix} Total CPU percent used by child process + subprocesses: {total_child_cpu_usage:.2f} % for sample {sample_number}",
                                file=sys.stdout,
                                flush=True,
                            )
                            print(
                                f"{prefix} System-wide CPU percent difference during CLASS computation: {system_cpu_usage:.2f} % for sample {sample_number}",
                                file=sys.stdout,
                                flush=True,
                            )
                            print(
                                f"{prefix} Total elapsed time for CLASS computation: {elapsed_time:.2f} s for sample {sample_number}",
                                file=sys.stdout,
                                flush=True,
                            )

                            print(
                                f"{prefix} Total CPU time used by all threads: {thread_cpu_time_used:.2f} s for sample {sample_number}",
                                file=sys.stdout,
                                flush=True,
                            )

                            print(
                                f"{prefix} CPU affinity: {child_proc.cpu_affinity()} for sample {sample_number}",
                                file=sys.stdout,
                                flush=True,
                            )

                            print(
                                f"{prefix} Active threads count: {len(child_proc.threads())} for sample {sample_number}",
                                file=sys.stdout,
                                flush=True,
                            )

                        if track_memory:
                            stop_event.set()  # Signal thread to stop
                            memory_thread.join()  # Wait for the thread to finish

                            print(
                                f"{prefix} Peak memory usage during CLASS computation for sample {sample_number}:\n"
                            )
                            print(
                                f"{prefix} {peak_memory_list[0] / (1024**3):.2f} GB",
                                file=sys.stdout,
                                flush=True,
                            )

                        if time_computation:
                            end_time = time.time()
                            time_cosmo = end_time - start_time

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
                            der = cosmo.get_current_derived_parameters(
                                param.output_derived
                            )
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

                        success_cosmo = True
                        result["success_cosmo"] = success_cosmo
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
                            error_msg = f"{prefix} The following model took too long to complete: \n"
                            error_msg += f"params: {params}\n"
                            print(error_msg, file=sys.stdout, flush=True)
                            print(error_msg, file=sys.stderr, flush=True)
                        else:
                            tb = traceback.format_exc()
                            error_msg = (
                                f"{prefix} The following model failed in CLASS: \n"
                            )
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

                        # Simulate a OOM kill, use a oom kill, do a sigkill
                        # os.kill(os.getpid(), signal.SIGKILL)
                        # sys.exit(1)
                        # sys.exit(0)
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

                    if not param.use_likelihood_filter:
                        signal.alarm(0)
                        if success_cosmo:
                            result["success"] = True
                            result["success_cosmo"] = True

                # Send the results back to the parent through MPI
                try:

                    if success_cosmo:

                        req = mpi_isend(
                            comm=child_parent,
                            data=("MODEL_DONE", result),
                            dest=0,
                            tag=0,
                            sleep=0.01,
                        )
                    else:
                        req = mpi_isend(
                            comm=child_parent,
                            data=("MODEL_FAILED", result),
                            dest=0,
                            tag=0,
                            sleep=0.01,
                        )
                        break
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
                    print(
                        f"The child process will now exit.", file=sys.stderr, flush=True
                    )
                    print(
                        f"{prefix} traceback: {traceback.format_exc()}",
                        file=sys.stderr,
                        flush=True,
                    )
                    sys.exit(0)
                except RuntimeError as e:
                    if str(e) == "timeout":
                        print(
                            f"{prefix} ERROR: Timeout while sending message to parent. Returning to PING PONG loop.",
                            file=sys.stderr,
                            flush=True,
                        )
                        break
                    else:
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
                        print(
                            f"The child process will now exit.",
                            file=sys.stderr,
                            flush=True,
                        )
                        print(
                            f"{prefix} traceback: {traceback.format_exc()}",
                            file=sys.stderr,
                            flush=True,
                        )
                        sys.exit(0)
                except Exception as e:
                    print(
                        f"{prefix} ERROR: Unexpected failure while sending message to parent: {repr(e)}",
                        file=sys.stderr,
                        flush=True,
                    )
                    print(
                        f"This happened after the model computation for model: {model}",
                        file=sys.stderr,
                        flush=True,
                    )
                    print(
                        f"The child process will now exit.", file=sys.stderr, flush=True
                    )
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
                        command, command_data = mpi_irecv(
                            comm=child_parent, source=0, tag=0, sleep=0.001
                        )
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
                        print(
                            f"The child process will now exit.",
                            file=sys.stderr,
                            flush=True,
                        )
                        print(
                            f"{prefix} traceback: {traceback.format_exc()}",
                            file=sys.stderr,
                            flush=True,
                        )

                        sys.exit(0)
                    except RuntimeError as e:
                        if str(e) == "timeout":
                            print(
                                f"{prefix} ERROR: Timeout while waiting for parent command. returning to PING PONG loop.",
                                file=sys.stderr,
                                flush=True,
                            )
                            break
                        else:
                            print(
                                f"{prefix} ERROR: Unexpected failure while receiving command: {repr(e)}. Exiting safely.",
                                file=sys.stderr,
                                flush=True,
                            )
                            print(
                                f"{prefix} traceback: {traceback.format_exc()}",
                                file=sys.stderr,
                                flush=True,
                            )
                            sys.exit(0)

                    except Exception as e:
                        print(
                            f"{prefix} ERROR: Unexpected failure while receiving command: {repr(e)}. Exiting safely.",
                            file=sys.stderr,
                            flush=True,
                        )
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
                            f"{prefix} Received None from parent",
                            file=sys.stderr,
                            flush=True,
                        )
                        sys.exit(0)

                    soft_limit_bytes = int(
                        command_data.get("soft_limit_bytes", hard_limit_bytes)
                    )
                    with memory_lock:
                        MEMORY_LIMIT_BYTES = int(soft_limit_bytes * 0.95)
                    resource.setrlimit(
                        resource.RLIMIT_AS, (soft_limit_bytes, hard_limit_bytes)
                    )

                    if command == "RUN_LIKELIHOOD":
                        # Compute likelihood

                        try:

                            if time_computation:
                                start_time = time.time()

                            # track memory usage:
                            if track_memory:
                                stop_event = threading.Event()
                                memory_thread = threading.Thread(
                                    target=track_peak_memory,
                                    args=(stop_event, peak_memory_list, memory_lock),
                                )
                                memory_thread.start()

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
                            result["success_lkl"] = success_lkl

                            if track_memory:
                                stop_event.set()
                                memory_thread.join()
                                print(
                                    f"{prefix} Peak memory usage during likelihood computation for sample {sample_number}:\n"
                                )
                                print(
                                    f"{prefix} {peak_memory_list[0] / (1024**3):.2f} GB",
                                    file=sys.stdout,
                                    flush=True,
                                )

                            if time_computation:
                                end_time = time.time()
                                time_lkl = end_time - start_time
                                if (
                                    time_computation
                                    and time_cosmo is not None
                                    and time_lkl is not None
                                    and success_cosmo
                                    and success_lkl
                                ):
                                    with open(time_file_path, "a") as time_file:
                                        time_file.write(
                                            f"{time_cosmo:.6f}\t{time_lkl:.6f}\n"
                                        )
                                        time_file.flush()  # Ensure the data is written to disk

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
                            error_msg += (
                                f"error: Likelihood computation returned None.\n"
                            )
                            print(error_msg, file=sys.stdout, flush=True)
                            print(error_msg, file=sys.stderr, flush=True)

                    signal.alarm(0)

                    if success_cosmo and success_lkl:
                        result["success"] = True
                        result["success_lkl"] = True
                    else:
                        result["success"] = False
                        result["success_lkl"] = False

                    # Send the results back to the parent through MPI
                    try:

                        if success_lkl:
                            req = mpi_isend(
                                comm=child_parent,
                                data=("LIKELIHOOD_DONE", result),
                                dest=0,
                                tag=0,
                                sleep=0.01,
                            )
                            break
                        else:
                            req = mpi_isend(
                                comm=child_parent,
                                data=("LIKELIHOOD_FAILED", result),
                                dest=0,
                                tag=0,
                                sleep=0.01,
                            )
                            break
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
                        print(
                            f"The child process will now exit.",
                            file=sys.stderr,
                            flush=True,
                        )
                        print(
                            f"{prefix} traceback: {traceback.format_exc()}",
                            file=sys.stderr,
                            flush=True,
                        )
                        sys.exit(0)
                else:
                    break

            except RuntimeError as e:
                if str(e) == "timeout":
                    print(
                        f"{prefix} ERROR: Timeout during computations. Returning to PING PONG loop.",
                        file=sys.stderr,
                        flush=True,
                    )
                    break
            except Exception as e:
                print(
                    f"{prefix} ERROR: Unexpected failure during computations: {repr(e)}. Returning to PING PONG loop.",
                    file=sys.stderr,
                    flush=True,
                )
                print(
                    f"{prefix} traceback: {traceback.format_exc()}",
                    file=sys.stderr,
                    flush=True,
                )
                break
            finally:
                signal.alarm(0)
                break
        if cosmo is not None:
            cosmo.struct_cleanup()

        # Stop monitoring memory
    # stop_memory_monitor.set()
    # # Close monitor thread
    # monitor_thread.join()

    # End of main child loop
    # If we get here, presumably we got "DONE" or the parent pipe closed
    sys.exit(0)


def mpi_isend(comm, data, dest, tag, sleep=0.01, timeout=20):
    """
    Non-blocking MPI send with explicit pickling.
    """

    # Pickle the data before sending
    pickled_data = pickle.dumps(data)

    # Non-blocking send
    req = comm.Isend(pickled_data, dest=dest, tag=tag)

    start_time = time.time()
    # Ensure message is fully sent (non-blocking check)
    while not req.Test():
        time.sleep(sleep)  # Reduce CPU usage
        if time.time() - start_time > timeout:
            print(
                f"{prefix} ERROR: Timeout while sending message.",
                file=sys.stderr,
                flush=True,
            )
            print(f"Exiting safely.", file=sys.stderr, flush=True)
            sys.exit(0)

    return req  # Returning request in case the caller wants to track it


def mpi_irecv(comm, source, tag, sleep=0.01, timeout=20):
    """
    Non-blocking MPI receive with dynamic buffer allocation.
    """
    global prefix, DISCONNECTED
    start_time = time.time()
    status = MPI.Status()

    if DISCONNECTED:
        print(
            f"{prefix} [DEBUG] ERROR: Parent has disconnected, yet child is still trying to receive messages.\n"
            f"Entering while not comm.Iprobe loop.",
            file=sys.stderr,
            flush=True,
        )

    # Wait for a message (non-blocking)
    while not comm.Iprobe(source=source, tag=tag, status=status):

        time.sleep(sleep)  # Reduce CPU usage
        if DISCONNECTED:
            print(
                f"{prefix} [DEBUG] ERROR: Parent has disconnected, but child is still expecting messages inside Iprobe loop.\n",
                file=sys.stderr,
                flush=True,
            )

        if time.time() - start_time > timeout:
            print(
                f"{prefix} ERROR: Timeout while waiting for message.",
                file=sys.stderr,
                flush=True,
            )
            print(f"Exiting safely.", file=sys.stderr, flush=True)
            sys.exit(0)

    # Get the message size
    message_size = status.Get_count(MPI.BYTE)

    # Allocate a buffer of the correct size
    buffer = bytearray(message_size)

    # Start the non-blocking receive
    req = comm.Irecv(buffer, source=source, tag=tag)

    # Wait for full message completion (non-blocking)
    while not req.Test():
        time.sleep(sleep)  # Reduce CPU usage
        if time.time() - start_time > timeout:
            print(
                f"{prefix} ERROR: Timeout while waiting for message.",
                file=sys.stderr,
                flush=True,
            )
            print(f"Exiting safely.", file=sys.stderr, flush=True)
            sys.exit(0)

    # Unpickle the received data
    received_data = pickle.loads(buffer)

    return received_data  # Fully received, unpickled message


if __name__ == "__main__":
    # We could parse sys.argv. If we see "--child-mode", run main_child.
    # E.g.:
    if "--child-mode" in sys.argv:
        main_child()
    else:
        print("This script is meant to run in --child-mode.")
        sys.exit(1)
