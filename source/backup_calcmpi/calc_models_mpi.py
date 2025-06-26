import os
import sys
import time
import itertools
import signal
import traceback
import copy

import pickle
import psutil

os.environ["UCX_LOG_LEVEL"] = "error"
param_file = sys.argv[1]
CONNECT_PATH = sys.argv[2]
sampling = sys.argv[3]
unique_id = sys.argv[4]
sys.path.insert(0, CONNECT_PATH)

import numpy as np
import classy
from scipy.interpolate import CubicSpline
from mpi4py import MPI

from source.default_module import Parameters
from source.tools import get_computed_cls, get_z_idx, get_covmat

param_file = os.path.join(CONNECT_PATH, param_file)
param = Parameters(param_file)
param_names = list(param.parameters.keys())
mapped_to_names = [
    param.custom_parameters[custom_name]["maps_to"]
    for custom_name in param.custom_parameters
]
if mapped_to_names:
    param_names += list(mapped_to_names)


if (
    param.mcmc_sampler == "montepython"
    and param.use_likelihood_filter
    and param.sampling == "iterative"
):
    from source.lkl_filter_module.likelihood_calc_montepython import (
        MontePythonLikelihoodCalculator as likelihood_calculator,
    )
elif (
    param.mcmc_sampler == "cobaya"
    and param.use_likelihood_filter
    and param.sampling == "iterative"
):
    # from source.lkl_filter_module.likelihood_calc_cobaya import CobayaLikelihoodCalculator as likelihood_calculator
    pass

path = os.path.join(CONNECT_PATH, f"data/{param.jobname}")
if sampling == "iterative":
    try:
        iteration = max(
            [
                int(f.split("number_")[-1])
                for f in os.listdir(path)
                if f.startswith("number")
            ]
        )
        directory = os.path.join(path, f"number_{iteration}")
    except:
        directory = os.path.join(path, f"N-{param.N}")
elif sampling in ["lhc", "hypersphere", "pickle"]:
    directory = os.path.join(path, f"N-{param.N}")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
N_slaves = comm.Get_size() - 1

get_slave = itertools.cycle(range(1, N_slaves + 1))

prefix = f"[PARENT][Rank {rank}] |"


def excepthook(etype, value, tb):
    if tb is not None and value not in [None, ""]:
        print("Traceback (most recent call last):", file=sys.stderr)
        traceback.print_tb(tb, file=sys.stderr)
        print(f"{etype.__name__}: {value.args[0]}", file=sys.stderr)
    comm.Abort(1)


sys.excepthook = excepthook


if param.use_likelihood_filter and param.sampling == "iterative":
    likelihood_calc = likelihood_calculator(param, rank)
    if param.mcmc_sampler == "montepython":
        initial_cosmo_arguments = copy.deepcopy(
            likelihood_calc.mp["data"].cosmo_arguments
        )
        # print(f"cosmo_arguments: {initial_cosmo_arguments}", flush=True)


if len(param.output_Cl) > 0:
    cosmo = classy.Class()
    input_params = {"output": "tCl, lCl, pCl", "lensing": "yes"}
    if (
        param.use_likelihood_filter
        and param.mcmc_sampler == "montepython"
        and param.sampling == "iterative"
    ):
        input_params.update(
            {
                "l_max_scalars": likelihood_calc.mp["data"].cosmo_arguments[
                    "l_max_scalars"
                ]
            }
        )
    if "l_max_scalars" in param.extra_input:
        input_params.update({"l_max_scalars": param.extra_input["l_max_scalars"]})
        if (
            param.use_likelihood_filter
            and param.mcmc_sampler == "montepython"
            and param.sampling == "iterative"
        ):
            print(
                (
                    f"Your user-defined l_max_scalars={param.extra_input['l_max_scalars']} "
                    f"is being overwritten by the likelihood filter with the value l_max_scalars={likelihood_calc.mp['data'].cosmo_arguments['l_max_scalars']} "
                    "to accommodate the likelihood calculation used by the likelihood filter."
                ),
                flush=True,
            )

            input_params.update(
                {
                    "l_max_scalars": likelihood_calc.mp["data"].cosmo_arguments[
                        "l_max_scalars"
                    ]
                }
            )

    cosmo.set(input_params)
    cosmo.compute()
    cls = get_computed_cls(cosmo)
    global_ell = cls["ell"]


# --------------------------- Memory-safe implementation ---------------------------#
if param.memory_safe and rank > 0:

    max_respawn = 3  # Maximum number of retries to respawn a child process
    respawn_count = 0  # Number of retries so far
    import random

    # Random sleep to avoid all children starting at the same time. ucx does not like that.
    time.sleep(random.uniform(0, 180))

    child_id = random.randint(
        0, 1000
    )  # Random ID for the child process used to instantiate a unique likelihood calculator

    MPI.COMM_WORLD.Set_errhandler(MPI.ERRORS_RETURN)
    MPI.COMM_SELF.Set_errhandler(MPI.ERRORS_RETURN)
    os.environ["UCX_LOG_LEVEL"] = "error"
    os.environ["UCX_HANDLE_ERRORS"] = "no"
    os.environ["UCX_ERROR_SIGNALS"] = "no"
    os.environ["UCX_TLS"] = "tcp"

    info = MPI.Info.Create()

    import socket

    # Force the child to run on the same node (host) as the parent:
    short_host = socket.gethostname().split(".")[0]
    info.Set("host", short_host)

    child_comm = MPI.COMM_SELF.Spawn(
        sys.executable,
        args=[
            os.path.join(CONNECT_PATH, "source", "calc_models_mpi_child.py"),
            "--child-mode",
            param_file,
            CONNECT_PATH,
            str(rank),
            str(child_id),
            str(unique_id),
        ],
        maxprocs=1,  # Spawn only one child process
        info=info,
    )
    child_comm.Set_errhandler(
        MPI.ERRORS_RETURN
    )  # Avoid MPI abort on child process error

    def get_soft_limit():
        """Calculate the soft memory limit for the child process."""

        hard_limit_bytes = (
            int(os.environ.get("SLURM_MEM_PER_CPU", "7000")) * 1024**2
        )  # Read in bytes
        parent_usage_bytes = psutil.Process().memory_info().rss  # Already in bytes

        # Compute available memory
        available_bytes = max(0, hard_limit_bytes - parent_usage_bytes)

        # Set soft limit as 80% of available memory
        soft_limit_bytes = int(0.85 * available_bytes)

        return soft_limit_bytes

    def respawn_child():
        """Handles respawning a new child process safely."""

        global child_comm, respawn_count, max_respawn
        child_id = random.randint(0, 1000)

        if respawn_count >= max_respawn:
            print(
                f"{prefix} ERROR: Maximum respawn attempts reached. Marking rank as failed.",
                file=sys.stderr,
                flush=True,
            )
            return None

        respawn_count += 1

        def timeout_handler(signum, frame):
            """Handles timeout if the child process is unresponsive."""
            raise TimeoutError("Timeout while waiting for child to respond.")

        import signal

        signal.signal(signal.SIGALRM, timeout_handler)

        signal.alarm(20)  # Set a timeout for the child process to respond

        try:
            # Clean up the old child process
            if child_comm is not None:
                child_comm.Free()
                child_comm = None
        except TimeoutError as e:
            print(
                f"{prefix} ERROR: Timeout while waiting for old child to disconnect: {repr(e)}",
                file=sys.stderr,
                flush=True,
            )
        except MPI.Exception as e:
            print(
                f"{prefix} ERROR: Failed to disconnect from old child: {repr(e)}",
                file=sys.stderr,
                flush=True,
            )
        finally:
            signal.alarm(0)

        try:
            print(
                f"{prefix} WARNING: Child process died or is unresponsive. Respawning...",
                file=sys.stderr,
                flush=True,
            )

            info = MPI.Info.Create()
            import socket

            short_host = socket.gethostname().split(".")[0]
            info.Set("host", short_host)

            child_comm = MPI.COMM_SELF.Spawn(
                sys.executable,
                args=[
                    os.path.join(CONNECT_PATH, "source", "calc_models_mpi_child.py"),
                    "--child-mode",
                    param_file,
                    CONNECT_PATH,
                    str(rank),
                    str(child_id),
                    str(unique_id),
                ],
                maxprocs=1,
                info=info,
            )
            child_comm.Set_errhandler(MPI.ERRORS_RETURN)

            print(
                f"{prefix} [INFO] Successfully respawned child process.",
                file=sys.stderr,
                flush=True,
            )
            return child_comm

        except MPI.Exception as e:
            print(
                f"{prefix} ERROR: MPI failed to spawn a new child process: {repr(e)}",
                file=sys.stderr,
                flush=True,
            )
            return None  # Return None to indicate failure

        except Exception as e:
            print(
                f"{prefix} ERROR: Unexpected failure during child respawn: {repr(e)}",
                file=sys.stderr,
                flush=True,
            )
            return None  # Return None to indicate failure

    def mpi_isend(comm, data, dest, tag, sleep=0.01):
        """
        Non-blocking MPI send with explicit pickling.
        """

        # Pickle the data before sending
        pickled_data = pickle.dumps(data)

        # Non-blocking send
        req = comm.Isend(pickled_data, dest=dest, tag=tag)

        # Ensure message is fully sent (non-blocking check)
        while not req.Test():
            time.sleep(sleep)  # Reduce CPU usage

        return req  # Returning request in case the caller wants to track it

    def mpi_irecv(comm, source, tag, sleep=0.01):
        """
        Non-blocking MPI receive with dynamic buffer allocation.
        """

        status = MPI.Status()

        # Wait for a message (non-blocking)
        while not comm.Iprobe(source=source, tag=tag, status=status):
            time.sleep(sleep)  # Reduce CPU usage

        # Get the message size
        message_size = status.Get_count(MPI.BYTE)

        # Allocate a buffer of the correct size
        buffer = bytearray(message_size)

        # Start the non-blocking receive
        req = comm.Irecv(buffer, source=source, tag=tag)

        # Wait for full message completion (non-blocking)
        while not req.Test():
            time.sleep(sleep)  # Reduce CPU usage

        # Unpickle the received data
        received_data = pickle.loads(buffer)

        return received_data  # Fully received, unpickled message


# --------------------------------- End of memory-safe implementation ---------------------------------#


## rank == 0 (master)
if rank == 0:
    if sampling == "iterative":
        exec(
            f"from source.mcmc_samplers.{param.mcmc_sampler} import {param.mcmc_sampler}"
        )
        _locals = {}
        exec(f"mcmc = {param.mcmc_sampler}(param, CONNECT_PATH)", locals(), _locals)
        mcmc = _locals["mcmc"]

        data = mcmc.import_points_from_chains(iteration)
        if param.use_likelihood_filter and param.sampling == "iterative":
            loglkl_chains = mcmc.import_loglkl_from_chains(iteration)

    elif sampling == "lhc":
        from source.ini_samplers import LatinHypercubeSampler

        lhc = LatinHypercubeSampler(param)
        data = lhc.run()
        if param.use_likelihood_filter and param.sampling == "iterative":
            loglkl_chains = np.full(len(data), np.nan)

    elif sampling == "hypersphere":
        from source.ini_samplers import HypersphereSampler

        hs = HypersphereSampler(param)
        data = hs.run()
        if param.use_likelihood_filter and param.sampling == "iterative":
            loglkl_chains = np.full(len(data), np.nan)

    elif sampling == "pickle":
        from source.ini_samplers import PickleSample

        ps = PickleSampler(param)
        data = ps.run()
        if param.use_likelihood_filter and param.sampling == "iterative":
            loglkl_chains = np.full(len(data), np.nan)

    sleep_short = 0.0001
    sleep_long = 0.1
    sleep_dict = {}
    failed_ranks = set()  # Track failed ranks
    # ranks that has tried respawning the child too many times, is set to idle until the job is done.
    idle_ranks = set()
    active_ranks = set(range(1, N_slaves + 1))
    last_comm = (
        {}
    )  # Dictionary to record last successful communication time for each rank
    for r in range(1, N_slaves + 1):
        sleep_dict[r] = 1
        last_comm[r] = time.time()

    TIMEOUT_SECONDS = 800  # Timeout for rank response

    data_idx = 0
    while len(active_ranks) > 0 and len(data) > data_idx:
        r = next(get_slave)

        # Skip failed ranks
        if r in failed_ranks:
            continue

        try:

            # Check if worker sent a response
            if r not in idle_ranks and comm.Iprobe(r):
                last_comm[r] = time.time()
                useless_info = comm.recv(source=r)  # Receive the "I am done" message

                # Prepare data to send
                if param.use_likelihood_filter and param.sampling == "iterative":
                    DATA = {
                        "model": data[data_idx],
                        "loglkl_chains": loglkl_chains[data_idx],
                    }
                else:
                    DATA = data[data_idx]

                comm.send(DATA, dest=r)  # Send next data point
                data_idx += 1
                sleep_dict[r] = 1  # Mark worker as active

            elif r not in idle_ranks and comm.Iprobe(r, tag=500):
                idle_message = comm.recv(source=r, tag=500)
                if idle_message == "IDLE":
                    idle_ranks.add(r)
                    print(
                        f"[Master] Rank {r} is set to idle after too many child respawn attempts. Skipping to next rank.",
                        flush=True,
                    )
                    active_ranks.remove(r)
                else:
                    print(
                        f"[Master] Rank {r} sent an unexpected message: {idle_message}. Skipping to next rank.",
                        flush=True,
                        file=sys.stderr,
                    )
            else:
                if time.time() - last_comm[r] > TIMEOUT_SECONDS:
                    print(
                        f"[Master] WARNING: Rank {r} is unresponsive (elapsed {time.time() - last_comm[r]:.2f}s). Marking as failed.",
                        flush=True,
                        file=sys.stderr,
                    )
                    failed_ranks.add(r)
                    active_ranks.remove(r)
                sleep_dict[r] = 0  # Mark worker as inactive

        except MPI.Exception as e:
            print(
                f"[Master] ERROR: MPI error with rank {r}: {repr(e)}",
                flush=True,
                file=sys.stderr,
            )
            failed_ranks.add(r)

        # Adaptive sleep
        if all(value == 0 for value in sleep_dict.values()):
            time.sleep(sleep_long)
        else:
            time.sleep(sleep_short)

    if len(active_ranks) == 0:
        print(
            f"[Master] Something went wrong. All workers are either idle or failed. Exiting...",
            flush=True,
            file=sys.stderr,
        )
        # Mpi abort
        MPI.COMM_WORLD.Abort(1)

    # Send termination signal to remaining active workers
    for r in range(1, N_slaves + 1):
        if r not in failed_ranks:  # Only send "Done" to active ranks
            comm.send("Done", dest=r)

    if len(failed_ranks) > 0:
        print(
            f"[Master] Finished distributing work. Active workers: {N_slaves - len(failed_ranks)}, Failed workers: {len(failed_ranks)}",
            flush=True,
        )

## rank > 0 (slaves)
else:
    # Directories for input (model parameters) and output (Cl data) data
    in_dir = os.path.join(directory, f"model_params_data/model_params_{rank}.txt")
    out_dirs_Cl = []
    out_dirs_Pk = []
    out_dirs_bg = []
    out_dirs_th = []
    out_dirs_ex = []
    if param.use_likelihood_filter and param.sampling == "iterative":
        out_dirs_loglkl = os.path.join(
            directory, f"likelihood_data/likelihood_data_{rank}.txt"
        )

    for Cl in param.output_Cl:
        out_dirs_Cl.append(
            os.path.join(directory, f"Cl_{Cl}_data/Cl_{Cl}_data_{rank}.txt")
        )
    for Pk in param.output_Pk:
        out_dirs_Pk.append(
            os.path.join(directory, f"Pk_{Pk}_data/Pk_{Pk}_data_{rank}.txt")
        )
    for bg in param.output_bg:
        bg = bg.replace("/", "\\")
        out_dirs_bg.append(
            os.path.join(directory, f"bg_{bg}_data/{bg}_data_{rank}.txt")
        )
    for th in param.output_th:
        out_dirs_th.append(
            os.path.join(directory, f"th_{th}_data/{th}_data_{rank}.txt")
        )
    if len(param.output_derived) > 0:
        out_dir_derived = os.path.join(
            directory, f"derived_data/derived_data_{rank}.txt"
        )
    for ex in param.extra_output:
        out_dirs_ex.append(
            os.path.join(directory, f"extra_{ex}_data/{ex}_data_{rank}.txt")
        )

    param_header = "# "
    for par_name in param_names:
        if par_name == param_names[-1]:
            param_header += par_name + "\n"
        else:
            param_header += par_name + "\t"

    derived_header = "# "
    for der_name in param.output_derived:
        if der_name == param.output_derived[-1]:
            derived_header += der_name + "\n"
        else:
            derived_header += der_name + "\t"

    # Initialise data files
    with open(in_dir, "w") as f:
        f.write(param_header)

    for out_dir in out_dirs_Cl + out_dirs_Pk + out_dirs_bg + out_dirs_th:
        with open(out_dir, "w") as f:
            f.write("")
    try:
        with open(out_dir_derived, "w") as f:
            f.write(derived_header)
    except:
        pass
    if param.use_likelihood_filter and param.sampling == "iterative":
        loglkl_header = "# true_loglkl\tchain_loglkl\n"
        with open(out_dirs_loglkl, "w") as f:
            f.write(loglkl_header)

    # Initialise timeout signal
    def timeout_handler(num, stack):
        raise Exception("timeout")

    signal.signal(signal.SIGALRM, timeout_handler)

    measure_cpu = False
    time_computation = False
    time_cosmo = None
    time_lkl = None
    if time_computation:
        time_file_path = f"{CONNECT_PATH}/data/{param.jobname}/time_computation.txt"
        if rank == 1:
            with open(time_file_path, "a") as time_file:
                time_file.write("# Cosmo time (s)\tLkl time (s)\n")

    def final_sync():
        """
        Final sync to ensure all data is written to disk.

        This function uses the local file path variables (in_dir, out_dirs_Cl, out_dirs_Pk,
        out_dir_derived, etc.) from the parent scope. It opens each file in read-write mode,
        does a flush and fsync, and closes. No new data is written.
        """

        # Collect all possible file paths into one list:
        # NOTE: Make sure these variables (in_dir, out_dirs_Cl, etc.) are defined in this scope!
        file_paths = []

        # Always sync model_params:
        if in_dir:
            file_paths.append(in_dir)

        # Cℓ files
        if len(param.output_Cl) > 0:
            file_paths.extend(out_dirs_Cl)
        # P(k) files
        if len(param.output_Pk) > 0:
            file_paths.extend(out_dirs_Pk)
        # Background, thermodynamics
        if len(param.output_bg) > 0:
            file_paths.extend(out_dirs_bg)
        if len(param.output_th) > 0:
            file_paths.extend(out_dirs_th)
        # Possibly derived:
        if len(param.output_derived) > 0:
            file_paths.append(out_dir_derived)

        # Extra outputs
        if len(param.extra_output) > 0:
            file_paths.extend(out_dirs_ex)

        # If using likelihood filter, also sync likelihood_data:
        if param.use_likelihood_filter and param.sampling == "iterative":
            file_paths.append(out_dirs_loglkl)

        # Now try opening each file in r+ mode and fsync it
        for fname in file_paths:
            # Some files might not exist if they were never used, so handle that safely.
            try:
                with open(fname, "r+") as f:
                    # flush() ensures no leftover user-space buffering (though we're not writing now)
                    f.flush()
                    # fsync() ensures the OS flushes to disk
                    os.fsync(f.fileno())
            except FileNotFoundError:
                # In case the file legitimately does not exist
                print(f"File {fname} does not exist. Skipping sync.", file=sys.stderr)
                pass
            except PermissionError:
                # or any other error you might want to handle
                print(
                    f"Permission error for file {fname}. Skipping sync.",
                    file=sys.stderr,
                )
                pass
            except Exception as e:
                print(f"Error syncing file {fname}: {repr(e)}", file=sys.stderr)
                pass

    # Iterate over each model
    while True:
        comm.send("I am done", dest=0)

        if param.use_likelihood_filter and param.sampling == "iterative":
            received_data = comm.recv(source=0)

            # Check if 'Done' message is received and break if so
            if type(received_data).__name__ == "str":
                try:
                    final_sync()
                except Exception as e:
                    print(
                        f"{prefix} ERROR: Failed to sync files at the end: {repr(e)}",
                        file=sys.stderr,
                    )
                break

            # Process the received data as dictionary if not 'Done'
            model = received_data["model"]
            loglkl_chains = received_data.get("loglkl_chains")

        else:
            model = comm.recv(source=0)

        # Check if 'Done' message was received in non-filter mode
        if type(model).__name__ == "str":
            try:
                final_sync()
            except Exception as e:
                print(
                    f"{prefix} ERROR: Failed to sync files at the end: {repr(e)}",
                    file=sys.stderr,
                )
            break

        # Set required CLASS parameters
        params = {}

        if (
            param.use_likelihood_filter
            and param.mcmc_sampler == "montepython"
            and param.sampling == "iterative"
        ):
            params.update(initial_cosmo_arguments)

        if len(param.output_Cl) > 0:
            params["output"] = "tCl,lCl"
            params["lensing"] = "yes"
            if any("b" in s or "e" in s for s in param.output_Cl):
                params["output"] += ",pCl"

        params.update(param.extra_input)
        for i, par_name in enumerate(param_names):
            params[par_name] = model[i]

        if "P_k_max_h/Mpc" in params:
            val = params.pop("P_k_max_h/Mpc")
            params["P_k_max_1/Mpc"] = val * 0.67556
        if len(param.output_Pk) > 0:
            if "output" in params:
                params["output"] += ",mPk"
            else:
                params["output"] = "mPk"
            params["P_k_max_1/Mpc"] = 2.5 * max(param.k_grid)
            params["z_max_pk"] = max(param.z_Pk_list)

        if "sigma8" in param.output_derived:
            if not "mPk" in params["output"]:
                if len(params["output"]) > 0:
                    params["output"] += ",mPk"
                else:
                    params["output"] = "mPk"
            if not "P_k_max_1/Mpc" in params:
                params["P_k_max_1/Mpc"] = 1.0

        # Simulate a parent that gets stuck:
        # if rank == 10:
        #     time.sleep(100000)

        signal.alarm(200)  # CLASS computations must not take longer than 200 seconds
        try:

            if param.memory_safe:

                signal.alarm(120)
                """
                The main loop for rank>0 that:
                - spawns one child
                - sends models to the child and let child compute CLASS and likelihood
                - recieves the results from the child
                - if child dies, respawn it.
                """

                result = None
                success = False
                recently_respawned = False

                while True:

                    if respawn_count >= max_respawn:

                        comm.send(
                            "IDLE", dest=0, tag=500
                        )  # Send idle message to master

                        print(
                            f"{prefix} [IDLE] Waiting for master to send 'Done'.",
                            file=sys.stderr,
                        )

                        while not comm.Iprobe(source=0):
                            time.sleep(0.5)
                        done_msg = comm.recv(source=0)  # Wait for master to send "Done"

                        if done_msg == "Done":
                            break

                    # ---------------------------- Compute CLASS ----------------------------#

                    soft_limit = get_soft_limit()

                    command_data = {
                        "params": params,
                        "model": model,
                        "global_ell": global_ell,
                        "soft_limit_bytes": soft_limit,
                    }

                    try:

                        if recently_respawned:
                            time.sleep(5)  # Wait for child to fully initialize

                        if child_comm == MPI.COMM_NULL:
                            print(
                                f"{prefix} ERROR: Child communicator invalid. Respawning immediately.",
                                file=sys.stderr,
                                flush=True,
                            )
                            child_comm = respawn_child()
                            continue

                        # Send a PING message to check if the child process is alive (non-blocking send)

                        req = mpi_isend(
                            comm=child_comm, data="PING", dest=0, tag=99, sleep=0.01
                        )

                        command = mpi_irecv(
                            comm=child_comm, source=0, tag=99, sleep=0.1
                        )

                        # Check if response is valid
                        if command != "PONG":
                            print(
                                f"{prefix} WARNING: Unexpected response from child: {command}. Expected 'PONG'. Retrying...",
                                file=sys.stderr,
                                flush=True,
                            )

                            req = mpi_isend(
                                comm=child_comm, data="PING", dest=0, tag=99, sleep=0.01
                            )

                            command = mpi_irecv(
                                comm=child_comm, source=0, tag=99, sleep=0.1
                            )

                            if command != "PONG":
                                print(
                                    f"{prefix} ERROR: Unexpected response again: {command}. Respawning child...",
                                    file=sys.stderr,
                                    flush=True,
                                )
                                child_comm = respawn_child()
                                recently_respawned = True
                                continue

                    except MPI.Exception as e:
                        print(
                            f"{prefix} ERROR: {repr(e)}. Child process is likely dead or unresponsive.\n"
                            f"Respawning child process...",
                            file=sys.stderr,
                            flush=True,
                        )
                        child_comm = respawn_child()
                        recently_respawned = True
                        continue

                    except Exception as e:
                        if str(e) == "timeout":
                            print(
                                f"{prefix} ERROR: Timeout while waiting for PONG response from child process. Child may be unresponsive.\n"
                                f"Respawning child process...",
                                file=sys.stderr,
                                flush=True,
                            )
                            child_comm = respawn_child()
                            recently_respawned = True
                            continue
                        else:
                            print(
                                f"{prefix} ERROR: Unexpected error during PING-PONG check: {repr(e)}"
                                f"Attemping to respawn child process...",
                                file=sys.stderr,
                                flush=True,
                            )
                            child_comm = respawn_child()
                            recently_respawned = True
                            continue
                    finally:
                        signal.alarm(0)

                    signal.alarm(
                        330
                    )  # CLASS computations must not take longer than 200 seconds

                    # ----------------- Send model and receive CLASS results -----------------#

                    # send command to child to run the model (cosmo.compute() with CLASS)
                    try:

                        if measure_cpu:
                            monitor_duration = 10.0
                            parent_proc = psutil.Process()

                            # Prime CPU measurement for the parent process
                            parent_proc.cpu_percent(interval=None)

                            # Prime CPU measurement for all subprocesses
                            subproc_list = parent_proc.children(recursive=True)
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
                                """Measure total CPU usage of parent process + all its threads and subprocesses."""
                                total_cpu = parent_proc.cpu_percent(
                                    interval=None
                                )  # Main process
                                for subproc in parent_proc.children(
                                    recursive=True
                                ):  # Include all subprocesses
                                    total_cpu += subproc.cpu_percent(interval=None)
                                return total_cpu

                            cpu_time_threads_before = get_total_thread_cpu_time(
                                parent_proc
                            )

                            # Also track system-wide CPU usage before CLASS starts
                            system_cpu_before = psutil.cpu_percent(interval=None)

                            # Measure CPU time before CLASS computation
                            cpu_time_before = parent_proc.cpu_times()

                        # Send the command to run the model

                        req = mpi_isend(
                            comm=child_comm,
                            data=("RUN_MODEL", command_data),
                            dest=0,
                            tag=0,
                            sleep=0.01,
                        )

                        command, result = mpi_irecv(
                            comm=child_comm, source=0, tag=0, sleep=0.1
                        )

                        if measure_cpu:
                            # Measure CPU time after CLASS computation
                            cpu_time_after = parent_proc.cpu_times()

                            # Get CPU usage **immediately after** CLASS computation (reflects what happened during)
                            parent_cpu_usage = parent_proc.cpu_percent(interval=None)

                            # Capture total CPU usage including all subprocesses
                            total_parent_cpu_usage = get_total_cpu_usage()

                            # Compute system-wide CPU usage difference
                            system_cpu_after = psutil.cpu_percent(interval=None)
                            system_cpu_usage = system_cpu_after - system_cpu_before

                            # Compute CPU time breakdown
                            cpu_time_user = cpu_time_after.user - cpu_time_before.user
                            cpu_time_system = (
                                cpu_time_after.system - cpu_time_before.system
                            )
                            cpu_time = cpu_time_user + cpu_time_system

                            cpu_time_threads_after = get_total_thread_cpu_time(
                                parent_proc
                            )
                            thread_cpu_time_used = (
                                cpu_time_threads_after - cpu_time_threads_before
                            )

                            # Print results
                            print(
                                f"{prefix} CPU time used during CLASS computation: {cpu_time:.2f} s",
                                file=sys.stderr,
                                flush=True,
                            )
                            print(
                                f"{prefix} CPU user time used during CLASS computation: {cpu_time_user:.2f} s",
                                file=sys.stderr,
                                flush=True,
                            )
                            print(
                                f"{prefix} CPU system time used during CLASS computation: {cpu_time_system:.2f} s",
                                file=sys.stderr,
                                flush=True,
                            )
                            print(
                                f"{prefix} CPU percent used by parent process: {parent_cpu_usage:.2f} %",
                                file=sys.stderr,
                                flush=True,
                            )
                            print(
                                f"{prefix} Total CPU percent used by parent process + subprocesses: {total_parent_cpu_usage:.2f} %",
                                file=sys.stderr,
                                flush=True,
                            )
                            print(
                                f"{prefix} System-wide CPU percent difference during CLASS computation: {system_cpu_usage:.2f} %",
                                file=sys.stderr,
                                flush=True,
                            )

                            print(
                                f"{prefix} Total CPU time used by all threads: {thread_cpu_time_used:.2f} s",
                                file=sys.stderr,
                                flush=True,
                            )

                            print(
                                f"{prefix} CPU affinity: {parent_proc.cpu_affinity()}",
                                file=sys.stderr,
                                flush=True,
                            )

                            print(
                                f"{prefix} Active threads count: {len(parent_proc.threads())}",
                                file=sys.stderr,
                                flush=True,
                            )

                    except MPI.Exception as e:
                        print(
                            f"{prefix} ERROR: Failed to send or receive data to/from child process. Child process may have crashed."
                            f"Possibly due to out-of-memory or other critical error during CLASS computation.\n"
                            f"Model: {params}\n"
                            f"skipping to the next model...",
                            file=sys.stderr,
                            flush=True,
                        )
                        print(
                            f"{prefix} Exception: {repr(e)}",
                            file=sys.stderr,
                            flush=True,
                        )
                        success = False
                        break  # Skip to next model
                    except Exception as e:
                        if str(e) == "timeout":
                            print(
                                f"{prefix} ERROR: Timeout while waiting for response from child process after CLASS computations. Child may be unresponsive.\n",
                                file=sys.stderr,
                                flush=True,
                            )
                            success = False
                            break  # Skip to next model
                        else:
                            print(
                                f"{prefix} ERROR: Failed to send or receive data to/from child process. Child process may have crashed."
                                f"Possibly due to out-of-memory or other critical error during CLASS computation    .\n"
                                f"Model: {params}\n"
                                f"skipping to the next model...",
                                file=sys.stderr,
                                flush=True,
                            )
                            print(
                                f"{prefix} Exception: {repr(e)}",
                                file=sys.stderr,
                                flush=True,
                            )
                            success = False
                            break

                    # ---------------------- HANDLE CHILD PROCESS FAILURE ----------------------#

                    # If the result is None, it means the child process failed mid-execution
                    if command and result is None:
                        print(
                            f"{prefix} ERROR: Child process terminated unexpectedly. Suspected cause: Out-of-memory (limit set for child: {soft_limit/(1024**3):.2f} GB) or other critical error.\n"
                            f"This happened during the CLASS computation step for the model: {params}"
                            f"skipping to the next model...",
                            file=sys.stderr,
                            flush=True,
                        )
                        success = False
                        break

                    if command == "MEMORY_EXCEEDED":
                        print(
                            f"{prefix} ERROR: Child process failed due to out-of-memory error during CLASS computation\n"
                            f"Model: {params}\n"
                            f"skipping to the next model...",
                            file=sys.stderr,
                            flush=True,
                        )
                        success = False
                        break
                    elif command == "MODEL_FAILED":
                        print(
                            f"{prefix} ERROR: Child process failed during CLASS computation\n"
                            f"Model: {params}\n"
                            f"skipping to the next model...",
                            file=sys.stderr,
                            flush=True,
                        )
                        success = False
                        break

                    if command != "MODEL_DONE":
                        print(
                            f"{prefix} ERROR: Unexpected command received from child process: {command}",
                            file=sys.stderr,
                            flush=True,
                        )
                        success = False
                        break

                    # if result['success_cosmo'] is False, the child process failed during cosmo.compute()
                    if not result.get("success_cosmo", False):
                        print(
                            f"{prefix} ERROR: Child process failed during CLASS computation\n"
                            f"Possibly due to out-of-memory or other critical error.\n"
                            f"Model: {params}\n"
                            f"skipping to the next model...",
                            file=sys.stderr,
                            flush=True,
                        )
                        success = False
                        break

                    success = result.get(
                        "success", False
                    )  # Would still be False if the likelihood filter is enabled or if the child process failed
                    # If the likelihood filter is enabled, the success flag will be set after the likelihood calculation step below
                    # ---------------------------- Compute Likelihood ----------------------------#
                    if (
                        param.use_likelihood_filter
                        and param.sampling == "iterative"
                        and result.get("success_cosmo", False)
                    ):

                        command = None
                        result = None
                        soft_limit = get_soft_limit()
                        command_data["soft_limit_bytes"] = soft_limit

                        # Send the command to run the likelihood calculation
                        try:

                            req = mpi_isend(
                                comm=child_comm,
                                data=("RUN_LIKELIHOOD", command_data),
                                dest=0,
                                tag=0,
                                sleep=0.01,
                            )

                            # Wait for the child to finish the likelihood calculation
                            command, result = mpi_irecv(
                                comm=child_comm, source=0, tag=0, sleep=0.05
                            )

                        except MPI.Exception as e:
                            print(
                                f"{prefix} ERROR: Likelihood computation failed. Child process may have crashed."
                                f"Possibly due to out-of-memory or other critical error during likelihood computation.\n"
                                f"Model: {params}\n"
                                f"skipping to the next model...",
                                file=sys.stderr,
                                flush=True,
                            )
                            print(
                                f"{prefix} Exception: {repr(e)}",
                                file=sys.stderr,
                                flush=True,
                            )
                            success = False
                            break  # Skip to next model
                        except Exception as e:
                            if str(e) == "timeout":
                                print(
                                    f"{prefix} ERROR: Timeout while waiting for response from child process. Child may be unresponsive.\n"
                                    f"this happened during the likelihood calculation step for the model: {params}\n"
                                    f"skipping to the next model...",
                                    file=sys.stderr,
                                    flush=True,
                                )
                                success = False
                                break
                            else:
                                print(
                                    f"{prefix} ERROR: Failed to send or receive data to/from child process. Child process may have crashed."
                                    f"Possibly due to out-of-memory or other critical error during likelihood computation.\n"
                                    f"Model: {params}\n"
                                    f"skipping to the next model...",
                                    file=sys.stderr,
                                    flush=True,
                                )
                                print(
                                    f"{prefix} Exception: {repr(e)}",
                                    file=sys.stderr,
                                    flush=True,
                                )
                                success = False
                                break

                        # Receive results step 2; after likelihood calculation
                        # This is the final results after both steps, it contains both computed CLASS and likelihood results

                        # If the result is None, it means the child process failed mid-execution
                        if command and result is None:
                            print(
                                f"{prefix} ERROR: Child process terminated unexpectedly. Suspected cause: Out-of-memory (limit: {soft_limit} bytes) or other critical error.\n"
                                f"This happened during the likelihood calculation step for the model: {params}\n"
                                f"skipping to the next model...",
                                file=sys.stderr,
                                flush=True,
                            )
                            success = False
                            break

                        if command == "MEMORY_EXCEEDED":
                            print(
                                f"{prefix} ERROR: Child process failed due to out-of-memory error during likelihood calculation\n"
                                f"Model: {params}\n"
                                f"skipping to the next model...",
                                file=sys.stderr,
                                flush=True,
                            )
                            success = False
                            break
                        elif command == "LIKELIHOOD_FAILED":
                            print(
                                f"{prefix} ERROR: Child process failed during likelihood calculation\n"
                                f"Model: {params}\n"
                                f"skipping to the next model...",
                                file=sys.stderr,
                                flush=True,
                            )
                            success = False
                            break

                        if command != "LIKELIHOOD_DONE":
                            print(
                                f"{prefix} ERROR: Unexpected command received from child process: {command}",
                                file=sys.stderr,
                                flush=True,
                            )
                            success = False
                            break

                        if not result.get("success_lkl", False):
                            print(
                                f"{prefix} ERROR: Child process failed during likelihood calculation\n"
                                f"Model: {params}\n"
                                f"skipping to the next model...",
                                file=sys.stderr,
                                flush=True,
                            )
                            success = False
                            break

                    # If the result contains 'success', check if the computation succeeded
                    if result.get("success", False):
                        # Store computed results
                        cls = result["cls"]
                        ell = result["ell"]
                        pks = result["pks"]
                        der = result["der"]
                        bg = result["bg"]
                        bg_idx = result["bg_idx"]
                        z_bg = result["z_bg"]
                        th = result["th"]
                        th_idx = result["th_idx"]
                        z_th = result["z_th"]
                        extra_output = result["extra_output"]
                        model = result["model"]
                        loglkl_true = result["loglkl_true"]
                        success = True
                    else:
                        print(
                            f"{prefix} ERROR: Model {params} failed. Error: {result.get('error_msg', 'Unknown error.')}",
                            file=sys.stderr,
                            flush=True,
                        )
                        success = False

                    break  # Finish loop after processing one model

            else:  # Not memory safe. Initial implementation

                if time_computation:
                    time_cosmo = time.time()

                if measure_cpu:
                    monitor_duration = 10.0
                    parent_proc = psutil.Process()

                    # Prime CPU measurement for the parent process
                    parent_proc.cpu_percent(interval=None)

                    # Prime CPU measurement for all subprocesses
                    subproc_list = parent_proc.children(recursive=True)
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
                        """Measure total CPU usage of parent process + all its threads and subprocesses."""
                        total_cpu = parent_proc.cpu_percent(
                            interval=None
                        )  # Main process
                        for subproc in parent_proc.children(
                            recursive=True
                        ):  # Include all subprocesses
                            total_cpu += subproc.cpu_percent(interval=None)
                        return total_cpu

                    cpu_time_threads_before = get_total_thread_cpu_time(parent_proc)

                    # Also track system-wide CPU usage before CLASS starts
                    system_cpu_before = psutil.cpu_percent(interval=None)

                    # Measure CPU time before CLASS computation
                    cpu_time_before = parent_proc.cpu_times()

                cosmo = classy.Class()
                cosmo.set(params)
                start_time2 = time.time()
                cosmo.compute()
                end_time2 = time.time()

                if measure_cpu:
                    # Measure CPU time after CLASS computation
                    cpu_time_after = parent_proc.cpu_times()

                    # Get CPU usage **immediately after** CLASS computation (reflects what happened during)
                    parent_cpu_usage = parent_proc.cpu_percent(interval=None)

                    # Capture total CPU usage including all subprocesses
                    total_parent_cpu_usage = get_total_cpu_usage()

                    # Compute system-wide CPU usage difference
                    system_cpu_after = psutil.cpu_percent(interval=None)
                    system_cpu_usage = system_cpu_after - system_cpu_before

                    # Compute CPU time breakdown
                    cpu_time_user = cpu_time_after.user - cpu_time_before.user
                    cpu_time_system = cpu_time_after.system - cpu_time_before.system
                    cpu_time = cpu_time_user + cpu_time_system
                    elapsed_time = end_time2 - start_time2

                    cpu_time_threads_after = get_total_thread_cpu_time(parent_proc)
                    thread_cpu_time_used = (
                        cpu_time_threads_after - cpu_time_threads_before
                    )

                    # Print results
                    print(
                        f"{prefix} CPU time used during CLASS computation: {cpu_time:.2f} s",
                        file=sys.stdout,
                        flush=True,
                    )
                    print(
                        f"{prefix} CPU user time used during CLASS computation: {cpu_time_user:.2f} s",
                        file=sys.stdout,
                        flush=True,
                    )
                    print(
                        f"{prefix} CPU system time used during CLASS computation: {cpu_time_system:.2f} s",
                        file=sys.stdout,
                        flush=True,
                    )
                    print(
                        f"{prefix} CPU percent used by parent process: {parent_cpu_usage:.2f} %",
                        file=sys.stdout,
                        flush=True,
                    )
                    print(
                        f"{prefix} Total CPU percent used by parent process + subprocesses: {total_parent_cpu_usage:.2f} %",
                        file=sys.stdout,
                        flush=True,
                    )
                    print(
                        f"{prefix} System-wide CPU percent difference during CLASS computation: {system_cpu_usage:.2f} %",
                        file=sys.stdout,
                        flush=True,
                    )
                    print(
                        f"{prefix} Total elapsed time for CLASS computation: {elapsed_time:.2f} s",
                        file=sys.stdout,
                        flush=True,
                    )

                    print(
                        f"{prefix} Total CPU time used by all threads: {thread_cpu_time_used:.2f} s",
                        file=sys.stdout,
                        flush=True,
                    )

                    print(f"{prefix} CPU affinity: {parent_proc.cpu_affinity()}")

                    print(
                        f"{prefix} Active threads count: {len(parent_proc.threads())}"
                    )

                if time_computation:
                    time_cosmo = time.time() - time_cosmo

                if len(param.output_bg) > 0:
                    bg = cosmo.get_background()
                    if len(param.z_bg_list) > 0:
                        z_bg = param.z_bg_list
                    else:
                        bg_idx = get_z_idx(bg["z"])
                        z_bg = bg["z"][bg_idx]
                if len(param.output_th) > 0:
                    th = cosmo.get_thermodynamics()
                    if len(param.z_th_list) > 0:
                        z_th = param.z_th_list
                    else:
                        th_idx = get_z_idx(th["z"])
                        z_th = th["z"][th_idx]
                if len(param.output_derived) > 0:
                    der = cosmo.get_current_derived_parameters(param.output_derived)
                if len(param.output_Cl) > 0:
                    cls = get_computed_cls(cosmo, ell_array=global_ell)
                    if any(np.isnan(cls[key]).any() for key in cls):
                        raise classy.CosmoComputationError(
                            "Class computation completed with NaN values in CMB power spectra."
                        )
                    ell = cls["ell"][2:]
                if len(param.output_Pk) > 0:
                    pks = {}
                    for pk in param.output_Pk:
                        pks[pk] = {}
                        for z in param.z_Pk_list:
                            pks[pk][z] = []
                            for k in param.k_grid:
                                pks[pk][z].append(eval(f"cosmo.{pk}(k,z)"))

                if param.use_likelihood_filter and param.sampling == "iterative":
                    try:

                        if time_computation:
                            time_lkl = time.time()

                        # compute likelihood
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

                        if time_computation:
                            time_lkl = time.time() - time_lkl

                            if time_cosmo is not None and time_lkl is not None:
                                with open(time_file_path, "a") as time_file:
                                    time_file.write(
                                        f"{time_cosmo:.6f}\t{time_lkl:.6f}\n"
                                    )
                                    time_file.flush()  # Ensure the data is written to the file

                    except Exception as e:
                        print(
                            "The following model failed in likelihood calculation:",
                            flush=True,
                        )
                        print(params, flush=True)
                        success = False
                        print(f"Exception: {repr(e)}", flush=True)
                        print(traceback.format_exc(), flush=True)
                        break

                    if loglkl_true == None:
                        print(
                            "The following model failed as a result of likelihood calculation returning None:",
                            flush=True,
                        )
                        print(params, flush=True)
                        success = False
                        break

                    if np.isnan(loglkl_true):
                        print(
                            "The following model failed with NaN in likelihood calculation:",
                            flush=True,
                        )
                        print(params, flush=True)
                        success = False
                        break

                success = True
        except classy.CosmoComputationError as e:
            print("The following model failed in CLASS:", flush=True)
            print(params, flush=True)
            success = False
            print(e.message)
        except classy.CosmoSevereError as e:
            print("The following model failed in CLASS:", flush=True)
            print(params, flush=True)
            success = False
            print(e.message)
        except Exception as e:
            if str(e) == "timeout":
                print("The following model took too long to complete:", flush=True)
                print(params, flush=True)
                success = False
            else:
                raise e
        finally:
            signal.alarm(0)

        if success:
            # Write data to data files
            for out_dir, output in zip(out_dirs_Cl, param.output_Cl):
                par_out = cls[output][2:] * ell * (ell + 1) / (2 * np.pi)
                with open(out_dir, "a") as f:
                    for i, l in enumerate(ell):
                        if i != len(ell) - 1:
                            f.write(str(l) + "\t")
                        else:
                            f.write(str(l) + "\n")
                    for i, p in enumerate(par_out):
                        if i != len(par_out) - 1:
                            f.write(str(p) + "\t")
                        else:
                            f.write(str(p) + "\n")

            for out_dir, output in zip(out_dirs_Pk, param.output_Pk):
                with open(out_dir, "a") as f:
                    for i, k in enumerate(param.k_grid):
                        if i != len(param.k_grid) - 1:
                            f.write(str(k) + "\t")
                        else:
                            f.write(str(k) + "\n")
                    for z in param.z_Pk_list:
                        par_out = pks[output][z]
                        for i, p in enumerate(par_out):
                            if i != len(par_out) - 1:
                                f.write(str(p) + "\t")
                            else:
                                f.write(str(p) + "\n")

            if len(param.output_derived) > 0:
                par_out = []
                for output in param.output_derived:
                    par_out.append(der[output])
                with open(out_dir_derived, "a") as f:
                    for i, p in enumerate(par_out):
                        if i != len(par_out) - 1:
                            f.write(str(p) + "\t")
                        else:
                            f.write(str(p) + "\n")

            for out_dir, output in zip(out_dirs_bg, param.output_bg):
                if len(param.z_bg_list) > 0:
                    par_out = CubicSpline(
                        np.flip(bg["z"]), np.flip(bg[output]), bc_type="natural"
                    )(param.z_bg_list)
                else:
                    par_out = bg[output][bg_idx]
                with open(out_dir, "a") as f:
                    for i, z in enumerate(z_bg):
                        if i != len(z_bg) - 1:
                            f.write(str(z) + "\t")
                        else:
                            f.write(str(z) + "\n")
                    for i, p in enumerate(par_out):
                        if i != len(par_out) - 1:
                            f.write(str(p) + "\t")
                        else:
                            f.write(str(p) + "\n")

            for out_dir, output in zip(out_dirs_th, param.output_th):
                if len(param.z_th_list) > 0:
                    par_out = CubicSpline(th["z"], th[output], bc_type="natural")(
                        param.z_th_list
                    )
                else:
                    par_out = th[output][th_idx]
                with open(out_dir, "a") as f:
                    for i, z in enumerate(z_th):
                        if i != len(z_th) - 1:
                            f.write(str(z) + "\t")
                        else:
                            f.write(str(z) + "\n")
                    for i, p in enumerate(par_out):
                        if i != len(par_out) - 1:
                            f.write(str(p) + "\t")
                        else:
                            f.write(str(p) + "\n")

            for out_dir, output in zip(out_dirs_ex, param.extra_output):
                if param.memory_safe:
                    par_out = extra_output[output]
                else:
                    par_out = eval(param.extra_output[output])
                try:
                    len(par_out)
                except:
                    par_out = [par_out]
                with open(out_dir, "a") as f:
                    for i, p in enumerate(par_out):
                        if i != len(par_out) - 1:
                            f.write(str(p) + "\t")
                        else:
                            f.write(str(p) + "\n")

            with open(in_dir, "a") as f:
                for i, m in enumerate(model):
                    if i != len(model) - 1:
                        f.write(str(m) + "\t")
                    else:
                        f.write(str(m) + "\n")

            if param.use_likelihood_filter and param.sampling == "iterative":
                with open(out_dirs_loglkl, "a") as f:
                    f.write(f"{loglkl_true}\t{loglkl_chains}\n")

        cosmo.struct_cleanup()

    if param.memory_safe:
        # Send the final "Done" message to the child process

        flag_normal_child_exit = False

        def timeout_handler(sig, frame):
            raise RuntimeError("timeout")

        signal.signal(signal.SIGALRM, timeout_handler)

        signal.alarm(345)  # 10 seconds timeout

        # Ping Pong
        try:
            req = mpi_isend(comm=child_comm, data="PING", dest=0, tag=99, sleep=0.01)

            command = mpi_irecv(comm=child_comm, source=0, tag=99, sleep=0.1)

            if command != "PONG":
                print(
                    f"{prefix} WARNING: Unexpected response from child: {command}. Expected 'PONG', before closing the child process.",
                    file=sys.stderr,
                    flush=True,
                )
        except MPI.Exception as e:
            print(
                f"{prefix} WARNING: MPI communicator may have already been cleaned up. Error: {repr(e)}",
                file=sys.stderr,
                flush=True,
            )
        except RuntimeError as e:
            if str(e) == "timeout":
                print(
                    f"{prefix} WARNING: Timeout while waiting for child process to acknowledge 'PING' message.",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                raise e
        except Exception as e:
            print(
                f"{prefix} WARNING: Failed to send 'PING' message to child process. Error: {repr(e)}",
                file=sys.stderr,
                flush=True,
            )
        finally:
            signal.alarm(0)

        signal.alarm(100)  # 10 seconds timeout
        # Send final "DONE" message to child to do proper close down
        try:
            # placeholder command_data
            command_data = {
                "params": None,
                "model": None,
                "global_ell": None,
                "soft_limit_bytes": None,
            }

            req = mpi_isend(
                comm=child_comm, data=("DONE", command_data), dest=0, tag=0, sleep=0.01
            )

            command, _ = mpi_irecv(comm=child_comm, source=0, tag=0, sleep=0.1)

            if command == "DONE_ACK":
                child_comm.Disconnect()
                flag_normal_child_exit = True
            else:
                print(
                    f"{prefix} WARNING: Unexpected response from child: {command}. Expected 'DONE_ACK'.",
                    file=sys.stderr,
                    flush=True,
                )

        except MPI.Exception as e:
            print(
                f"{prefix} WARNING: MPI communicator may have already been cleaned up. Error: {repr(e)}",
                file=sys.stderr,
                flush=True,
            )
        except RuntimeError as e:
            if str(e) == "timeout":
                print(
                    f"{prefix} WARNING: Timeout while waiting for child process to acknowledge 'DONE' message."
                    f"Most likely the child process has already terminated or is unresponsive or has crashed."
                    f"The parent will shut down without waiting for the child to respond.",
                    file=sys.stderr,
                    flush=True,
                )

        except Exception as e:
            print(
                f"{prefix} WARNING: Failed to send 'DONE' message to child process. Error: {repr(e)}",
                file=sys.stderr,
                flush=True,
            )
        finally:
            signal.alarm(0)


TIMEOUT_SECONDS = 600  # Timeout: 5 minutes

# -------------------- Worker Ranks (> 0) --------------------
if rank > 0:

    if param.memory_safe:
        if flag_normal_child_exit and respawn_count == 0:
            # Send done signal to master
            comm.send("DONE", dest=0, tag=999)
        else:
            # send ABORT signal to master
            comm.send("ABORT", dest=0, tag=999)
    else:
        comm.send("DONE", dest=0, tag=999)

    # Non-blocking wait for shutdown signal from master
    start_wait = time.time()

    while not comm.Iprobe(source=0, tag=1000):
        time.sleep(0.1)  # Prevent CPU overuse
        if time.time() - start_wait > TIMEOUT_SECONDS + 10:  # Failsafe extra timeout
            print(
                f"[Worker {rank}] WARNING: Master unresponsive. Exiting.",
                flush=True,
                file=sys.stderr,
            )
            time.sleep(3)  # wait to ensure buffer is flushed and written to all files
            MPI.COMM_WORLD.Abort(1)

    go_to_finalize = comm.recv(source=0, tag=1000)

    if go_to_finalize == "FINALIZE":
        MPI.Finalize()
    else:
        print(
            f"[Worker {rank}] Received ABORT signal. Exiting immediately.",
            flush=True,
            file=sys.stderr,
        )
        time.sleep(3)  # wait to ensure buffer is flushed and written to all files
        MPI.COMM_WORLD.Abort(1)

# -------------------- Master Rank (0) --------------------
elif rank == 0:
    done_ranks = set()
    abort_ranks = set()
    start_time = time.time()

    # Wait for all worker ranks to signal "DONE"
    while len(done_ranks) < N_slaves:
        status = MPI.Status()
        if comm.Iprobe(source=MPI.ANY_SOURCE, tag=999, status=status):
            msg = comm.recv(source=status.source, tag=999)
            if msg == "DONE":
                done_ranks.add(status.source)
            elif msg == "ABORT":
                abort_ranks.add(status.source)

        # Check timeout
        if time.time() - start_time > TIMEOUT_SECONDS:
            print(
                f"[Master] Timeout reached! Some ranks are stuck. Aborting MPI.",
                flush=True,
                file=sys.stderr,
            )
            # Save a file to directory called "calculations_completed" to indicate that the calculation was completed
            path_calc_complete = os.path.join(directory, "calc_completed.txt")
            with open(path_calc_complete, "w") as f:
                f.write("COMPLETED")

            time.sleep(3)  # wait to ensure buffer is flushed and written to all files

            MPI.COMM_WORLD.Abort(1)

        if len(abort_ranks) > 0:
            print(
                f"[Master] [WARNING] Received ABORT command from ranks: {abort_ranks}.\n"
                f"This is due to the parent failing closing down the child process properly.\n"
                f"Or the child process crashed or was unresponsive at some point.\n"
                f"Exiting through MPI.Abort() to prevent hanging/deadlock.",
                flush=True,
                file=sys.stderr,
            )
            # Save a file to directory called "calculations_completed" to indicate that the calculation was completed
            path_calc_complete = os.path.join(directory, "calc_completed.txt")
            with open(path_calc_complete, "w") as f:
                f.write("COMPLETED")

            time.sleep(3)  # wait to ensure buffer is flushed and written to all files
            MPI.COMM_WORLD.Abort(1)

    path_calc_complete = os.path.join(directory, "calc_completed.txt")
    with open(path_calc_complete, "w") as f:
        f.write("COMPLETED")

    for r in range(1, N_slaves + 1):
        comm.send("FINALIZE", dest=r, tag=1000)
        time.sleep(0.005)  # 5ms delay per message

    MPI.Finalize()

print(f"{prefix} Exiting normally.", flush=True, file=sys.stdout)


time.sleep(1)

os._exit(0)  # Force exit because sys.exit(0) sometimes hangs

# sys.exit(0)
