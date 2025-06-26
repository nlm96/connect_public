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

os.environ["UCX_LOG_LEVEL"] = "error"
param_file = sys.argv[1]
CONNECT_PATH = sys.argv[2]
sampling = sys.argv[3]
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


if (
    not param.memory_safe
    and param.mcmc_sampler == "montepython"
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


# --------------------------- Memory-safe implementation ---------------------------#
if param.memory_safe and rank > 0:

    print(
        f"{prefix} [DEBUG] Current conda environment: {os.environ.get('CONDA_DEFAULT_ENV', 'Not set')}",
        file=sys.stderr,
        flush=True,
    )

    import random

    time.sleep(
        random.uniform(0, 2)
    )  # Random sleep to avoid all children starting at the same time

    MPI.COMM_WORLD.Set_errhandler(MPI.ERRORS_RETURN)
    os.environ["UCX_LOG_LEVEL"] = "error"
    os.environ["UCX_HANDLE_ERRORS"] = "no"
    os.environ["UCX_ERROR_SIGNALS"] = "no"

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

    import subprocess
    import signal
    import sys
    import os
    import random
    import time
    import traceback

    # Suppose these are globals:
    child_proc = None
    respawn_count = 0
    max_respawn = 3

    def spawn_child(rank, CONNECT_PATH, param_file):
        """
        Spawns the child process using srun on the same node.
        Returns the Popen object for the child.
        """
        mem = int(int(os.environ.get("SLURM_MEM_PER_CPU", "7000")) * 0.95)  # Read in MB
        import socket

        short_host = socket.gethostname().split(".")[0]

        child_id = random.randint(0, 1000)

        cmd = [
            "srun",
            "--nodelist",
            short_host,  # same node
            "--nodes=1",
            "--ntasks=1",
            "--overlap",
            "--cpu-bind=none",
            f"--mem={mem}M",
            "--mpi=none",
            "--no-kill",
            "--export=ALL",
            "--unbuffered",
            # possibly add: "--exclusive", "--exact", etc. if your cluster allows it
            sys.executable,
            os.path.join(CONNECT_PATH, "source", "calc_models_mpi_child.py"),
            "--child-mode",
            param_file,
            CONNECT_PATH,
            str(rank),
            str(child_id),
        ]

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # or None if you want child errors in your main .err
            bufsize=0,
            text=False,  # we will do raw bytes if we want to pickle directly
            env=os.environ,
        )

        return proc

    print(f"{prefix} [DEBUG] Starting child process...", file=sys.stderr, flush=True)
    child_proc = spawn_child(rank, CONNECT_PATH, param_file)

    def respawn_child(
        rank, CONNECT_PATH, param_file, prefix="[Rank X]", child_proc_ref=None
    ):
        """
        Handles respawning a new child process safely with srun + subprocess.
        We remove all MPI references and do a kill on the old Popen child if it exists.
        """

        global child_proc, respawn_count, max_respawn

        if child_proc_ref is not None:
            # If you want to pass in which child_proc you want to replace
            child_proc = child_proc_ref

        # If we have reached max respawn attempts, mark as failed
        if respawn_count >= max_respawn:
            print(
                f"{prefix} ERROR: Maximum respawn attempts reached. Marking rank as failed.",
                file=sys.stderr,
                flush=True,
            )
            return None

        respawn_count += 1

        # Optional: Put a small timeout around the child killing
        def timeout_handler(signum, frame):
            raise TimeoutError("Timeout while waiting to kill old child.")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)  # Up to 5s to kill old child

        try:
            # If there was an old child process, kill it
            if child_proc is not None:
                try:
                    # Attempt a graceful terminate
                    child_proc.terminate()
                    ret = child_proc.wait(timeout=3)
                    print(
                        f"{prefix} Old child process terminated with code {ret}",
                        file=sys.stderr,
                        flush=True,
                    )
                except subprocess.TimeoutExpired:
                    # Force kill
                    child_proc.kill()
                    ret = child_proc.wait(timeout=3)
                    print(
                        f"{prefix} Had to kill old child. Return code {ret}",
                        file=sys.stderr,
                        flush=True,
                    )

            child_proc = None
        except Exception as e:
            print(
                f"{prefix} WARNING: Failed to properly kill old child process: {repr(e)}",
                file=sys.stderr,
                flush=True,
            )
        finally:
            signal.alarm(0)  # disable the kill-timeout

        # Now spawn a new child
        try:
            print(
                f"{prefix} WARNING: Child process died or is unresponsive. Respawning...",
                file=sys.stderr,
                flush=True,
            )

            new_child = spawn_child(rank, CONNECT_PATH, param_file)
            child_proc = new_child

            print(
                f"{prefix} [INFO] Successfully respawned child process with PID {new_child.pid}.",
                file=sys.stderr,
                flush=True,
            )

            return child_proc

        except Exception as e:
            print(
                f"{prefix} ERROR: Unexpected failure during child respawn: {repr(e)}",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc()
            child_proc = None
            return None

    import pickle
    import tempfile

    # Find the correct scratch directory
    SCRATCH_DIR = os.getenv("SLURM_JOB_ID")
    if SCRATCH_DIR:
        SCRATCH_DIR = f"/scratch/{SCRATCH_DIR}/"
        os.makedirs(SCRATCH_DIR, exist_ok=True)  # Ensure the directory exists
    else:
        raise RuntimeError(
            "SLURM_JOB_ID not found! This script must be run in a SLURM job."
        )

    def write_msg(command, output_stream, data=None):
        """
        Sends `command` (with or without `data` data) through `output_stream`.

        - If `data` is None, sends only `command`.
        - Otherwise, pickles `data` to a temp file and sends `command FILE_PATH`.

        Parameters:
            command (str): The command string to send.
            output_stream (stream): The output pipe to write to.
            data (any picklable object, optional): The data to send. If None, only command is sent.

        Example:
            - write_one_message("PING", output_stream)  # Sends "PING" without data
            - write_one_message("UPDATE", output_stream, some_object)  # Sends "UPDATE FILE_PATH"
        """
        if data is None:
            # Send only the command
            message = f"{command}\n".encode("utf-8")
        else:
            # Create a temporary file in SCRATCH for the pickled data
            with tempfile.NamedTemporaryFile(
                dir=SCRATCH_DIR, delete=False, suffix=".pkl"
            ) as temp_file:
                pickle.dump(data, temp_file, protocol=pickle.HIGHEST_PROTOCOL)
                file_path = temp_file.name  # Store the file path

            # Send the command along with the file path
            message = f"{command} {file_path}\n".encode("utf-8")

        # Write and flush to ensure it's sent immediately
        output_stream.write(message)
        output_stream.flush()

    def read_msg(input_stream):
        """
        Reads a command (with or without data) from `input_stream`, loads the pickled data if provided.

        - If only a command is received, returns (`command`, None).
        - If `command FILE_PATH` is received, loads and deletes the file, then returns (`command`, data).

        Parameters:
            input_stream (stream): The input pipe to read from.

        Returns:
            tuple: (command, data or None)

        Example:
            - read_one_message(input_stream)  # ("PING", None) if only a command was sent
            - read_one_message(input_stream)  # ("UPDATE", some_object) if data was sent
        """
        # Read the command and file path from the pipe
        line = input_stream.readline().strip()

        if not line:
            return None, None  # Handle EOF

        # Decode the binary input into a string (important for Python3)
        if isinstance(line, bytes):
            line = line.decode("utf-8")  # Convert bytes to string

        # Check if there is a file path attached
        parts = line.split(" ", 1)
        command = parts[0]

        if len(parts) == 1:
            return command, None  # No data, only command

        if len(parts) != 2:
            raise ValueError(
                f"[ERROR] Malformed message received: {line}"
            )  # Does ValueError work with flush and file or should it be print?

        file_path = parts[1]

        # Read the pickle file
        with open(file_path, "rb") as file:
            data = pickle.load(file)

        # Delete the temporary pickle file
        os.remove(file_path)

        return command, data


# --------------------------------- End of memory-safe implementation ---------------------------------#


if (
    not param.memory_safe
    and param.use_likelihood_filter
    and param.sampling == "iterative"
    and rank > 0
):
    likelihood_calc = likelihood_calculator(param, rank)
    if param.mcmc_sampler == "montepython":
        initial_cosmo_arguments = copy.deepcopy(
            likelihood_calc.mp["data"].cosmo_arguments
        )
        # print(f"cosmo_arguments: {initial_cosmo_arguments}", flush=True)
elif (
    param.memory_safe
    and param.use_likelihood_filter
    and param.sampling == "iterative"
    and rank > 0
):

    # Read manually:

    # message = child_proc.stdout.readline().strip()
    # print(f"Message: {message}", flush=True, file=sys.stderr)
    print(f"{prefix} [DEBUG] Reading initial cosmo arguments...", file=sys.stderr)
    command, initial_cosmo_arguments = read_msg(child_proc.stdout)
    if command != "COSMO_ARGS":
        print(
            f"{prefix} ERROR: Unexpected command received from child process: {command}",
            file=sys.stderr,
            flush=True,
        )
        MPI.COMM_WORLD.Abort(1)

    print(
        f"{prefix} [DEBUG] Initial cosmo arguments received successfully with command: {command}",
        file=sys.stderr,
    )


if len(param.output_Cl) > 0 and rank > 0:
    cosmo = classy.Class()
    input_params = {"output": "tCl, lCl, pCl", "lensing": "yes"}
    if (
        param.use_likelihood_filter
        and param.mcmc_sampler == "montepython"
        and param.sampling == "iterative"
    ):
        input_params.update({"l_max_scalars": initial_cosmo_arguments["l_max_scalars"]})
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
                {"l_max_scalars": initial_cosmo_arguments["l_max_scalars"]}
            )

    cosmo.set(input_params)
    cosmo.compute()
    cls = get_computed_cls(cosmo)
    global_ell = cls["ell"]


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

    # Iterate over each model
    while True:
        comm.send("I am done", dest=0)

        if param.use_likelihood_filter and param.sampling == "iterative":
            received_data = comm.recv(source=0)

            # Check if 'Done' message is received and break if so
            if type(received_data).__name__ == "str":
                break

            # Process the received data as dictionary if not 'Done'
            model = received_data["model"]
            loglkl_chains = received_data.get("loglkl_chains")

        else:
            model = comm.recv(source=0)

        # Check if 'Done' message was received in non-filter mode
        if type(model).__name__ == "str":
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

                        if child_proc.poll() is not None:  # Child has exited
                            print(
                                f"{prefix} ERROR: Child process died. Respawning immediately...",
                                file=sys.stderr,
                                flush=True,
                            )
                            child_proc = respawn_child(
                                rank, CONNECT_PATH, param_file, prefix, child_proc
                            )
                            recently_respawned = True
                            continue  # Skip the rest of this loop iteration and retry

                    except Exception as e:
                        print(
                            f"{prefix} ERROR: Unexpected error during check of child process {repr(e)}"
                            f"Attemping to respawn child process...",
                            file=sys.stderr,
                            flush=True,
                        )
                        child_proc = respawn_child(
                            rank, CONNECT_PATH, param_file, prefix, child_proc
                        )
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

                        write_msg("RUN_MODEL", child_proc.stdin, command_data)

                        command, result = read_msg(child_proc.stdout)

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

                    except Exception as e:
                        if str(e) == "timeout":
                            print(
                                f"{prefix} ERROR: Timeout while waiting for response from child process. Child may be unresponsive.\n",
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
                                f"Exception: {repr(e)}",
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

                            write_msg("RUN_LIKELIHOOD", child_proc.stdin, command_data)

                            # Wait for the child to finish the likelihood calculation
                            command, result = read_msg(child_proc.stdout)

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
        # Send the final "DONE" message to the child process
        def timeout_handler(sig, frame):
            raise RuntimeError("timeout")

        signal.signal(signal.SIGALRM, timeout_handler)

        signal.alarm(100)  # 10 seconds timeout

        command_data = {
            "params": None,
            "model": None,
            "global_ell": None,
            "soft_limit_bytes": None,
        }

        if child_proc.poll() is None:  # Check if child is still running
            try:
                write_msg("DONE", child_proc.stdin, data=command_data)

                # Read acknowledgment from child
                command, _ = read_msg(child_proc.stdout)

                if command == "DONE_ACK":
                    child_proc.stdin.close()  # Signal EOF, allowing child to exit
                else:
                    print(
                        f"{prefix} WARNING: Child process did not acknowledge 'DONE' message.",
                        flush=True,
                    )
            except (RuntimeError, BrokenPipeError) as e:
                print(
                    f"{prefix} WARNING: Issue while sending 'DONE' message: {repr(e)}",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"{prefix} ERROR: Unexpected failure sending 'DONE': {repr(e)}",
                    flush=True,
                )

        # Graceful shutdown with timeout handling
        try:
            child_proc.wait(timeout=10)  # Wait for up to 10 seconds
        except subprocess.TimeoutExpired:
            print(
                f"{prefix} WARNING: Child did not exit in time. Sending SIGTERM...",
                flush=True,
            )
            child_proc.terminate()  # SIGTERM

            try:
                child_proc.wait(timeout=5)  # Wait another 5 seconds
            except subprocess.TimeoutExpired:
                print(
                    f"{prefix} ERROR: Child still alive after SIGTERM. Sending SIGKILL...",
                    flush=True,
                )
                child_proc.kill()  # SIGKILL

        except Exception as e:
            print(
                f"{prefix} ERROR: Failed to wait for child process: {repr(e)}",
                flush=True,
            )
        finally:
            signal.alarm(0)


TIMEOUT_SECONDS = 600  # Timeout: 5 minutes

# -------------------- Worker Ranks (> 0) --------------------
if rank > 0:

    # Send done signal to master
    comm.send("DONE", dest=0, tag=999)

    # Non-blocking wait for shutdown signal from master
    start_wait = time.time()
    while not comm.Iprobe(source=0, tag=1000):
        time.sleep(0.1)  # Prevent CPU overuse
        if time.time() - start_wait > TIMEOUT_SECONDS + 10:  # Failsafe extra timeout
            print(f"[Worker {rank}] WARNING: Master unresponsive. Exiting.", flush=True)
            MPI.COMM_WORLD.Abort(0)

    go_to_finalize = comm.recv(source=0, tag=1000)

    if go_to_finalize == "FINALIZE":
        MPI.Finalize()
    else:
        print(
            f"[Worker {rank}] Received ABORT signal. Exiting immediately.", flush=True
        )
        MPI.COMM_WORLD.Abort(0)

# -------------------- Master Rank (0) --------------------
elif rank == 0:
    done_ranks = set()
    start_time = time.time()

    # Wait for all worker ranks to signal "DONE"
    while len(done_ranks) < N_slaves:
        status = MPI.Status()
        if comm.Iprobe(source=MPI.ANY_SOURCE, tag=999, status=status):
            msg = comm.recv(source=status.source, tag=999)
            done_ranks.add(status.source)

        # Check timeout
        if time.time() - start_time > TIMEOUT_SECONDS:
            print(
                f"[Master] Timeout reached! Some ranks are stuck. Aborting MPI.",
                flush=True,
            )
            # Save a file to directory called "calculations_completed" to indicate that the calculation was completed
            path_calc_complete = os.path.join(directory, "calc_completed.txt")
            with open(path_calc_complete, "w") as f:
                f.write("COMPLETED")

            MPI.COMM_WORLD.Abort(0)

    path_calc_complete = os.path.join(directory, "calc_completed.txt")
    with open(path_calc_complete, "w") as f:
        f.write("COMPLETED")

    for r in range(1, N_slaves + 1):
        comm.send("FINALIZE", dest=r, tag=1000)
        time.sleep(0.005)  # 5ms delay per message

    MPI.Finalize()

print(f"{prefix} Exiting normally.", flush=True)


time.sleep(1)

os._exit(0)  # Force exit because sys.exit(0) sometimes hangs

# sys.exit(0)
