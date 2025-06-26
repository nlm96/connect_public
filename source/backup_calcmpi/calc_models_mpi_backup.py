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

    import pickle
    import struct

    import pickle
    import tempfile
    import os

    def spawn_child(rank):
        """
        Launch the child process that runs calc_models_mpi_child.py --child-mode
        Returns the Popen object
        """

        cmd = [
            sys.executable,
            os.path.join(CONNECT_PATH, "source", "calc_models_mpi_child.py"),
            "--child-mode",
            param_file,
            CONNECT_PATH,
            str(rank),
        ]

        """
        cmd = [
            "srun", "--cpu-bind=core", "-n", "1",  # SLURM ensures child runs within allocated cores
            sys.executable,
            os.path.join(CONNECT_PATH, "source", "calc_models_mpi_child.py"),
            "--child-mode",
            param_file,
            CONNECT_PATH,
            str(rank),
        ]            
        """

        proc = subprocess.Popen(
            cmd,
            env=os.environ,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # Child writes errors directly to jobs .err file, not through the pipe
            bufsize=0,  # unbuffered
        )

        return proc

    child_proc = spawn_child(rank)

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

    def get_soft_limit():
        """Calculate the soft memory limit for the child process."""

        hard_limit_bytes = (
            int(os.environ.get("SLURM_MEM_PER_CPU", "7000")) * 1024**2
        )  # Read in bytes
        parent_usage_bytes = psutil.Process().memory_info().rss  # Already in bytes

        # Compute available memory
        available_bytes = max(0, hard_limit_bytes - parent_usage_bytes)

        # Set soft limit as 90% of available memory
        soft_limit_bytes = int(0.9 * available_bytes)

        return soft_limit_bytes

    def handle_error(e, rank, params, step, soft_limit=None, transfer_type="sent"):
        """
        Logs an error message for different exception types and returns (success) False.

        Args:
            e (Exception): The caught exception.
            rank (int): The MPI rank (worker ID).
            params (dict): The parameters of the failed model.
            step (str): The step where the error occurred ("cosmo" or "likelihood").
            soft_limit (int, optional): The memory limit in bytes.
            transfer_type (str): "sent" if the error occurred while sending data, "received" if it occurred while receiving data.

        Returns:
            bool: Always returns False to indicate failure.
        """

        # Map step names to descriptive labels
        step_description = {
            "cosmo": "CLASS computation",
            "likelihood": "Likelihood evaluation",
        }.get(step, "Unknown step")

        # Determine whether the failure happened during sending or receiving
        step_io = "Sending" if transfer_type == "sent" else "Receiving"

        if isinstance(e, BrokenPipeError):
            error_type = f"{prefix} BrokenPipeError - Child process terminated unexpectedly during {step_io}."
            guess_cause = (
                f"Possible Causes:\n"
                f"  - Child process was killed (e.g., out of memory, segfault, SLURM preemption).\n"
                f"  - Parent tried to write to a closed pipe.\n"
                f"  - Sudden network failure (unlikely).\n"
            )
        elif isinstance(e, EOFError):
            error_type = f"{prefix} EOFError - Child process unexpectedly closed connection during {step_io}."
            guess_cause = (
                f"Possible Causes:\n"
                f"  - Child process crashed (e.g., memory exceeded, segmentation fault).\n"
                f"  - Child process was forcefully terminated by SLURM or OS.\n"
            )
        elif isinstance(e, MemoryError):
            error_type = f"{prefix} MemoryError - Process exceeded memory limit during {step_description}."
            guess_cause = (
                f"Possible Causes:\n"
                f"  - The model required more memory than allocated.\n"
                f"  - CLASS or another library caused excessive memory usage.\n"
                f"  - System OOM-killed the process.\n"
            )
        elif isinstance(e, TimeoutError):
            error_type = f"{prefix} TimeoutError - Child process took too long during {step_description}."
            guess_cause = (
                f"Possible Causes:\n"
                f"  - Computational complexity of the model exceeded the timeout.\n"
                f"  - Child process entered an infinite loop.\n"
                f"  - Child process was stuck in a blocking operation.\n"
            )

        else:
            error_type = f"{prefix} Unexpected {type(e).__name__} occurred during {step_io} of {step_description}."
            guess_cause = "See traceback below for more details."

        # Print structured error message
        print("=" * 80, file=sys.stdout)
        print(f"{prefix} ERROR: {error_type}", file=sys.stdout, flush=True)
        print(
            f"{prefix} Failed Step: {step_description}",
            file=sys.stdout,
            flush=True,
        )
        print(
            f"{prefix} Step I/O: {step_io} data with the child process",
            file=sys.stdout,
            flush=True,
        )
        print(
            f"{prefix} Model parameters: {params}",
            file=sys.stdout,
            flush=True,
        )

        if soft_limit is not None:
            print(
                f"{prefix} Memory Limit: {soft_limit} bytes",
                file=sys.stdout,
                flush=True,
            )

        print(f"{prefix} {guess_cause}", file=sys.stdout, flush=True)

        print(
            f"{prefix} Skipping to the next model...",
            file=sys.stdout,
            flush=True,
        )

        # Print traceback only for unexpected errors
        print(
            f"{prefix} Exception Details: {repr(e)}",
            file=sys.stdout,
            flush=True,
        )
        print(
            f"{prefix} Traceback:\n{traceback.format_exc()}",
            file=sys.stdout,
            flush=True,
        )

        print("=" * 80, file=sys.stdout)

        return False  # Always return False to indicate failure


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
    for r in range(1, N_slaves + 1):
        sleep_dict[r] = 1

    data_idx = 0
    while len(data) > data_idx:
        r = next(get_slave)
        if comm.iprobe(r):
            useless_info = comm.recv(source=r)

            if param.use_likelihood_filter and param.sampling == "iterative":
                DATA = {
                    "model": data[data_idx],
                    "loglkl_chains": loglkl_chains[data_idx],
                }
                comm.send(DATA, dest=r)
            else:
                comm.send(data[data_idx], dest=r)
            data_idx += 1
            sleep_dict[r] = 1
        else:
            sleep_dict[r] = 0

        if all(value == 0 for value in sleep_dict.values()):
            time.sleep(sleep_long)
        else:
            time.sleep(sleep_short)

    for r in range(1, N_slaves + 1):
        comm.send("Done", dest=r)


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

        signal.alarm(200)  # CLASS computations must not take longer than 200 seconds
        try:

            if param.memory_safe:

                """
                The main loop for rank>0 that:
                - spawns one child
                - sends models to the child and let child compute CLASS and likelihood
                - recieves the results from the child
                - if child dies, respawn it.
                -
                """

                result = None
                success = False

                while True:

                    # ---------------------------- Compute CLASS ----------------------------#

                    soft_limit = get_soft_limit()

                    command_data = {
                        "params": params,
                        "model": model,
                        "global_ell": global_ell,
                        "soft_limit_bytes": soft_limit,
                    }

                    # Check if the child died and respawn if needed
                    if child_proc.poll() is not None:
                        print(
                            f"{prefix} WARNING: Child process was terminated by the previous model. Likely cause: Out-of-memory or fatal error.",
                            file=sys.stdout,
                            flush=True,
                        )
                        print(
                            f"{prefix} Respawning child process...",
                            file=sys.stdout,
                            flush=True,
                        )
                        child_proc = spawn_child(rank)

                    # send command to child to run the model (cosmo.compute() with CLASS)
                    try:
                        write_one_message(
                            command="RUN_MODEL",
                            output_stream=child_proc.stdin,
                            obj=command_data,
                        )
                    except Exception as e:
                        success = handle_error(
                            e, rank, params, "cosmo", soft_limit, transfer_type="sent"
                        )
                        break  # Move to next model

                    # Receive results step 1; after cosmo.compute(). Check if the child process failed
                    try:
                        command, result = read_one_message(child_proc.stdout)
                    except Exception as e:
                        success = handle_error(
                            e,
                            rank,
                            params,
                            "cosmo",
                            soft_limit,
                            transfer_type="received",
                        )
                        break  # Move to next model
                    # If the result is None, it means the child process failed mid-execution

                    if command and result is None:
                        print(
                            f"{prefix} ERROR: Child process terminated unexpectedly. Suspected cause: Out-of-memory (limit: {soft_limit} bytes) or other critical error.\n"
                            f"This happened during the CLASS computation step for the model: {params}"
                            f"skipping to the next model...",
                            file=sys.stdout,
                            flush=True,
                        )
                        print(f"{prefix} Model: {params}", file=sys.stdout, flush=True)

                        success = False
                        break

                    if command != "MODEL_DONE":
                        print(
                            f"{prefix} ERROR: Unexpected command received from child process: {command}",
                            file=sys.stdout,
                            flush=True,
                        )
                        success = False
                        break

                    # if result['finished_cosmo'] is False, the child process failed during cosmo.compute()
                    if not result.get("finished_cosmo", False):
                        print(
                            f"{prefix} ERROR: Child process failed during CLASS computation\n"
                            f"Possibly due to out-of-memory or other critical error.\n"
                            f"Model: {params}\n"
                            f"Error message from child:\n {result.get('error_msg', 'Unknown error.')}"
                            f"skipping to the next model...",
                            file=sys.stdout,
                            flush=True,
                        )
                        success = False
                        break

                    if result.get("error_msg", None) is not None:
                        print(
                            f"{prefix} ERROR: Child process failed during CLASS computation\n"
                            f"Model: {params}\n"
                            f"Error message from child:\n {result.get('error_msg', 'Unknown error.')}"
                            f"skipping to the next model...",
                            file=sys.stdout,
                            flush=True,
                        )
                        success = False
                        break

                    success = result.get(
                        "success", False
                    )  # Would still be False if the likelihood filter is enabled or if the child process failed
                    # If the likelihood filter is enabled, the success flag will be set after the likelihood calculation step below

                    # ---------------------------- Compute Likelihood ----------------------------#
                    if param.use_likelihood_filter and param.sampling == "iterative":

                        command = None
                        result = None
                        soft_limit = get_soft_limit()
                        command_data["soft_limit_bytes"] = soft_limit

                        # Send the command to run the likelihood calculation
                        print(
                            f"{prefix} Sending likelihood calculation command to child process: {params}",
                            flush=True,
                        )
                        try:
                            write_one_message(
                                command="RUN_LIKELIHOOD",
                                output_stream=child_proc.stdin,
                                obj=command_data,
                            )
                        except Exception as e:
                            success = handle_error(
                                e,
                                rank,
                                params,
                                "likelihood",
                                soft_limit,
                                transfer_type="sent",
                            )
                            break  # Move to next model

                        # Receive results step 2; after likelihood calculation
                        # This is the final results after both steps, it contains both computed CLASS and likelihood
                        try:
                            command, result = read_one_message(child_proc.stdout)
                        except Exception as e:
                            success = handle_error(
                                e,
                                rank,
                                params,
                                "likelihood",
                                soft_limit,
                                transfer_type="received",
                            )
                            break

                        print(
                            f"{prefix} Received result from child process: {result}",
                            flush=True,
                        )

                        # If the result is None, it means the child process failed mid-execution
                        if command and result is None:
                            print(
                                f"{prefix} ERROR: Child process terminated unexpectedly. Suspected cause: Out-of-memory (limit: {soft_limit} bytes) or other critical error.\n"
                                f"This happened during the likelihood calculation step for the model: {params}\n"
                                f"skipping to the next model...",
                                file=sys.stdout,
                                flush=True,
                            )
                            print(
                                f"{prefix} Model: {params}", file=sys.stdout, flush=True
                            )
                            success = False
                            break

                        if command != "LIKELIHOOD_DONE":
                            print(
                                f"{prefix} ERROR: Unexpected command received from child process: {command}",
                                file=sys.stdout,
                                flush=True,
                            )
                            success = False
                            break

                        if not result.get("finished_lkl", False):
                            print(
                                f"{prefix} ERROR: Child process failed during likelihood calculation\n"
                                f"Model: {params}\n"
                                f"Error: {result.get('error_msg', 'Unknown error.')}\n"
                                f"skipping to the next model...",
                                file=sys.stdout,
                                flush=True,
                            )
                            success = False
                            break

                        if result.get("error_msg", None) is not None:
                            print(
                                f"{prefix} ERROR: Child process failed during likelihood calculation\n"
                                f"Model: {params}\n"
                                f"Error message from child:\n {result.get('error_msg', 'Unknown error.')}"
                                f"skipping to the next model...",
                                file=sys.stdout,
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
                            file=sys.stdout,
                            flush=True,
                        )
                        success = False

                    break  # Finish loop after processing one model

            else:  # Not memory safe. Initial implementation

                cosmo = classy.Class()
                cosmo.set(params)
                cosmo.compute()

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
        command_data = {"params": None, "model": None, "global_ell": None}
        if child_proc.poll() is None:  # Check if child is still running
            try:
                write_one_message("DONE", child_proc.stdin, obj=command_data)
                child_proc.stdin.close()  # Signal EOF
                print(
                    f"{prefix} [DEBUG] Sent 'DONE' message to child process.",
                    flush=True,
                )
            except BrokenPipeError:
                print(
                    f"{prefix} WARNING: Child process already closed. Skipping 'DONE' message.",
                    file=sys.stdout,
                    flush=True,
                )
            except Exception as e:
                print(
                    f"{prefix} ERROR: Failed to send 'DONE' message: {repr(e)}",
                    file=sys.stdout,
                    flush=True,
                )

        if child_proc.poll() is None:  # Check if child is still running
            try:
                child_proc.terminate()  # Ensure child is terminated
                child_proc.wait()  # Wait for clean exit
                print(
                    f"{prefix} [DEBUG] Terminated child process.",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"{prefix} ERROR: Failed to properly close child process: {repr(e)}",
                    file=sys.stdout,
                    flush=True,
                )

MPI.Finalize()
sys.exit(0)
