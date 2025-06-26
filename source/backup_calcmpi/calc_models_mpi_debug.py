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

print(f"{prefix} [DEBUG] Python executable:", sys.executable)
print(f"{prefix} [DEBUG] Python path:", sys.path)
print(f"{prefix} [DEBUG] PYTHONPATH:", os.environ.get("PYTHONPATH"))


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

    MPI.COMM_WORLD.Set_errhandler(MPI.ERRORS_RETURN)
    MPI.COMM_SELF.Set_errhandler(MPI.ERRORS_RETURN)
    os.environ["UCX_LOG_LEVEL"] = "error"
    os.environ["UCX_HANDLE_ERRORS"] = "no"
    os.environ["UCX_ERROR_SIGNALS"] = "no"
    os.environ["UCX_TLS"] = "tcp"

    print(f"{prefix} [DEBUG]: Spawning child process...", flush=True, file=sys.stderr)
    child_comm = MPI.COMM_SELF.Spawn(
        sys.executable,
        args=[
            os.path.join(CONNECT_PATH, "source", "calc_models_mpi_child.py"),
            "--child-mode",
            param_file,
            CONNECT_PATH,
            str(rank),
        ],
        maxprocs=1,  # Spawn only one child process
    )
    print(
        f"{prefix} [DEBUG]: Successfully spawned child process.",
        flush=True,
        file=sys.stderr,
    )

    print(
        f"{prefix} [DEBUG]: Setting error handler for child process...",
        flush=True,
        file=sys.stderr,
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

        # Set soft limit as 90% of available memory
        soft_limit_bytes = int(0.95 * available_bytes)

        return soft_limit_bytes


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

                    try:
                        flag = (
                            child_comm.Get_size()
                        )  # Raises MPI.Exception if the child has died
                    except MPI.Exception:
                        print(
                            f"{prefix} WARNING: Child process died unexpectedly. most likely due to out-of-memory or other critical error from previous model.\n"
                            f"Respawning child process...",
                            file=sys.stderr,
                            flush=True,
                        )
                        child_comm = MPI.COMM_SELF.Spawn(
                            sys.executable,
                            args=[
                                os.path.join(
                                    CONNECT_PATH, "source", "calc_models_mpi_child.py"
                                ),
                                "--child-mode",
                                param_file,
                                CONNECT_PATH,
                                str(rank),
                            ],
                            maxprocs=1,
                        )
                        child_comm.Set_errhandler(MPI.ERRORS_RETURN)
                        print(
                            f"{prefix} [INFO] Successfully respawned child process.",
                            file=sys.stderr,
                            flush=True,
                        )

                    # ----------------- Send model and receive CLASS results -----------------#
                    print(
                        f"[DEBUG] {prefix} Step 1: Compute CLASS",
                        flush=True,
                        file=sys.stderr,
                    )

                    # send command to child to run the model (cosmo.compute() with CLASS)
                    try:
                        print(
                            f"[DEBUG] {prefix} Sending model to child process: {params}",
                            flush=True,
                            file=sys.stderr,
                        )
                        # Check if the child process is still alive
                        child_comm.send(("RUN_MODEL", command_data), dest=0, tag=0)
                        print(
                            f"[DEBUG] {prefix} Sent model to child, now waiting for results...",
                            flush=True,
                            file=sys.stderr,
                        )
                        # Wait for the child to finish computing the model, check every 0.1 seconds
                        while not child_comm.iprobe(source=0, tag=0):
                            time.sleep(0.1)

                        command, result = child_comm.recv(source=0, tag=0)
                        result = pickle.loads(result)
                        print(
                            f"[DEBUG] {prefix} Received result from child process: {result.keys()}",
                            flush=True,
                            file=sys.stderr,
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

                    print(
                        f"[DEBUG] {prefix} Received result from child process: ...",
                        flush=True,
                        file=sys.stderr,
                    )
                    print(
                        f"[DEBUG] {prefix} Received command: {command}",
                        flush=True,
                        file=sys.stderr,
                    )

                    # ---------------------- HANDLE CHILD PROCESS FAILURE ----------------------#

                    # If the result is None, it means the child process failed mid-execution
                    if command and result is None:
                        print(
                            f"{prefix} ERROR: Child process terminated unexpectedly. Suspected cause: Out-of-memory (limit set for child: {soft_limit/(1024**3)} GB) or other critical error.\n"
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

                    if command != "MODEL_DONE":
                        print(
                            f"{prefix} ERROR: Unexpected command received from child process: {command}",
                            file=sys.stderr,
                            flush=True,
                        )
                        success = False
                        break

                    print(
                        f"[DEBUG] {prefix} Checking finished_cosmo flag: {result.get('finished_cosmo', False)}",
                        flush=True,
                        file=sys.stderr,
                    )

                    # if result['finished_cosmo'] is False, the child process failed during cosmo.compute()
                    if not result.get("finished_cosmo", False):
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

                    print(
                        f"[DEBUG] {prefix} Setting the success flag to: {result.get('success', False)}",
                        flush=True,
                        file=sys.stderr,
                    )
                    success = result.get(
                        "success", False
                    )  # Would still be False if the likelihood filter is enabled or if the child process failed
                    # If the likelihood filter is enabled, the success flag will be set after the likelihood calculation step below
                    print(
                        f"[DEBUG] {prefix} Now moving to likelihood calculation step",
                        flush=True,
                        file=sys.stderr,
                    )
                    # ---------------------------- Compute Likelihood ----------------------------#
                    if param.use_likelihood_filter and param.sampling == "iterative":

                        command = None
                        result = None
                        soft_limit = get_soft_limit()
                        command_data["soft_limit_bytes"] = soft_limit

                        # Send the command to run the likelihood calculation
                        print(
                            f"{prefix} [DEBUG] Sending likelihood calculation command to child process: {params}",
                            flush=True,
                        )
                        try:
                            child_comm.send(
                                ("RUN_LIKELIHOOD", command_data), dest=0, tag=0
                            )

                            if not child_comm.iprobe(source=0, tag=0):
                                time.sleep(0.1)

                            command, result = child_comm.recv(source=0, tag=0)
                            result = pickle.loads(result)
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

                        print(
                            f"{prefix} [DEBUG] Received result from child process: {result.keys()}",
                            flush=True,
                        )

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

                        if command != "LIKELIHOOD_DONE":
                            print(
                                f"{prefix} ERROR: Unexpected command received from child process: {command}",
                                file=sys.stderr,
                                flush=True,
                            )
                            success = False
                            break

                        if not result.get("finished_lkl", False):
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
                    print(
                        f"[DEBUG] {prefix} checking success flag: {success}",
                        flush=True,
                        file=sys.stderr,
                    )
                    if result.get("success", False):
                        print(
                            f"[DEBUG] {prefix} Model {params} was successfully computed by the child process. Now extracting results...",
                            flush=True,
                            file=sys.stderr,
                        )
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
                        print(
                            f"[DEBUG] {prefix} Model {params} results extracted successfully, now writes data to files.",
                            flush=True,
                            file=sys.stderr,
                        )
                    else:
                        print(
                            f"{prefix} ERROR: Model {params} failed. Error: {result.get('error_msg', 'Unknown error.')}",
                            file=sys.stderr,
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
            print(
                f"[DEBUG] {prefix} writing data to files", flush=True, file=sys.stderr
            )
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
        try:
            print(
                f"{prefix} [DEBUG] Sending 'DONE' message to child...",
                flush=True,
                file=sys.stderr,
            )
            # placeholder command_data
            command_data = {
                "params": None,
                "model": None,
                "global_ell": None,
                "soft_limit_bytes": None,
            }
            pickled_data = pickle.dumps(command_data)
            child_comm.send(("DONE", pickled_data), dest=0, tag=0)

            # Wait for an acknowledgment if possible
            # Add some timeout to avoid infinite loop:
            wait_time = 20  # seconds
            start_time = time.time()

            while not child_comm.iprobe(source=0, tag=0):
                time.sleep(0.001)
                if (time.time() - start_time) > wait_time:
                    print(
                        f"{prefix} [DEBUG] Timeout reached while waiting for child acknowledgment. Proceeding with disconnect...",
                        flush=True,
                        file=sys.stderr,
                    )
                    break

            command, _ = child_comm.recv(source=0, tag=0)

            if command == "DONE_ACK":  # Child acknowledged shutdown
                print(
                    f"{prefix} [DEBUG] Child acknowledged shutdown.",
                    flush=True,
                    file=sys.stderr,
                )

            # Attempt to disconnect
            print(
                f"{prefix} [DEBUG] Attempting to disconnect MPI child communicator...",
                flush=True,
                file=sys.stderr,
            )
            child_comm.Disconnect()
            print(
                f"{prefix} [DEBUG] Successfully disconnected child communicator.",
                flush=True,
                file=sys.stderr,
            )

        except MPI.Exception as e:
            print(
                f"{prefix} WARNING: MPI communicator may have already been cleaned up. Error: {repr(e)}",
                file=sys.stderr,
                flush=True,
            )

MPI.Finalize()
sys.exit(0)
