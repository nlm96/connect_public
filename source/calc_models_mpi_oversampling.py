import os
import sys
import time
import itertools
import signal
import traceback
import copy

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
                    f"to accommodate the likelihood calculation used by the likelihood filter."
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

    # ----------- Get oversampling-filtered data for research purposes -----------
    if param.compute_oversampled_likelihoods and iteration > 0:
        full_chains = mcmc.import_points_from_chains(iteration, import_full_chains=True)
        loglkl_full_chains = mcmc.import_loglkl_from_chains(
            iteration, import_full_chains=True
        )

        import pandas as pd

        data_df = pd.DataFrame(data, columns=param_names)
        full_chains_df = pd.DataFrame(full_chains, columns=param_names)
        loglkl_full_chains_df = pd.DataFrame(loglkl_full_chains, columns=["loglkl"])

        from source.tools import compare_dataframes

        oversampled_data, oversampled_fake_loglkl = compare_dataframes(
            full_chains_df,
            data_df,
            df_likelihood=loglkl_full_chains_df,
            df_likelihood2=None,
            comparison_type="new",  # <-- THIS is the key
            verbose=1,
            compare_context={
                "context": "df1_minus_df2",
                "df1": "df1",
                "df2": "df2",
                "msg1": "",
                "msg2": "",
            },
        )

    sleep_short = 0.0001
    sleep_long = 0.1
    sleep_dict = {}
    for r in range(1, N_slaves + 1):
        sleep_dict[r] = 1

    data_idx = 0

    if param.compute_oversampled_likelihoods and iteration > 0:
        data_length = len(data) + len(oversampled_data)
        models_length = len(data)
        oversampled_data_length = len(oversampled_data)
    else:
        data_length = len(data)

    while data_length > data_idx:
        r = next(get_slave)
        if comm.iprobe(r):
            useless_info = comm.recv(source=r)

            if param.use_likelihood_filter and param.sampling == "iterative":

                if not param.compute_oversampled_likelihoods:
                    DATA = {
                        "model": data[data_idx],
                        "loglkl_chains": loglkl_chains[data_idx],
                    }
                else:
                    if data_idx < models_length:
                        DATA = {
                            "model": data[data_idx],
                            "loglkl_chains": loglkl_chains[data_idx],
                            "oversampling_flag": False,
                        }
                    else:
                        DATA = {
                            "model": oversampled_data[data_idx - models_length],
                            "loglkl_chains": oversampled_fake_loglkl[
                                data_idx - models_length
                            ],
                            "oversampling_flag": True,
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

    if param.compute_oversampled_likelihoods and iteration > 0:
        oversampled_data_dir = os.path.join(
            directory, f"oversampled_data/oversampled_data_{rank}.txt"
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

    if param.compute_oversampled_likelihoods and iteration > 0:

        oversampling_header = (
            "# "
            + "\t".join(
                filter(
                    None,
                    [
                        loglkl_header.strip()[1:].strip(),
                        param_header.strip()[1:].strip(),
                        derived_header.strip()[1:].strip(),
                    ],
                )
            )
            + "\n"
        )

        with open(oversampled_data_dir, "w") as f:
            f.write(param_header)

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
                position = {}
                for i, par_name in enumerate(param_names):
                    position[par_name] = [model[i]]
                # set nuisance parameters to fixed values based the input parameter nuisance_params_lkl
                for nuisance_name, nuisance_value in param.nuisance_params_lkl.items():
                    position[nuisance_name] = [nuisance_value]
                # compute likelihood
                loglkl_true = likelihood_calc.loglkl(position, cosmo)

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

MPI.Finalize()
sys.exit(0)
