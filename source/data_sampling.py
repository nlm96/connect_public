import os
import subprocess as sp
import fileinput
import shutil
import sys
import datetime
import time
import pandas as pd
import re

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np

from .join_output import CreateSingleDataFile
from .train_network import Training
from .default_module import Parameters
from .tools import (
    create_output_folders,
    join_data_files,
    combine_iterations_data,
    compare_dataframes,
)
from .lkl_filter_module.likelihood_filter import LikelihoodFilter


class Sampling:
    def __init__(self, param_file, CONNECT_PATH):
        self.param_file = param_file
        self.param = Parameters(param_file)
        self.CONNECT_PATH = CONNECT_PATH
        self.N_tasks, self.N_cpus_per_task = np.int64(
            sp.run(["./source/shell_scripts/number_of_tasks.sh"], stdout=sp.PIPE)
            .stdout.decode("utf-8")
            .split(",")
        )
        if self.param.sampling in ["lhc", "hypersphere", "pickle"]:
            self.data_path = f"data/{self.param.jobname}/N-{self.param.N}"
        elif self.param.sampling == "iterative":
            self.data_path = f"data/{self.param.jobname}"

        self.consecutive_bad_states_count = 0
        self.final_iteration = False

    def create_hypersphere_data(self):
        self.copy_param_file()
        self.call_calc_models(sampling="hypersphere")

    def create_lhc_data(self):
        self.copy_param_file()
        self.call_calc_models(sampling="lhc")

    def create_pickle_data(self):
        self.copy_param_file()
        self.call_calc_models(sampling="pickle")

    def create_iterative_data(self):
        print(f"{datetime.datetime.now()}", flush=True)
        exec(
            f"from source.mcmc_samplers.{self.param.mcmc_sampler} import {self.param.mcmc_sampler}"
        )
        _locals = {}
        exec(
            f"mcmc = {self.param.mcmc_sampler}(self.param, self.CONNECT_PATH)",
            locals(),
            _locals,
        )
        mcmc = _locals["mcmc"]

        annealing = isinstance(self.param.temperature, list)
        if annealing:
            temp_len = len(self.param.temperature)
            temperature = self.param.temperature[0]
        else:
            temp_len = 0
            temperature = self.param.temperature

        mcmc.check_version()
        i_converged = 10000
        if self.param.resume_iterations:
            resume_from_initial = False
            try:
                i = max(
                    [
                        int(f.split("number_")[-1])
                        for f in os.listdir(self.data_path)
                        if f.startswith("number")
                    ]
                )
            except:
                if os.path.isfile(
                    os.path.join(self.data_path, f"N-{self.param.N}/model_params.txt")
                ):
                    i = 1  # It will set i=0 below and resume from initial sampling
                else:
                    raise NotImplementedError(
                        "You do not have any computed iterations to resume from. Please run again without resume_iterations=True"
                    )

            # check if the subfolder lkl_calc/montepython exists and delete it and its content if it does
            path_lkl_calc = os.path.join("data", self.param.jobname, "lkl_calc")
            if os.path.isdir(os.path.join(path_lkl_calc, "montepython")):
                shutil.rmtree(os.path.join(path_lkl_calc, "montepython"))
            data_is_computed = os.path.isfile(
                os.path.join(self.data_path, f"number_{i}", "training.log")
            )

            if self.param.use_likelihood_filter:
                try:
                    log_path = os.path.join(self.data_path, "output.log")
                    self.consecutive_bad_states_count = (
                        self.recover_bad_state_count_from_log(log_path)
                    )
                except Exception as e:
                    print(f"Error reading log file: {e}", flush=True)

            if not data_is_computed:
                # Check if model_params.txt exists
                if os.path.isfile(
                    os.path.join(self.data_path, f"number_{i}", "model_params.txt")
                ):
                    data_is_computed = self.check_calc_completed(
                        sampling="iterative",
                        skip_calc_completed_file_check=True,
                        check_rank_files=False,
                        success_threshold=0.9,
                    )

            if data_is_computed:
                print("Resuming iterative sampling", flush=True)
                print(f"Retraining neural network from iteration {i}", flush=True)
            else:
                i -= 1
                if i == 0:
                    if os.path.isfile(
                        os.path.join(self.data_path, f"N-{self.param.N}/training.log")
                    ):
                        resume_from_initial = True
                        os.system(f"rm -rf {self.data_path}/number_{i+1}")
                        print("Resuming iterative sampling", flush=True)
                        print(
                            f"Retraining neural network from initial sampling",
                            flush=True,
                        )
                    else:

                        if os.path.isfile(
                            os.path.join(
                                self.data_path, f"N-{self.param.N}/model_params.txt"
                            )
                        ):
                            if self.check_calc_completed(
                                sampling="lhc",
                                skip_calc_completed_file_check=True,
                                check_rank_files=False,
                                success_threshold=0.9,
                            ):

                                resume_from_initial = True
                                os.system(f"rm -rf {self.data_path}/number_{i+1}")
                                print("Resuming iterative sampling", flush=True)
                                print(
                                    f"Retraining neural network from initial sampling",
                                    flush=True,
                                )
                            else:
                                raise NotImplementedError(
                                    "You do not have any computed iterations to resume from. Please run again without resume_iterations=True"
                                )
                        else:
                            raise NotImplementedError(
                                "You do not have any computed iterations to resume from. Please run again without resume_iterations=True"
                            )
                else:
                    os.system(f"rm -rf {self.data_path}/number_{i+1}")
                    print("Resuming iterative sampling", flush=True)
                    print(f"Retraining neural network from iteration {i}", flush=True)

            if resume_from_initial:
                model = self.train_neural_network(
                    sampling=self.param.initial_sampling,
                    output_file=os.path.join(
                        self.data_path, f"N-{self.param.N}/training.log"
                    ),
                )
            else:
                model = self.train_neural_network(
                    sampling="iterative",
                    output_file=os.path.join(
                        self.data_path, f"number_{i}/training.log"
                    ),
                    mcmc=mcmc,
                )
            if not os.path.isdir(os.path.join(self.data_path, "compare_iterations")):
                os.system(f"mkdir {self.data_path}/compare_iterations")
                mcmc.Gelman_Rubin_log_ini()
            with open(os.path.join(self.data_path, "output.log"), "r") as f:
                for line in f:
                    if (
                        "will be the last with same temperature since convergence in"
                        in line
                    ):
                        i_converged = int(line.split(" ")[1])
                        break
            i += 1
        else:
            self.copy_param_file()
            if not os.path.isdir(self.data_path):
                os.system(f"mkdir {self.data_path}")
            os.system(f"rm -rf {self.data_path}/compare_iterations")
            os.system(f"mkdir {self.data_path}/compare_iterations")
            i = 1
            if self.param.initial_model == None:
                os.system(f"rm -rf {self.CONNECT_PATH}/{self.data_path}/number_*")
                mcmc.Gelman_Rubin_log_ini()
                print("No initial model given", flush=True)
                print(f"Calculating {self.param.N} initial CLASS models", flush=True)
                self.call_calc_models(sampling=self.param.initial_sampling)
                if self.param.mcmc_sampler == "montepython":
                    path_lkl_calc = os.path.join("data", self.param.jobname, "lkl_calc")
                    self.cleanup_montepython_folders()
                    if os.path.isdir(os.path.join(path_lkl_calc, "montepython")):
                        shutil.move(
                            os.path.join(path_lkl_calc, "montepython"),
                            os.path.join(path_lkl_calc, f"montepython_initial"),
                        )

                # print("Training neural network", flush=True)
                model = self.train_neural_network(
                    sampling=self.param.initial_sampling,
                    output_file=os.path.join(
                        self.data_path, f"N-{self.param.N}/training.log"
                    ),
                )
            else:
                model = self.param.initial_model
            print(f"Initial model is {model}", flush=True)

        kill_iteration = False
        while True:
            if i > i_converged and annealing:
                temperature = self.param.temperature[i - i_converged]
            print(f"\n\n{datetime.datetime.now()}", flush=True)
            print(f"Beginning iteration no. {i}", flush=True)
            print(f"Temperature is now {temperature}", flush=True)
            print(f"Running MCMC sampling no. {i}...", flush=True)
            mcmc.temperature = temperature
            mcmc.run_mcmc_sampling(model, i)
            print(
                f"MCMC sampling stopped since R-1 less than {self.param.mcmc_tol} has been reached.",
                flush=True,
            )

            N_acc = mcmc.get_number_of_accepted_steps(i)
            print(f"Number of accepted steps: {N_acc}", flush=True)
            if i == 1 and not self.param.keep_first_iteration:
                N_keep = 5000
            elif (
                i == 1
                and self.param.use_likelihood_filter
                and self.param.keep_first_iteration
            ):
                N_keep = self.param.N_first_iteration  # default is 5000
            else:
                N_keep = self.param.N_max_points
            N_keep = mcmc.filter_chains(N_keep, i)
            print(
                f"Keeping only last {N_keep} of the accepted Markovian steps",
                flush=True,
            )
            print("Comparing latest iterations...", flush=True)
            if i > 1 and not kill_iteration:
                kill_iteration = mcmc.compare_iterations(i)

            #  Apply overlap filter to discard points that are too close to existing training data
            if i > int(not self.param.keep_first_iteration) + 1 and i <= i_converged:
                N_accepted = mcmc.discard_oversampled_points(i)
                N_in_data_set = mcmc.get_number_of_data_points(i - 1) + N_accepted
                print(f"Accepted {N_accepted} points out of {N_keep}", flush=True)
            elif i == 1 and self.param.keep_initial_data:
                N_accepted = mcmc.discard_oversampled_points(i)
                N_in_data_set = mcmc.get_number_of_data_points(i - 1) + N_accepted
                print(f"Accepted {N_accepted} points out of {N_keep}", flush=True)
            else:
                N_accepted = N_keep
                N_in_data_set = 0

            if kill_iteration and N_accepted < 0.1 * N_in_data_set:
                i_converged = i
                if annealing:
                    print(
                        f"Iteration {i} will be the last with same temperature since convergence in",
                        flush=True,
                    )
                    print(
                        f"data has been reached and less than 10% of the data was added in this iteration.",
                        flush=True,
                    )
                    print(
                        f"Iterations will now continue according to temperature schedule.",
                        flush=True,
                    )
                else:
                    print(
                        f"Iteration {i} will be the last since convergence in data has been reached",
                        flush=True,
                    )
                    print(
                        f"and less than 10% of the data was added in this iteration.",
                        flush=True,
                    )
                    self.final_iteration = True
            else:
                kill_iteration = False

            if i == i_converged + temp_len - 1 and annealing:
                self.final_iteration = True
                print(f"Final iteration {i} reached", flush=True)

            print(
                f"{datetime.datetime.now()}\n" f"Calculating {N_accepted} CLASS models",
                flush=True,
            )
            create_output_folders(self.param, iter_num=i, reset=False)
            self.call_calc_models(sampling="iterative")
            if self.param.mcmc_sampler == "montepython":
                path_lkl_calc = os.path.join("data", self.param.jobname, "lkl_calc")
                self.cleanup_montepython_folders()
                if os.path.isdir(os.path.join(path_lkl_calc, "montepython")):
                    shutil.move(
                        os.path.join(path_lkl_calc, "montepython"),
                        os.path.join(path_lkl_calc, f"montepython_{i}"),
                    )
            join_data_files(self.param)

            if self.param.use_likelihood_filter:
                # Logic for combining data with likelihood filtering
                if i == 1:
                    if self.param.keep_initial_data:
                        combine_iterations_data(self.param, i)
                        print(
                            f"Copied initial data from data/{self.param.jobname}/N-{self.param.N} into data/{self.param.jobname}/number_{i}",
                            flush=True,
                        )
                    else:
                        # print(f"Skipping combining initial data and iteration {i} as keep_initial_data=False.", flush=True)
                        pass
                elif i > 1:
                    if self.param.keep_first_iteration or i > 2:
                        combine_iterations_data(self.param, i)
                        print(
                            f"Copied data from data/{self.param.jobname}/number_{i-1} into data/{self.param.jobname}/number_{i}",
                            flush=True,
                        )
                    else:
                        # print(f"Skipping combining data for iteration {i} as keep_first_iteration=False and keep_initial_data=False.", flush=True)
                        pass
            else:
                # Logic for standard iterative sampling without likelihood filtering
                if (
                    i > int(not self.param.keep_first_iteration) + 1
                    and i <= i_converged
                ):
                    combine_iterations_data(self.param, i)
                    print(
                        f"Copied data from data/{self.param.jobname}/number_{i-1} into data/{self.param.jobname}/number_{i}",
                        flush=True,
                    )

            model = self.train_neural_network(
                sampling="iterative",
                output_file=os.path.join(self.data_path, f"number_{i}/training.log"),
                mcmc=mcmc,
            )

            if (
                self.consecutive_bad_states_count
                >= self.param.max_consecutive_bad_states
            ):
                if not annealing:
                    print(
                        f"⚠️  WARNING: Maximum consecutive 'bad states' reached. Exiting the iterative sampling process.",
                        flush=True,
                    )
                    print(f"Final model is {model}", flush=True)
                    print(
                        f"{datetime.datetime.now()}",
                        flush=True,
                    )
                    break
                else:
                    if i == i_converged + temp_len - 1:
                        print(
                            f"⚠️  WARNING: Maximum consecutive 'bad states' reached after annealing. Exiting the iterative sampling process.",
                            flush=True,
                        )
                        print(f"Final model is {model}", flush=True)
                        print(
                            f"{datetime.datetime.now()}",
                            flush=True,
                        )
                        break
                    else:
                        print(
                            f"⚠️  WARNING: Maximum consecutive 'bad states' reached. Continuing annealing with next temperature.",
                            flush=True,
                        )
                        self.consecutive_bad_states_count = 0
                        i += 1
                        continue

            if kill_iteration and N_accepted < 0.1 * N_in_data_set and not annealing:
                print(f"Final model is {model}", flush=True)
                print(
                    f"{datetime.datetime.now()}",
                    flush=True,
                )
                break
            if i == i_converged + temp_len - 1 and annealing:
                print(f"Final model is {model}", flush=True)
                print(
                    f"{datetime.datetime.now()}",
                    flush=True,
                )
                break
            else:
                print(f"New model is {model}", flush=True)
                i += 1

    def copy_param_file(self):
        os.system(
            f"cp {self.CONNECT_PATH+'/'+self.param_file} {self.CONNECT_PATH+'/'+self.data_path+'/log_connect.param'}"
        )
        with open(
            self.CONNECT_PATH + "/" + self.data_path + "/log_connect.param", "a+"
        ) as f:
            jobname_specified = False
            for line in f:
                if line.startswith("jobname"):
                    jobname_specified = True
            if not jobname_specified:
                f.write(f"\njobname = '{self.param.jobname}'")
        self.param.param_file = (
            self.CONNECT_PATH + "/" + self.data_path + "/log_connect.param"
        )

    def call_calc_models(self, sampling="lhc"):

        unique_id = str(int(time.time()))

        os.environ["export OMP_NUM_THREADS"] = str({self.N_cpus_per_task})
        os.environ["PMIX_MCA_gds"] = "hash"

        if not self.param.memory_safe:
            sp.check_call(
                f"mpirun -np {self.N_tasks - 1} python {self.CONNECT_PATH}/source/calc_models_mpi.py {self.param.param_file} {self.CONNECT_PATH} {sampling}".split()
            )

        else:
            try:

                # Copy current environment
                env2 = os.environ.copy()

                # Set ONLY the variables for this mpirun subprocess
                # Infiniband is unstable with dynamic spawning of MPI child processes (not always, but sometimes). Use TCP instead:
                env2["OMPI_MCA_btl"] = "^openib"
                env2["UCX_TLS"] = "tcp,sm,self"
                # Set the timeout for PMIx exchange and server to 500ms to better handle dynamic spawning
                env2["OMPI_MCA_pmix_base_exchange_timeout"] = "500"
                env2["OMPI_MCA_pmix_server_max_wait"] = "500"

                # Construct the command
                cmd = f"mpirun --mca orte_abort_on_non_zero_status 0 --mca orte_allowed_exit_without_sync 1 -np {self.N_tasks - 1} python {self.CONNECT_PATH}/source/calc_models_mpi_memsafe.py {self.param.param_file} {self.CONNECT_PATH} {sampling} {unique_id}"

                # Run it with the modified env
                sp.check_call(cmd.split(), env=env2)

                if not self.check_calc_completed(sampling):
                    print(
                        "ERROR: Calculations did not complete successfully. Exiting...",
                        flush=True,
                        file=sys.stderr,
                    )
                    sys.exit(1)

            except sp.CalledProcessError as e:
                if self.check_calc_completed(sampling):
                    print(
                        f"WARNING: mpirun failed with exit code {e.returncode}, but calculations completed successfully. Checking for lingering MPI processes...",
                        flush=True,
                        file=sys.stderr,
                    )
                    # Give MPI abort some time to finish terminating processes
                    time.sleep(20)

                    self.cleanup_mpi_processes(unique_id)
                    print(
                        "Check complete, continuing...",
                        flush=True,
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"ERROR: mpirun failed with exit code {e.returncode}, and calculations did not complete. Exiting...",
                        flush=True,
                        file=sys.stderr,
                    )

                    sys.exit(
                        e.returncode
                    )  # **Ensure script exits with the correct failure code**
            except Exception as e:
                if self.check_calc_completed(sampling):
                    print(
                        f"WARNING: mpirun failed with exception {repr(e)}, but calculations completed successfully. Checking for lingering MPI processes...",
                        flush=True,
                        file=sys.stderr,
                    )
                    time.sleep(20)
                    self.cleanup_mpi_processes(unique_id)
                    print(
                        "Check complete, continuing...",
                        flush=True,
                        file=sys.stderr,
                    )

                else:
                    print(
                        f"ERROR: mpirun failed with exception {repr(e)}, and calculations did not complete. Exiting...",
                        flush=True,
                        file=sys.stderr,
                    )

                    sys.exit(1)

        os.environ["export OMP_NUM_THREADS"] = "1"

    def train_neural_network(
        self,
        sampling="lhc",
        output_file=None,
        mcmc=None,
    ):
        if sampling in ["lhc", "hypersphere", "pickle"]:
            folder = f"N-{self.param.N}"
            iter_num = 0
        elif sampling == "iterative":
            i = max(
                [
                    int(f.split("number_")[-1])
                    for f in os.listdir(os.path.join(self.CONNECT_PATH, self.data_path))
                    if f.startswith("number")
                ]
            )
            folder = f"number_{i}"
            iter_num = i
        if not os.path.isfile(
            os.path.join(self.CONNECT_PATH, self.data_path, folder, "model_params.txt")
        ):
            CSDF = CreateSingleDataFile(self.param, self.CONNECT_PATH)
            CSDF.join()
        if self.param.use_likelihood_filter and self.param.sampling == "iterative":
            if not self.param.strict_filtering_final_iteration:
                strict_filtering = False
            else:
                strict_filtering = self.final_iteration

            if not self.param.keep_initial_data and iter_num == 0:
                print(
                    "Skipping likelihood filter for first iteration as keep_initial_data=False",
                    flush=True,
                )
            elif not self.param.keep_first_iteration and iter_num == 1:
                print(
                    "Skipping likelihood filter for first iteration as keep_first_iteration=False",
                    flush=True,
                )
            else:
                print("Applying likelihood filter to data", flush=True)
                Likelihood_Filter = LikelihoodFilter(
                    param=self.param,
                    iter_num=iter_num,
                    CONNECT_PATH=self.CONNECT_PATH,
                    strict_filtering=strict_filtering,
                )
                Likelihood_Filter.run()
                if iter_num > 0:
                    self.check_likelihood_filter_health(
                        i=iter_num,
                        mcmc=mcmc,
                    )
                    if (
                        self.param.strict_filtering_final_iteration
                        and self.consecutive_bad_states_count
                        >= self.param.max_consecutive_bad_states
                    ):

                        strict_filtering = True
                        Likelihood_Filter = LikelihoodFilter(
                            param=self.param,
                            iter_num=iter_num,
                            CONNECT_PATH=self.CONNECT_PATH,
                            strict_filtering=strict_filtering,
                        )
                        Likelihood_Filter.run()

        print("Training neural network", flush=True)
        tr = Training(self.param, self.CONNECT_PATH)
        tr.train_model(output_file=output_file)
        tr.save_model()
        tr.save_history()
        tr.save_test_data()

        if self.param.save_name != None:
            model_name = self.param.save_name
        else:
            model_name = (
                f"{tr.param.jobname}_N{tr.N}_bs{tr.param.batchsize}_e{tr.param.epochs}"
            )
        if not self.param.overwrite_model:
            M = 1
            if os.path.isdir(os.path.join("trained_models", model_name)):
                while os.path.isdir(
                    os.path.join("trained_models", model_name + f"_{M}")
                ):
                    M += 1
                if M - 1 > 0:
                    model_name += f"_{M-1}"
        return model_name

    def cleanup_montepython_folders(self):
        """
        Cleanup the MontePython initialization folders/files after the iteration finishes.
        This is only cleaning up output files from the likelihood calculator, not files related to the MCMC sampling.
        It is intended to be called after the iteration finishes, and before the next iteration starts.
        It will:
        - Remove all rank_{id} folders from the montepython folder.
        - Merge all rank_{id}.out/.err logs into output_log.out/error_log.err.
        - Copy chain, .paramnames, log.* files from rank_0 to the main montepython folder.
        - Leave any other files untouched. (But there should be no other files in the montepython folder.)

        It is a very ad-hoc cleanup. Perhaps it would be better to look at the likelihood_calc_montepython.py code
        and see if there is a smarter way to handle this, ie. avoid creating rank_{id} folders in the first place.
        However, I was afraid of potential conflicts during parallel runs, so I opted for letting the code create seperate rank_{id} folders and outputs.
        And then cleaning up afterwards. This probably adds a little overhead each iteration, but it should be negligible.

        """
        path_lkl_calc = os.path.join("data", self.param.jobname, "lkl_calc")
        mp_folder = os.path.join(path_lkl_calc, "montepython")

        # 1) Check if main MontePython folder exists
        if not os.path.isdir(mp_folder):
            # Nothing to clean up
            return

        # 2) Copy rank_0's files to mp_folder (so we retain one copy)
        rank_0_folder = os.path.join(mp_folder, "rank_0")
        if os.path.isdir(rank_0_folder):
            for fname in os.listdir(rank_0_folder):
                src = os.path.join(rank_0_folder, fname)
                dst = os.path.join(mp_folder, fname)
                # We only want certain files (chain, .paramnames, log.*)
                # Adjust these patterns as needed:
                if (
                    fname.endswith("__0.txt")
                    or fname.endswith(".paramnames")
                    or fname.startswith("log.")  # e.g. log.conf, log.param
                ):
                    try:
                        # Copy (or move) the file up one level. Using copy2 to preserve metadata:
                        shutil.copy2(src, dst)
                    except Exception as e:
                        print(
                            f"Warning: Could not copy {fname} from rank_0. Reason: {e}",
                            flush=True,
                        )

        # 3) Remove all rank_{id} folders
        #    This will also remove rank_0_folder, including anything left inside.
        for item in os.listdir(mp_folder):
            rank_path = os.path.join(mp_folder, item)
            if item.startswith("rank_") and os.path.isdir(rank_path):
                try:
                    shutil.rmtree(rank_path, ignore_errors=True)
                except Exception as e:
                    print(
                        f"Warning: Could not remove folder {rank_path}. Reason: {e}",
                        flush=True,
                    )

        # 4) Collect rank_{id}.out/.err logs into two consolidated files, then delete them
        output_log_path = os.path.join(mp_folder, "output_log.out")
        error_log_path = os.path.join(mp_folder, "error_log.err")

        # Open both logs in 'append' mode, so we keep adding each iteration
        with open(output_log_path, "a") as out_log, open(
            error_log_path, "a"
        ) as err_log:
            for fname in os.listdir(mp_folder):
                full_path = os.path.join(mp_folder, fname)
                # Skip directories (we just removed rank_*, but in case anything else remains)
                if not os.path.isfile(full_path):
                    continue

                # 4a) Gather *.out into output_log.out
                if fname.startswith("rank_") and fname.endswith(".out"):
                    rank_name = fname.replace(".out", "")  # e.g. "rank_0"
                    out_log.write(
                        f"\n{'='*50}\nOutput messages from {rank_name}:\n{'='*50}\n"
                    )
                    try:
                        with open(full_path, "r") as temp_out:
                            out_log.write(temp_out.read())
                    except Exception as e:
                        out_log.write(
                            f"[Warning: Could not read file {fname}. Reason: {e}]\n"
                        )
                    # Delete the file after merging
                    try:
                        os.remove(full_path)
                    except Exception as e:
                        print(
                            f"Warning: Could not remove file {full_path}. Reason: {e}",
                            flush=True,
                        )

                # 4b) Gather *.err into error_log.err
                elif fname.startswith("rank_") and fname.endswith(".err"):
                    rank_name = fname.replace(".err", "")
                    err_log.write(
                        f"\n{'='*50}\nError messages from {rank_name}:\n{'='*50}\n"
                    )
                    try:
                        with open(full_path, "r") as temp_err:
                            err_log.write(temp_err.read())
                    except Exception as e:
                        err_log.write(
                            f"[Warning: Could not read file {fname}. Reason: {e}]\n"
                        )
                    # Delete the file after merging
                    try:
                        os.remove(full_path)
                    except Exception as e:
                        print(
                            f"Warning: Could not remove file {full_path}. Reason: {e}",
                            flush=True,
                        )

        # Done!
        # At this point:
        #  - rank_{id} subfolders are removed.
        #  - .out and .err logs from each rank_{id}.* are merged into output_log.out/error_log.err
        #  - leftover chain or paramnames from rank_0 are copied to mp_folder (if they existed).
        #  - everything else is deleted or left alone if not recognized.

    def check_calc_completed(
        self,
        sampling="lhc",
        success_threshold=0.8,
        skip_calc_completed_file_check=False,
        check_rank_files=True,
    ):
        job_path = os.path.join(self.CONNECT_PATH, f"data/{self.param.jobname}")
        if sampling in ["lhc", "hypersphere"]:
            iter_directory = os.path.join(job_path, f"N-{self.param.N}")
            target_samples = self.param.N
        elif sampling == "iterative":
            iteration = max(
                [
                    int(f.split("number_")[-1])
                    for f in os.listdir(job_path)
                    if f.startswith("number")
                ]
            )
            iter_directory = os.path.join(job_path, f"number_{iteration}")

        if not skip_calc_completed_file_check:
            calc_completed = os.path.isfile(
                os.path.join(iter_directory, "calc_completed.txt")
            )
            if calc_completed:
                os.remove(os.path.join(iter_directory, "calc_completed.txt"))
            else:
                return False

        def get_number_of_samples(iter_directory, check_rank_files=check_rank_files):
            total_samples = 0

            if check_rank_files:
                folder_path = os.path.join(iter_directory, "model_params_data")
                file_list = [
                    f
                    for f in os.listdir(folder_path)
                    if f.startswith("model_params_") and f.endswith(".txt")
                ]
            else:
                folder_path = iter_directory
                file_list = [
                    f
                    for f in os.listdir(folder_path)
                    if f.startswith("model_params") and f.endswith(".txt")
                ]

            for filename in file_list:
                file_path = os.path.join(folder_path, filename)
                try:
                    with open(file_path, "r") as f:
                        total_samples += max(
                            0, len(f.readlines()) - 1
                        )  # Ignore the header line
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}", flush=True)

            return total_samples

        if sampling == "iterative":
            exec(
                f"from source.mcmc_samplers.{self.param.mcmc_sampler} import {self.param.mcmc_sampler}"
            )
            _locals = {}
            exec(
                f"mcmc = {self.param.mcmc_sampler}(self.param, self.CONNECT_PATH)",
                locals(),
                _locals,
            )
            mcmc = _locals["mcmc"]

            data = mcmc.import_points_from_chains(iteration)
            target_samples = len(data)

        total_samples = get_number_of_samples(iter_directory)

        if total_samples < success_threshold * target_samples:
            print(
                f"ERROR / WARNING: Only {total_samples} samples were generated, less than {success_threshold}% of the target {target_samples}.\n"
                f"This indicates a significant percentage of the calculations failed. Please check the logs for errors.\n"
                f"Marking calculations as incomplete.",
                flush=True,
                file=sys.stderr,
            )
            return False

        return True

    def cleanup_mpi_processes(self, unique_id):
        num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", "1"))
        slurm_nodelist = os.environ.get("SLURM_JOB_NODELIST", "Not set")

        check_cmd = (
            f"srun --ntasks={num_nodes} --ntasks-per-node=1 --nodelist=$SLURM_JOB_NODELIST "
            f'bash -c \'processes=$(pgrep -a -u $USER | grep "{unique_id}" | grep -v pgrep | grep -v grep | grep -v srun | grep -v "mpirun" | sort -k2 | uniq); '
            f'if [[ -n "$processes" ]]; then echo -e "==== Node: $(hostname) ===="; echo "$processes"; echo ""; fi\''
        )

        try:
            result = sp.run(
                check_cmd,
                shell=True,
                stdout=sp.PIPE,
                stderr=sp.PIPE,
                text=True,
                timeout=10,
            )

            if result.stdout.strip():
                kill_unsuccessful = False
                print("Found lingering MPI processes:", flush=True, file=sys.stderr)
                print(result.stdout, flush=True, file=sys.stderr)

                from collections import defaultdict

                # Store PIDs per node
                pids_by_node = defaultdict(list)

                # Parse the result.stdout to extract node names and PIDs
                current_node = None
                for line in result.stdout.splitlines():
                    node_match = re.match(r"==== Node: (\S+?)\.", line)
                    if node_match:
                        current_node = node_match.group(1)
                    elif current_node and line.strip():
                        parts = line.split()
                        if parts and parts[0].isdigit():
                            pid = parts[0]
                            pids_by_node[current_node].append(pid)

                for node, pids in pids_by_node.items():
                    print(
                        f"Node: {node}, Killing PIDs: {', '.join(pids)}",
                        flush=True,
                        file=sys.stderr,
                    )

                    # Construct a per-node kill command
                    kill_cmd = f"srun --nodes=1 --ntasks=1 --nodelist={node} bash -c 'kill -9 {' '.join(pids)}'"

                    # Run the kill command
                    kill_result = sp.run(
                        kill_cmd,
                        shell=True,
                        stdout=sp.PIPE,
                        stderr=sp.PIPE,
                        text=True,
                        timeout=10,
                    )

                    if kill_result.returncode != 0 or kill_result.stderr.strip():
                        print(
                            f"Failed to kill PIDs due to error: {kill_result.stderr}",
                            flush=True,
                            file=sys.stderr,
                        )
                        kill_unsuccessful = True
                    else:
                        print(f"Kill command successful.", flush=True, file=sys.stderr)

                if not kill_unsuccessful:
                    print(
                        "Killed all lingering MPI processes.",
                        flush=True,
                        file=sys.stderr,
                    )
                # Wait for majority of processes to be terminated before continuing
                time.sleep(10)
            else:
                print("No lingering MPI processes found.", flush=True, file=sys.stderr)

        except sp.CalledProcessError as e:
            print(f"Error during process check: {e}", flush=True, file=sys.stderr)
        except Exception as e:
            print(
                f"Exception during process check: {repr(e)}",
                flush=True,
                file=sys.stderr,
            )

    def load_data_file(self, file_path, verbose=1):
        """
        Load a data file that has a header line starting with '#' and returns a DataFrame.
        """
        if not os.path.isfile(file_path):
            if verbose >= 1:
                print(f"[load_data_file] File {file_path} does not exist.", flush=True)
            return None

        header_line = None
        with open(file_path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    header_line = line.lstrip("#").strip()
                    break

        if header_line is None:
            raise ValueError(f"No header line starting with '#' found in {file_path}")

        columns = header_line.split()
        if verbose >= 3:
            print(f"[load_data_file] Columns for {file_path}: {columns}", flush=True)

        df = pd.read_csv(
            file_path,
            sep=r"\s+",
            comment="#",
            names=columns,
            index_col=False,
            dtype=np.float32,
        )

        # Optional sanity checks
        if df.empty and verbose >= 2:
            print(
                f"[load_data_file] Warning: Loaded DataFrame from {file_path} is empty.",
                flush=True,
            )

        return df

    def check_likelihood_filter_health(
        self,
        i,
        mcmc,
    ):

        # ---------------- 2nd CONVERGENCE CHECK ----------------
        # Import model_params from current iteration. The accepted points after the likelihood filter
        all_accepted_after_lklfilter_df = self.load_data_file(
            os.path.join(
                self.CONNECT_PATH,
                self.data_path,
                f"number_{i}",
                "model_params.txt",
            ),
            verbose=2,
        )
        all_accepted_after_lklfilter_likelihood_df = self.load_data_file(
            os.path.join(
                self.CONNECT_PATH,
                self.data_path,
                f"number_{i}",
                "likelihood_data.txt",
            ),
            verbose=2,
        )
        N_all_accepted_after_lkl_filter = len(all_accepted_after_lklfilter_df)

        # Import data from chains; i.e. data accepted from oversampling filter
        accepted_by_oversampling = mcmc.import_points_from_chains(i)
        N_accepted_by_oversampling = len(accepted_by_oversampling)
        N_in_data_set = (
            mcmc.get_number_of_data_points(i - 1) + N_accepted_by_oversampling
        )

        param_cols = all_accepted_after_lklfilter_df.columns
        accepted_by_oversampling_df = pd.DataFrame(
            accepted_by_oversampling, columns=param_cols
        )

        # Compare the two dataframes to find the overlap
        compare_context = {
            "context": "Finding overlap between points accepted by likelihood filter and new points from chains",
            "df1": f"all_accepted_after_lklfilter",
            "df2": f"accepted_by_oversampling",
            "msg1": "all_accepted_after_lklfilter is empty",
            "msg2": "accepted_by_oversampling is empty",
        }
        # New data that survived the oversampling filter and also survived the likelihood filter. i.e. all new points added to the accepted pool this iteration.
        new_accepted_df, new_accepted_likelihood_df = compare_dataframes(
            df1=all_accepted_after_lklfilter_df,
            df2=accepted_by_oversampling_df,
            df_likelihood=all_accepted_after_lklfilter_likelihood_df,
            comparison_type="common",  # Find common points between the two dataframes
            compare_context=compare_context,
            verbose=1,
        )

        N_new_accepted_after_lkl_filter = len(new_accepted_df)
        print(
            f"    The likelihood-filter accepted {N_new_accepted_after_lkl_filter}/{N_accepted_by_oversampling} new points of the points that survived the oversampling filter.\n"
            f"    The new points added constitutes {N_new_accepted_after_lkl_filter/N_all_accepted_after_lkl_filter*100:.1f}% of the total accepted pool after applying the likelihood-filter.",
            flush=True,
        )

        # Note for any future developer: I did experiment with using 'The new points added constitutes less than 10% of the total accepted pool after applying the likelihood-filter' as a convergence criterion.
        # I.e. if {N_new_accepted_after_lkl_filter/N_all_accepted_after_lkl_filter*100:.1f} < 10% then set i_converged = i and stop the iterative sampling.
        # However, this is not a good criterion, since this percentage can fluctuate up and down. Some early iterations it might be below 10%, but later it accepts significantly more points.
        # This is why I let it use the default build-in convergence criterion and instead opted for the consecutive_bad_states_count and printing a warning message that tries to explain the situation to the user.
        # Perhaps there is a better way to handle this, but I did not find a good solution.

        if (
            N_accepted_by_oversampling > 0.1 * N_in_data_set
            and N_new_accepted_after_lkl_filter < 0.1 * N_in_data_set
        ):
            self.consecutive_bad_states_count += 1

            if self.consecutive_bad_states_count >= 3:

                print(
                    f"""
                    ────────────────────────────────────────────────────────────────────
                    ⚠️  \033[1;31mWARNING:\033[0m Iterative process may have entered a "bad state".
                    ────────────────────────────────────────────────────────────────────
                    
                    🔁 The "bad state" has occurred for the last {self.consecutive_bad_states_count} consecutive iterations.
                    
                    The likelihood filter might be too agressive:
                    - Among the new samples that did survive the oversampling filter, which were {N_accepted_by_oversampling} points,
                    which equals {N_accepted_by_oversampling/N_in_data_set*100:.1f}% of the pre-likelihood-filter accepted data pool.
                    - {N_new_accepted_after_lkl_filter} points survived the likelihood filter, which constitutes only {N_new_accepted_after_lkl_filter/N_in_data_set*100:.1f}% 
                    of the pre-likelihood-filter accepted data pool (accepted data from iteration {i-1} + new oversampling accepted data).
                    
                    ⚠️  Risk:
                    This indicates that the likelihood filter MIGHT be too agressive and the MCMC sampling might continue to generate new data that is rejected by the likelihood filter.
                    This can lead to a situation where the MCMC sampling is stuck in a infinite loop, where it keeps generating the same data being discarded by the likelihood filter.
                    In a nutshell, the MCMC sampler might keep sampling points in the same region and too many points are outside the likelihood filter's acceptance region.
                    CONNECT might never reach its built-in convergence criterion, which is based on the oversampling filter accepting less than 10% of the pre-likelihood-filter accepted data pool.
                    
                    ✅ Recommendation:
                    If this keeps happening for multiple consecutive iterations, you might want to consider increasing the threshold for the likelihood-filter or disable it completely. 
                    Analyze the iterative outputs carefully: You may use the associated the "plot_iterations.py" tool.
                    You might want to try and run CONNECT with a Δχ²-threshold = 1e+32, keep_intial_data=False, keep_first_iteration=False, and use_likelihood_filter=True, to emulate a normal CONNECT run without the likelihood filter.
                    Then you can analyze the Δχ²-distribution and consider how agressive you need the likelihood filter to be.
                    
                    🔍This "bad state" is not a definitive indicator that the trained neural network is a bad emulator diverging from the true posterior distribution.
                    But you need to proceed with caution and analyze the situation carefully if it keeps occurring. An overly agressive filter have previously caused poor results.
                    
                    ⏳ CONNECT will continue to run the iterative sampling for minimum {self.param.max_consecutive_bad_states - self.consecutive_bad_states_count} iterations more, to see if the situation improves.
                    
                    """,
                    flush=True,
                )
                print(
                    f"""
                    ────────────────────────────────────────────────────────────────────
                    ⚠️  WARNING: Iterative process may have entered a "bad state".
                    ────────────────────────────────────────────────────────────────────
                    The likelihood filter might be too agressive. 
                    Please check the output.log in CONNECT's data folder under this running project for more information.
                    """,
                    flush=True,
                    file=sys.stderr,
                )

                if self.param.auto_update_threshold:

                    print(
                        f"    CONNECT is run with auto_update_threshold=True. Updating the likelihood-filter threshold to avoid the 'bad state'.",
                        flush=True,
                    )
                    print(
                        f"    The likelihood filter threshold is currently set to: {self.param.delta_chi2_threshold}",
                        flush=True,
                    )
                    print(
                        f"    The likelihood filter threshold will be set to 12th percentile of the Δχ²-values of the accepted points by the oversampling-filter.",
                        flush=True,
                    )

                    # Update the threshold for the filter to be less agressive, above the lowest 10% delta_chi2 value of the accepted points by the oversampling-filter.
                    # Load discarded data from the likelihood filter
                    discarded_by_lklfilter_df = self.load_data_file(
                        os.path.join(
                            self.CONNECT_PATH,
                            self.data_path,
                            f"number_{i}",
                            "loglkl_discarded_data",
                            "model_params.txt",
                        ),
                        verbose=2,
                    )
                    discarded_by_lklfilter_likelihood_df = self.load_data_file(
                        os.path.join(
                            self.CONNECT_PATH,
                            self.data_path,
                            f"number_{i}",
                            "loglkl_discarded_data",
                            "likelihood_data.txt",
                        ),
                        verbose=2,
                    )

                    # Compare the discarded data from the likelihood filter with the new data from the chains
                    compare_context = {
                        "context": "Finding overlap between points discarded by likelihood filter and new points from chains",
                        "df1": f"discarded_by_lklfilter",
                        "df2": f"accepted_by_oversampling",
                        "msg1": "discarded_by_lklfilter is empty",
                        "msg2": "accepted_by_oversampling is empty",
                    }

                    # New data that survived the oversampling filter but was discarded by the likelihood filter. i.e. all points discarded by the likelihood filter this iteration, which was generated by the chains.
                    new_discarded_df, new_discarded_likelihood_df = compare_dataframes(
                        df1=discarded_by_lklfilter_df,
                        df2=accepted_by_oversampling_df,
                        df_likelihood=discarded_by_lklfilter_likelihood_df,
                        comparison_type="common",
                        compare_context=compare_context,
                        verbose=1,
                    )

                    # Find lowest loglkl value in all_accepted_after_lklfilter_likelihood_df
                    bestfit_loglkl = all_accepted_after_lklfilter_likelihood_df[
                        "true_loglkl"
                    ].min()

                    # Now append the new_discarded_likelihood_df and new_accepted_likelihood_df into one dataframe: accepted_by_oversampling_likelihood_df

                    accepted_by_oversampling_likelihood_df = pd.concat(
                        [new_discarded_likelihood_df, new_accepted_likelihood_df]
                    )

                    # Now compute the delta_chi2 values for accepted_by_oversampling_likelihood_df["true_loglkl"] and bestfit_loglkl
                    delta_chi2 = 2 * (
                        accepted_by_oversampling_likelihood_df["true_loglkl"]
                        - bestfit_loglkl
                    )

                    # Compute the 12.5th percentile of the delta_chi2 values
                    threshold = np.percentile(delta_chi2, 12)

                    percentile_10 = np.percentile(delta_chi2, 10)

                    # Update the likelihood filter threshold
                    self.param.delta_chi2_threshold = threshold
                    print(
                        f"    The 10% percentile of the Δχ²-values is: {percentile_10}"
                    )
                    print(
                        f"    The likelihood filter threshold has been updated to: {self.param.delta_chi2_threshold}",
                        flush=True,
                    )

        else:
            self.consecutive_bad_states_count = 0

    def recover_bad_state_count_from_log(self, log_path):
        """
        Returns the most recent consecutive_bad_states_count found in output.log,
        or zero if none found.
        """
        if not os.path.isfile(log_path):
            return 0

        # read entire file
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        # Split into blocks by iteration
        # This will give you a list of iteration blocks, each starting with
        # 'Beginning iteration no. X' line and continuing until the next iteration or end of file.
        blocks = re.split(
            r"(?=^Beginning iteration no\.\s*\d+)", text, flags=re.MULTILINE
        )

        # We only care about the last *complete* iteration block, i.e. one that has "New model is" or "Final model is"
        # near the end. So let's find the last block that has that text.
        complete_blocks = []
        for block in blocks:
            # If block has "New model is" or "Final model is", we call it "complete".
            if re.search(r"(New model is|Final model is)", block):
                complete_blocks.append(block)

        if not complete_blocks:
            # No complete blocks found
            return 0

        last_complete_block = complete_blocks[-1]

        # Now search in last_complete_block for the line:
        # "The "bad state" has occurred for the last (\d+) consecutive iterations."
        # We can do a simple pattern:
        pattern = (
            r'The "bad state" has occurred for the last (\d+) consecutive iterations'
        )
        match = re.search(pattern, last_complete_block)
        if match:
            return int(match.group(1))
        else:
            return 0
