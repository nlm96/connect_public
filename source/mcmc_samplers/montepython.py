import os
import sys
import subprocess as sp
import re
import fileinput
import shutil
import time
import numpy as np

from ..mcmc_base import MCMC_base_class


class montepython(MCMC_base_class):

    def __init__(self, param, CONNECT_PATH):
        super(montepython, self).__init__(param, CONNECT_PATH)
        path = {}
        with open(os.path.join(CONNECT_PATH, "mcmc_plugin/connect.conf"), "r") as f:
            for line in f:
                exec(line)
        self.montepython_path = path["montepython"]

    def run_mcmc_sampling(self, model, iteration):
        MP_param_file = self.create_montepython_param(iteration)

        with fileinput.input(MP_param_file, inplace=True) as file:
            for line in file:
                if "data.cosmo_arguments['connect_model']" in line:
                    line = f"data.cosmo_arguments['connect_model'] = '{model}'\n"
                print(line, end="")

        output_dir = f"{self.CONNECT_PATH}/data/{self.param.jobname}/number_{iteration}"

        if self.param.superupdate > 0:

            restart = None
            # N_normal is the number of steps to run without superupdate as normal run
            # I found using superupdate immediately can lead to the chains getting stuck in a local minimum
            N_normal = self.param.N_mcmc_burnin
            # run without superupdate (and then only run it N= 100000 )
            time.sleep(1)  # Wait a bit before running the normal run
            sp.run(
                f"{self.CONNECT_PATH}/source/mcmc_samplers/run_scripts/run_montepython_iteration.sh {output_dir} {MP_param_file} {N_normal} {os.path.join(self.CONNECT_PATH,'mcmc_plugin/connect.conf')} 0 {self.param.temperature} {self.param.mcmc_tol} {self.mcmc_node}",
                shell=True,
                cwd=self.montepython_path,
            )

            # Check if the run converged before N=100000
            if not self.check_convergence(iteration, output_dir, max_steps=N_normal):
                time.sleep(1)  # Wait a bit before running the superupdate

                # Copy all files in output_dir to a subdirectory
                os.mkdir(f"{output_dir}/mcmc_normal_run")
                for f in os.listdir(output_dir):
                    if f == "mcmc_normal_run":
                        continue  # Skip the new directory
                    shutil.copy(
                        os.path.join(output_dir, f),
                        os.path.join(output_dir, "mcmc_normal_run", f),
                    )

                def existing_chain(output_dir):
                    """Find the correct chain file to pass to MontePython's -r option."""
                    # Look for the chain file ending with __1.txt (which is the one MontePython expects)
                    # But do it in output_dir/normal_run

                    for f in sorted(os.listdir(f"{output_dir}")):
                        if re.match(
                            r".*__1\.txt$", f
                        ):  # Matches files like "2025-03-10_1000000__1.txt"
                            # Return the full path to the chain file
                            return os.path.join(output_dir, f)

                    print(
                        "Warning: No valid chain found to resume from.",
                        flush=True,
                        file=sys.stderr,
                    )
                    return None  # No valid chain found, start fresh

                restart = existing_chain(output_dir)

                # If it did not converge before N=100000, run it again with superupdate
                N_remaining = self.param.N_mcmc - N_normal
                sp.run(
                    f"{self.CONNECT_PATH}/source/mcmc_samplers/run_scripts/run_montepython_iteration.sh {output_dir} {MP_param_file} {N_remaining} {os.path.join(self.CONNECT_PATH,'mcmc_plugin/connect.conf')} {self.param.superupdate} {self.param.temperature} {self.param.mcmc_tol} {self.mcmc_node} {restart}",
                    shell=True,
                    cwd=self.montepython_path,
                )

                # Move jumping_factor.txt and jumping_factors.txt to a subdirectory
                try:
                    os.mkdir(f"{output_dir}/jumping_factors")
                    os.rename(
                        f"{output_dir}/jumping_factor.txt",
                        f"{output_dir}/jumping_factors/jumping_factor.txt",
                    )
                    os.rename(
                        f"{output_dir}/jumping_factors.txt",
                        f"{output_dir}/jumping_factors/jumping_factors.txt",
                    )
                except Exception as e:
                    print(
                        f"Failed to move jumping factors: {e}",
                        flush=True,
                        file=sys.stderr,
                    )

                def delete_identical_filenames(output_dir):
                    backup_dir = os.path.join(output_dir, "mcmc_normal_run")
                    if not os.path.isdir(backup_dir):
                        print(
                            f"No backup directory found at {backup_dir}",
                            flush=True,
                            file=sys.stderr,
                        )
                        return

                    # List all files in the parent folder once
                    parent_files = os.listdir(output_dir)

                    for fname in os.listdir(backup_dir):
                        # Skip files we don't want to delete
                        if (
                            fname.endswith(".bestfit")
                            or fname.endswith(".covmat")
                            or fname.endswith(".log")
                            or fname.endswith(".param")
                        ):
                            continue

                        # Count occurrences in the parent folder
                        count = parent_files.count(fname)
                        if count == 1:
                            parent_file = os.path.join(output_dir, fname)
                            try:
                                os.remove(parent_file)
                            except Exception as e:
                                print(
                                    f"Error deleting {parent_file}: {e}",
                                    flush=True,
                                    file=sys.stderr,
                                )
                        elif count > 1:
                            print(
                                f"Multiple files found for {fname} in parent folder; skipping deletion.",
                                flush=True,
                                file=sys.stderr,
                            )

                try:
                    delete_identical_filenames(output_dir)
                except Exception as e:
                    print(
                        f"Failed to delete identical files: {e}",
                        flush=True,
                        file=sys.stderr,
                    )

        else:
            sp.run(
                f"{self.CONNECT_PATH}/source/mcmc_samplers/run_scripts/run_montepython_iteration.sh {output_dir} {MP_param_file} {self.param.N_mcmc} {os.path.join(self.CONNECT_PATH,'mcmc_plugin/connect.conf')} {self.param.superupdate} {self.param.temperature} {self.param.mcmc_tol} {self.mcmc_node}",
                shell=True,
                cwd=self.montepython_path,
            )

    def get_number_of_accepted_steps(self, iteration):
        directory = os.path.join(
            self.CONNECT_PATH, f"data/{self.param.jobname}/number_{iteration}"
        )
        txt_files = [f for f in os.listdir(directory) if f.endswith(".txt")]
        N_acc = 0
        for f in txt_files:
            N_acc += sum(
                1
                for line in open(os.path.join(directory, f))
                if line[0] != "#" and line.strip()
            )
        return N_acc

    def filter_chains(self, N, iteration):
        path = f"data/{self.param.jobname}/number_{iteration}"
        N_max_points = N

        list_of_files = []
        # for filename in [f for f in os.listdir(path) if f.endswith(".txt")]:
        for filename in [
            f
            for f in os.listdir(path)
            if f.endswith(".txt")
            and f not in {"jumping_factor.txt", "jumping_factors.txt"}
        ]:
            list_of_files.append(path + "/" + filename)
        lines_of_use_in_files = {}
        for filename in list_of_files:
            lines = self.useful_points_in_file(filename)
            lines_of_use_in_files[filename] = lines
        N_left = [lines_of_use_in_files[filename] for filename in list_of_files]
        N_to_use = self.filter_steps(N_left, N_max_points)

        output_lines = {}
        for filename, N_lines in zip(list_of_files, N_to_use):
            with open(filename, "r") as f:
                f_list = list(f)
            output_lines[filename] = []
            count_noncomment = 0
            for i, line in enumerate(reversed(f_list), 1):
                if i <= N_lines and count_noncomment < N_lines:
                    if line[0] != "#":
                        count_noncomment += 1
                        output_lines[filename].append(line)

        for filename in list_of_files:
            with open(filename, "w") as f:
                for line in output_lines[filename]:
                    f.write(line)

        # Don't double append, when using resume_iterations = True
        if os.path.isfile(
            f"data/{self.param.jobname}/compare_iterations/chain__{iteration}.txt"
        ):
            os.remove(
                f"data/{self.param.jobname}/compare_iterations/chain__{iteration}.txt"
            )

        for filename in list_of_files:
            os.system(
                f"cat {filename} >> {f'data/{self.param.jobname}/compare_iterations/chain__{iteration}.txt'}"
            )
        os.system(
            f"cp data/{self.param.jobname}/number_{iteration}/log.param data/{self.param.jobname}/compare_iterations/log.param"
        )

        return int(np.sum(N_to_use))

    def compare_iterations(self, iteration):
        chain1 = (
            f"data/{self.param.jobname}/compare_iterations/chain__{iteration-1}.txt"
        )
        chain2 = f"data/{self.param.jobname}/compare_iterations/chain__{iteration}.txt"
        kill_iteration = self.Gelman_Rubin_log(iteration, all_chains=[chain1, chain2])
        return kill_iteration

    def create_montepython_param(self, iteration):
        path = os.path.join(
            self.CONNECT_PATH, "data", self.param.jobname, "montepython_input"
        )
        os.system(f"mkdir -p {path}")
        with open(
            "mcmc_plugin/mp_param_templates/connect_lite.param.template", "r"
        ) as f:
            with open(os.path.join(path, f"number_{iteration}.param"), "w") as g:
                for line in f:
                    g.write(line)
                    if (
                        "#------Experiments to test (separated with commas)-----"
                        in line
                    ):
                        g.write("")
                        experiments_line = "data.experiments=["
                        for lkl in self.param.sampling_likelihoods:
                            if lkl == "Planck_lite":
                                lkl = "Planck_highl_TTTEEE_lite"
                            elif lkl == "Planck_lowl_EE":
                                lkl = "Planck_lowl_EE_connect"
                            experiments_line += f"'{lkl}', "
                        experiments_line = experiments_line[:-2] + "]"
                        g.write(experiments_line)
                    elif "# Cosmological parameters list" in line:
                        g.write("")

                        all_params = dict(
                            self.param.parameters
                        )  # copy or dict() for a separate copy
                        # Add each custom param into 'all_params'
                        for custom_par, cdict in self.param.custom_parameters.items():
                            all_params[custom_par] = cdict["range"]

                        for par, interval in all_params.items():
                            if par in self.param.prior_ranges:
                                xmin = self.param.prior_ranges[par][0]
                                xmax = self.param.prior_ranges[par][1]
                            elif par in self.param.custom_parameters:
                                xmin = self.param.custom_parameters[par]["range"][0]
                                xmax = self.param.custom_parameters[par]["range"][1]
                            else:
                                xmin = "None"
                                xmax = "None"
                            if par in self.param.bestfit_guesses:
                                guess = self.param.bestfit_guesses[par]
                            else:
                                guess = (interval[0] + interval[1]) / 2
                            if par in self.param.sigma_guesses:
                                sig = self.param.sigma_guesses[par]
                            else:
                                sig = abs((interval[1] - interval[0]) / 50)
                            if par == "omega_b" or par == "Omega_b":
                                scale = 0.01
                                guess *= 1 / scale
                                sig *= 1 / scale
                                if not isinstance(xmin, str):
                                    xmin *= 1 / scale
                                if not isinstance(xmax, str):
                                    xmax *= 1 / scale
                            else:
                                scale = 1
                            if par in self.param.log_priors:
                                if not isinstance(xmin, str):
                                    xmin = np.log10(xmin)
                                if not isinstance(xmax, str):
                                    xmax = np.log10(xmax)
                                guess = np.log10(guess)
                                if not isinstance(xmin, str) and not isinstance(
                                    xmax, str
                                ):
                                    sig = sig * (xmax - xmin) / (10**xmax - 10**xmin)
                                else:
                                    sig = 0.01
                                par = par + "_log10_prior"

                            g.write(
                                f"data.parameters['{par}'] = [{guess}, {xmin}, {xmax}, {sig}, {scale}, 'cosmo']\n"
                            )
                    elif "# Derived parameter list" in line:
                        g.write("")
                        for par in self.param.output_derived:
                            if par == "A_s":
                                scale = 1e-9
                            else:
                                scale = 1
                            g.write(
                                f"data.parameters['{par}'] = [1, None, None, 0, {scale}, 'derived']\n"
                            )
        return os.path.join(path, f"number_{iteration}.param")

    def get_Rminus1_of_chains(
        self, all_chains, iteration  # list of paths to chains  # Iteration number
    ):

        output = sp.run(
            f"python {os.path.join(self.montepython_path, 'montepython/MontePython.py')} info {all_chains[0]} {all_chains[1]} --noplot --minimal",
            shell=True,
            stdout=sp.PIPE,
        ).stdout.decode("utf-8")
        output = list(iter(output.splitlines()))
        kill_iteration = False
        if any("Removed everything: chain not converged" in s for s in output):
            print(
                f"chains from iterations {iteration} and {iteration-1} did not have sufficient overlap"
            )
            Rm1_line = (
                "\t"
                + f"chains from iterations {iteration} and {iteration-1} did not have sufficient overlap\n"
            )
        else:
            Rm1_line = ""
            try:
                index = [idx for idx, s in enumerate(output) if "-> R-1 is" in s][0]
            except:
                raise RuntimeError(
                    "chains are much too short for Monte Python to analyse. Increase N_max_points in the parameter file"
                )
            kill_iteration = True
            largest_Rm1 = 0
            native_params = list(self.param.parameters.keys())
            custom_params = list(self.param.custom_parameters.keys())
            param_names = native_params + custom_params
            for par in param_names:
                for out in output[index:]:
                    if par == out[-len(par) :] and out[-len(par) - 1] == " ":
                        Rm1 = np.float32(re.findall("\d+\.\d+", out)[0])
                        if Rm1 > largest_Rm1:
                            largest_Rm1 = Rm1
                        Rm1_line += "\t" + str(Rm1)
                        if Rm1 > self.param.iter_tol:
                            kill_iteration = False
                        break
            Rm1_line += "\t" + str(largest_Rm1) + "\n"
            print(
                f"Iteration {iteration} and {iteration-1} has the following R-1 values"
            )
            for line in output[index:]:
                if any(par in line for par in param_names):
                    print(line)
        return kill_iteration, Rm1_line

    def check_version(self):
        with open(os.path.join(self.montepython_path, "VERSION"), "r") as f:
            version = list(f)[0].strip()
        if int(version.split(".")[0]) < 3:
            err_msg = f"Your version of MontePython is {version}, which is not compatible with python 3. Your MontePython version must be at least 3.0"
            print(err_msg, flush=True)
            raise NotImplementedError(err_msg)
        else:
            print(f"Your version of MontePython is {version}", flush=True)
        if np.float64(".".join(version.split(".")[:2])) > 3.5:
            libs = "".join(
                os.listdir(os.path.join(self.CONNECT_PATH, "mcmc_plugin/python/build"))
            )
            if libs.find(sys.version) == -1:
                os.mkdir(
                    os.path.join(
                        self.CONNECT_PATH, f"mcmc_plugin/python/build/lib.{sys.version}"
                    )
                )
                os.link(
                    os.path.join(
                        self.CONNECT_PATH,
                        "mcmc_plugin/python/build/lib.connect_disguised_as_classy/classy.py",
                    ),
                    os.path.join(
                        self.CONNECT_PATH,
                        f"mcmc_plugin/python/build/lib.{sys.version}/classy.py",
                    ),
                )

    def save_accepted_points(
        self,
        indices_accepted,  # indices of accepted points
        iteration,  # Iteration Number
    ):
        path = f"data/{self.param.jobname}/number_{iteration}"
        files = sorted([f for f in os.listdir(path) if f.endswith(".txt")])

        indices = indices_accepted
        for filename in files:
            N = self.useful_points_in_file(path + "/" + filename)
            indices_chain = [i for i in indices if i < N]
            indices = [i - N for i in indices if i >= N]
            with open(path + "/" + filename, "r") as f:
                f_list = list(f)
            with open(path + "/" + filename, "w") as f:
                line_number = 0
                for line in f_list:
                    if line_number in indices_chain:
                        f.write(line)
                    line_number += 1

    def useful_points_in_file(self, filename):
        with open(filename, "r") as f:
            all_lines = f.readlines()

        if self.param.superupdate == 0:
            lines = 0
            for i, line in enumerate(all_lines):
                if line[0] != "#":
                    lines += 1
                else:
                    lines = 0
            return lines
        elif self.param.superupdate > 0:
            total_non_comment = sum(
                1 for line in all_lines if not line.lstrip().startswith("#")
            )
            lines = 0
            for i, line in enumerate(all_lines):
                if line[0] == "#":
                    # Count non-comment lines remaining after this line:
                    remaining = total_non_comment - i
                    if remaining >= self.param.N_max_points / 4:
                        lines = 0
                else:
                    lines += 1
            return lines

    def backup_full_chains(self, iteration):
        files = sorted(
            [
                f
                for f in os.listdir(f"data/{self.param.jobname}/number_{iteration}")
                if f.endswith(".txt")
            ]
        )
        os.mkdir(f"data/{self.param.jobname}/number_{iteration}/full_chains")
        for filename in files:
            os.system(
                f"cp data/{self.param.jobname}/number_{iteration}/{filename} data/{self.param.jobname}/number_{iteration}/full_chains/{filename}"
            )

    def import_points_from_chains(self, iteration, import_full_chains=False):

        paramnames_native = self.param.parameters.keys()
        paramnames_custom = self.param.custom_parameters.keys()
        paramnames = list(paramnames_native) + list(paramnames_custom)
        paramnames = ["100theta_s" if s == "100*theta_s" else s for s in paramnames]
        model_param_scales = []
        with open(f"data/{self.param.jobname}/number_{iteration}/log.param", "r") as f:
            lines = list(f)
        for i, name in enumerate(paramnames):
            for line in lines:
                if name in self.param.log_priors:
                    if (
                        line.startswith(f"data.parameters['{name}_log10_prior']")
                        and line.split("'")[-2] == "cosmo"
                    ):
                        model_param_scales.append(
                            np.float32(
                                line.split("=")[-1].replace(" ", "").split(",")[4]
                            )
                        )
                        break
                else:
                    if (
                        line.startswith(f"data.parameters['{name}']")
                        and line.split("'")[-2] == "cosmo"
                    ):
                        model_param_scales.append(
                            np.float32(
                                line.split("=")[-1].replace(" ", "").split(",")[4]
                            )
                        )
                        break
        model_param_scales = np.array(model_param_scales)
        mp_names = []
        paramnames_file = [
            f
            for f in os.listdir(f"data/{self.param.jobname}/number_{iteration}")
            if f.endswith(".paramnames")
        ][0]
        with open(
            f"data/{self.param.jobname}/number_{iteration}/" + paramnames_file, "r"
        ) as f:
            list_f = list(f)
        for line in list_f[0 : len(paramnames)]:
            name = line.replace(" ", "").split("\t")[0]
            if name[-12:] == "_log10_prior" and name[:-12] in self.param.log_priors:
                name = name[:-12]
            mp_names.append(name)

        lines = []
        i = 0
        data = []

        if not import_full_chains:
            files = sorted(
                [
                    f
                    for f in os.listdir(f"data/{self.param.jobname}/number_{iteration}")
                    if f.endswith(".txt") and "__" in f
                ]
            )
        else:
            files = sorted(
                [
                    f
                    for f in os.listdir(
                        f"data/{self.param.jobname}/number_{iteration}/full_chains"
                    )
                    if f.endswith(".txt")
                ]
            )

        for filename in files:
            with open(
                f"data/{self.param.jobname}/number_{iteration}/" + filename, "r"
            ) as f:
                list_f = list(f)
            lines += list_f
            for line in list_f:
                if line[0] != "#":
                    params = np.float32(
                        line.replace("\n", "").split("\t")[1 : len(paramnames) + 1]
                    )
                    for name in self.param.log_priors:
                        k = mp_names.index(name)
                        params[k] = np.power(10.0, params[k])
                    params *= model_param_scales
                    data.append(
                        [params[j] for j in [mp_names.index(n) for n in paramnames]]
                    )
                    i += 1

        data = np.array(data, dtype=np.float32)

        from ..tools import transform_custom_parameters

        final_data, final_names = transform_custom_parameters(
            data,
            paramnames,
            self.param.parameters,  # native_params
            self.param.custom_parameters,  # custom_params
        )

        return final_data

    def import_loglkl_from_chains(self, iteration, import_full_chains=False):
        # Initialize list to store -log(likelihood) values from the chain files
        loglkl_values = []

        # Get sorted list of chain files for the specified iteration

        if not import_full_chains:
            files = sorted(
                [
                    f
                    for f in os.listdir(f"data/{self.param.jobname}/number_{iteration}")
                    if f.endswith(".txt")
                ]
            )
        else:
            files = sorted(
                [
                    f
                    for f in os.listdir(
                        f"data/{self.param.jobname}/number_{iteration}/full_chains"
                    )
                    if f.endswith(".txt")
                ]
            )

        for filename in files:
            # Open each chain file
            with open(
                f"data/{self.param.jobname}/number_{iteration}/" + filename, "r"
            ) as f:
                for line in f:
                    # Skip comment lines starting with '#'
                    if line[0] != "#":
                        # The first column of each line contains both multiplicity and -log(likelihood),
                        # separated by two spaces. The remaining columns (cosmological parameters) are
                        # ignored as they are separated by tabs and are not needed here.
                        parts = line.split(
                            "  ", 1
                        )  # Split once to isolate multiplicity and -log(likelihood)

                        # Skip this line if it does not contain both multiplicity and -log(likelihood) in the expected format
                        if len(parts) < 2:
                            continue

                        # multiplicity = int(parts[0].strip())  # Extract the multiplicity
                        loglkl = np.float32(
                            parts[1].split()[0]
                        )  # Extract -log(likelihood)

                        loglkl_values.append(loglkl)  # Store -log(likelihood) value

        # Return -log(likelihood) values as a NumPy array
        return np.array(loglkl_values)

    def check_convergence(self, iteration, output_dir, max_steps=100000):
        """Check if MontePython chains have converged early.

        Parameters:
            output_dir (str): Path to the directory containing MontePython output.
            max_steps (int): Maximum number of steps allowed in the MCMC run.
            R1_threshold (float): Threshold for Gelman-Rubin (R-1) convergence.

        Returns:
            bool: True if converged early, False otherwise.
        """
        log_file = os.path.join(output_dir, f"number_{iteration}.log")

        # Step 1: Check if any chain reached the max steps (means MCMC did NOT stop early)
        try:
            with open(log_file, "r") as f:
                for line in f:
                    if "Number of steps:" in line:
                        num_steps = int(line.split("Number of steps:")[-1].split()[0])
                        if num_steps >= max_steps:
                            return False  # Did not converge early, run superupdate
        except FileNotFoundError:
            print(
                f"Warning: {log_file} not found. Assuming no early convergence.",
                flush=True,
                file=sys.stderr,
            )
            return False

        return True  # Assuming it converged early, do not run superupdate
