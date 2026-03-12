from importlib.machinery import SourceFileLoader
import difflib as dl

import numpy as np


class Parameters:
    def __init__(self, param_file):
        param = SourceFileLoader(param_file, param_file).load_module()
        jobname = param_file.split("/")[-1].split(".")[0]

        self.param_file = param_file

        default = {
            ### Training parameters
            "train_ratio": (0.9, float),
            "val_ratio": (0.01, float),
            "epochs": (200, int),
            "batchsize": (64, int),
            "N_hidden_layers": (4, int),
            "N_nodes": (512, int),
            "loss_function": ("cosmic_variance", str),
            "activation_function": ("alsing", str),
            "normalisation_method": ("standardisation", str),
            # Advanced training parameters
            "use_advanced_training": (True, bool),
            "initial_lr": (0.01, float),
            "use_lr_scheduler": (True, bool),
            "lr_scheduler": ("ReduceLROnPlateau", str),
            "use_early_stopping": (True, bool),
            "early_stopping_patience": (200, int),
            "monitor_prog_start": (50, int),
            "monitor_prog_force_thres": (200, float),
            "monitor_prog_active_above_loss": (50, float),
            "monitor_prog_min_delta": (10, float),
            "monitor_prog_patience": (40, int),
            ### Sampling parameters
            "parameters": ({}, dict),
            "custom_parameters": ({}, dict),
            "extra_input": ({}, dict),
            "output_Cl": (["tt"], list),
            "output_Pk": ([], list),
            "k_grid": (self.get_k_grid(param), list),
            "z_Pk_list": ([0.0], list),
            "output_bg": ([], list),
            "z_bg_list": ([], list),
            "output_th": ([], list),
            "z_th_list": ([], list),
            "extra_output": ({}, dict),
            "output_derived": ([], list),
            "N": (10000, int),
            "sampling": ("lhc", str),
            ### Additional parameters for iterative sampling
            "N_max_points": (10000, int),
            "mcmc_sampler": ("cobaya", str),
            "initial_model": (None, str),
            "initial_sampling": ("hypersphere", str),
            "mcmc_tol": (0.01, float),
            "iter_tol": (0.1, float),
            "temperature": (5.0, (float, int, list)),
            "sampling_likelihoods": (["Planck_lite"], list),
            "prior_ranges": ({}, dict),
            "bestfit_guesses": ({}, dict),
            "sigma_guesses": ({}, dict),
            "log_priors": ([], list),
            "keep_first_iteration": (True, bool),
            "resume_iterations": (False, bool),
            "extra_cobaya_lkls": ({}, dict),
            "superupdate": (0, int),  # Use superupdate if > 0
            "N_mcmc": (1000000, int),  # Number of steps to use in the MCMC sampler
            "N_mcmc_burnin": (
                250000,
                int,
            ),  # Number of steps to use for "burn-in" before doing the superupdate
            ### Parameters for the likelihood filter
            "use_likelihood_filter": (True, bool),
            "delta_chi2_threshold": (200, (float, int, list, str)),
            "min_points_to_keep": (5000, (int, list)),
            "nuisance_params_lkl": ({"A_planck": 1.0}, dict),
            "keep_initial_data": (True, bool),
            "discard_worst_first": (True, bool),
            "max_consecutive_bad_states": (3, int),
            "strict_filtering_final_iteration": (False, bool),
            "auto_update_threshold": (False, bool),
            "compute_oversampled_likelihoods": (False, bool),
            "N_first_iteration": (
                5000,
                int,
            ),
            "auto_threshold_percentile": (99.3, (float, int)),
            "auto_bulk_anchor_percentile": (30.0, (float, int)),
            
            # Number of points to generate in the first iteration
            ### Additional parameters for other kinds of sampling
            "hypersphere_surface": (False, bool),
            "hypersphere_covmat": (None, str),
            "pickle_data_file": (None, str),
            ### Saving parameters
            "jobname": (jobname, str),
            "save_name": (None, str),
            "overwrite_model": (False, bool),
            ### Memory overload parameters
            "memory_safe": (False, bool),
            "max_time_for_theory": (200, int),
        }

        for key, val in default.items():
            setattr(self, key, getattr(param, key, val[0]))
        self.error_handling(param, default)

        self.adjust_defaults_based_on_likelihood_filter()

    def get_k_grid(self, param):
        if hasattr(param, "output_Pk") and len(param.output_Pk) > 0:
            k_grid = np.sort(
                np.concatenate(
                    [
                        np.logspace(np.log10(5e-6), np.log10(5e-5), 15),
                        np.logspace(np.log10(1e-4), np.log10(0.008), 8),
                        np.logspace(np.log10(0.76), np.log10(5), 17),
                        np.logspace(np.log10(0.009), np.log10(0.75), 60),
                    ]
                )
            )
            k_grid *= 0.67556  # value of h in LCDM
            return k_grid
        return None

    def try_cast_to_native_type(self, value, target_type):
        try:
            return target_type(value)
        except:
            pass

    def error_handling(self, param, default):
        sep = "\n" + "    " * 2 + "- "
        name_errors = self.get_name_errors(param, default)
        type_errors = self.get_type_errors(param, default)
        input_error = "Error with inputs in parameter file"
        if len(type_errors) > 1 and len(name_errors) > 1:
            input_errors = (
                input_error
                + "\n"
                + sep.join(name_errors)
                + "\n"
                + sep.join(type_errors)
            )
            raise Exception(input_errors)
        elif len(name_errors) > 1:
            name_errors = input_error + "\n" + sep.join(name_errors)
            raise Exception(name_errors)
        elif len(type_errors) > 1:
            type_errors = input_error + "\n" + sep.join(type_errors)
            raise Exception(type_errors)
        if not self.parameters:
            raise ValueError(
                "No set of parameters were given. Please provide a dictionary with lower and upper bounds."
            )

    def get_name_errors(self, param, default):
        name_errors = ["NameErrors:"]
        for name in [
            n for n in dir(param) if not n.startswith("__") and not n.endswith("__")
        ]:
            if not name in default:
                matches = dl.get_close_matches(name, default.keys())
                if len(matches) == 0:
                    name_error = f"'{name}' is not a recognised parameter and no close matches exist."
                else:
                    name_error = f"'{name}' is not a recognised parameter. Did you mean '{matches[0]}'?"
                    input_val = getattr(param, name)
                    input_type = type(input_val)
                    target_type = default[matches[0]][1]
                    if type(target_type) is tuple:
                        type_bool = input_type not in target_type
                    else:
                        type_bool = input_type != target_type
                    if type_bool:
                        if target_type == bool:
                            name_error += f" The type should then be 'bool'."
                        elif self.try_cast_to_native_type(
                            input_val, target_type
                        ) == None or (
                            target_type in [list, tuple, set] and input_type == str
                        ):
                            name_error += (
                                f" The type should then be '{target_type.__name__}'."
                            )
                name_errors.append(name_error)
        return name_errors

    def get_type_errors(self, param, default):
        type_errors = ["TypeErrors:"]
        for key, val in default.items():
            input_val = getattr(self, key)
            input_type = type(input_val)
            target_val = val[0]
            target_type = val[1]
            if type(target_type) is tuple:
                type_bool = input_type not in target_type
            else:
                type_bool = input_type != target_type
            if hasattr(param, key) and type_bool and target_val != None:
                if target_type == bool:
                    type_errors.append(
                        f"'{key}' is of type '{input_type.__name__}' and should be of type 'bool'."
                    )
                    continue
                casting = self.try_cast_to_native_type(input_val, target_type)
                if casting == None or (
                    target_type in [list, tuple, set] and input_type == str
                ):
                    if type(target_type) is tuple:
                        msg = "types "
                        for t in target_type:
                            msg += f"'{t.__name__}' or "
                        msg = msg[:-4]
                    else:
                        msg = f"type '{target_type.__name__}'"
                    type_errors.append(
                        f"'{key}' is of type '{input_type.__name__}' and could not be converted to {msg}."
                    )
                else:
                    setattr(self, key, casting)
        return type_errors

    def adjust_defaults_based_on_likelihood_filter(self):
        """
        Adjust default values of keep_initial_data and keep_first_iteration
        based on the value of use_likelihood_filter. But only if keep_initial_data and keep_first_iteration
        are not already specified in the input parameter file.

        If they are not specified, the following rules apply:
        - If use_likelihood_filter is True, keep_initial_data and keep_first_iteration are set to True.
        - If use_likelihood_filter is False, keep_initial_data and keep_first_iteration are set to False.

        Furthermore, if use_likelihood_filter is True and keep_initial_data is True, keep_first_iteration is also set to True.
        To ensure logical consistency.
        """
        if not hasattr(self, "keep_initial_data") or not hasattr(
            self, "keep_first_iteration"
        ):
            # Ensure these parameters exist before modifying them
            return

        if "use_likelihood_filter" in self.param_file:
            if self.use_likelihood_filter:
                # Default behavior when likelihood filter is used
                if "keep_initial_data" not in self.param_file:
                    self.keep_initial_data = True
                if "keep_first_iteration" not in self.param_file:
                    self.keep_first_iteration = True
            else:
                # Default behavior when likelihood filter is not used
                if "keep_initial_data" not in self.param_file:
                    self.keep_initial_data = False
                if "keep_first_iteration" not in self.param_file:
                    self.keep_first_iteration = False

        # Enforce logical consistency: if keep_initial_data is True, keep_first_iteration must also be True
        if (
            self.keep_initial_data
            and not self.keep_first_iteration
            and self.use_likelihood_filter
        ):
            print(
                "Warning: Enforcing keep_first_iteration=True because keep_initial_data=True.\n"
                "This is to ensure logical consistency.",
                flush=True,
            )
            self.keep_first_iteration = True
