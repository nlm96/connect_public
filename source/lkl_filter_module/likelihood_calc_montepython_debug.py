import ast
import contextlib
import copy
import os
import shutil
import sys
from typing import Any
import numpy as np
from Speciale.connectv2.source.lkl_filter_module.likelihood_calc_base import (
    LikelihoodCalculator as BaseLikelihoodCalculator,
)
import importlib.util
from types import ModuleType


class InitialiseMontePython:
    """
    Singleton class that initialises MP on first init and returns
    that instance at every other initialisation, meaning
    MP is only ever initialised once
    (thus does not support MP inits with different input...)
    """

    def __new__(cls, param_connect, id):

        print(
            f"[DEBUG] [Rank {id}] Entering InitialiseMontePython.__new__()",
            file=sys.stderr,
            flush=True,
        )

        # Get the .param file and .conf file paths
        path_lkl_calc = os.path.join("data", param_connect.jobname, "lkl_calc")
        param_file = os.path.join(path_lkl_calc, "likelihood_calc.param")
        conf_file = "source/lkl_filter_module/likelihood_calc.conf"
        output_folder = path_lkl_calc

        print(
            f"[DEBUG] [Rank {id}] Paths set: path_lkl_calc={path_lkl_calc}, param_file={param_file}, conf_file={conf_file}, output_folder={output_folder}",
            file=sys.stderr,
            flush=True,
        )

        os.makedirs(output_folder, exist_ok=True)

        path = {}
        """
        with open(conf_file, 'r') as f:
            for line in f:
                exec(line)
        montepython_path = path['montepython']
        """

        # DEBUG DELETE LATER:
        try:
            with open(conf_file, "r") as f:
                for line in f:
                    exec(line)
                    print(
                        f"[DEBUG] [Rank {id}] Read line from conf file: {line}",
                        file=sys.stderr,
                        flush=True,
                    )
            montepython_path = path["montepython"]
            print(
                f"[DEBUG] [Rank {id}] Read montepython_path from conf file: {montepython_path}",
                file=sys.stderr,
                flush=True,
            )

        except Exception as e:
            print(
                f"[ERROR] [Rank {id}] Failed to read conf file {conf_file}: {repr(e)}",
                file=sys.stderr,
                flush=True,
            )
            raise

        if not hasattr(cls, "mp"):
            print(
                f"[DEBUG] [Rank {id}] First-time MontePython initialization",
                file=sys.stderr,
                flush=True,
            )

            print("Initializing MontePython on process for the first time.")
            os.makedirs(os.path.join(output_folder, f"montepython"), exist_ok=True)
            # with contextlib.redirect_stdout(open(os.path.join(output_folder, f'montepython/rank_{id}.out'), 'w')):
            #    with contextlib.redirect_stderr(open(os.path.join(output_folder, f'montepython/rank_{id}.err'), 'w')):
            #        cls.mp = cls.initialise_montepython(param_file, conf_file, montepython_path, output_folder, id)

            try:
                print(
                    f"[DEBUG] [Rank {id}] Trying to initialize MontePython",
                    file=sys.stderr,
                    flush=True,
                )
                cls.mp = cls.initialise_montepython(
                    param_file, conf_file, montepython_path, output_folder, id
                )
                print(
                    f"[DEBUG] [Rank {id}] MontePython initialized successfully",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[ERROR] [Rank {id}] MontePython initialization failed: {repr(e)}",
                    file=sys.stderr,
                    flush=True,
                )
                raise

        else:
            print(
                f"[DEBUG] [Rank {id}] MontePython was already initialized, returning existing instance",
                file=sys.stderr,
                flush=True,
            )
            print("MontePython was already initialized, returning existing instance.")
            if param_file != cls.mp["param"]:
                raise ValueError(
                    f"Tried to initialize MontePython with .param file {param_file}, which is different from {cls.mp['param']}, which was used initially."
                )
            elif conf_file != cls.mp["conf"]:
                raise ValueError(
                    f"Tried to initialize MontePython with .conf file {conf_file}, which is different from {cls.mp['conf']}, which was used initially."
                )
        return cls.mp

    def __reduce__(self) -> str | tuple[Any, ...]:
        return (None, None)

    def initialise_montepython(
        param_file, conf_file, montepython_path, output_folder, id
    ):
        print(
            f"[DEBUG] [Rank {id}] Entering initialise_montepython()",
            file=sys.stderr,
            flush=True,
        )
        """
        Perform the actual initialization of MontePython.

        Args:
            param_file: Path to the MontePython .param file.
            conf_file: Path to the MontePython .conf file.
            montepython_path: Path to the MontePython directory
            output_folder: Directory for MontePython outputs.
            id: Task identifier (e.g., MPI rank).

        Returns:
            A dictionary containing the MontePython instance and relevant attributes.
        """

        mp = {}
        sys.path.append(os.path.join(montepython_path, "montepython"))
        print(
            f"[DEBUG] [Rank {id}] Added {os.path.join(montepython_path, 'montepython')} to sys.path",
            file=sys.stderr,
            flush=True,
        )

        try:
            print(
                f"[DEBUG] [Rank {id}] Trying to import MontePython",
                file=sys.stderr,
                flush=True,
            )
            from initialise import initialise as mp_initialise
            import sampler

            print(
                f"[DEBUG] [Rank {id}] MontePython modules imported",
                file=sys.stderr,
                flush=True,
            )
            mp["compute_lkl"] = sampler.compute_lkl
            from io_mp import CosmologicalModuleError

            print(
                f"[DEBUG] [Rank {id}] Imported CosmologicalModuleError and compute_lkl",
                file=sys.stderr,
                flush=True,
            )

            mp["cosmo_soft_exception"] = CosmologicalModuleError
        except Exception:
            print(
                f"[ERROR] [Rank {id}] Failed to import MontePython: {repr(e)}",
                file=sys.stderr,
                flush=True,
            )
            raise ImportError(
                f"Could not import MontePython modules. Is the path in connect_public/source/lkl_filter_module/likelihood_calc.conf, {montepython_path}, correctly pointing to the /montepython_public directory?"
            )

        os.makedirs(os.path.join(output_folder, f"montepython"), exist_ok=True)
        os.makedirs(
            os.path.join(output_folder, f"montepython/rank_{id}"), exist_ok=True
        )

        print(
            f"[DEBUG] [Rank {id}] Created output directories",
            file=sys.stderr,
            flush=True,
        )

        mp_dir = os.path.join(output_folder, f"montepython/rank_{id}")

        print(
            f"[DEBUG] [Rank {id}] MontePython output directory: {mp_dir}",
            file=sys.stderr,
            flush=True,
        )

        mp_command_input = (
            f"run -p {param_file} --conf {conf_file} -o {mp_dir} --chain-number 0"
        )
        print(
            f"[DEBUG] [Rank {id}] MontePython command: {mp_command_input}",
            file=sys.stderr,
            flush=True,
        )
        mp["param"], mp["conf"] = param_file, conf_file
        # _, mp['data'], mp['command_line'], _ = mp_initialise(mp_command_input)

        print(
            f"[DEBUG] [Rank {id}] Trying to initialize MontePython with command: {mp_command_input}",
            file=sys.stderr,
            flush=True,
        )
        try:
            _, mp["data"], mp["command_line"], _ = mp_initialise(mp_command_input)
            print(
                f"[DEBUG] [Rank {id}] MontePython initialization completed",
                file=sys.stderr,
                flush=True,
            )
        except Exception as e:
            print(
                f"[ERROR] [Rank {id}] MontePython init function failed: {repr(e)}",
                file=sys.stderr,
                flush=True,
            )
            raise

        mp["err_dir"] = os.path.join(output_folder, f"montepython/rank_{id}.err")
        mp["out_dir"] = os.path.join(output_folder, f"montepython/rank_{id}.out")
        return mp


class MontePythonLikelihoodCalculator(BaseLikelihoodCalculator):
    def initialise(self, param_connect, id):
        self.mp = InitialiseMontePython(param_connect, id)
        self.dimension = len(list(self.mp["data"].mcmc_parameters.keys()))
        self.output_folder = os.path.join("data", param_connect.jobname, "lkl_calc")
        self.param_connect = param_connect
        # self.config_kernel = config_kernel
        self.id = id

        from classy import CosmoSevereError

        self.severe_exception = CosmoSevereError
        self.computation_exception = self.mp["cosmo_soft_exception"]

    def _loglkl(self, position, cosmo):
        """
        Compute the likelihood for a given parameter set and CLASS instance.

        Args:
            position: Dictionary of parameter values (including fixed values).
            cosmo: Pre-computed CLASS instance with cosmological outputs.

        Returns:
            Negative log-likelihood (-log(likelihood)).
        """

        for param_name, param in self.param["varying"].items():
            self.mp["data"].mcmc_parameters[self.connect_to_mp_name[param_name]][
                "current"
            ] = position[param_name][0]
        for param_name, param in self.param["fixed"].items():
            self.mp["data"].mcmc_parameters[self.connect_to_mp_name[param_name]][
                "current"
            ] = param["fixed_value"]
        self.mp["data"].update_cosmo_arguments()
        self.mp["data"].need_cosmo_update = False  # Avoids recomputing cosmo every time
        with contextlib.redirect_stdout(open(self.mp["out_dir"], "a+")):
            with contextlib.redirect_stderr(open(self.mp["err_dir"], "a+")):
                out = self.mp["compute_lkl"](cosmo, self.mp["data"])
        return -out

    def set_parameter_dict(self):
        self.mp_to_connect_name = {}
        for param_name, param_dict in self.varying_param_dict.items():
            connect_param_name = self.format_param_name(param_name)
            self.param["varying"][connect_param_name] = {
                "range": param_dict["prior"].prior_range
            }
            self.mp_to_connect_name[param_name] = connect_param_name
        for param_name in self.get_mp_param_names("derived"):
            connect_param_name = self.format_param_name(param_name)
            self.param["derived"][connect_param_name] = {}
            self.mp_to_connect_name[param_name] = connect_param_name
        self.connect_to_mp_name = {
            val: key for key, val in self.mp_to_connect_name.items()
        }

    def format_param_name(self, param_name):
        # Translate a parameter name from MontePython format to CONNECT format
        connect_param = param_name
        if "*" in param_name:
            connect_param = param_name.replace("*", "")
        return connect_param

    def save_config(self):
        os.makedirs(
            os.path.join(self.output_folder, f"montepython/rank_{self.id}"),
            exist_ok=True,
        )
        shutil.copy(
            self.mp["conf"],
            os.path.join(self.output_folder, f"montepython/rank_{self.id}/log.conf"),
        )
        shutil.copy(
            self.mp["param"],
            os.path.join(self.output_folder, f"montepython/rank_{self.id}/log.param"),
        )

    def get_mp_param_names(self, type):
        return self.mp["data"].get_mcmc_parameters([type])

    @property
    def varying_param_dict(self):
        return {
            param_name: self.mp["data"].mcmc_parameters[param_name]
            for param_name in self.mp["data"].get_mcmc_parameters(["varying"])
        }
