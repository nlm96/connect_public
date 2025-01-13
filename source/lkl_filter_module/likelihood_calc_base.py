from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import numpy as np 
from typing import Any, Type
from types import NoneType
import yaml
from pathlib import Path

class LikelihoodCalculator(ABC):
    """
    Abstract base class for implementing likelihood calculators. 
    Subclasses must implement methods for initialization, parameter setup, 
    and likelihood computation. This class serves as the framework for 
    modular integration with various MCMC samplers or likelihood codes.
    """
    def __init__(self, param_connect, task_id, output_folder=None):
        """
        Initialize the likelihood calculator.

        Args:
            param_connect: Object containing CONNECT configuration parameters.
            task_id: Identifier for the current task (e.g., MPI rank).
            output_folder: Directory to save any configuration or output files.
        """
        self.set_default_errors()
        self.id = task_id
        self.param_connect = param_connect  # Store the configuration for later use
        self.output_folder = output_folder
        self.initialise(param_connect, self.id)

        self.param = {
            'varying': {},
            'fixed': {},
            'derived': {}
        }
        self.set_parameter_dict() 
        self.save_config()
    
    class NullException(Exception):
        pass

    def set_default_errors(self):
        """
            Two error types exist: 
                Severe exceptions will stop the process
                Computation exceptions will assign -inf as the 
                likelihood value of the proposed point and then continue the run

            Derivatives of BaseKernel can define their own specific error types.
        """
        # By default, all exceptions are severe
        self.severe_exception = Exception 
        self.computation_exception = self.NullException

    @abstractmethod
    def initialise(self, param_connect, task_id):
        pass

    def set_fixed_parameters(self, fixed_param_dict):
        for param_name, fixed_value in fixed_param_dict.items():
            # Move parameter to the fixed dict
            self.param['fixed'][param_name] = self.param['varying'][param_name]
            self.param['fixed'][param_name]['fixed_value'] = fixed_value
            del self.param['varying'][param_name]
    
    @property
    def varying_param_names(self):
        return list(self.param['varying'].keys())
    
    @abstractmethod
    def set_parameter_dict(self):
        """
            Must set self.param, a dict with keys that are parameter names
        """
        pass
    
    def save_config(self):
        """
        Saves kernel-specific data in the kernel subdir based on type
            - Analytical: Nothing saved
            - MontePython: .conf and .param
        """
        pass


    def loglkl(self, prop, cosmo):
        for fixed_param, param in self.param['fixed'].items():
            prop[fixed_param] = [param['fixed_value']]
        try:
            return self._loglkl(prop, cosmo)
        except self.computation_exception as e:
            print(f"Soft exception {e} occurred. Assigning inf to loglkl.")
            return np.inf
        except self.severe_exception:
            raise ValueError("Severe exception occurred in likelihood computation. Stopping process.")
    

    @abstractmethod
    def _loglkl(self, position):
        # Should return -log(likelihood); note the minus sign!
        pass

    

def create_likelihood_calc_input_files(param):
    """
    Create input files for initializing the likelihood calculator based on the MCMC sampler.
    Supports MontePython and Cobaya.

    Args:
        param: An object containing configuration parameters, including `param.mcmc_sampler`.
    """
    # Determine the path to save the input files
    path_lkl_calc = os.path.join("data", param.jobname, "lkl_calc")
    os.makedirs(path_lkl_calc, exist_ok=True)
    
    path = {}
    with open('source/lkl_filter_module/likelihood_calc.conf','r') as f:
        for line in f:
            exec(line)

    # Handle MontePython-specific input file creation
    if param.mcmc_sampler == "montepython":
        likelihood_calc_param_file = os.path.join(path_lkl_calc, "likelihood_calc.param")
        param_template_file = "mcmc_plugin/mp_param_templates/connect_lite.param.template"

        # Read template file and create the .param file
        with open(param_template_file, "r") as template:
            with open(likelihood_calc_param_file, "w") as output:
                skip_line = False
                for line in template:
                    # Check if the line or the next lines should be skipped
                    if "data.cosmo_arguments['connect_model']" in line:
                        continue
                    elif "# Connect model name used by plugin" in line:
                        continue

                    # Copy the template content and customize it for likelihood calculations
                    output.write(line)

                    # Add experiments
                    if "#------Experiments to test (separated with commas)-----" in line:
                        output.write('\n')  # Match existing function's formatting
                        experiments_line = "data.experiments=["
                        for lkl in param.sampling_likelihoods:
                            if lkl == "Planck_lite":
                                lkl = "Planck_highl_TTTEEE_lite"
                            elif lkl == "Planck_lowl_EE":
                                lkl = "Planck_lowl_EE_connect"
                            experiments_line += f"'{lkl}', "
                        experiments_line = experiments_line[:-2] + "]"
                        output.write(experiments_line + "\n")

                    # Add cosmological parameters
                    elif "# Cosmological parameters list" in line:
                        output.write('\n')  # Match existing function's formatting
                        for par, interval in param.parameters.items():
                            xmin = param.prior_ranges.get(par, [None, None])[0]
                            xmax = param.prior_ranges.get(par, [None, None])[1]
                            guess = param.bestfit_guesses.get(par, (interval[0] + interval[1]) / 2)
                            sig = param.sigma_guesses.get(par, abs((interval[1] - interval[0]) / 50))

                            scale = 0.01 if par in ["omega_b", "Omega_b"] else 1
                            if scale != 1:
                                guess *= 1 / scale
                                sig *= 1 / scale
                                xmin = xmin * 1 / scale if xmin is not None else None
                                xmax = xmax * 1 / scale if xmax is not None else None

                            if par in param.log_priors:
                                xmin = np.log10(xmin) if xmin is not None else None
                                xmax = np.log10(xmax) if xmax is not None else None
                                guess = np.log10(guess)
                                if xmin is not None and xmax is not None:
                                    sig = sig * (xmax - xmin) / (10 ** xmax - 10 ** xmin)
                                else:
                                    sig = 0.01
                                par += "_log10_prior"

                            output.write(f"data.parameters['{par}'] = [{guess}, {xmin}, {xmax}, {sig}, {scale}, 'cosmo']\n")

                    # Add derived parameters
                    elif "# Derived parameter list" in line:
                        output.write('\n')  # Match existing function's formatting
                        for par in param.output_derived:
                            scale = 1e-9 if par == "A_s" else 1
                            output.write(f"data.parameters['{par}'] = [1, None, None, 0, {scale}, 'derived']\n")
        print(f"MontePython input file created: {likelihood_calc_param_file}")

    elif param.mcmc_sampler == "cobaya":
        # Initialize the Cobaya info dictionary
        info = {
            'likelihood': {},
            'params': {},
            'theory': {}
        }

        # Paths configuration
        # Assuming 'path' dictionary is defined somewhere
        # You need to define 'path['clik']' to point to the Planck likelihoods

        # Define the path to the Planck likelihoods
        path_clik = os.path.join(Path(path['clik']).parents[2], 'baseline/plc_3.0/')

        # Map CONNECT likelihood names to Cobaya likelihood names and paths
        lkls = {'Planck_highl_TTTEEE_lite': {'name': 'planck_2018_highl_plik.TTTEEE_lite',
                                            'clik': os.path.join(path_clik, 'hi_l/plik_lite')},
                'Planck_lowl_TT':           {'name': 'planck_2018_lowl.TT_clik',
                                            'clik': os.path.join(path_clik, 'low_l/commander')},
                'Planck_lowl_EE':           {'name': 'planck_2018_lowl.EE_clik',
                                            'clik': os.path.join(path_clik, 'low_l/simall')}}




        for lkl in param.sampling_likelihoods:
            if lkl == 'Planck_lite':
                lkl = 'Planck_highl_TTTEEE_lite'
            if lkl in lkls:
                for name in os.listdir(lkls[lkl]['clik']):
                    if 'TTTEEE' in lkl and name.endswith('TTTEEE.clik'):
                        clik_file = os.path.join(lkls[lkl]['clik'], name)
                        break
                    elif 'EE' in lkl and name.endswith('.clik') and 'BB' not in name:
                        clik_file = os.path.join(lkls[lkl]['clik'], name)
                        break
                    elif 'TT' in lkl and name.endswith('.clik'):
                        clik_file = os.path.join(lkls[lkl]['clik'], name)
                        break
                info['likelihood'][lkls[lkl]['name']] = {'clik_file':   clik_file}

            else:
                raise NotImplementedError(f"For now, only the following three likelihoods are available during training:\n{' '*4}Planck_highl_TTTEEE_lit, Planck_lowl_TT, Planck_lowl_EE\nYou can manually add extra likelihoods as a nested dictionary using Cobaya syntax in the parameter file, e.g.\n{' '*4}"+"extra_cobaya_lkls = {'Likelihood_name': {'path': path/to/likelihood,\n"+f"{' '*52}'options': other_options,\n{' '*52}"+"...},\n"+f"{' '*44}"+"...}")

        for name, item in param.extra_cobaya_lkls.items():
            info['likelihood'][name] = item

        print(info['likelihood'])

        for par,interval in param.parameters.items():
            if par in param.prior_ranges:
                if param.prior_ranges[par][0] != 'None':
                    xmin = param.prior_ranges[par][0]
                else:
                    xmin = -1e+32
                if param.prior_ranges[par][1] != 'None':
                    xmax = param.prior_ranges[par][1]
                else:
                    xmax =  1e+32
            else:
                xmin = -1e+32
                xmax =  1e+32
            if par in param.bestfit_guesses:
                guess = param.bestfit_guesses[par]
            else:
                guess = (interval[0] + interval[1])/2
            if par in param.sigma_guesses:
                sig = param.sigma_guesses[par]
            else:
                sig = abs((interval[1] - interval[0])/10)
            if par in param.log_priors:
                if xmin != -1e+32:
                    xmin = np.log10(xmin)
                else:
                    xmin = 0
                if xmax != 1e+32:
                    xmax = np.log10(xmax)
                guess = np.log10(guess)
                if xmin != -1e+32 and xmax != 1e+32:
                    sig = sig * (xmax-xmin)/(10**xmax-10**xmin)
                else:
                    sig = sig * ( abs( np.log10( abs(interval[1]) ) - np.log10( abs(interval[0]) ) ) /
                                abs( interval[1] - interval[0] ) )
                par = par + '_log10_prior'
            proposal = sig
            if par.startswith('ln10^{10}A_s'):
                info['params']['logA'] = {}
                info['params']['logA']['prior'] = {}
                info['params']['logA']['ref'] = {'dist': 'norm'}
                info['params']['logA']['prior']['min'] = xmin
                info['params']['logA']['prior']['max'] = xmax
                info['params']['logA']['ref']['loc']   = guess
                info['params']['logA']['ref']['scale'] = sig
                info['params']['logA']['proposal'] = proposal
                info['params']['logA']['latex'] = par
                info['params']['logA']['drop'] = True
                info['params']['A_s'] = {}
                info['params']['A_s']['latex'] = 'A_s'
                info['params']['A_s']['value'] = 'lambda logA: 1e-10*np.exp(logA)'
            elif par.startswith('100*theta_s'):
                info['params']['theta_s_100'] = {}
                info['params']['theta_s_100']['prior'] = {}
                info['params']['theta_s_100']['ref'] = {'dist': 'norm'}
                info['params']['theta_s_100']['prior']['min'] = xmin
                info['params']['theta_s_100']['prior']['max'] = xmax
                info['params']['theta_s_100']['ref']['loc']   = guess
                info['params']['theta_s_100']['ref']['scale'] = sig
                info['params']['theta_s_100']['proposal'] = proposal
                info['params']['theta_s_100']['latex'] = par
                info['params']['theta_s_100']['drop'] = True
                info['params']['100*theta_s'] = {}
                info['params']['100*theta_s']['latex'] = par
                info['params']['100*theta_s']['value'] = 'lambda theta_s_100: theta_s_100'
                info['params']['100*theta_s']['derived'] = False
            else:
                info['params'][par] = {}
                info['params'][par]['prior'] = {}
                info['params'][par]['ref'] = {'dist': 'norm'}
                info['params'][par]['prior']['min'] = xmin
                info['params'][par]['prior']['max'] = xmax
                info['params'][par]['ref']['loc']   = guess
                info['params'][par]['ref']['scale'] = sig
                info['params'][par]['proposal'] = proposal
                info['params'][par]['latex'] = par

        for par in param.output_derived:
            if par == 'A_s' and 'ln10^{10}A_s' in param.parameters:
                info['params']['A'] = {}
                info['params']['A']['derived'] = 'lambda A_s: 1e9*A_s'
                info['params']['A']['latex'] = '10^9 A_s'
            elif par == '100*theta_s':
                info['params']['theta_s_100'] = {}
                info['params']['theta_s_100']['latex'] = par
            else:
                info['params'][par] = {}
                info['params'][par]['latex'] = par

        # Add theory code
        info['theory'] = {'classy': {}}

        # Write the 'info' dictionary to a YAML file
        yaml_file = os.path.join(path_lkl_calc, "likelihood_calc.yaml")
        with open(yaml_file, "w") as f:
            yaml.dump(info, f, default_flow_style=False)

        print(f"Cobaya input file created: {yaml_file}")