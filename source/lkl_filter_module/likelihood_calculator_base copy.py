from abc import ABC, abstractmethod  # Standard library: Abstract Base Classes
from dataclasses import dataclass     # Standard library: For data classes
import os                             # Standard library: Operating system interfaces
import numpy as np                    # Common external library: Numerical operations (requires NumPy package)
from typing import Any, Type          # Standard library: Type hinting support
from types import NoneType            # Standard library: Type definitions (Note: NoneType may require Python 3.10+)
from prospect.input import InputArgument  # Custom module: Specific to PROSPECT (may need to adjust or replace)

class BaseKernel(ABC):
    """
    Abstract base class for the likelihood calculator kernel.
    It defines the interface and common methods for different backends (e.g., MontePython, Cobaya).
    """
    def __init__(self, config_kernel, task_id, output_folder=None):
        self.set_default_errors()  # Initialize default error handling
        self.id = task_id          # Identifier for the task (may not be needed if only computing likelihoods)
        self.initialise(config_kernel, output_folder)  # Backend-specific initialization

        # Dictionary to store parameters categorized as varying, fixed, or derived
        self.param = {
            'varying': {},
            'fixed': {},
            'derived': {}
        }
        self.set_parameter_dict()  # Set up the parameter dictionary (to be implemented in subclasses)
        self.save_config()         # Save kernel-specific configuration (may not be needed for your use case)
    
    class NullException(Exception):
        """
        Custom exception class used for handling computation exceptions.
        """
        pass

    def set_default_errors(self):
        """
        Defines default exceptions for error handling.
        There are two types of exceptions:
            - severe_exception: Stops the process (e.g., critical errors)
            - computation_exception: Assigns infinity to the likelihood value and continues (e.g., invalid parameters)
        """
        # By default, all exceptions are considered severe
        self.severe_exception = Exception 
        self.computation_exception = self.NullException  # Custom exception for computation errors

    @abstractmethod
    def initialise(self, kernel_param):
        """
        Abstract method to initialize the kernel.
        Must be implemented in subclasses for specific backends (MontePython, Cobaya).
        """
        pass

    def set_fixed_parameters(self, fixed_param_dict):
        """
        Moves parameters from 'varying' to 'fixed' category and assigns fixed values.
        """
        for param_name, fixed_value in fixed_param_dict.items():
            # Move parameter to the 'fixed' dictionary
            self.param['fixed'][param_name] = self.param['varying'][param_name]
            self.param['fixed'][param_name]['fixed_value'] = fixed_value
            del self.param['varying'][param_name]
    
    @property
    def varying_param_names(self):
        """
        Returns a list of names of the varying parameters.
        """
        return list(self.param['varying'].keys())
    
    @abstractmethod
    def set_parameter_dict(self):
        """
        Abstract method to set up the parameter dictionary.
        Must populate self.param with parameter information.
        """
        pass

    def save_config(self):
        """
        Saves kernel-specific configuration data.
        For example, saves .conf and .param files for MontePython.
        May not be necessary if configurations are managed elsewhere.
        """
        pass

    def loglkl(self, prop):
        """
        Computes the negative log-likelihood (-logL) for a given set of parameters.
        Adds fixed parameters to the proposal and calls the backend-specific _loglkl method.
        Handles exceptions accordingly.
        """
        # Add fixed parameters to the proposal dictionary
        for fixed_param, param in self.param['fixed'].items():
            prop[fixed_param] = [param['fixed_value']]
        try:
            return self._loglkl(prop)  # Call the backend-specific likelihood computation
        except self.computation_exception as e:
            print(f"Soft exception {e} occurred. Trying a new proposal.")
            return np.inf  # Assign infinity to the likelihood (indicates failure)
        except self.severe_exception:
            raise ValueError("Severe exception occurred in likelihood computation. Stopping process.")
    
    def log_uniform_prior(self, position):
        """
        Computes the log of a uniform prior for the given position.
        Returns zero if within bounds, infinity if outside bounds.
        """
        if self.outside_of_prior_bound(position):
            return np.inf  # Outside prior bounds
        # Note: Different normalization than Cobaya, same as MontePython
        return 0.0  # Uniform prior contributes zero to the log-prior

    def outside_of_prior_bound(self, prop):
        """
        Checks if the proposed parameters are outside the prior bounds.
        """
        outside = False
        for param_name, param_val in prop.items():
            if param_name in self.param['varying']:
                prior = self.param['varying'][param_name]['range']
                if prior[0] is not None:
                    if param_val[0] < prior[0]:
                        outside = True
                        break
                if prior[1] is not None:
                    if param_val[0] > prior[1]:
                        outside = True
                        break
        return outside

    @abstractmethod
    def _loglkl(self, position):
        """
        Abstract method for computing the negative log-likelihood.
        Must be implemented in subclasses for specific backends.
        Should return -log(likelihood) (note the negative sign).
        """
        pass

    @abstractmethod
    def logprior(self, position):
        """
        Abstract method for computing the negative log-prior.
        Must be implemented in subclasses if non-uniform priors are used.
        Should return -log(prior).
        """
        pass

    def logpost(self, position):
        """
        Computes the negative log-posterior by combining likelihood and prior.
        """
        return self.loglkl(position) + self.logprior(position)

    @abstractmethod
    def get_default_initial_position(self):
        """
        Abstract method to provide a default initial position for parameters.
        """
        pass

    @abstractmethod
    def read_initial_position(self, config_initial_position):
        """
        Abstract method to read initial positions from user input.
        """
        pass

    @abstractmethod
    def get_default_covmat(self):
        """
        Abstract method to provide a default covariance matrix.
        May not be necessary if you're not using MCMC sampling.
        """
        pass

    @abstractmethod
    def read_covmat(self, config_covmat):
        """
        Abstract method to read covariance matrix from user input.
        May not be necessary if you're not using MCMC sampling.
        """
        pass

    def finalize(self):
        """
        Finalizes the kernel, performing any necessary cleanup.
        """
        pass

"""
Definition of user arguments related to base_kernel.py.
These are input parameters that can be provided in configuration files or scripts.
"""

@dataclass
class Arguments:
    # Each class variable represents an input argument with validation.

    class type(InputArgument):
        """
        Specifies the type of kernel to use (e.g., 'analytical', 'montepython', 'cobaya').
        """
        val_type = str
        allowed_values = ['analytical', 'montepython', 'cobaya']
    
    class param(InputArgument):
        """
        Path to the parameter file (.param for MontePython, .yaml for Cobaya).
        """
        val_type = str  # Either a path to a file or a dictionary if 'analytical' type
        def validate(self, config: dict[str, Any]) -> None:
            if not os.path.isfile(os.path.join(os.getcwd(), config['param'])):
                raise ValueError(f"The file pointed to in the 'param' field of the 'kernel' input, which has the value {config['param']}, could not be found.")
    
    class conf(InputArgument):
        """
        Path to the configuration file (.conf for MontePython).
        Not used for 'analytical' or 'cobaya' types.
        """
        val_type = str
        def get_default(self, config_yaml: dict[str, Any]):
            if config_yaml['kernel']['type'] in ['analytical', 'cobaya']:
                return ''
            return None
        def validate(self, config: dict[str, Any]) -> None:
            if config['type'] == 'montepython':
                if not os.path.isfile(os.path.join(os.getcwd(), config['conf'])):
                    raise ValueError(f"The file pointed to in the 'conf' field of the 'kernel' input, which has the value {config['conf']}, could not be found.")
    
    class path(InputArgument):
        """
        Path to the backend software (e.g., MontePython directory).
        Not used for 'analytical' or 'cobaya' types.
        """
        val_type = str
        def get_default(self, config_yaml: dict[str, Any]):
            if config_yaml['kernel']['type'] in ['analytical', 'cobaya']:
                return ''
            return None
        def validate(self, config: dict[str, Any]) -> None:
            if config['type'] == 'montepython':
                if not os.path.isdir(os.path.join(os.getcwd(), config['path'])):
                    raise ValueError(f"The directory pointed to in the 'path' field of the 'kernel' input, which has the value {config['path']}, could not be found.")

    class debug(InputArgument):
        """
        Debug mode flag for Cobaya.
        Drops debug files in the kernel subfolder when True.
        """
        val_type = bool
        def get_default(self, config_yaml: dict[str, Any]):
            return False

    # Definition of the arguments as class variables
    type: type
    param: param
    conf: conf
    path: path
    debug: debug


"""


### **Analysis of the Code and Suggestions**

**Imports:**

- **Standard Libraries:**
  - `abc`: Provides support for abstract base classes.
  - `dataclasses`: Provides a decorator and functions for creating data classes.
  - `os`: Provides a way of using operating system dependent functionality.
  - `typing`: Provides support for type hints.
  - `types`: Defines utility functions and names for built-in types.

- **External Libraries:**
  - `numpy` (`np`): A fundamental package for numerical computations. You'll need to have NumPy installed (usually already installed in most environments).

- **Custom Modules:**
  - `from prospect.input import InputArgument`: This is a custom module specific to PROSPECT.

**Notes on `NoneType` Import:**

- **`from types import NoneType`**:
  - In Python versions prior to 3.10, `NoneType` is not directly available from `types`.
  - You may need to adjust this import or define `NoneType` as `type(None)`.

**Code Structure:**

- The `BaseKernel` class serves as an abstract base class that defines the interface for different kernels (backends). It includes abstract methods that must be implemented by subclasses (e.g., MontePython and Cobaya kernels).

**Methods and Attributes:**

1. **`__init__` Method:**
   - Initializes error handling and calls `initialise`, `set_parameter_dict`, and `save_config`.

2. **`set_default_errors`:**
   - Defines default exceptions for error handling.

3. **`initialise`:**
   - Abstract method to be implemented by subclasses for backend-specific initialization.

4. **`set_parameter_dict`:**
   - Abstract method to set up the parameter dictionary (`self.param`).

5. **`save_config`:**
   - Saves kernel-specific configuration data (may not be necessary for computing likelihood values).

6. **`loglkl`:**
   - Computes the negative log-likelihood by adding fixed parameters and calling `_loglkl`.

7. **`_loglkl`:**
   - Abstract method to compute the negative log-likelihood; must be implemented in subclasses.

8. **`logprior`, `logpost`:**
   - Methods related to prior computations (may not be needed if using uniform priors or if prior handling is done elsewhere).

9. **`get_default_initial_position`, `read_initial_position`, `get_default_covmat`, `read_covmat`:**
   - Methods related to MCMC sampling and covariance matrices (likely not needed for your use case).

10. **`finalize`:**
    - Placeholder method for any cleanup (optional).

**Relevance to Your Use Case:**

- **Methods to Keep:**
  - `__init__`: Essential for initializing the class.
  - `set_default_errors`: Useful for error handling.
  - `initialise`: Must be implemented in subclasses.
  - `set_parameter_dict`: Necessary to set up parameters.
  - `loglkl`: Core method for computing likelihoods.
  - `_loglkl`: Must be implemented in subclasses.
  - `outside_of_prior_bound`, `log_uniform_prior`: If you need to handle prior bounds.

- **Methods You Can Consider Removing or Simplifying:**
  - `logprior`, `logpost`: If you are only computing likelihoods and not dealing with priors or posterior computations.
  - `get_default_initial_position`, `read_initial_position`: Related to MCMC sampling; likely unnecessary.
  - `get_default_covmat`, `read_covmat`: Covariance matrices are used in MCMC proposals; may not be needed.
  - `save_config`: If you don't need to save configuration files.

**Dependencies on Custom Modules:**

- **`prospect.input.InputArgument`:**
  - This is a custom class from the PROSPECT package.
  - If you aim to minimize dependencies, you may need to define `InputArgument` yourself or adjust the code to work without it.

**Adjustments for Simplification:**

- **Remove Unnecessary Methods:**
  - If you are not performing MCMC sampling or handling priors, you can remove or comment out methods related to these functionalities.

- **Simplify `Arguments` Class:**
  - If you are not using PROSPECT's configuration system, you may replace `InputArgument` with a simple data class or another configuration method.

- **Handle Imports:**
  - Ensure all imports are available in your environment.
  - Remove or replace imports from PROSPECT if not needed.

**Cobaya Considerations:**

- The code includes placeholders for Cobaya (`allowed_values = ['analytical', 'montepython', 'cobaya']`).

- To ensure compatibility with Cobaya later, keep the structure that allows for different kernel types, but you can focus on implementing the MontePython-related parts first.

---

### **Next Steps**

- **Implement Subclasses:**
  - Create a subclass of `BaseKernel` for MontePython (e.g., `MontePythonLikelihoodCalculator`) and implement the abstract methods (`initialise`, `set_parameter_dict`, `_loglkl`, etc.).

- **Adjust Imports and Dependencies:**
  - Replace or adjust imports from PROSPECT as needed to eliminate unnecessary dependencies.

- **Simplify Configuration Handling:**
  - If you prefer, replace the `Arguments` class with a simpler configuration system that fits your use case.

- **Test with MontePython:**
  - Focus on getting the likelihood computation working with MontePython before adding Cobaya support.

- **Plan for Cobaya Integration:**
  - Keep in mind the structure needed to add Cobaya support later, but you can defer implementing it until after MontePython is working.

---

I hope these comments and explanations help you understand the code and decide what to keep or remove for your likelihood computation use case. If you have any further questions or need assistance with specific parts of the code, feel free to ask!



"""