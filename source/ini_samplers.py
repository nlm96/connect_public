import os
import pickle as pkl
import itertools

import numpy as np
from scipy.stats import qmc

from source.tools import get_covmat, transform_custom_parameters


class LatinHypercubeSampler:
    def __init__(self, param):

        self.N = param.N
        self.native_parameters = param.parameters
        self.custom_parameters = (
            param.custom_parameters
        )  # dict of { 'log10_lifetime_dcdm': {'maps_to':..., 'range':[...]}, ... }
        self.native_param_names = list(self.native_parameters.keys())
        self.custom_param_names = list(self.custom_parameters.keys())
        self.param_names = list(self.native_param_names) + list(self.custom_param_names)
        self.d = len(self.param_names)
        self.log_priors = param.log_priors

    def run(self):
        sampler = qmc.LatinHypercube(d=self.d)
        sample = sampler.random(n=self.N)
        data = sample.T
        for i, name in enumerate(self.param_names):
            if name in self.log_priors:
                data[i] *= np.log10(self.native_parameters[name][1]) - np.log10(
                    self.native_parameters[name][0]
                )
                data[i] += np.log10(self.native_parameters[name][0])
                data[i] = np.power(10.0, data[i])

            elif name in self.custom_param_names:
                data[i] *= (
                    self.custom_parameters[name]["range"][1]
                    - self.custom_parameters[name]["range"][0]
                )
                data[i] += self.custom_parameters[name]["range"][0]
            else:
                data[i] *= (
                    self.native_parameters[name][1] - self.native_parameters[name][0]
                )
                data[i] += self.native_parameters[name][0]

        data = data.T

        # Post-process the data to transform custom parameters
        # 2) Now transform custom → native
        final_data, final_names = transform_custom_parameters(
            data, self.param_names, self.native_parameters, self.custom_parameters
        )

        return final_data


class HypersphereSampler:
    """
    Authors: Andreas Nygaard and Thomas Tram (2024)
    """

    def __init__(self, param):

        self.N = param.N
        self.native_parameters = param.parameters
        self.custom_parameters = param.custom_parameters

        self.native_param_names = list(self.native_parameters.keys())
        self.custom_param_names = list(self.custom_parameters.keys())
        self.param_names = self.native_param_names + self.custom_param_names

        self.d = len(self.param_names)
        self.surface = param.hypersphere_surface
        self.buffer_size = 100000

        bestfit_guesses = []
        for name in self.param_names:
            try:
                bestfit_guesses.append(param.bestfit_guesses[name])
            except:
                if name in self.native_param_names:
                    bestfit_guesses.append(
                        (param.parameters[name][0] + param.parameters[name][1]) / 2
                    )
                elif name in self.custom_param_names:
                    bestfit_guesses.append(
                        (
                            param.custom_parameters[name]["range"][0]
                            + param.custom_parameters[name]["range"][1]
                        )
                        / 2
                    )
                else:
                    raise ValueError(
                        f"Parameter {name} not found in native 'parameters' or 'custom_parameters' dicts."
                    )
        if type(param.temperature) in [int, float]:
            fac = (
                param.temperature * 2
            ) ** 2  # This factor is multiplied on the covmat and ensures the boundary of the
        else:  # hyperellipsoid to be at T*sigma, where T is the sampling temperature.
            fac = (param.temperature[0] * 2) ** 2

        seed = None
        self.rng = np.random.default_rng(seed=seed)
        if param.hypersphere_covmat is not None:
            cov_file = param.hypersphere_covmat
            cov = get_covmat(cov_file, param)
            self.L = np.linalg.cholesky(cov)

        self.bounds = np.array(list(self.native_parameters.values()))
        # Collect custom parameter bounds as a list of [min, max]
        custom_bounds = [
            self.custom_parameters[name]["range"] for name in self.custom_param_names
        ]

        # If there are custom parameters, convert to array and stack
        if custom_bounds:
            custom_bounds = np.array(custom_bounds)
            self.bounds = np.vstack((self.bounds, custom_bounds))

        self.bounds = np.insert(self.bounds, 1, bestfit_guesses, axis=1)
        # Compute bounding box around 0
        self.bbox = np.empty((self.bounds.shape[0], 2))
        self.bbox[:, 0] = self.bounds[:, 0] - self.bounds[:, 1]
        self.bbox[:, 1] = self.bounds[:, 2] - self.bounds[:, 1]

    def points_in_bbox(self, A, B):
        # Ensure that A and B have the same number of dimensions
        assert A.shape[0] == B.shape[0], "Dimensions of A and B must match"
        # Use boolean indexing for each dimension
        mask = np.all((A.T >= B[:, 0]) & (A.T <= B[:, 1]), axis=1)
        # Select points that satisfy the condition
        return A[:, mask]

    def gaussian_hypersphere(self, M):
        # Sample D vectors of N Gaussian coordinates
        samples = self.rng.normal(loc=0.0, scale=1.0, size=M * self.d).reshape(
            (self.d, M)
        )
        # Normalise all distances (radii) to 1
        radii = np.sqrt(np.sum(samples * samples, axis=0))
        samples = samples / radii
        # Sample N radii with exponential distribution
        # (unless points are to be on the surface)
        if self.surface:
            return samples
        new_radii = self.rng.uniform(low=0.0, high=1.0, size=M) ** (1.0 / self.d)
        return samples * new_radii

    def get_transformed_hypersphere_vector_with_bbox(self):
        while True:
            buffer = self.gaussian_hypersphere(self.buffer_size)
            if hasattr(self, "L"):
                buffer = self.L @ buffer
            selected_points = self.points_in_bbox(buffer, self.bbox).T
            for c in selected_points:
                yield c

    def run(self):
        data = np.array(
            list(
                itertools.islice(
                    self.get_transformed_hypersphere_vector_with_bbox(), self.N
                )
            )
        )
        data += self.bounds[:, 1]

        final_data, final_names = transform_custom_parameters(
            data, self.param_names, self.native_parameters, self.custom_parameters
        )
        return final_data


class PickleSampler:
    def __init__(self, param):
        self.pickle_data_file = param.pickle_data_file

    def run(self):
        with open(self.pickle_data_file, "rb") as f:
            data = pkl.load(f)
        return data
