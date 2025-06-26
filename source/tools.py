import os
import subprocess as sp

import numpy as np

CONNECT_PATH = os.path.split(os.path.realpath(os.path.dirname(__file__)))[0]


def get_computed_cls(cosmo, ell_array=[]):  # A computed CLASS model

    cls = cosmo.lensed_cl()

    if len(ell_array) == 0:
        # get parameters from CLASS model
        BA = cosmo.get_background()
        conformal_age_ = BA["conf. time [Mpc]"][-1]
        der_pars = cosmo.get_current_derived_parameters(["ra_rec", "tau_rec"])
        l_max = len(cls["ell"]) - 1
        angular_rescaling_ = der_pars["ra_rec"] / (conformal_age_ - der_pars["tau_rec"])
        l_linstep = 40
        l_logstep = 1.12

        # compute necessary ells like CLASS (see transfer module in CLASS)
        increment = max(int(2 * (np.power(l_logstep, angular_rescaling_) - 1)), 1)
        l = [0, 1, 2]

        lin_increment = int(l_linstep * angular_rescaling_)
        while l[-1] + increment < l_max:
            if increment < lin_increment:
                l.append(l[-1] + increment)
                increment = max(
                    int(l[-1] * (np.power(l_logstep, angular_rescaling_) - 1)), 1
                )
            else:
                l.append(l[-1] + lin_increment)
        ell = np.array(l)
    else:
        ell = np.array(ell_array)

    # create reduced dict of cls
    new_cls = {"ell": ell}
    for key, arr in cls.items():
        if key != "ell":
            new_cls[key] = arr[ell]

    return new_cls


def get_z_idx(z):
    z = z.copy()
    if z[-1] > z[0]:
        grid = np.logspace(np.log10(z[1]), np.log10(z[-1]), 99)
    else:
        grid = np.logspace(np.log10(z[0]), np.log10(z[-2]), 99)
    z_idx = []
    for zg in grid:
        i = np.abs(z - zg).argmin()
        z[i] = np.inf
        z_idx.append(i)
    j = len(z) - 1
    while len(z_idx) < 100:
        if j in z_idx:
            j -= 1
        else:
            z_idx.append(j)
    return np.array(sorted(z_idx))


def create_output_folders(
    param,  # Parameter object
    iter_num=None,  # Current iteration number
    reset=True,  # Resets output folders
    resume=False,  # Resume from iteration
):
    if not resume:
        path = os.path.join(CONNECT_PATH, f"data/{param.jobname}")
        if not os.path.isdir(path):
            os.mkdir(path)
        if param.use_likelihood_filter and param.sampling == "iterative":
            path_lkl_calc = os.path.join(path, "lkl_calc")
            if not os.path.isdir(path_lkl_calc):
                os.mkdir(path_lkl_calc)

        if iter_num == None:
            if (reset and param.initial_model == None) or param.sampling == "lhc":
                for name in os.listdir(path):
                    if not name.startswith("N-") or name == f"N-{param.N}":
                        os.system(f"rm -rf {os.path.join(path, name)}")
                os.mkdir(os.path.join(path, f"N-{param.N}"))
                os.mkdir(os.path.join(path, f"N-{param.N}/model_params_data"))
                if param.use_likelihood_filter and param.sampling == "iterative":
                    os.mkdir(os.path.join(path, f"N-{param.N}/likelihood_data"))
                for output in param.output_Cl:
                    os.mkdir(os.path.join(path, f"N-{param.N}/Cl_{output}_data"))
                for output in param.output_Pk:
                    os.mkdir(os.path.join(path, f"N-{param.N}/Pk_{output}_data"))
                for output in param.output_bg:
                    output = output.replace("/", "\\")
                    os.mkdir(os.path.join(path, f"N-{param.N}/bg_{output}_data"))
                for output in param.output_th:
                    os.mkdir(os.path.join(path, f"N-{param.N}/th_{output}_data"))
                if len(param.output_derived) > 0:
                    os.mkdir(os.path.join(path, f"N-{param.N}/derived_data"))
                for output in param.extra_output:
                    os.mkdir(os.path.join(path, f"N-{param.N}/extra_{output}_data"))

            elif reset:
                for name in os.listdir(path):
                    if not name.startswith("N-"):
                        os.system(f"rm -rf {os.path.join(path, name)}")

        else:
            if reset:
                os.system(f"rm -rf {os.path.join(path, f'number_{iter_num}')}")
                os.mkdir(os.path.join(path, f"number_{iter_num}"))
            os.mkdir(os.path.join(path, f"number_{iter_num}/model_params_data"))
            if param.use_likelihood_filter and param.sampling == "iterative":
                os.mkdir(os.path.join(path, f"number_{iter_num}/likelihood_data"))
            for output in param.output_Cl:
                os.system(
                    f"rm -rf {os.path.join(path, f'number_{iter_num}/Cl_{output}_data')}"
                )
                os.mkdir(os.path.join(path, f"number_{iter_num}/Cl_{output}_data"))
            for output in param.output_Pk:
                os.system(
                    f"rm -rf {os.path.join(path, f'number_{iter_num}/Pk_{output}_data')}"
                )
                os.mkdir(os.path.join(path, f"number_{iter_num}/Pk_{output}_data"))
            for output in param.output_bg:
                output = output.replace("/", "\\")
                os.system(
                    f"rm -rf {os.path.join(path, f'number_{iter_num}/bg_{output}_data')}"
                )
                os.mkdir(os.path.join(path, f"number_{iter_num}/bg_{output}_data"))
            for output in param.output_th:
                os.system(
                    f"rm -rf {os.path.join(path, f'number_{iter_num}/th_{output}_data')}"
                )
                os.mkdir(os.path.join(path, f"number_{iter_num}/th_{output}_data"))
            if len(param.output_derived) > 0:
                os.system(
                    f"rm -rf {os.path.join(path, f'number_{iter_num}/derived_data')}"
                )
                os.mkdir(os.path.join(path, f"number_{iter_num}/derived_data"))
            for output in param.extra_output:
                os.system(
                    f"rm -rf {os.path.join(path, f'number_{iter_num}/extra_{output}_data')}"
                )
                os.mkdir(os.path.join(path, f"number_{iter_num}/extra_{output}_data"))


def join_data_files(param):  # Parameter object

    from source.join_output import CreateSingleDataFile

    CSDF = CreateSingleDataFile(param, CONNECT_PATH)
    CSDF.join()


def combine_sets_of_data_files(
    new_data,  # Data file for the new data (destination for combined data)
    old_data,  # Data file for the old data
    Pk=False,
    no_header=False,
):
    if not Pk:
        with open(new_data, "a") as f:
            with open(old_data, "r") as g:
                if no_header:
                    start_idx = 0
                else:
                    start_idx = 1
                for line in list(g)[start_idx:]:
                    f.write(line)
    else:
        with open(new_data, "r") as f:
            new_lines = list(f)
        with open(old_data, "r") as f:
            old_lines = list(f)

        lines_separated = {}
        z_keys = []
        for line in new_lines[1:] + old_lines[1:]:
            if line[0] == "#":
                z_keys.append(line)
                if z_keys[-1] not in lines_separated:
                    lines_separated[z_keys[-1]] = []
            else:
                lines_separated[z_keys[-1]].append(line)

        N = int(len(z_keys) / 2)
        with open(new_data, "w") as f:
            f.write(new_lines[0])
            for key in z_keys[:N]:
                f.write(key)
                for line in lines_separated[key]:
                    f.write(line)


def combine_iterations_data(
    param, iter_num  # Parameter object  # Current iteration number
):

    path_i = os.path.join(CONNECT_PATH, f"data/{param.jobname}/number_{iter_num}")
    if iter_num == 1:
        path_j = os.path.join(CONNECT_PATH, f"data/{param.jobname}/N-{param.N}")
    else:
        path_j = os.path.join(CONNECT_PATH, f"data/{param.jobname}/number_{iter_num-1}")

    combine_sets_of_data_files(
        os.path.join(path_i, "model_params.txt"),
        os.path.join(path_j, "model_params.txt"),
    )

    if len(param.output_derived) > 0:
        combine_sets_of_data_files(
            os.path.join(path_i, "derived.txt"), os.path.join(path_j, "derived.txt")
        )
    for output in param.output_Cl:
        combine_sets_of_data_files(
            os.path.join(path_i, f"Cl_{output}.txt"),
            os.path.join(path_j, f"Cl_{output}.txt"),
        )
    for output in param.output_Pk:
        combine_sets_of_data_files(
            os.path.join(path_i, f"Pk_{output}.txt"),
            os.path.join(path_j, f"Pk_{output}.txt"),
            Pk=True,
        )
    for output in param.output_bg:
        output = output.replace("/", "\\")
        combine_sets_of_data_files(
            os.path.join(path_i, f"bg_{output}.txt"),
            os.path.join(path_j, f"bg_{output}.txt"),
        )
    for output in param.output_th:
        combine_sets_of_data_files(
            os.path.join(path_i, f"th_{output}.txt"),
            os.path.join(path_j, f"th_{output}.txt"),
        )
    for output in param.extra_output:
        combine_sets_of_data_files(
            os.path.join(path_i, f"extra_{output}.txt"),
            os.path.join(path_j, f"extra_{output}.txt"),
            no_header=True,
        )

    if param.use_likelihood_filter and param.sampling == "iterative":
        combine_sets_of_data_files(
            os.path.join(path_i, "likelihood_data.txt"),
            os.path.join(path_j, "likelihood_data.txt"),
            no_header=False,
        )


def get_node_with_most_cpus():

    job_info_list = sp.run(
        "scontrol show job -d $SLURM_JOB_ID", shell=True, stdout=sp.PIPE
    ).stdout.decode("utf-8")

    node_info_list = []
    for line in iter(job_info_list.splitlines()):
        if "CPU_IDs" in line:
            node_info_list.append(line)

    num_cpu = {}
    for i, line in enumerate(node_info_list):
        node_name = line.split("Nodes=")[-1].split(" CPU_IDs")[0]
        cpu_list = line.split("CPU_IDs=")[-1].split(" Mem")[0].split(",")
        n = 0
        for c in cpu_list:
            if "-" in c:
                cc = c.split("-")
                n += int(cc[1]) - int(cc[0]) + 1
            else:
                n += 1
        num_cpu[node_name] = n

    max_node = max(num_cpu, key=num_cpu.get)
    while "[" in max_node:
        _ = num_cpu.pop(max_node)
        max_node = max(num_cpu, key=num_cpu.get)

    return max_node


def get_covmat(path, param):
    cov = np.loadtxt(path)
    with open(path, "r") as f:
        header = f.readline().strip().replace("#", "").replace(" ", "").split(",")

    params_native = list(param.parameters.keys())
    params_custom = list(param.custom_parameters.keys())
    params = params_native + params_custom
    cov_new = np.zeros((len(params), len(params)))
    indices = []
    for i, p in enumerate(params):
        try:
            indices.append(header.index(p))
        except ValueError:
            indices.append(None)
            if p in params_native:
                cov_new[i, i] = (
                    (param.parameters[p][1] - param.parameters[p][0]) / 10
                ) ** 2
            elif p in params_custom:
                cov_new[i, i] = (
                    (
                        param.custom_parameters[p]["range"][1]
                        - param.custom_parameters[p]["range"][0]
                    )
                    / 10
                ) ** 2
            else:
                raise ValueError(f"Parameter {p} not found in parameter list")

    for i, idx_i in enumerate(indices):
        for j, idx_j in enumerate(indices):
            if idx_i is not None and idx_j is not None:
                cov_new[i, j] = cov[idx_i, idx_j]

    for i, p in enumerate(params):
        if p in param.sigma_guesses:
            cov_new[i, i] = param.sigma_guesses[p] ** 2

    return cov_new


def transform_custom_parameters(
    data: np.ndarray,
    all_param_names: list,
    native_params: dict,
    custom_parameters: dict,
):
    """
    Post-process an (N, d) array of both native + custom parameters.

    data: shape (N, d)
    all_param_names: list of length d, in the same order as data's columns
    native_params: dict of the form { 'omega_b': [min, max], ... }
    custom_parameters: dict of the form {
        'log10_lifetime_dcdm': {
            'maps_to': 'Gamma_dcdm',
            'range': [min, max],
        },
        ...
    }

    Returns: final_data (N, D) and final_names (list of length D)
    """
    N, d = data.shape

    # A dictionary that will hold the final columns for each "true" parameter name
    data_dict = {}

    # 1) Initialize the dictionary for any known "native_params" keys
    #    so we can store arrays by name
    for p in native_params.keys():
        data_dict[p] = None  # to be filled

    # Also add placeholders for each custom mapped name
    for custom_p, info in custom_parameters.items():
        mapped_name = info["maps_to"]
        data_dict[mapped_name] = None

    # 2) Go through each column in data
    for col_idx, param_name in enumerate(all_param_names):
        column = data[:, col_idx]  # shape (N,)

        # If it's already in native_params, store directly
        if param_name in native_params:
            data_dict[param_name] = column

        # Otherwise it must be a custom param
        elif param_name in custom_parameters:
            mapped_to = custom_parameters[param_name]["maps_to"]

            # ----------------------- Implement custom transformations here -----------------------#

            # Example: log10_lifetime_dcdm → Gamma_dcdm
            if param_name == "log10_lifetime_dcdm":
                # Hard-coded transformation: Gamma = const / (10^col * sec_per_year)
                SEC_PER_YEAR = np.float32(3.15576e7)
                SINV_TO_KM_S_MPC = np.float32(3.0856776e19)
                # Convert log10(lifetime_yr) -> lifetime_yr
                lifetime_yr = np.power(np.float32(10.0), column).astype(np.float32)
                # Then gamma_km_s_Mpc = ...
                gamma_km_s_Mpc = (
                    SINV_TO_KM_S_MPC / (lifetime_yr * SEC_PER_YEAR)
                ).astype(np.float32)
                data_dict[mapped_to] = gamma_km_s_Mpc.astype(np.float32)

            # Implement your own custom transformation here transforming from custom param → native param (maps_to)
            # Do the same transformation as the one implemented in MontePython's update_cosmo_arguments but modified
            # to the logic in the above example

            else:
                # if the user eventually wants more custom params, handle them here
                raise NotImplementedError(
                    f"No transformation defined for custom param '{param_name}'!"
                )

        else:
            # If it's in neither, raise an error or ignore
            raise ValueError(
                f"Parameter '{param_name}' not found in native 'parameters' or 'custom_parameters' dicts."
            )

    # 3) Build final_names in the exact same length/order as all_param_names
    #    but swap custom param name -> its mapped name in place
    final_names = []
    for param_name in all_param_names:
        if param_name in custom_parameters:
            final_names.append(custom_parameters[param_name]["maps_to"])
        else:
            final_names.append(param_name)

    # 4) Now column-stack in final_names order
    #    This means final_data.shape == (N, d) exactly
    final_data = np.column_stack([data_dict[name] for name in final_names])

    return final_data, final_names


def compare_dataframes(
    df1,
    df2,
    df_likelihood=None,
    df_likelihood2=None,
    comparison_type="new",
    verbose=1,
    compare_context=None,
):

    import pandas as pd

    """
    
    #######
    This function was initially developed for the 'plot_iterations.py' module used to analyze the iterative sampling process.
    But it can be used here as well to compare the data overlap between the final accepted data by the likelihood-filter and the percentage of new data from the chains.
    This gives information of how large percentage of the data is actually accepted, and helps us track if the likelihood-filter is too strict for the procedure to converge naturally.
    #######

    Compare two DataFrames (df1, df2) to identify samples that are 'new', 'removed', or 'common',
    while preserving one-to-one matching of duplicates. Also re-aligns likelihood data if provided.

    Parameters
    ----------
    df1 : pd.DataFrame
        The first DataFrame (e.g., the "current" iteration's data).
    df2 : pd.DataFrame
        The second DataFrame (e.g., the "previous" iteration's data).
    df_likelihood : pd.DataFrame, optional
        Likelihood rows aligned with df1 (same length, same row order as df1 BEFORE sorting).
    df_likelihood2 : pd.DataFrame, optional
        Likelihood rows aligned with df2 (same length, same row order as df2 BEFORE sorting).
    comparison_type : {'new','removed','common'}, default='new'
        - 'new': return rows in df1 that are not in df2.
        - 'removed': return rows in df2 that are not in df1.
        - 'common': return rows present in both df1 and df2.
    verbose : int, optional
        If >0, prints some debugging info.

    Returns
    -------
    matched_params : pd.DataFrame
        Subset of parameter rows that match the requested relationship,
        extracted from the correct perspective (df1 or df2, or intersection).
    matched_likelihood : pd.DataFrame or None
        Subset of likelihood rows that align with matched_params. If none given,
        returns None.
    """

    # --------------------------------------------------
    # Step 1: Validate input parameters
    # --------------------------------------------------
    valid_types = ["new", "removed", "common"]
    if comparison_type not in valid_types:
        raise ValueError(
            f"comparison_type must be one of {valid_types}, got: {comparison_type}"
        )

    # --------------------------------------------------
    # Step 2: Handle edge cases (if df1 or df2 is empty)
    # --------------------------------------------------
    if df1 is None or df1.empty:
        if verbose > 0:
            print(
                f'\n[compare_dataframes] [{compare_context["context"]}] df1 ({compare_context["df1"]}) is empty; returning trivial result:\n {compare_context["msg1"]}'
            )
        if comparison_type == "removed" and df2 is not None:
            return df2.copy().reset_index(drop=True), df_likelihood2
        return None, None

    if df2 is None or df2.empty:
        if verbose > 0:
            print(
                f'\n[compare_dataframes] [{compare_context["context"]}] df2 ({compare_context["df2"]}) is empty; returning trivial result:\n {compare_context["msg2"]}'
            )
        if comparison_type == "new":
            return df1.copy().reset_index(drop=True), df_likelihood
        return None, None

    # --------------------------------------------------
    # Step 2b: Ensure likelihood data and dataframes match in length
    if df_likelihood is not None and len(df_likelihood) != len(df1):
        raise ValueError(
            f'\nLength mismatch: df_likelihood ({len(df_likelihood)}) and df1 ({compare_context["df1"]}) ({len(df1)}) are not equal.'
        )
    if df_likelihood2 is not None and len(df_likelihood2) != len(df2):
        raise ValueError(
            f'\nLength mismatch: df_likelihood2 ({len(df_likelihood2)}) and df2 ({compare_context["df2"]}) ({len(df2)}) are not equal.'
        )

    # --------------------------------------------------
    # Step 3: Preprocess df1 (current iteration's data)
    # --------------------------------------------------
    df1 = df1.copy()
    df1["_temp_idx1"] = df1.index  # Store original row index before sorting

    # Identify the relevant parameter columns (excluding helper columns)
    param_cols = [c for c in df1.columns if c not in ["_temp_idx1", "dup_id"]]

    # Sort df1 so that identical samples appear together
    df1_sorted = df1.sort_values(param_cols, kind="mergesort").reset_index(drop=True)

    # Assign a 'dup_id' to each duplicate row so they can be matched one-to-one
    df1_sorted["dup_id"] = df1_sorted.groupby(param_cols).cumcount()

    # Reorder the likelihood data to match this new sorted order
    df_likelihood_sorted = (
        df_likelihood.iloc[df1_sorted["_temp_idx1"]].reset_index(drop=True)
        if df_likelihood is not None
        else None
    )

    # --------------------------------------------------
    # Step 4: Preprocess df2 (previous iteration's data)
    # --------------------------------------------------
    df2 = df2.copy()
    df2["_temp_idx2"] = df2.index

    df2_sorted = df2.sort_values(param_cols, kind="mergesort").reset_index(drop=True)
    df2_sorted["dup_id"] = df2_sorted.groupby(param_cols).cumcount()

    df_likelihood2_sorted = (
        df_likelihood2.iloc[df2_sorted["_temp_idx2"]].reset_index(drop=True)
        if df_likelihood2 is not None
        else None
    )

    # --------------------------------------------------
    # Step 5: Perform Merge to Find Matches
    # --------------------------------------------------
    """
    We now compare df1_sorted and df2_sorted to determine which samples belong to which category:
    
    - 'new': Samples in df1 but not in df2 (found using a LEFT JOIN)
    - 'removed': Samples in df2 but not in df1 (found using a RIGHT JOIN)
    - 'common': Samples that exist in both df1 and df2 (found using an INNER JOIN)

    The 'merge' function combines both dataframes based on their common parameter columns + 'dup_id'.
    This ensures that duplicate rows match correctly and one-to-one.
    
    The 'how' parameter controls which type of comparison we perform:
    
    - 'left' (for 'new'): Keeps all rows from df1_sorted, adds matches from df2_sorted.
    - 'right' (for 'removed'): Keeps all rows from df2_sorted, adds matches from df1_sorted.
    - 'inner' (for 'common'): Keeps only rows that exist in BOTH df1_sorted and df2_sorted.

    The 'indicator=True' adds a new column `_merge`, which labels each row as:
    - 'left_only'  → Present only in df1 (new sample)
    - 'right_only' → Present only in df2 (removed sample)
    - 'both'       → Present in both (common sample)
    """
    if comparison_type == "new":
        merge_type = "left"
        indicator = True
    elif comparison_type == "removed":
        merge_type = "right"
        indicator = True
    else:  # 'common'
        merge_type = "inner"
        indicator = False

    merged = df1_sorted.merge(
        df2_sorted,
        on=param_cols + ["dup_id"],
        how=merge_type,
        indicator=indicator,
        suffixes=("_df1", "_df2"),
    )

    # --------------------------------------------------
    # Step 6: Extract the Matching Rows from the Merge
    # --------------------------------------------------
    """
    Now that we have merged df1_sorted and df2_sorted, we extract the rows based on `_merge`:

    - For 'new': We filter only rows labeled as 'left_only' (i.e., samples that appear in df1 but not df2).
    - For 'removed': We filter only rows labeled as 'right_only' (samples in df2 but not df1).
    - For 'common': We take all merged rows, since they exist in both dataframes.
    """
    if comparison_type == "new":
        matched_df = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
    elif comparison_type == "removed":
        matched_df = merged[merged["_merge"] == "right_only"].drop(columns=["_merge"])
    else:  # 'common'
        matched_df = merged

    # Extract the original rows from df1 or df2
    matched_params = (
        df1.iloc[matched_df["_temp_idx1"]].copy()
        if comparison_type != "removed"
        else df2.iloc[matched_df["_temp_idx2"]].copy()
    )

    # Remove helper columns
    matched_params.drop(
        columns=["dup_id", "_temp_idx1", "_temp_idx2"],
        inplace=True,
        errors="ignore",
    )
    matched_params.reset_index(drop=True, inplace=True)

    # --------------------------------------------------
    # Step 7: Extract Aligned Likelihood Data
    # --------------------------------------------------
    matched_likelihood = None
    if comparison_type in ["new", "common"] and df_likelihood_sorted is not None:
        matched_likelihood = (
            df_likelihood.iloc[matched_df["_temp_idx1"]].copy().reset_index(drop=True)
        )
    elif comparison_type == "removed" and df_likelihood2_sorted is not None:
        matched_likelihood = (
            df_likelihood2.iloc[matched_df["_temp_idx2"]].copy().reset_index(drop=True)
        )

    return matched_params, matched_likelihood
