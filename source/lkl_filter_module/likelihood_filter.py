import os
import numpy as np


class LikelihoodFilter:
    """
    A class to filter training data samples based on their "true" calculated -log(likelihood) values
    during iterative sampling in CONNECT.
    The samples are filtered by:
    - Comparing the difference in chi-squared values between a point and the best-fit point.
    - Discarding points that exceed a specified threshold:
        if delta_chi2(point) > delta_chi2_threshold, the point is discarded.
    - Retaining a minimum number of points to avoid discarding too many points:
        if the number of points to retain is less than min_points_to_keep, the filter stops early.

    In short, the filter updates the training data files by:
    - saving the retained points (which meet the threshold) in the original files.
    - saving the discarded points in a separate subdirectory, loglkl_discarded_data, but with the same file names as the original files.

    Parameters:
    - param: The parameter object from CONNNECT containing all input parameters, including filter settings, such as:
        - delta_chi2_threshold: The threshold for the difference in chi-squared values between a point and the best-fit point.
        - min_points_to_keep: The minimum number of points to retain after filtering, to avoid discarding too many points.
        - discard_worst_first: If True, the filter will discard the worst points first. Otherwise, it will filter sequentially, discarding older iteration points first.
    - iter_num: The iteration number for which the filter is being applied. 0 for initial sampling.
    - CONNECT_PATH: Path to the root directory of the CONNECT data files.
    """

    def __init__(self, param, iter_num, CONNECT_PATH, strict_filtering=False):
        """
        Initializes the LikelihoodFilter instance.

        Args:
            param: CONNECT configuration parameters, including filter settings.
            iter_num: Current iteration number; None for initial sampling.
            CONNECT_PATH: Path to the CONNECT working directory.
        """

        self.iter_num = iter_num
        self.param = param
        self.strict_filtering = strict_filtering
        self.delta_chi2_threshold = self.param.delta_chi2_threshold
        self.delta_chi2_accepted = None
        self.delta_chi2_discarded = None
        self.verbosity_level = 2
        self.filtering_strategy = ("absolute", None)  # or percentile or std deviations

        # If self.delta_chi2_threshold is a list, then it is a list of thresholds for each iteration.
        if isinstance(self.delta_chi2_threshold, list):
            if self.iter_num < len(self.delta_chi2_threshold):
                self.delta_chi2_threshold = self.delta_chi2_threshold[self.iter_num]
            else:
                self.delta_chi2_threshold = self.delta_chi2_threshold[-1]
        self.min_points_to_keep = self.param.min_points_to_keep

        # If self.min_points_to_keep is a list, then it is a list of minimum points to keep for each iteration.
        if isinstance(self.min_points_to_keep, list):
            if self.iter_num < len(self.min_points_to_keep):
                self.min_points_to_keep = self.min_points_to_keep[self.iter_num]
            else:
                self.min_points_to_keep = self.min_points_to_keep[-1]

        if self.strict_filtering:
            self.min_points_to_keep = 1000
            print(
                f'    Likelihood-filter: Strict filtering enabled. "Ignoring" min_points_to_keep by setting it to 1000.',
                flush=True,
            )

        self.path = os.path.join(CONNECT_PATH, f"data/{self.param.jobname}")
        if iter_num == 0:
            self.iteration_path = os.path.join(self.path, f"N-{self.param.N}")
        else:
            self.iteration_path = os.path.join(self.path, f"number_{iter_num}")
        self.discarded_data_path = os.path.join(
            self.iteration_path, "loglkl_discarded_data"
        )
        self.model_params_file = os.path.join(self.iteration_path, "model_params.txt")
        self.derived_file = os.path.join(self.iteration_path, "derived.txt")
        self.likelihood_file = os.path.join(self.iteration_path, "likelihood_data.txt")
        self.best_fit_file = os.path.join(self.iteration_path, "best_fit_point.txt")
        self.N_samples = self.get_total_samples()
        self.min_points_reached = False

        # Create subdirectory to store the filtered data if it does not exist
        if not os.path.exists(self.discarded_data_path):
            os.makedirs(self.discarded_data_path)

    def run(self):
        """
        Executes the likelihood filtering process by:
        - Finding the best-fit value and index in likelihood_data.txt.
        - Creating the best_fit_point.txt file with the best-fit parameters and derived values.
        - Finding the indices of points that do not meet the threshold, "discarded_indices".
        - Filtering each type of file based on the discard indices:
            - model_params.txt, derived.txt, Cl_output.txt, Pk_output.txt, bg_output.txt, th_output.txt, extra_output.txt, likelihood_data.txt.
        - Print statements to update the user on the filtering process.
        """

        best_fit_loglkl, best_fit_index = self.find_best_fit_loglkl()
        self.create_best_fit_file(best_fit_loglkl, best_fit_index)

        # Find the indices of points that do not meet the threshold
        discard_indices, total_potential_discard = self.find_discard_indices(
            best_fit_loglkl
        )

        # Filter each type of file based on discard indices
        self.filter_data("model_params.txt", discard_indices, Pk=False, no_header=False)

        if len(self.param.output_derived) > 0:
            self.filter_data("derived.txt", discard_indices, Pk=False, no_header=False)

        for output in self.param.output_Cl:
            self.filter_data(
                f"Cl_{output}.txt", discard_indices, Pk=False, no_header=False
            )

        for output in self.param.output_Pk:
            self.filter_data(
                f"Pk_{output}.txt", discard_indices, Pk=True, no_header=False
            )

        """ Commented out for now. As I encountered a case where one of the bg files was missing for some unknown reason during testing, I implemented the code below to handle such cases.
        But you probably want to use this commented out code instead:
            
        for output in self.param.output_bg:
            output = output.replace('/', '\\')
            self.filter_data(f'bg_{output}.txt', discard_indices, Pk=False, no_header=False)
        """

        for output in self.param.output_bg:
            output = output.replace("/", "\\")
            filename = f"bg_{output}.txt"
            original_file = os.path.join(self.iteration_path, filename)

            if os.path.exists(original_file):
                self.filter_data(filename, discard_indices, Pk=False, no_header=False)
            else:
                print(
                    f"    ERROR: {filename} does not exist and will be skipped for filtering.",
                    flush=True,
                )

        for output in self.param.output_th:
            self.filter_data(
                f"th_{output}.txt", discard_indices, Pk=False, no_header=False
            )

        for output in self.param.extra_output:
            self.filter_data(
                f"extra_{output}.txt", discard_indices, Pk=False, no_header=True
            )

        self.filter_data(
            "likelihood_data.txt", discard_indices, Pk=False, no_header=False
        )

        num_filtered = len(discard_indices)
        iteration = (
            f"iteration {self.iter_num}" if self.iter_num > 0 else "initial sampling"
        )
        formatted_best_fit = f"{best_fit_loglkl:.1f}"
        filter_method = (
            "Discarded worst points first."
            if self.param.discard_worst_first
            else "Filtered sequentially, discarding older iteration points first."
        )

        if self.delta_chi2_accepted is not None and self.delta_chi2_accepted.size > 0:
            worst_delta_chi2_accepted = np.max(self.delta_chi2_accepted)
        else:
            worst_delta_chi2_accepted = 0

        # Create summary statistics of self.delta_chi2_accepted and self.delta_chi2_discarded
        delta_chi2_summary = self.format_delta_chi2_summary()

        kind, v = self.filtering_strategy
        if kind == "percentile":
            note = f"({v*100:.0f}th percentile)"
        elif kind == "std deviations":
            note = f"(μ + {v:.1f}σ)"
        else:
            note = ""

        delta_chi2_string = (
            f"Δχ²-threshold: {self.delta_chi2_threshold} {note}."
            if self.delta_chi2_threshold < 99999
            else f"Δχ²-threshold: {self.delta_chi2_threshold:.1e} {note}."
        )
        min_points_to_keep_string = (
            f"Minimum points to keep: {self.min_points_to_keep}."
        )

        indent = "    "  # 4-space indent

        summary_lines = [
            f"{delta_chi2_string}",
            f"{min_points_to_keep_string}",
            f"Likelihood-filter discarded {num_filtered}/{self.N_samples} points in {iteration} ({total_potential_discard} exceeded threshold).",
            (
                f"Stopped early to retain ≥{self.min_points_to_keep} points."
                if self.min_points_reached
                else ""
            ),
            f"{filter_method}",
            f"Current global best-fit -log(lkl): {formatted_best_fit}.",
            (
                f"Worst 'accepted' Δχ²: {worst_delta_chi2_accepted:.1f}."
                if self.min_points_reached
                else ""
            ),
        ]

        if self.verbosity_level > 0:
            for line in summary_lines:
                if line:  # Skip empty lines
                    print(f"{indent}{line}", flush=True)

        if self.verbosity_level > 1:
            print(f"")
            for line in delta_chi2_summary.split("\n"):
                print(f"{indent}{line}", flush=True)

    def find_best_fit_loglkl(self):
        """Finds and returns the minimum -log(likelihood) value and its index in likelihood_data.txt.
        This corresponds to the -log(likelihood) value of the current global best-fit point.
        """
        likelihood_values = []
        with open(self.likelihood_file, "r") as f:
            for line in f:
                if not line.startswith("#"):
                    # Split the line into columns
                    columns = line.strip().split("\t")
                    # Extract the first column as the likelihood value
                    loglkl_value = float(columns[0])
                    likelihood_values.append(loglkl_value)

        best_fit_loglkl = min(likelihood_values)
        best_fit_index = likelihood_values.index(best_fit_loglkl)
        return best_fit_loglkl, best_fit_index

    def create_best_fit_file(self, best_fit_loglkl, best_fit_index):
        """Creates the best_fit_point.txt file using the best-fit "true" likelihood value and index.
        This file contains the calculated -log(likelihood) value, the model input parameters, and the derived values.
        """

        # Retrieve headers from model_params.txt and derived.txt
        with open(self.model_params_file, "r") as f:
            model_params_header = f.readline().strip().lstrip("#").strip().split("\t")
        if len(self.param.output_derived) > 0:
            with open(self.derived_file, "r") as f:
                derived_header = f.readline().strip().lstrip("#").strip().split("\t")

        # Combine headers with loglkl as the first entry
        if len(self.param.output_derived) > 0:
            full_header = "# " + "\t".join(
                ["loglkl"] + model_params_header + derived_header
            )
        else:
            full_header = "# " + "\t".join(["loglkl"] + model_params_header)

        # Retrieve the best-fit parameters and derived values
        with open(self.model_params_file, "r") as f:
            model_params_lines = f.readlines()
            best_fit_params = model_params_lines[best_fit_index + 1].strip().split("\t")

        if len(self.param.output_derived) > 0:
            with open(self.derived_file, "r") as f:
                derived_lines = f.readlines()
                best_fit_derived = derived_lines[best_fit_index + 1].strip().split("\t")

        # Combine the best-fit data with the likelihood value as the first entry
        if len(self.param.output_derived) > 0:
            best_fit_data = [str(best_fit_loglkl)] + best_fit_params + best_fit_derived
        else:
            best_fit_data = [str(best_fit_loglkl)] + best_fit_params

        # Write the combined header and best-fit data to `best_fit_point.txt`
        with open(self.best_fit_file, "w") as f:
            f.write(full_header + "\n")
            f.write("\t".join(best_fit_data) + "\n")

    def find_discard_indices(self, best_fit_loglkl):
        """
        Identifies indices of points to discard based on the Δχ² threshold:
            if Δχ²(point) > Δχ²_threshold, the point's index is added to the discard_indices list.
            as long as the number of points to retain is greater than min_points_to_keep.

        Parameters:
        - best_fit_loglkl: The global best-fit -log(likelihood) value.
        Returns:
        - discard_indices: Indices of points that exceed the Δχ² threshold.
        - total_potential_discard: Total number of points that exceed the threshold, for informational purposes when printing.
            it's the number of points that could be discarded if the minimum number of points to retain was 0.
        Additional Attributes:
        - min_points_to_keep: The minimum number of points to retain after filtering, to avoid discarding too many points.
        - min_points_reached: A flag indicating whether the minimum number of points to retain has been reached. Only used for informational purposes for printing.
        - discard_worst_first: If True, the filter will discard the worst points first. Otherwise, it will filter sequentially, discarding older iteration points first.
        - delta_chi2_threshold: The threshold for the difference in chi-squared values between a point and the best-fit point.
        """

        """
        In summary, this function:
        1. Reads the likelihood values from the likelihood_data.txt file.
        2. Calculates the delta_chi2 value for each data point.
        3. Sorts the delta_chi2 values from worst to best (if discard_worst_first is True).
        4. Filters out points that do not meet the threshold, while retaining a minimum number of points.
        It also counts the total number of points that exceed the threshold, which is used for informational purposes.
        5. Returns the indices of points to discard and the total number of points that exceed the threshold.
        """

        likelihood_values = []
        header_count = 0

        # Read likelihood values from the likelihood_data.txt file
        with open(self.likelihood_file, "r") as f:
            for idx, line in enumerate(f):
                if line.startswith("#"):
                    header_count += (
                        1  # Count header lines (likelihood_data.txt has 1 header line)
                    )
                    continue

                # Split the line into columns, first column is "true" -log(likelihood) value. 2nd column is "fake" -log(likelihood) value.
                columns = line.strip().split("\t")
                # Extract the first column as the likelihood value
                loglkl_value = float(columns[0])
                data_index = idx - header_count  # Adjust for header lines
                likelihood_values.append((data_index, loglkl_value))

        # Calculate delta_chi2 for each data point (Δχ² = 2 * (loglkl - best_fit_loglkl))
        # Stores the index and delta_chi2 value for each data point. The index corresponds to sample's position across all original data files.
        delta_chi2_values = [
            (idx, (loglkl - best_fit_loglkl) * 2) for idx, loglkl in likelihood_values
        ]

        if self.param.discard_worst_first:
            # Sort from highest to lowest delta_chi2 (worst to best)
            delta_chi2_values.sort(key=lambda x: x[1], reverse=True)
        else:
            # Keep the original order (as read from the file), i.e. sequential order.
            pass  # No action needed since delta_chi2_values are already in original order

        # Set self.delta_chi2_threshold if it is a decimale or string

        if (
            isinstance(self.delta_chi2_threshold, (int, float))
            and self.delta_chi2_threshold <= 1
        ):
            # If the threshold is a decimal determine the delta_chi2_threshold of that percentile of the data
            self.filtering_strategy = ("percentile", self.delta_chi2_threshold)
            self.delta_chi2_threshold = np.percentile(
                [delta_chi2 for _, delta_chi2 in delta_chi2_values],
                self.delta_chi2_threshold * 100,
            )
        elif isinstance(self.delta_chi2_threshold, str):
            # Assume the string is of the form "N sigma", where N is a number and keep the points up to mu + N * sigma
            try:
                N_sigma = float(self.delta_chi2_threshold.split()[0])
                mu = np.mean([delta_chi2 for _, delta_chi2 in delta_chi2_values])
                sigma = np.std([delta_chi2 for _, delta_chi2 in delta_chi2_values])
                self.delta_chi2_threshold = mu + N_sigma * sigma
                self.filtering_strategy = ("std deviations", N_sigma)
            except ValueError:
                raise ValueError(
                    f"Invalid delta_chi2_threshold format: {self.delta_chi2_threshold}. "
                    "Expected a decimal or a string of the form 'N sigma'."
                )

        # Filter out points that do not meet the threshold
        discard_indices = []  # List to store indices of points that will be discarded
        total_potential_discard = 0  # Count points that exceed the threshold
        for idx, delta_chi2 in delta_chi2_values:
            if delta_chi2 > self.delta_chi2_threshold:
                total_potential_discard += 1

                # Check if discarding this point would still retain the minimum required points
                if (self.N_samples - len(discard_indices)) > self.min_points_to_keep:
                    discard_indices.append(idx)
                else:
                    # Stop adding to discard_indices to ensure we retain enough points, but continue counting potential discards
                    self.min_points_reached = (
                        True  # Used for informational purposes only
                    )
            else:
                # If data is sorted, stop early since all subsequent points are within the threshold
                if self.param.discard_worst_first:
                    break  # Only applicable if sorted; else continue
                # If not sorted, we still need to check all points
                continue

        self.delta_chi2_accepted = np.array(
            [
                delta_chi2
                for idx, delta_chi2 in delta_chi2_values
                if idx not in discard_indices
            ]
        )
        self.delta_chi2_discarded = np.array(
            [
                delta_chi2
                for idx, delta_chi2 in delta_chi2_values
                if idx in discard_indices
            ]
        )
        self.delta_chi2_all = np.array(
            [delta_chi2 for _, delta_chi2 in delta_chi2_values]
        )

        return discard_indices, total_potential_discard

    def filter_data(self, filename, discard_indices, no_header=False, Pk=False):
        """
        Filters data in a file by copying discarded points to a subdirectory and removing them from the main file.
        Handles different structures for regular files and Pk files.

        Args:
            filename: Name of the file to filter.
            discard_indices: List of indices corresponding to points to be discarded.
            no_header: If True, indicates that the file does not contain a header row (e.g., extra_output.txt).
            Pk: If True, indicates that the file has a specific format for P(k) data.
        """
        # Original file path
        original_file = os.path.join(self.iteration_path, filename)
        # Filtered file path
        discarded_file = os.path.join(self.discarded_data_path, filename)
        # Check if the discard file already exists
        discarded_file_exists = os.path.exists(discarded_file)

        # Read all lines in the original file
        with open(original_file, "r") as f:
            lines = f.readlines()  # Read all lines in the original file

        if not Pk:
            # Process non-Pk files
            start_idx = 1 if not no_header else 0  # Skip header if present

            # Open in append (a) if it exists, else write (w)
            # Added append mode if the likehood filter is applied twice within the same iteration
            # This was added to accomplish a "strict filtering" mode, when the iterative process reached the maximum number of consecutive bad states
            # and enabled the strict filtering mode to force the filter discarding "ignoring" the min_points_to_keep parameter.
            # In most cases, the append mode will not be used.
            mode = "a" if discarded_file_exists else "w"

            # Write discarded points to the filtered file
            with open(discarded_file, mode) as f_discarded:
                if not no_header and not discarded_file_exists:
                    f_discarded.write(lines[0])  # Copy header if it exists
                for idx in discard_indices:
                    f_discarded.write(lines[idx + start_idx])

            # Write non-discarded points back to the original file
            with open(original_file, "w") as f_accepted:
                if not no_header:
                    f_accepted.write(lines[0])  # Retain header if it exists
                for i, line in enumerate(lines[start_idx:], start=start_idx):
                    if i - start_idx not in discard_indices:  # Skip discarded points
                        f_accepted.write(line)

        else:
            """
            Handle P(k) files with a specific structure

            P(k) File Structure:
            ---------------------
            1. **Header Line**: The first line contains k-values, common across all redshifts.
            2. **Redshift Header**: Each block starts with a line like `# z = <redshift_value>`, indicating the redshift.
            3. **Data Rows**: Following the redshift header, there are `N_samples` rows of power spectrum values,
               one row per sample, with one value for each k-value.
            4. **Repeating Blocks**: This pattern (redshift header + data rows) repeats for each redshift in the file.
            ---------------------

            # - Discarded samples are handled per redshift block.
            """

            mode = "a" if discarded_file_exists else "w"
            # Note the append mode for P(k) files disrupts the original structure of the file in the discarded file.
            # This is because it will append the discarded samples to the end of the file, which mean it will now append the redshift blocks at the end of the file.
            # This means it will end up having multiple redshift blocks with the same redshift value, which is not the original structure of the file.
            # This is not an issue for the filtered/accepted file, so it shouldn't affect the filtering process or CONNECT's iterative process in any way.
            # But it's something to be aware of if you need to use the discarded P(k) files for other purposes or analysis.
            # In most cases, the append mode will not be used since the likelihood-filter is typically applied once per iteration. That is why I haven't bothered to fix this issue.
            # I am also not certain if this is a bug, it depends how the discarded P(k) files are used and how it handles repeated redshift blocks.

            if mode == "a":
                print(
                    f"""
                    \033[1;33mNOTICE:\033[0m The likelihood-filter is applied twice within the same iteration because of strict filtering after reaching the 'bad state' limit. 
                    It is appending new discarded P(k) data to an existing discard file. 
                    This will produce multiple '# z =' blocks for the same redshift. 
                    If you need to parse these discards later, be aware of the duplicated blocks. You can easily fix this manually by moving the blocks to the correct position in the file.
                    The file affected: {discarded_file}
                    This will not affect CONNECT's iterative process or the 'accepted' files.
                    So you can ignore this message if you are not using the discarded P(k) files for further analysis.
                    """,
                    flush=True,
                )

            with open(discarded_file, mode) as f_discarded, open(
                original_file, "w"
            ) as f_accepted:
                # Write the k-values header to both discarded and retained files
                if not discarded_file_exists:
                    f_accepted.write(lines[0])
                f_discarded.write(lines[0])

                line_idx = 1  # Start processing after the k-values header

                """
                This block processes the P(k) file by iterating through the original file stored lines, line by line, starting after the k-values header.
                It identifies redshift headers and processes data rows associated with each redshift block, by:
                - Writing the redshift header to both files, discarded and retained file
                - iterating over all the next `N_samples` lines, writing each line to the appropriate file.
                - If the sample index is in `discard_indices`, the sample is written to the filtered file.
                - Otherwise, the sample is written to the original file.
                - The current line index is continually updated to keep track of the current position in the original file.
                such that by the end of the loop, it should be at the next redshift header or the end of the file.
                """

                while line_idx < len(lines):  # Iterate through all lines in the file
                    # Identify the redshift header, which begins a new block of data
                    if lines[line_idx].startswith("# z ="):
                        z_header = lines[line_idx]
                        f_accepted.write(z_header)
                        f_discarded.write(z_header)
                        line_idx += 1  # Move to the next line

                        # Process data rows associated with this redshift block
                        for sample_idx in range(
                            self.N_samples
                        ):  # Iterate over all samples
                            if line_idx >= len(lines):
                                break  # Stop if we reach the end of the file
                            line = lines[line_idx]

                            if sample_idx in discard_indices:
                                f_discarded.write(
                                    line
                                )  # Write discarded sample to the filtered file
                            else:
                                f_accepted.write(
                                    line
                                )  # Write retained sample to the original file
                            line_idx += 1  # Keep track of the current line index.
                            # By the end of the loop, it should be at the next redshift header or the end of the file.
                    else:
                        # Handle unexpected lines (should not occur in properly formatted Pk files)
                        line_idx += 1

    def get_total_samples(self):
        """Determine the total number of samples based on the length of the likelihood_data.txt file."""
        with open(self.likelihood_file, "r") as f:
            return sum(1 for line in f if not line.startswith("#"))

    def format_delta_chi2_summary(self):
        """
        Returns a clean, color-formatted, column-aligned string summary of Δχ² statistics
        for accepted and discarded points (NumPy arrays).
        """

        def compute_stats(arr):
            if arr is None or arr.size == 0:
                return None
            return {
                "N": arr.size,
                "Mean": np.mean(arr),
                "Std": np.std(arr),
                "Min": np.min(arr),
                "Max": np.max(arr),
                "Median": np.median(arr),
            }

        stats_accepted = compute_stats(self.delta_chi2_accepted)
        stats_discarded = compute_stats(self.delta_chi2_discarded)
        stats_all = compute_stats(self.delta_chi2_all)

        # Define column order and spacing
        columns = ["N", "Mean", "Std", "Min", "Max", "Median"]
        col_width = 12
        label_width = 14

        # Header
        header = " " * label_width + "".join(f"{col:>{col_width}}" for col in columns)

        def format_float_for_table(val, width=10):
            """
            Formats the float `val` so that it fits nicely
            in `width` columns, using either
            - scientific notation when |val| >= 1e5, or
            - normal float with one decimal place otherwise.
            """
            if abs(val) >= 99999:
                # scientific notation with 3 decimal
                return f"{val:{width}.3e}"
            else:
                # normal float with 2 decimal
                return f"{val:{width}.2f}"

        def format_row(label, stats, threshold=None, accepted=False, discarded=False):
            if stats is None:
                return f"{label:<{label_width}}" + "  [no data]".rjust(col_width)

            # Choose base color
            if discarded:
                base_color = "\033[1;31m"  # red
            elif accepted:
                base_color = "\033[1;32m"  # green (used for label and N)
            else:
                base_color = "\033[1;30m"  # black

            reset = "\033[0m"
            row = f"{base_color}{label:<{label_width}}{stats['N']:{col_width}d}{reset}"

            # Other columns
            for col in columns[1:]:  # skip 'N' (already printed)
                val = stats[col]

                # Accepted: check if stat exceeds threshold
                if accepted:
                    if threshold is not None and val > threshold:
                        color = "\033[1;33m"  # yellow
                    else:
                        color = "\033[1;32m"  # green
                elif discarded:
                    color = "\033[1;31m"  # red
                else:
                    color = "\033[1;30m"  # black

                formatted_val = format_float_for_table(val, width=col_width)
                row += f"{color}{formatted_val}{reset}"

            return row

        # Compose table
        title = "\033[1;4;37mΔχ² Summary Statistics:\033[0m"

        summary = title + "\n"
        summary += header + "\n"
        summary += (
            format_row(
                "Accepted",
                stats_accepted,
                threshold=self.delta_chi2_threshold,
                accepted=True,
            )
            + "\n"
        )
        summary += format_row("Discarded", stats_discarded, discarded=True) + "\n"
        summary += format_row("All", stats_all) + "\n"

        return summary
