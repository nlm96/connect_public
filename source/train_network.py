import os
import sys
import pickle

import tensorflow as tf

tf.random.set_seed(42)
import numpy as np
from copy import deepcopy

from .custom_functions import LossFunctions
from .architecture.sequential_models import Dense_model
from .callbacks import KeepBestEpoch, RecoverOnBadLoss, CustomReduceLROnPlateau


class Training:

    def __init__(self, param, CONNECT_PATH):
        self.param = param
        self.CONNECT_PATH = CONNECT_PATH

        path = os.path.join(CONNECT_PATH, f"data/{param.jobname}")
        if param.sampling == "iterative":
            try:
                i = max(
                    [
                        int(f.split("number_")[-1])
                        for f in os.listdir(path)
                        if f.startswith("number")
                    ]
                )
                self.load_path = os.path.join(path, f"number_{i}")
                self.N = (
                    sum(
                        1
                        for line in open(
                            os.path.join(self.load_path, "model_params.txt")
                        )
                    )
                    - 1
                )
            except:
                self.load_path = os.path.join(path, f"N-{self.param.N}")
                self.N = (
                    sum(
                        1
                        for line in open(
                            os.path.join(self.load_path, "model_params.txt")
                        )
                    )
                    - 1
                )
        else:
            self.load_path = os.path.join(path, f"N-{self.param.N}")
            self.N = (
                sum(
                    1 for line in open(os.path.join(self.load_path, "model_params.txt"))
                )
                - 1
            )

        self.N_train = int(np.floor(self.N * self.param.train_ratio))
        input_dim = len(self.param.parameters.keys())
        input_dim += len(self.param.custom_parameters.keys())

        model_params = []
        outputs = []
        ell_weights = []
        output_interval = {}
        output_size_iter = 0
        normalise_method = self.param.normalisation_method

        ### Load input ###
        with open(os.path.join(self.load_path, "model_params.txt"), "r") as f:
            for line in f:
                if line[0] == "#":
                    paramnames = line[2:].replace("\n", "").split("\t")
                else:
                    line_vals = []
                    params = list(self.param.parameters.keys())
                    params_mapped_to = []
                    for custom_par, cdict in self.param.custom_parameters.items():
                        # Add cdict["maps_to"] to params_mapped_to
                        params_mapped_to.append(cdict["maps_to"])

                    if len(params_mapped_to) > 0:
                        params = params + params_mapped_to

                    for name in params:
                        idx = paramnames.index(name)
                        line_vals.append(line.replace("\n", "").split("\t")[idx])
                    line_vals = np.float32(line_vals)
                    model_params.append(line_vals)

        ### Load output data ###
        if len(self.param.output_Cl) > 0:
            output_interval["Cl"] = {}
        for output in self.param.output_Cl:
            out = []
            with open(os.path.join(self.load_path, f"Cl_{output}.txt"), "r") as f:
                for i, line in enumerate(f):
                    if i == 0:
                        ell = np.int32(line.replace("\n", "").split("\t"))
                        if not hasattr(self, "output_ell"):
                            self.output_ell = ell
                    else:
                        if i == 1:
                            ell_weights.append(ell)
                        out.append(np.float32(line.replace("\n", "").split("\t")))
            outputs.append(out)
            output_interval["Cl"][output] = [
                output_size_iter,
                output_size_iter + len(out[0]),
            ]
            output_size_iter += len(out[0])

        if len(self.param.output_Pk) > 0:
            output_interval["Pk"] = {}
        for output in self.param.output_Pk:
            output_interval["Pk"][output] = {}
            out = {}
            for z in self.param.z_Pk_list:
                out[z] = []
            z_idx = 0
            with open(os.path.join(self.load_path, f"Pk_{output}.txt"), "r") as f:
                for i, line in enumerate(f):
                    if i == 0:
                        k_grid = np.float32(line.replace("\n", "").split("\t"))
                        if not hasattr(self, "output_k_grid"):
                            self.output_k_grid = k_grid
                    else:
                        if line[0] == "#":
                            z = self.param.z_Pk_list[z_idx]
                            z_idx += 1
                            j = 0
                        if j > 0:
                            if j == 1:
                                ell_weights.append(k_grid ** (1 / 3) * 1500)
                            out[z].append(
                                np.float32(line.replace("\n", "").split("\t"))
                            )
                        j += 1
            for z in self.param.z_Pk_list:
                outputs.append(out[z])
                output_interval["Pk"][output][str(z)] = [
                    output_size_iter,
                    output_size_iter + len(out[z][0]),
                ]
                output_size_iter += len(out[z][0])

        if len(self.param.output_bg) > 0:
            output_interval["bg"] = {}
        for output in self.param.output_bg:
            file_output = output.replace("/", "\\")
            out = []
            with open(os.path.join(self.load_path, f"bg_{file_output}.txt"), "r") as f:
                for i, line in enumerate(f):
                    if i == 0:
                        if not hasattr(self, "output_z_bg"):
                            self.output_z_bg = np.float32(
                                line.replace("\n", "").split("\t")
                            )
                    else:
                        if i == 1:
                            ell_weights.append(
                                5 * 10**2 * np.ones(line.count("\t") + 1)
                            )
                        out.append(np.float32(line.replace("\n", "").split("\t")))
            outputs.append(np.array(out))
            output_interval["bg"][output] = [
                output_size_iter,
                output_size_iter + len(out[0]),
            ]
            output_size_iter += len(out[0])

        if len(self.param.output_th) > 0:
            output_interval["th"] = {}
        for output in self.param.output_th:
            out = []
            with open(os.path.join(self.load_path, f"th_{output}.txt"), "r") as f:
                for i, line in enumerate(f):
                    if i == 0:
                        if not hasattr(self, "output_z_th"):
                            self.output_z_th = np.float32(
                                line.replace("\n", "").split("\t")
                            )
                    else:
                        if i == 1:
                            ell_weights.append(
                                5 * 10**2 * np.ones(line.count("\t") + 1)
                            )
                        out.append(np.float32(line.replace("\n", "").split("\t")))
            outputs.append(np.array(out))
            output_interval["th"][output] = [
                output_size_iter,
                output_size_iter + len(out[0]),
            ]
            output_size_iter += len(out[0])

        if len(self.param.output_derived) > 0:
            output_interval["derived"] = {}
            output_derived = []
            with open(os.path.join(self.load_path, "derived.txt"), "r") as f:
                lines = list(f)
                derived_names = lines[0].replace("# ", "").replace("\n", "").split("\t")
                for i, line in enumerate(lines[1:]):
                    out = []
                    line_vals = []
                    for name in self.param.output_derived:
                        idx = derived_names.index(name)
                        line_vals.append(line.replace("\n", "").split("\t")[idx])
                    line_vals = np.float32(line_vals)
                    if i == 0:
                        ell_weights.append(10**4 * np.ones(line.count("\t") + 1))
                    for j, output in enumerate(self.param.output_derived):
                        out.append(np.array(line_vals[j]))
                        if i == 0:
                            output_interval["derived"][output] = output_size_iter
                            output_size_iter += 1
                    output_derived.append(out)
            output_derived = np.array(output_derived)
            outputs.append(output_derived)

        if len(self.param.extra_output) > 0:
            output_interval["extra"] = {}
        for output in self.param.extra_output:
            out = []
            with open(os.path.join(self.load_path, f"extra_{output}.txt"), "r") as f:
                for i, line in enumerate(f):
                    if i == 0:
                        ell_weights.append(10**4 * np.ones(line.count("\t") + 1))
                    out.append(np.float32(line.replace("\n", "").split("\t")))
            outputs.append(np.array(out))
            output_interval["extra"][output] = [
                output_size_iter,
                output_size_iter + len(out[0]),
            ]
            output_size_iter += len(out[0])

        self.output_dim = output_size_iter

        outputs = np.concatenate(outputs, axis=1)

        if normalise_method == "standardisation":

            def normalise_with_moments(x, epsilon=1e-20):
                delta = 1e-3
                mean = np.mean(x)
                var = np.var(x)
                variance = var + epsilon  # epsilon to avoid dividing by zero
                x_normed = (x - mean) / np.sqrt(variance)
                var = np.var(x_normed)
                if var == 0:
                    return x_normed, mean, variance
                while var > 1 + delta or var < 1 - delta:
                    x_normed = x_normed / np.sqrt(var + epsilon)
                    variance *= var + epsilon
                    var = np.var(x_normed)
                return x_normed, mean, variance

            out_nodes = outputs.T
            out_mean = []
            out_var = []
            outputs = []

            for n in out_nodes:
                data, mean, var = normalise_with_moments(n)
                outputs.append(data)
                out_mean.append(mean)
                out_var.append(var)

            outputs = np.array(outputs).T
            std = tf.sqrt(tf.constant(out_var, dtype=tf.float32))
            mean = tf.constant(out_mean, dtype=tf.float32)
            self.unnormalise = lambda x: x * std + mean

            out_nodes = []
            out_mean = []
            out_var = []
            del out_nodes
            del out_mean
            del out_var

        elif normalise_method == "min-max":

            def normalise_with_minmax(x, epsilon=1e-20):
                x_min = min(x)
                x_max = max(x) - epsilon  # epsilon to avoid dividing by zero
                x_normed = (x - x_min) / (x_max - x_min)
                return x_normed, x_min, x_max

            out_nodes = outputs.T
            out_x_min = []
            out_x_max = []
            outputs = []

            for n in out_nodes:
                data, x_min, x_max = normalise_with_minmax(n)
                outputs.append(data)
                out_x_min.append(x_min)
                out_x_max.append(x_max)

            outputs = np.array(outputs).T
            x_min = tf.constant(out_x_min, dtype=tf.float32)
            x_max = tf.constant(out_x_max, dtype=tf.float32)
            self.unnormalise = lambda x: x * (x_max - x_min) + x_max

            out_nodes = []
            out_x_min = []
            out_x_max = []
            del out_nodes
            del out_x_min
            del out_x_max

        elif normalise_method == "log":

            def normalise_with_log(x, epsilon=1e-20):
                offset1 = -min(np.min(x), epsilon) * 1.01
                x2 = np.log(x + offset1)
                offset2 = -min(np.min(x2), epsilon) * 1.01
                x_normed = np.log(x2 + offset2)
                return x_normed, offset1, offset2

            out_nodes = outputs.T
            out_offset1 = []
            out_offset2 = []
            outputs = []

            for n in out_nodes:
                data, offset1, offset2 = normalise_with_log(n)
                outputs.append(data)
                out_offset1.append(offset1)
                out_offset2.append(offset2)

            outputs = np.array(outputs).T
            offset1 = tf.constant(out_offset1, dtype=tf.float32)
            offset2 = tf.constant(out_offset2, dtype=tf.float32)
            self.unnormalise = lambda x: tf.exp(tf.exp(x) - offset2) - offset1

            out_nodes = []
            out_offset1 = []
            out_offset1 = []
            del out_nodes
            del out_offset1
            del out_offset1

        elif normalise_method == "factor":

            def normalise_with_factor(x, epsilon=1e-20):
                factor = np.mean(x) + epsilon  # epsilon to avoid dividing by zero
                x_normed = x / factor
                return x_normed, factor

            out_nodes = outputs.T
            out_factor = []
            outputs = []

            for n in out_nodes:
                data, factor = normalise_with_factor(n)
                outputs.append(data)
                out_factor.append(factor)

            outputs = np.array(outputs).T
            factor = tf.constant(out_factor, dtype=tf.float32)
            self.unnormalise = lambda x: x * factor

            out_nodes = []
            out_factor = []
            del out_nodes
            del out_factor

        ### Prepare output info dictionary for model ###
        self.output_info = {"input_names": paramnames, "interval": output_interval}
        if len(self.param.output_Cl) > 0:
            self.output_info["ell"] = list(self.output_ell)
            self.output_info["output_Cl"] = self.param.output_Cl
        if len(self.param.output_Pk) > 0:
            self.output_info["k_grid"] = list(self.output_k_grid)
            self.output_info["output_Pk"] = self.param.output_Pk
        if len(self.param.output_bg) > 0:
            self.output_info["z_bg"] = list(self.output_z_bg)
            self.output_info["output_bg"] = self.param.output_bg
        if len(self.param.output_th) > 0:
            self.output_info["z_th"] = list(self.output_z_th)
            self.output_info["output_th"] = self.param.output_th
        if len(self.param.output_derived) > 0:
            self.output_info["output_derived"] = self.param.output_derived
        if len(self.param.extra_output) > 0:
            self.output_info["extra_output"] = self.param.extra_output
        if len(self.param.extra_input) > 0:
            self.output_info["extra_input"] = self.param.extra_input

        ### Convert data to tensors ###
        model_params_tf = tf.constant(model_params)
        model_params = []
        del model_params
        output_tf = tf.constant(outputs)
        outputs = []
        del outputs
        dataset = tf.data.Dataset.from_tensor_slices((model_params_tf, output_tf))
        model_params_tf = []
        output_tf = []
        del model_params_tf
        del output_tf

        ### Shuffle dataset and split in training, test and validation ###
        dataset = dataset.shuffle(
            buffer_size=10 * self.param.batchsize,
            seed=42,
            reshuffle_each_iteration=False,
        )
        self.train_dataset = dataset.take(self.N_train)
        test_raw_dataset = dataset.skip(self.N_train)
        self.test_dataset = test_raw_dataset.batch(self.param.batchsize)
        # Build a second dataset for evaluation in the training loop:
        self.test_dataset_loop = test_raw_dataset.batch(self.param.batchsize).repeat()
        import math

        self.steps_for_test = math.ceil((self.N - self.N_train) / self.param.batchsize)

        dataset = []
        del dataset
        test_raw_dataset = []
        del test_raw_dataset

        self.val_dataset = self.train_dataset.take(
            int(self.param.val_ratio * self.N)
        ).batch(self.param.batchsize)
        self.train_dataset = self.train_dataset.skip(
            int(self.param.val_ratio * self.N)
        ).batch(self.param.batchsize)

        ### Define strategy for training on multiple GPUs ###
        if len(tf.config.list_physical_devices("GPU")) > 0:
            self.training_strategy = tf.distribute.MirroredStrategy(
                devices=["/gpu:0", "/gpu:1"],
                cross_device_ops=tf.distribute.HierarchicalCopyAllReduce(),
            ).scope()
        else:

            class NoneStrategy(object):
                def __init__(self):
                    pass

                def __enter__(self):
                    pass

                def __exit__(self, a, b, c):
                    pass

            self.training_strategy = NoneStrategy()

        ### Set loss function ###
        ell_weights = np.concatenate(ell_weights)
        Loss = LossFunctions(ell_weights)
        if callable(getattr(Loss, self.param.loss_function, None)):
            _locals = {}
            exec("loss_fun = Loss." + self.param.loss_function, locals(), _locals)
            self.loss_fun = _locals["loss_fun"]
        else:
            self.loss_fun = self.param.loss_function

        ### Normalise input data ###
        self.input_normaliser = (
            tf.keras.layers.experimental.preprocessing.Normalization(
                input_dim=input_dim
            )
        )
        self.input_normaliser.adapt(
            np.concatenate([x for x, y in self.train_dataset], axis=0)
        )

    def train_model(self, epochs=None, output_file=None):
        if epochs != None:
            self.param.epochs = int(epochs)

        if not self.param.use_advanced_training:
            self.train_model_simple(epochs=epochs, output_file=output_file)
        else:
            self.train_model_advanced(epochs=epochs, output_file=output_file)

    def train_model_advanced(self, epochs=None, output_file=None):

        # Set epochs if provided.
        if epochs is not None:
            self.param.epochs = int(epochs)

        # Initialize training state variables.
        max_attempts = 40
        attempts = 0
        global_best_test_loss = float("inf")
        self.global_best_weights = None
        # Start with the initial learning rate from your parameters.
        current_lr = self.param.initial_lr
        self.current_lr = current_lr
        self.current_best_epoch = 0
        self.best_weights = None
        self.best_optimizer_state = None
        self.training_success = False
        saved_best_model = False
        # We use these attributes for state restoration.
        # self.TI = self  # For convenience: our training instance is self.

        # This list defines the order in which we try recovery methods.

        if output_file is not None:
            stdout_backup = sys.stdout
            sys.stdout = open(output_file, "w")

        # Main outer loop: try until a successful training epoch is found or max_attempts reached.
        while attempts < max_attempts and not self.training_success:

            print(f"\n{'='*60}")
            print(f" Attempt {attempts}")
            print(f"{'='*60}\n", flush=True)

            if attempts > 0:
                restore_modes = ["optimizer_and_weights", "weights_only", "none"]
            else:
                restore_modes = ["none"]

            attempts += 1
            # Cycle through the three recovery modes.
            for mode in restore_modes:
                print(f"\n {'-'*30} trying {mode} {'-'*30}", flush=True)

                # Create a new optimizer with the current learning rate.
                adam = tf.keras.optimizers.Adam(
                    learning_rate=current_lr,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-4,
                    amsgrad=True,
                    name="Adam",
                    global_clipnorm=1.0,
                )

                # Optionally, create LR scheduler and early stopping callbacks if enabled.
                callbacks = [KeepBestEpoch()]
                if self.param.use_lr_scheduler:
                    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
                        monitor="val_loss",
                        factor=0.8,
                        patience=20,
                        verbose=1,
                        mode="min",
                        min_lr=1e-6,
                        min_delta=0.0001,
                    )
                    monitor_progression = CustomReduceLROnPlateau(
                        monitor="val_loss",
                        factor=0.8,
                        patience=self.param.monitor_prog_patience,
                        active_epoch_start=self.param.monitor_prog_start,
                        verbose=1,
                        mode="min",
                        min_lr=1e-6,
                        min_delta=self.param.monitor_prog_min_delta,
                        apply_only_if_above=self.param.monitor_prog_active_above_loss,
                        force_threshold=self.param.monitor_prog_force_thres,
                    )
                    callbacks.extend([lr_schedule, monitor_progression])
                if self.param.use_early_stopping:
                    early_stopping = tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=self.param.early_stopping_patience,
                        restore_best_weights=True,
                        mode="min",
                        verbose=1,
                        baseline=None,
                        min_delta=0.0001,
                    )
                    callbacks.append(early_stopping)

                # Add our custom recovery callback. We pass a flag based on the mode.
                # (Here we pass restore_optimizer_state=True only if mode=='optimizer_and_weights')
                input_dim = len(self.param.parameters.keys()) + len(
                    self.param.custom_parameters.keys()
                )
                recover_cb = RecoverOnBadLoss(
                    training_instance=self,
                    threshold=1e20,
                    reduce_lr_on_bad=False,
                    reduce_lr_factor=0.8,
                    min_lr=1e-6,
                    monitor="val_loss",
                    verbose=1,
                    input_dim=input_dim,
                    output_dim=self.output_dim,
                    restore_mode=mode,
                )
                callbacks.append(recover_cb)

                # Build the model inside your distribution strategy.
                with self.training_strategy:
                    self.model = Dense_model(
                        self.param.N_nodes,
                        input_dim,
                        self.output_dim,
                        input_normaliser=self.input_normaliser,
                        output_unnormaliser=self.unnormalise,
                        activation=self.param.activation_function,
                        num_hidden_layers=self.param.N_hidden_layers,
                    )
                    self.model.compile(optimizer=adam, loss=self.loss_fun)

                # Now run training.
                self.history = self.model.fit(
                    self.train_dataset,
                    epochs=max(self.param.epochs - self.current_best_epoch, 1),
                    validation_data=self.val_dataset,
                    callbacks=callbacks,
                )

                # Evaluate on test data.

                test_loss = self.model.evaluate(
                    self.test_dataset_loop, verbose=2, steps=self.steps_for_test
                )
                print(f"Evaluated test loss: {test_loss:.6f}", flush=True)

                # Check if training was successful (your custom callback should set self.training_success = True)
                if test_loss < global_best_test_loss:
                    print(f"✅ New global best test loss: {test_loss:.6f}", flush=True)
                    print(f"Saving global best weights...", flush=True)
                    global_best_test_loss = test_loss

                    self.global_best_weights = self.model.get_weights()

                    global_best_history_dict = self.history.history
                    # Save the history to a file

                # If training was successful, break out of the loop.
                if self.training_success:
                    break
                else:
                    print(
                        "⚠️  Training unsuccessful for this configuration. Trying next restore mode...\n",
                        flush=True,
                    )

            # If training was successful, break out of the outer loop.
            if self.training_success:
                print(f"\n\n{'='*60}")
                print("✅ Training successful!")
                print(f"Will now save the model and history...", flush=True)
                print(f"{'='*60}\n", flush=True)
                break
            else:
                current_lr *= 0.8
                self.current_lr = current_lr
                print(f"Lowering learning rate to {current_lr}", flush=True)

        if output_file is not None:
            sys.stdout.close()
            sys.stdout = stdout_backup

        if self.global_best_weights is not None:
            # Load the best model from the file
            self.model.set_weights(self.global_best_weights)
            self.global_best_history = global_best_history_dict

        test_loss = self.model.evaluate(self.test_dataset, verbose=2)
        print("\nTest loss:", test_loss, flush=True)

        self.model.train_normalise = False
        self.model.raw_info = str(self.output_info)
        self.model.info_dict = self.model.convert_types_to_tf(self.output_info)

    def train_model_simple(self, epochs=None, output_file=None):
        if epochs != None:
            self.param.epochs = int(epochs)

        adam = tf.keras.optimizers.Adam(
            learning_rate=0.0001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-4,
            amsgrad=True,
            name="Adam",
        )

        self.training_success = False
        training_try_number = 0

        if output_file != None:
            stdout_backup = sys.stdout
            sys.stdout = open(output_file, "w")

        with self.training_strategy:

            self.model = Dense_model(
                self.param.N_nodes,
                len(self.param.parameters.keys()),
                self.output_dim,
                input_normaliser=self.input_normaliser,
                output_unnormaliser=self.unnormalise,
                activation=self.param.activation_function,
                num_hidden_layers=self.param.N_hidden_layers,
            )

            self.model.compile(optimizer=adam, loss=self.loss_fun)

        self.history = self.model.fit(
            self.train_dataset,
            epochs=self.param.epochs,
            validation_data=self.val_dataset,
            callbacks=[KeepBestEpoch()],
        )

        if output_file != None:
            sys.stdout.close()
            sys.stdout = stdout_backup

        test_loss = self.model.evaluate(self.test_dataset, verbose=2)
        print("\nTest loss:", test_loss)

        self.model.train_normalise = False
        self.model.raw_info = str(self.output_info)
        self.model.info_dict = self.model.convert_types_to_tf(self.output_info)

    def save_model(self):
        if not getattr(self, "model", None):
            errmsg = "the class has no attribute 'model'. You should run the 'train_model' method before using the 'save_model' method"
            raise AttributeError(errmsg)

        if self.param.save_name:
            self.save_path = os.path.join(
                self.CONNECT_PATH, "trained_models", self.param.save_name
            )
        else:
            save_name = f"{self.param.jobname}"
            save_name += f"_N{self.N}"
            save_name += f"_bs{self.param.batchsize}"
            save_name += f"_e{self.param.epochs}"
            self.save_path = os.path.join(
                self.CONNECT_PATH, f"trained_models", save_name
            )

        if not self.param.overwrite_model:
            M = 1
            if os.path.isdir(self.save_path):
                while os.path.isdir(self.save_path + f"_{M}"):
                    M += 1
                self.save_path += f"_{M}"
        self.model.save(self.save_path)

        os.system(
            f"cp {self.param.param_file} {os.path.join(self.save_path, 'log.param')}"
        )

        with open(os.path.join(self.save_path, "log.param"), "a+") as f:
            jobname_specified = False
            for line in f:
                if line.startswith("jobname"):
                    jobname_specified = True
            if not jobname_specified:
                f.write(f"\njobname = '{self.param.jobname}'")

    def save_history(self):
        if not getattr(self, "save_path", None):
            errmsg = "the class has no attribute 'save_path'. You should run the 'save_model' method before using the 'save_history' method"
            raise AttributeError(errmsg)

        if not hasattr(self, "global_best_history"):
            loss = self.history.history["val_loss"]
            train_loss = self.history.history["loss"]
        else:
            loss = self.global_best_history["val_loss"]
            train_loss = self.global_best_history["loss"]

        with open(os.path.join(self.save_path, "val_loss.pkl"), "wb") as f:
            pickle.dump(loss, f)
        with open(os.path.join(self.save_path, "train_loss.pkl"), "wb") as f:
            pickle.dump(train_loss, f)

    def save_test_data(self):
        if not getattr(self, "save_path", None):
            errmsg = "the class has no attribute 'save_path'. You should run the 'save_model' method before using the 'save_history' method"
            raise AttributeError(errmsg)

        test_mp, test_out = tuple(zip(*self.test_dataset))

        try:
            dim1 = self.N - self.N_train
            mp_dim2 = len(test_mp[0][0])
            out_dim2 = len(test_out[0][0])
            test_model_params = tf.reshape(test_mp, (dim1, mp_dim2)).numpy()
            test_output = self.unnormalise(
                tf.reshape(test_out, (dim1, out_dim2)).numpy()
            )
        except:
            test_model_params = []
            test_output = []
            for tmp, tout in zip(test_mp, test_out):
                for tmp_i, tout_i in zip(tmp, tout):
                    test_model_params.append(tmp_i.numpy())
                    test_output.append(tout_i.numpy())
            test_model_params = np.array(test_model_params)
            test_output = self.unnormalise(np.array(test_output))

        test_data = [test_model_params, test_output]

        with open(os.path.join(self.save_path, "test_data.pkl"), "wb") as f:
            pickle.dump(test_data, f)
