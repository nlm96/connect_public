import tensorflow as tf
import numpy as np
from copy import deepcopy


class CheckNaN(tf.keras.callbacks.Callback):
    def __init__(self, Training_instance, success_param_name="training_success"):
        self.TI = Training_instance
        self.success_param_name = success_param_name
        exec(f"self.TI.{self.success_param_name} = True")

    def on_epoch_end(self, epoch, logs=None):
        if not np.isfinite(logs["loss"]):
            exec(f"self.TI.{self.success_param_name} = False")
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if not eval(f"self.TI.{self.success_param_name}"):
            print(
                "\n\nTraining has been stopped due to NaN encounter and will restart now\n\n"
            )


class KeepBestEpoch(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.best_weights = None
        self.best_loss = None
        self.best_epoch = 1

    def on_epoch_end(self, epoch, logs=None):
        loss = logs["loss"]
        if not np.isfinite(loss):
            self.model.stop_training = True
        elif self.best_loss == None or self.best_loss > loss:
            self.best_loss = loss
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch + 1

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            print(
                f"\nThe epoch with lowest validation loss was epoch {self.best_epoch}",
                flush=True,
            )
            print(
                f"with a loss of {self.best_loss} and this has been saved.", flush=True
            )
        else:
            print(
                f"No valid epoch was found. The model has not been saved.", flush=True
            )


class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

    Arguments:
        patience: Number of epochs to wait after min has been hit. After this
        number of no improvement, training stops.
    """

    def __init__(self, patience=10):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialise the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


class RecoverOnBadLoss(tf.keras.callbacks.Callback):
    """
    This callback monitors the loss (using a specified metric name)
    and, if it becomes non-finite or exceeds a specified threshold,
    it attempts to recover by restoring the best saved weights (and optimizer state).

    It also updates an external training instance’s flags:
      - training_success: set to True if a good epoch has been encountered,
                          or set to False if a bad state is detected.
      - current_lr: the current learning rate.

    """

    def __init__(
        self,
        training_instance,  # external training instance (any mutable object)
        threshold=1e20,
        reduce_lr_on_bad=False,
        reduce_lr_factor=0.8,
        min_lr=1e-6,
        monitor="loss",
        verbose=1,
        input_dim=None,
        output_dim=None,
        restore_mode=None,
    ):
        super().__init__()
        self.TI = training_instance  # save a reference to the training instance
        self.threshold = threshold
        self.reduce_lr_on_bad = reduce_lr_on_bad
        self.reduce_lr_factor = reduce_lr_factor
        self.min_lr = min_lr
        self.monitor = monitor
        self.verbose = verbose
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.restore_mode = restore_mode

        self.best_loss = np.inf
        self.current_best_epoch = 0
        # Initialize external flags
        self.TI.training_success = False

    def on_train_begin(self, logs=None):

        if (
            self.TI.best_weights is not None
            and self.TI.best_optimizer_state is not None
        ):
            if self.restore_mode == "optimizer_and_weights":
                try:
                    dummy_x = tf.zeros((2, self.input_dim))
                    dummy_y = tf.zeros((2, self.output_dim))
                    self.model.train_on_batch(dummy_x, dummy_y)  # build optimizer vars
                    self.model.optimizer.set_weights(self.TI.best_optimizer_state)
                    self.model.set_weights(self.TI.best_weights)
                    print(
                        "[RecoverOnBadLoss] Optimizer state and model weights restored successfully.",
                        flush=True,
                    )
                except Exception as e:
                    print(
                        "[RecoverOnBadLoss] Warning: could not restore optimizer state due to:",
                        repr(e),
                        flush=True,
                    )

            elif self.restore_mode == "weights_only":
                try:
                    dummy_input = tf.zeros((1, self.input_dim))
                    _ = self.model(dummy_input)
                    self.model.set_weights(self.TI.best_weights)
                    print(
                        "[RecoverOnBadLoss] Model weights restored successfully.",
                        flush=True,
                    )
                except Exception as e:
                    print(
                        "[RecoverOnBadLoss] Warning: could not restore model weights due to:",
                        repr(e),
                        flush=True,
                    )
            elif self.restore_mode == "none":
                print(
                    "[RecoverOnBadLoss] No restoration mode specified; attempting to use the default initializer specified in the model.",
                    flush=True,
                )

            tf.keras.backend.set_value(
                self.model.optimizer.learning_rate, self.TI.current_lr
            )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_loss = logs.get(self.monitor)
        current_lr = float(
            tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        )
        self.TI.current_lr = current_lr  # update external current LR

        # Check if loss is non-finite, NaN, or exceeds the threshold.
        if (
            not np.isfinite(current_loss)
            or np.isnan(current_loss)
            or current_loss >= self.threshold
        ):
            # For early epochs, do not attempt recovery—stop training immediately.
            self.TI.training_success = False
            if self.verbose:
                print(
                    f"\n[RecoverOnBadLoss] Epoch {epoch+1}: Bad loss detected early ({current_loss:.4g}). Stopping training.",
                    flush=True,
                )
            self.model.stop_training = True
            return

        else:
            # A "good" epoch: update the best state.
            if current_loss < self.best_loss:
                self.TI.training_success = True
                self.best_loss = current_loss
                if current_loss < 550:
                    self.TI.best_weights = self.model.get_weights()
                    self.TI.best_optimizer_state = deepcopy(
                        self.model.optimizer.get_weights()
                    )
                    self.current_best_epoch = epoch + 1

                # if self.verbose:
                #     print(
                #         f"\n[RecoverOnBadLoss] Epoch {epoch+1}: Improved loss to {current_loss:.4g}; state saved.",
                #         flush=True,
                #     )

    def on_train_end(self, logs=None):

        if self.current_best_epoch > 0:
            self.TI.current_best_epoch += self.current_best_epoch

        if not self.TI.training_success:
            print(
                "\n\nTraining has been stopped due to non-finite or bad loss; external logic should restart training with a lower Learning Rate.\n\n",
                flush=True,
            )


import warnings


class CustomReduceLROnPlateau(tf.keras.callbacks.Callback):
    """
    Reduce learning rate when a metric has stopped improving.

    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This callback monitors a
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Additionally, you can restrict the callback to only be active for a
    specific range of epochs (using `active_epoch_start` and `active_epoch_end`)
    and to only apply if the monitored metric is above (or below) a certain value.
    Additionally, you can force a reduction if the monitored metric is above (or below)
    a certain value.
    """

    def __init__(
        self,
        monitor="val_loss",
        factor=0.1,
        patience=10,
        verbose=0,
        mode="auto",
        min_delta=1e-4,
        cooldown=0,
        min_lr=0.0,
        active_epoch_start=0,
        active_epoch_end=None,
        apply_only_if_above=None,
        apply_only_if_below=None,
        force_threshold=None,
        **kwargs,
    ):
        super().__init__()
        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError(
                "CustomReduceLROnPlateau does not support a factor >= 1.0. "
                f"Received factor={factor}"
            )
        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = None
        self.mode = mode
        self.monitor_op = None

        # Extra optional parameters:
        self.active_epoch_start = active_epoch_start
        self.active_epoch_end = (
            active_epoch_end if active_epoch_end is not None else np.Inf
        )
        self.apply_only_if_above = apply_only_if_above
        self.apply_only_if_below = apply_only_if_below
        self.force_threshold = force_threshold

        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter."""
        if self.mode not in {"auto", "min", "max"}:
            warnings.warn(
                f"Learning rate reduction mode {self.mode} is unknown, fallback to auto mode.",
                stacklevel=2,
            )
            self.mode = "auto"
        # For 'min' mode (or auto when 'acc' not in monitor) we want lower values.
        if self.mode == "min" or (self.mode == "auto" and "acc" not in self.monitor):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def in_cooldown(self):
        return self.cooldown_counter > 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        # Record current learning rate in logs.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        logs["learning_rate"] = lr

        # Only run if we're within the specified epoch range.
        if epoch < self.active_epoch_start or epoch > self.active_epoch_end:
            return

        # Only apply if the current metric is in the specified range.
        if self.apply_only_if_above is not None and current <= self.apply_only_if_above:
            return
        if self.apply_only_if_below is not None and current >= self.apply_only_if_below:
            return

        if current is None:
            warnings.warn(
                "Learning rate reduction is conditioned on metric "
                f"`{self.monitor}` which is not available. Available metrics are: {','.join(list(logs.keys()))}.",
                stacklevel=2,
            )

        forced_reduce = False
        if self.force_threshold is not None:
            if self.mode == "min" and current > self.force_threshold:
                forced_reduce = True
            elif self.mode == "max" and current < self.force_threshold:
                forced_reduce = True

        if forced_reduce:
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = lr
                if old_lr > np.float32(self.min_lr):
                    new_lr = old_lr * self.factor
                    new_lr = max(new_lr, self.min_lr)
                    tf.keras.backend.set_value(
                        self.model.optimizer.learning_rate, new_lr
                    )
                    if self.verbose > 0:
                        print(
                            f"\nEpoch {epoch + 1}: CustomReduceLROnPlateau forced reduction: reducing learning rate to {new_lr}.",
                            flush=True,
                        )
                    self.cooldown_counter = self.cooldown
                    self.wait = 0

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = float(
                        tf.keras.backend.get_value(self.model.optimizer.learning_rate)
                    )
                    if old_lr > np.float32(self.min_lr):
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        tf.keras.backend.set_value(
                            self.model.optimizer.learning_rate, new_lr
                        )
                        if self.verbose > 0:
                            print(
                                f"\nEpoch {epoch + 1}: CustomReduceLROnPlateau reducing learning rate to {new_lr}.",
                                flush=True,
                            )
                        self.cooldown_counter = self.cooldown
                        self.wait = 0
