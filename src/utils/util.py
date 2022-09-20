import os
from typing import Any, List, Optional
import pytorch_lightning as pl

from utils.config import get_default_config


# Util classes
# ------------------------------------------------------------------------------------------

class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

def num_available_cpus():
    # Returns the number of usable cpus 
    return len(os.sched_getaffinity(0))

def load_config(cfg_path: str, add_args: List[str] = []):
    # Priority 3: get default configs
    cfg_base = get_default_config()

    # Priority 2: merge from yaml config
    if cfg_path is not None and os.path.isfile(cfg_path):
        cfg_base.merge_from_file(cfg_path)

    # Priority 1: merge from additional arguments
    if add_args:
        add_args = [arg.lstrip("--") if i%2==0 else arg for i, arg in enumerate(add_args)]
        cfg_base.merge_from_list(add_args)

    return cfg_base

class GracefulStopCallback(pl.Callback):
    # From EarlyStoppingCallback
    def __init__(
            self,
            watch_file="stop",
            prefix="",
            verbose: bool = False,
            check_on_train_epoch_end: Optional[bool] = None,
        ):
        super().__init__()
        self._watch_file = prefix + watch_file
        self.verbose = verbose
        self._check_on_train_epoch_end = check_on_train_epoch_end

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: Optional[str] = None) -> None:
        reason = f"File to watch for graceful stop: '{self._watch_file}'."
        self._log_info(trainer, reason, self.verbose)

        if os.path.exists(self._watch_file):
            os.remove(self._watch_file)
            reason = f"File to watch for graceful stop: '{self._watch_file}' exists during start and was therefore deleted."
            self._log_info(trainer, reason, self.verbose)

        if self._check_on_train_epoch_end is None:
            # if the user runs validation multiple times per training epoch or multiple training epochs without
            # validation, then we run after validation instead of on train epoch end
            self._check_on_train_epoch_end = trainer.val_check_interval == 1.0 and trainer.check_val_every_n_epoch == 1
    
    def _should_skip_check(self, trainer: pl.Trainer) -> bool:
        from pytorch_lightning.trainer.states import TrainerFn

        return trainer.state.fn != TrainerFn.FITTING or trainer.sanity_checking
    
    def _run_early_stopping_check(self, trainer: pl.Trainer) -> None:
        if os.path.exists(self._watch_file):
            trainer.strategy.reduce_boolean_decision(True)
            trainer.should_stop = True
            
            reason = f"File to watch for graceful stop: '{self._watch_file}' exists and training therefore finished early."
            self._log_info(trainer, reason, self.verbose)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        self._run_early_stopping_check(trainer)
    
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        self._run_early_stopping_check(trainer)
    
    @staticmethod
    def _log_info(trainer: Optional[pl.Trainer], message: str, verbose: bool) -> None:
        if not verbose:
            return
        if trainer is not None and trainer.world_size > 1:
            print(f"[rank: {trainer.global_rank}] {message}")
        else:
            print(message)
