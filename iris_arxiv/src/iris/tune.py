import json
from functools import partial
from typing import Callable

import optuna
import torch
import torch.nn as nn
from lancedb.table import LanceTable
from optuna import Trial
from optuna.pruners import HyperbandPruner
from optuna.samplers import QMCSampler, TPESampler

from iris.config import Config
from iris.train import Trainer, make_trainer, set_hyperparams


class Objective:
    def __init__(self, cfg: Config, trainer_fn: Callable) -> None:
        self.cfg = cfg
        self.best_value = 0.0
        self.trainer_fn = trainer_fn

    def sample_hyperparameters(self, trial: optuna.Trial):
        return {
            "max_epochs": trial.suggest_int(**self.cfg.hparams.max_epochs),
            "lr": trial.suggest_float(**self.cfg.hparams.lr),
            "weight_decay": trial.suggest_float(**self.cfg.hparams.weight_decay),
            "dropout": trial.suggest_float(**self.cfg.hparams.dropout),
            "hidden_dim": 2 ** trial.suggest_int(**self.cfg.hparams.hidden_dim),
            # "temperature": trial.suggest_float(**self.cfg.hparams.temperature),
        }

    def __call__(self, trial: Trial) -> float:
        hyperparams = self.sample_hyperparameters(trial=trial)
        cfg = set_hyperparams(cfg=self.cfg, **hyperparams)
        trainer: Trainer = self.trainer_fn()
        trainer.train()
        if trainer.value < self.best_value:
            self.best_value = trainer.val_loss
            with open(self.cfg.tuner.path, "w") as f:
                json.dump(hyperparams, f)
            torch.save(trainer.model.state_dict(), cfg.trainer.checkpoint)
        return trainer.value


def make_study(cfg: Config, sampler: optuna.samplers.BaseSampler) -> optuna.study.Study:
    if cfg.tuner.prune:
        pruner = HyperbandPruner(
            min_resource=cfg.trainer.max_epochs // 4,
            max_resource=cfg.trainer.max_epochs,
        )
    else:
        pruner = None
    storage = optuna.storages.RDBStorage(
        url="sqlite:///:memory:",
        engine_kwargs={"pool_size": 20, "connect_args": {"timeout": 10}},
    )
    return optuna.create_study(
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        study_name="tune",
    )


def tune(cfg: Config, tbl: LanceTable):
    trainer_fn = partial(make_trainer, cfg=cfg, tbl=tbl)
    objective = Objective(cfg=cfg, trainer_fn=trainer_fn)
    half_trials = cfg.tuner.n_trials // 2
    sampler = QMCSampler(seed=cfg.seed)
    study = make_study(cfg=cfg, sampler=sampler)
    study.optimize(func=objective, n_trials=half_trials)
    study.sampler = TPESampler(multivariate=True, seed=cfg.seed)
    study.optimize(func=objective, n_trials=half_trials)


def load_best_checkpoint(cfg: Config, model_class: type[nn.Module]) -> nn.Module:
    model_weights = torch.load(cfg.trainer.checkpoint, weights_only=True)
    with open(cfg.tuner.path, "r") as f:
        hyperparams = json.load(f)
    cfg = set_hyperparams(cfg=cfg, **hyperparams)
    exclude = {"num_queries", "query_ini_random", "device"}
    model_config = cfg.model.model_dump(exclude=exclude)
    model: nn.Module = model_class(**model_config, ini_queries=model_weights["queries"])
    model.load_state_dict(model_weights)
    model.to(cfg.trainer.device)
    return model
