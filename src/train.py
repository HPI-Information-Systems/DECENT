import os
import click
from dotenv import load_dotenv
import torch
from data.loading import FGNETSeparateDataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor

from model.model import FGNETSeparatePredictor
from utils.util import EasyDict, GracefulStopCallback, load_config, num_available_cpus
from utils.wandb_helper import try_wandb_login


def init_wandb(**wandb_params):
    wandb_params = EasyDict(wandb_params)
    if not wandb_params.offline and try_wandb_login():
        if not wandb_params.project:
            wandb_params.project = os.environ.get("WANDB_PROJECT")
        if not wandb_params.entity:
            wandb_params.entity = os.environ.get("WANDB_ENTITY")
        print("Wandb params:", wandb_params)
        wandb_logger = WandbLogger(**wandb_params, log_model=False)
        run_id = wandb_logger.experiment.id
        logger = wandb_logger
    else:
        print("No wandb")
        run_id = "0"
        logger = True
    return logger, run_id

def init_data_module(model_name, **data_module_params):
    data_module_params = EasyDict(data_module_params)
    if not data_module_params.num_workers:
        data_module_params.num_workers = num_available_cpus()
    print("Data module params:", data_module_params)
    data_module = FGNETSeparateDataModule(**data_module_params, tokenizer=model_name)
    data_module.prepare_data()
    data_module.setup()
    return data_module

def init_model(data_module: FGNETSeparateDataModule, max_epochs=None, **model_params):
    model_params = EasyDict(model_params)
    if model_params.scheduler_params:
        assert max_epochs
        num_updates = len(data_module.train_dataloader()) #// (effective_batch_size // args.train_batch_size)
        total_iters = num_updates * max_epochs
        model_params.scheduler_params["num_warmup_steps"] = model_params.scheduler_params["warmup"] * total_iters
        model_params.scheduler_params["num_training_steps"] = total_iters
        del model_params.scheduler_params["warmup"]
    
    add_model_params = dict(
        num_tokens=len(data_module.tokenizer),
        label_tokens_loader=data_module.label_dataloader(),
        eval_batch_size=data_module.eval_batch_size,
    )
    print("Model params:", model_params)
    print("Add model params:", add_model_params)

    ckpt_path = model_params.pop("from_ckpt")
    model = FGNETSeparatePredictor(**model_params, **add_model_params)
    if ckpt_path:
        print("Load from checkpoint:", ckpt_path)
        ckpt = torch.load(ckpt_path)
        assert "state_dict" in ckpt
        model.load_state_dict(ckpt["state_dict"])
    return model

def init_trainer(
        result_dir,
        logger,
        run_id,
        **trainer_params):
    trainer_params = EasyDict(trainer_params)        
    if trainer_params.val_check_interval >= 1:
        trainer_params.check_val_every_n_epoch = int(trainer_params.val_check_interval)
        del trainer_params.val_check_interval

    last_callback = ModelCheckpoint()
    checkpoint_callback = ModelCheckpoint(monitor="val_choi_f1", mode="max", save_weights_only=True, filename="choi-{epoch}-{step}-{val_choi_f1:.4f}")
    loss_checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_weights_only=True, filename="loss-{epoch}-{step}-{val_loss:.5f}")
    best_checkpoint_callback = ModelCheckpoint(monitor="val_best_choi_f1", mode="max", save_weights_only=True, filename="bestchoi-{epoch}-{step}-{val_best_choi_f1:.4f}")
    
    model_checkpoint_callbacks = [
        last_callback,
        checkpoint_callback,
        loss_checkpoint_callback,
        best_checkpoint_callback,
    ]

    lr_monitor_callback = LearningRateMonitor()
    stop_callback = GracefulStopCallback(prefix=run_id, verbose=True)
    callbacks = [*model_checkpoint_callbacks, lr_monitor_callback, stop_callback]

    print("Trainer params:", trainer_params)
    default_root_dir = trainer_params.pop("default_root_dir")
    result_dir = result_dir or default_root_dir
    assert result_dir, "Define the result through --result-dir or --trainer.default_root_dir"
    print("Result dir:", result_dir)
    
    trainer = pl.Trainer(
        **trainer_params,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=result_dir,
    )
    
    return trainer, checkpoint_callback

def run_config(add_args=[], **args):
    load_dotenv()

    args = EasyDict(args)
    print(args.config)
    assert os.path.isfile(args.config)
    config = load_config(args.config, add_args)

    # Init
    logger, run_id = init_wandb(**config.wandb)
    data_module = init_data_module(model_name=config.model.model_name, **config.data)
    max_epochs = config.trainer.max_epochs
    model = init_model(data_module=data_module, max_epochs=max_epochs, **config.model)
    trainer, checkpoint_callback = init_trainer(result_dir=args.result_dir, **config.trainer, logger=logger, run_id=run_id)
    
    # Train
    _threshold_step = model.threshold_step
    if _threshold_step:
        model.threshold_step = 0.5
    trainer.validate(model, data_module.val_dataloader())
    model.threshold_step = _threshold_step
    trainer.fit(model, data_module)
    if hasattr(model.classifier, "classifier"):
        print("Final output bias:", model.classifier.classifier.get_output_bias())

    # Test with best model
    try:
        model = FGNETSeparatePredictor.load_from_checkpoint(checkpoint_callback.best_model_path)
        model.label_tokens_loader = data_module.label_dataloader()
        model.eval_batch_size = data_module.eval_batch_size
        print("="*30)
        print("Test Results:")
        trainer.test(model, data_module.test_dataloader())
        print("="*30)
    except Exception as e:
        print(e)


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--config", required=True, help="Yaml with experiment configuration")
@click.option("--result-dir", "-o", default=None, help="Manual output path")
@click.pass_context
def train(ctx, **args):
    # To turn of wandb: --wandb.offline True
    run_config(add_args=ctx.args, **args)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    train()
