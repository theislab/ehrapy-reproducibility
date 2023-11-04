import click

from diabetic_retinopathy.latent_extractors.predictors.opts import ModelOptions

from diabetic_retinopathy.latent_extractors.predictors.models.LatentExtractor import (
    Models,
)
from diabetic_retinopathy.latent_extractors.predictors.NNExtractor import NNExtractor
from diabetic_retinopathy.constants import (
    DATASPLIT_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    PhaseType,
    TRAIN_CONFIG,
)

from diabetic_retinopathy.cli.utils.cli_utils import read_toml, display_exception
from pathlib import Path
import psutil

from copy import deepcopy
import toml
import numpy as np

np.random.seed(228)


@click.command(name="train", no_args_is_help=True)
@click.option("--config_path", type=str, default=None, help="path of data config")
@click.option(
    "--model",
    type=click.Choice([el for el in list(Models.__members__.keys()) if "FC" in el]),
    default="resnet18ImgNetFC",
)
@click.option(
    "--overwrite",
    help="Whether to overwrite the model if it exists",
    default=False,
    type=bool,
)


# General NN params


@click.option(
    "--batch_size",
    type=int,
    default=12,
    # default=36,
)
@click.option("--num_workers", type=int, default=4)
@click.option(
    "--lr_policy",
    type=click.Choice(["lambda", "step", "constant", "cyclic"]),
    default="lambda",
    help="type of learn rate decay",
)
@click.option("--lr", type=float, default=0.0001)
@click.option("--find_lr", type=bool, default=True)
@click.option(
    "--num_epochs", type=int, default=1200, help="number of epochs"
)  # 400 * d_iter
@click.option(
    "--n_ep_decay",
    type=int,
    default=600,
    help="epoch start decay learning rate, set -1 if no decay",
)  # 200 * d_iter
@click.option("--gpu", type=bool, default=False, help="gpu")
@click.option(
    "--limit_train_batches",
    type=int,
    default=None,
)
@click.option("--limit_val_batches", type=int, default=None)
@click.option(
    "--resume",
    type=bool,
    default=False,
    help="specified the dir of saved models for resume the training",
)
@click.option(
    "--nn_model_id",
    type=str,
    default=None,
)
@click.option(
    "--model_name",
    type=str,
    default=None,
)
@click.option(
    "--resume_model_type",
    type=click.Choice([el for el in list(Models.__members__.keys()) if "FC" in el]),
    default=None,
)
@click.option(
    "--nn_loss",
    type=click.Choice(["CrossEntropyLoss"]),
    default="CrossEntropyLoss",
    help="loss for classification task",
)
@click.option(
    "--weighted_sampler", type=bool, default=False, help="if True, balances sampling"
)

# weighted_sampler
# Reproducibility


@click.option(
    "--seed",
    help="Value to set the seed for all random operations."
    "Default will sample a random value for the seed.",
    default=5,
    type=int,
)
@click.option(
    "--deterministic",
    type=bool,
    default=True,
    help="Forces Pytorch to be deterministic even when using a GPU. "
    "Will raise a RuntimeError if a non-deterministic function is encountered.",
)
@click.option(
    "--compensation",
    help="Allow the user to choose how CUDA will compensate the deterministic behaviour.",
    default="memory",
    type=click.Choice(["memory", "time"]),
)
@click.option(
    "--profiler",
    help="Allow to find the bottlneck",
    default=None,
    type=click.Choice(["simple", "advanced"]),
)
@click.option(
    "--gradient_clip_val",
    help="Whether to clip gradient value",
    default=None,
    type=float,
)
def cli(**kwargs):

    if kwargs["config_path"] is not None:
        print(f"Readng config files from {kwargs['config_path']}")
        train_opts = read_toml(TRAIN_CONFIG / kwargs["config_path"], kwargs)
        print(f"Conf file was sucessfully loaded")

    else:
        train_opts = kwargs

    if train_opts["resume"]:
        if (
            train_opts["model_name"] is None
            and train_opts["resume_model_type"] is not None
        ):
            models = [
                mn.name
                for mn in Path(MODELS_DIR).rglob(f'*{train_opts["resume_model_type"]}*')
                if (mn.is_dir() and not (mn / "status.txt").exists())
            ]
            print("The training for the following models will be resumed: ", models)
            for model in models:
                train_opts_model = deepcopy(train_opts)
                train_opts_model["model_name"] = model
                train_opts_model = ModelOptions(
                    train_opts_model, resume=train_opts_model["resume"]
                )
                train_opts_model.prepare_commandline(train_opts_model.model_name)
                try:
                    train(train_opts_model)
                except BaseException as ex:
                    display_exception(ex, None)

    else:
        train_opts = ModelOptions(train_opts, resume=train_opts["resume"])
        train_opts.prepare_commandline(train_opts.model)
        print(train_opts.__dict__)
        try:
            train(train_opts)
        except BaseException as ex:
            display_exception(ex, train_opts.model_path)


def train(train_opts):

    classifier = NNExtractor(train_opts)
    classifier.train(resume=train_opts.resume)
    print(f"CPU usage: {psutil.cpu_percent()}%")

    with open(Path(train_opts.model_path) / "status.txt", "w") as f:
        f.write("Status: job was finished for all folds")


def check_if_training_finished(model_path, num_folds):

    return (Path(model_path) / "status.txt").exists()


if __name__ == "__main__":
    cli()
