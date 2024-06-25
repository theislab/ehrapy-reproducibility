import os

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
    RESULTS_DIR,
)

from diabetic_retinopathy.cli.utils.cli_utils import read_toml, display_exception
from copy import deepcopy


from pathlib import Path
import warnings

import numpy as np
import pandas as pd

np.random.seed(228)
warnings.simplefilter(action="ignore", category=FutureWarning)


@click.command(name="predict", no_args_is_help=True)
@click.option("--config_path", type=str, default=None, help="path of data config")
@click.option(
    "--models_for_inference",
    type=click.Choice(["all", "unpredicted"]),
    default="unpredicted",
    help="models for which inference will be made",
)
@click.option("--include_unfinished", type=bool, default=False, help="gpu")
@click.option(
    "--model_types",
    type=click.Choice(list(Models.__members__.keys())),
    # type = click.Choice([el for el in list(Models.__members__.keys()) if "FC" in el]),
    multiple=True,
    default=None,
)
@click.option(
    "--model_names",
    type=str,
    multiple=True,
    default=None,
)
@click.option(
    "--limit_predict_batches",
    type=int,
    default=None,
)
@click.option(
    "--batch_size",
    type=int,
    default=64,
    # default=36,
)
@click.option("--num_workers", type=int, default=4)
@click.option("--gpu", type=bool, default=False, help="gpu")
@click.option(
    "--phase",
    help="data phase on which to predict",
    default=["val"],
    multiple=True,
    type=click.Choice(["train", "val", "test"]),
)
@click.option(
    "--which_checkpoint",
    help="In case of nn: whether to predict from the best or the last checkpoint",
    default="last",
    type=click.Choice(["last", "best"]),
)
@click.option(
    "--prediction_type",
    help="Whether to predict latents or class",
    default="latents",
    type=click.Choice(["class", "latents"]),
)
def cli(**kwargs):

    inference_opts_all = setup_cli_params(kwargs)

    predict_from_model_names = len(inference_opts_all["model_names"]) > 0
    models_list = (
        inference_opts_all["model_names"]
        if predict_from_model_names
        else inference_opts_all["model_types"]
    )

    for model_type in models_list:
        for phase in inference_opts_all["phase"]:
            # print(phase)
            models = (
                get_model_path(
                    model_name=model_type,
                    include_unfinished=inference_opts_all["include_unfinished"],
                    model_status=inference_opts_all["models_for_inference"],
                    model_ckptn=inference_opts_all["which_checkpoint"],
                    phase=phase,
                )
                if predict_from_model_names
                else find_model_path(
                    model_type=model_type,
                    include_unfinished=inference_opts_all["include_unfinished"],
                    model_status=inference_opts_all["models_for_inference"],
                    model_ckptn=inference_opts_all["which_checkpoint"],
                    phase=phase,
                )
            )

            for model_dict in models:
                try:
                    predictor_ = ModelPredictor(
                        model_dict=model_dict,
                        inference_opts_all=inference_opts_all,
                        phase=phase,
                    )
                    predictor_.predict()

                except BaseException as ex:
                    display_exception(ex)


def setup_cli_params(kwargs):
    if kwargs["config_path"] is not None:
        inference_opts_all = read_toml(INFERENCE_CONFIG / kwargs["config_path"], kwargs)
    else:
        inference_opts_all = kwargs

    if (
        len(inference_opts_all["model_types"]) == 0
        and len(inference_opts_all["model_names"]) == 0
    ):
        if inference_opts_all["prediction_type"] == "class":
            inference_opts_all["model_types"] = [
                el for el in list(Models.__members__.keys()) if "FC" in el
            ]
        elif inference_opts_all["prediction_type"] == "latents":
            inference_opts_all["model_types"] = list(Models.__members__.keys())

    return inference_opts_all


class ModelPredictor:
    def __init__(self, model_dict, inference_opts_all, phase):
        self.model_dict = model_dict
        self.phase = PhaseType[phase]
        self.inference_opts_all = inference_opts_all

    def predict(self):
        print(f"Predicting from {self.model_dict['model_name']}")

        inference_opts_ = self.prepare_model_opts()
        self._prediction(inference_opts_)

    def prepare_model_opts(self):
        inference_opts_dict = deepcopy(self.inference_opts_all)
        inference_opts_dict.pop("model_types", None)
        if "model_type" in self.model_dict:
            inference_opts_dict["model_type"] = self.model_dict["model_type"]
        inference_opts_dict["model_name"] = self.model_dict["model_name"]

        inference_opts = ModelOptions(inference_opts_dict, train=False)
        inference_opts.prepare_commandline(
            self.model_dict["model_name"], load_pickle="FC" in inference_opts.model_name
        )
        return inference_opts

    def _prediction(self, inference_opts):
        classifier = NNExtractor(inference_opts)
        classifier.load_model()
        if inference_opts.prediction_type == "class":
            classifier.get_performance(phase=self.phase, save_pred=True)
        elif inference_opts.prediction_type == "latents":
            classifier.get_latents(phase=self.phase)


def get_model_path(
    model_name,
    include_unfinished=False,
    model_status="all",
    model_ckptn="best",
    phase="val",
):

    if include_unfinished:
        models = [mn for mn in Path(MODELS_DIR).rglob(f"*{model_name}*") if mn.is_dir()]
    else:
        models = [
            mn
            for mn in Path(MODELS_DIR).rglob(f"*{model_name}*")
            if (mn.is_dir() and (mn / "status.txt").exists())
        ]
    unpred_models = []
    if model_status == "unpredicted":
        for mn in models:
            prediction_exist = (
                RESULTS_DIR
                / mn.name
                / "class_predictions"
                / model_ckptn
                / f"{phase}_preds.csv"
            ).exists()

            if not prediction_exist:
                unpred_models.append({"model_name": mn.name, "model_type": model_type})
    else:
        unpred_models = [({"model_name": mn.name}) for mn in models]
        # print(f"All models {models}; already predicted models: {pred_models}")

    if len(unpred_models) > 0:
        print(
            f"predicting  {model_name}, {model_status} models, predictions for following models will be acquired: {unpred_models}"
        )

    return unpred_models


def find_model_path(
    model_type,
    include_unfinished=False,
    model_status="all",
    model_ckptn="best",
    phase="val",
):

    if include_unfinished:
        models = [mn for mn in Path(MODELS_DIR).rglob(f"*{model_type}*") if mn.is_dir()]
    else:
        models = [
            mn
            for mn in Path(MODELS_DIR).rglob(f"*{model_type}*")
            if (mn.is_dir() and mn / "status.txt").exists()
        ]

    unpred_models = []
    if model_status == "unpredicted":
        for mn in models:
            prediction_exist = (
                RESULTS_DIR
                / mn.name
                / "class_predictions"
                / model_ckptn
                / f"{phase}_preds.csv"
            ).exists()

            if not prediction_exist:
                unpred_models.append({"model_name": mn.name, "model_type": model_type})
    else:
        unpred_models = [
            ({"model_name": mn.name, "model_type": model_type}) for mn in models
        ]
        # print(f"All models {models}; already predicted models: {pred_models}")

    if len(unpred_models) > 0:
        print(
            f"predicting  {model_type}, {model_status} models, predictions for following models will be acquired: {unpred_models}"
        )

    return unpred_models


if __name__ == "__main__":
    cli()
