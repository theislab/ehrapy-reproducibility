import click
from lodepred.constants import MODELS_DIR, DATA_CONFIG, PREPROCESSED_DIR, DATASPLIT_DIR
import json
from lodepred.constants import (
    ReferencePointsDelta,
    VisitFilterType,
    VisitsCombination,
    PhaseType,
    CovariatesTransforms,
)

from lodepred.predictors.datasets.utils import Options
import toml
from lodepred.predictors.datasets.BaseDataset import RespondersDataset
import numpy as np

np.random.seed(228)


def ParseTimeRangeParams(ctx, param, values):
    timerange_params = [json.loads(value) for value in values]
    return timerange_params


def ParseVisitFilterType(ctx, param, value):
    return VisitFilterType[value]


def ParseReferencePointsDelta(ctx, param, value):
    return ReferencePointsDelta[value] if value is not None else value


def ParseReturnType(ctx, param, value):
    return VisitsCombination[value]


def ParseDeltaPattern(ctx, param, values):

    delta_pattern = json.loads(
        values,
        object_hook=lambda d: {
            int(k) if k.lstrip("-").isdigit() else k: v for k, v in d.items()
        },
    )
    return delta_pattern


def ParseCovariatesTransforms(ctx, param, value):
    return CovariatesTransforms[value]


def ResolvePath(ctx, param, value):
    if value is None:
        value = str((MODELS_DIR).resolve())
    return value


@click.command(name="config_data", no_args_is_help=True)
@click.option(
    "--results_path",
    type=str,
    default=None,
    help="path of data config",
    callback=ResolvePath,
)
@click.option("--config_path", type=str, default=None, help="path of data config")

# Process target columns
@click.option(
    "--targets_to_merge",
    type=click.Choice(["good", "poor", "non-responder"]),
    multiple=True,
    default=[],
)
@click.option(
    "--targets_to_drop",
    type=click.Choice(["good", "poor", "non-responder"]),
    multiple=True,
    default=[],
)

# Filter visits by time range/visit indexes
@click.option(
    "--timerange_params", multiple=True, default=[], callback=ParseTimeRangeParams
)
@click.option(
    "--visit_choice_type",
    type=click.Choice(list(VisitFilterType.__members__.keys())),
    default="none",
    callback=ParseVisitFilterType,
)
@click.option(
    "--delta_t",
    type=int,
    default=None,
)
@click.option(
    "--visits_of_interest",
    "-voi",
    type=int,
    multiple=True,
    default=None,
)

# Chosing columns that will be used:


@click.option(
    "--visit_dependent_features",
    "-vdf",
    type=str,
    multiple=True,
    default=[],
)
@click.option(
    "--region_features",
    type=bool,
    default=False,
)
@click.option(
    "--assembled_features",
    type=bool,
    default=True,
)
@click.option(
    "--meta_features",
    "-mf",
    type=str,
    multiple=True,
    default=[],
)
@click.option(
    "--excluded_biomarkers",
    "-eb",
    type=str,
    multiple=True,
    default=[],
)

# Extend table with absolute changes in specified biomarkers


@click.option(
    "--extend_with_delta",
    type=bool,
    default=False,
)
@click.option(
    "--delta_columns",
    "-dc",
    type=str,
    multiple=True,
    default=[],
)
@click.option(
    "--delta_to",
    type=click.Choice(list(ReferencePointsDelta.__members__.keys())),
    default=None,
    callback=ParseReferencePointsDelta,
)
@click.option(
    "--delta_as_rate",
    type=bool,
    default=False,
)
@click.option(
    "--delta_mapping_pattern",
    "-dmp",
    default=[],
    # multiple = True,
    callback=ParseDeltaPattern,
)
# Set normalization/standartization/encoding routine:


@click.option(
    "--robust_biomarkers",
    multiple=True,
    type=str,
    default=[],
)
@click.option(
    "--no_processing",
    multiple=True,
    type=str,
    default=[],
)
@click.option(
    "--one_hot",
    type=bool,
    default=False,
)
@click.option(
    "--additional_target",
    type=click.Choice(RespondersDataset.standart_additional_targets),
    default=None,
)
@click.option(
    "--additional_target_transformation",
    type=click.Choice(list(CovariatesTransforms.__members__.keys())),
    default=None,
    callback=ParseCovariatesTransforms,
)
@click.option(
    "--additional_target_norm_scalar",
    type=float,
    default=None,
)
@click.option(
    "--additional_target_norm_scalar",
    type=float,
    multiple=True,
    default=None,
)

#  Combine visits of the same trajectory into "one entry"


@click.option(
    "--return_type",
    type=click.Choice(list(VisitsCombination.__members__.keys())),
    default="pd_multiindex",
    callback=ParseReturnType,
)

# (Optionally) oversample data
@click.option(
    "--oversample",
    type=bool,
    default=False,
)

# (Optionally) oversample data
@click.option(
    "--data_id",
    type=str,
    default=None,
)
def cli(**kwargs):
    if kwargs["config_path"] is not None:
        opts_dict = read_toml(DATA_CONFIG / kwargs["config_path"], kwargs)
    else:
        opts_dict = kwargs

    opts = Options(opts_dict)
    opts.get_experiment_path(kwargs["results_path"])

    opts.save_opts()

    dataset_train = RespondersDataset.from_processed(
        preprocessed_filepath=PREPROCESSED_DIR,
        data_split_filepath=DATASPLIT_DIR,
        phase=PhaseType.train,
    )

    dataset_train.prepare_data(opts=opts)


def read_toml(config_path, cli_dict):
    config_dict = toml.load(config_path)
    for key in config_dict:
        if key == "timerange_params":
            value = ParseTimeRangeParams(None, key, config_dict[key])
        elif key == "visit_choice_type":
            value = ParseVisitFilterType(None, key, config_dict[key])
        elif key == "delta_to":
            value = ParseReferencePointsDelta(None, key, config_dict[key])
        elif key == "delta_mapping_pattern":
            value = ParseDeltaPattern(None, key, config_dict[key])
        elif key == "return_type":
            value = ParseReturnType(None, key, config_dict[key])
        else:
            value = config_dict[key]
        cli_dict[key] = value
    return cli_dict


if __name__ == "__main__":
    cli()
