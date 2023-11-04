# coding: utf8

import click


from diabetic_retinopathy.cli.train_cli import cli as train_cli
from diabetic_retinopathy.cli.utils.logwriter import setup_logging

from diabetic_retinopathy.cli.predict_cli import cli as predict_cli


CONTEXT_SETTINGS = dict(
    # Extend content width to avoid shortening of pipeline help.
    max_content_width=160,
    # Display help string with -h, in addition to --help.
    help_option_names=["-h", "--help"],
)


@click.group(context_settings=CONTEXT_SETTINGS, no_args_is_help=True)
@click.version_option()
@click.option(
    "-v", "--verbose", "verbosity", count=True, help="Increase logging verbosity."
)
def cli(verbosity):
    """diabetic_retinopathy command line."""
    setup_logging(verbosity=verbosity)


cli.add_command(train_cli)
cli.add_command(predict_cli)


if __name__ == "__main__":
    cli()
