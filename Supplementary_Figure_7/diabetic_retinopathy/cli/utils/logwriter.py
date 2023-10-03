import logging
from os import makedirs, path

import numpy as np
import pandas as pd


class StdLevelFilter(logging.Filter):
    def __init__(self, err=False):
        super().__init__()
        self.err = err

    def filter(self, record):
        if record.levelno <= logging.INFO:
            return not self.err
        return self.err


def setup_logging(verbosity: int = 0) -> None:
    """
    Setup ClinicaDL's logging facilities.
    Args:
        verbosity: The desired level of verbosity for logging.
            (0 (default): WARNING, 1: INFO, 2: DEBUG)
    """
    from logging import DEBUG, INFO, WARNING, Formatter, StreamHandler, getLogger
    from sys import stderr, stdout

    # Cap max verbosity level to 2.
    verbosity = min(verbosity, 2)

    # Define the module level logger.
    logger = getLogger("clinicadl")
    logger.setLevel([WARNING, INFO, DEBUG][verbosity])

    if logger.hasHandlers():
        logger.handlers.clear()

    stdout = StreamHandler(stdout)
    stdout.addFilter(StdLevelFilter())
    stderr = StreamHandler(stderr)
    stderr.addFilter(StdLevelFilter(err=True))
    # create formatter
    formatter = Formatter("%(asctime)s - %(levelname)s: %(message)s", "%H:%M:%S")
    # add formatter to ch
    stdout.setFormatter(formatter)
    stderr.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(stdout)
    logger.addHandler(stderr)
