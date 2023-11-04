import os
import json
import datetime
from pathlib import Path
import pickle as pkl
import shutil
import glob
import numpy as np
from diabetic_retinopathy.latent_extractors.predictors.utils import NNModel
from diabetic_retinopathy.constants import MODELS_DIR
from diabetic_retinopathy.latent_extractors.predictors.models.LatentExtractor import (
    Models,
)


class ModelOptions:

    evaluation_list = [
        "batch_size",
        "phase",
        "model_name",
        "task",
        "nThreads",
        "result_dir",
        "model_path",
        "model_dir",
        "train",
    ]
    computational_list = [
        "resume",
        "gpu",
        "gpu_device",
        "model_name",
    ]

    resume_list = ["model_name", "model_path"]

    def __init__(self, opts, train: bool = True, resume: bool = True):
        self.__dict__.update(**{key: item for key, item in opts.items()})
        self.train = train
        self.resume = resume

    def prepare_commandline(self, model_name=None, load_pickle=True):
        import os
        import json

        if (
            (not self.train)
            or (hasattr(self, "resume") and self.resume)
            and load_pickle
        ):
            if not hasattr(self, "model_path"):
                self.model_path = MODELS_DIR / f"{model_name}"

            with open(os.path.join(self.model_path, "params.pkl"), "rb") as f:
                opts_dict = pkl.load(f)

            for key, item in opts_dict.items():

                if key in self.computational_list:
                    continue

                elif (
                    hasattr(self, "resume")
                    and self.resume
                    and key in self.resume_list
                    and self.__dict__[key] is not None
                ):
                    continue

                elif not self.train and key in self.evaluation_list:
                    continue
                else:
                    setattr(self, key, item)

        elif self.train and (hasattr(self, "resume") and not self.resume):
            model_name = self.get_model_name(model_name)
            self.model_path, self.model_name = self._create_model_folder(model_name)
            self.save_opts()
            print(self.model_path, self.model_name)

        else:
            pass

    def _create_model_folder(self, name):
        model_dir = MODELS_DIR / name
        if os.path.exists(model_dir):
            if (not self.overwrite) and len(os.listdir(model_dir)) > 0:
                print(
                    f"Since overwrite={self.overwrite} and path already exists, creating new unique name"
                )
                name = self.change_model_name(name)
                # name = name + '_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                model_dir = os.path.join(MODELS_DIR, name)

            else:
                if not self.skip_finished_jobs:
                    print(
                        f"Since overwrite={self.overwrite} and path already exists, cleaning it"
                    )
                    shutil.rmtree(model_dir)

        os.makedirs(model_dir, exist_ok=True)
        print(model_dir)
        return model_dir, name

    def get_model_name(self, model_name):

        if self.nn_model_id is not None:
            model_name = f"{model_name}-{self.nn_model_id}"

        return f"{model_name}-version0"

    def change_model_name(self, model_name):

        name = model_name.split("-version")[0]
        # print(name, [mn for mn in  glob.glob(str(MODELS_DIR / f'{name}*'))])
        version = np.max(
            [
                int(mn.split("-version")[-1])
                for mn in glob.glob(str(MODELS_DIR / f"{name}*"))
            ]
        )
        return f"{name}-version{version+1}"

    def save_opts(self):

        with open(os.path.join(self.model_path, "params.pkl"), "wb") as f:
            pkl.dump(self.__dict__, f)

    def get_dict(self):
        return self.__dict__
