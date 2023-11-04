from diabetic_retinopathy.latent_extractors.predictors.NNExtractor import NNExtractor
from diabetic_retinopathy.latent_extractors.predictors.utils import (
    Stage,
    FinishCallback,
    NNModel,
)
from diabetic_retinopathy.constants import MODELS_DIR, RESULTS_DIR
from diabetic_retinopathy.latent_extractors.predictors.models.LatentExtractor import (
    Models as SpatialSSLModels,
)

# from ehrapylat.latent_extractors.predictors.utils import Stage
from diabetic_retinopathy.latent_extractors.predictors.opts import ModelOptions
from diabetic_retinopathy.constants import RESULTS_DIR, PhaseType

if __name__ == "__main__":
    model_opts = dict(
        model="ViTHF",
        model_type="SpatialSSL",
        model_name="ViTHF",
        model_path=RESULTS_DIR,
        batch_size=32,
        num_workers=2,
        which_checkpoint=None,
        limit_predict_batches=None,
        seed=2,
    )
    model_opts = ModelOptions(model_opts)
    for model in list(SpatialSSLModels.__members__.keys()):
        model_opts.model = model
        extractor = NNExtractor(model_opts)
        extractor.load_model()
        extractor.get_latents(phase=PhaseType.test)
