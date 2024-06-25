# Models wrapper

All models supported by the package are stored in the sslbio.models.LatentExtractor.Models as the class attributes.

To get a prediction model, call function get_latent_extractor() with model type specified.
E.g.:

```sh
from sslbio.models.LatentExtractor import Models, get_latent_extractor
torch_arr = torch.rand(1, 3, 224, 224)

model = get_latent_extractor(model_name=Models.resnet50HistDINO)
out = model(torch_arr)
print(out)

```

# Feature extraction

To extract features from the desired dataset, one needs to do the following:

1. Create dataset. For that, specify the experiment name, technology and resolution. So far, the following datasets are available:

Technology: 'stnet':

- Experiment name: 'st_human_breast_cancer'
  - Resolution: 'spots'
- Technology: 'visium':
  - Experiment name: 'V1_Human_Lymph_Node'
    - Resolution: 'spots', '5spots', '10spots'
  - Experiment name: 'Visium_Human_Breast_Cancer'
    - Resolution: 'spotss', '5spots'
- Technology: 'dbitx':
  - Experiment name: 'dbitx_37_48_A1'
    - Resolution: 'spots'

Given the dataset details, one needs to create and set up a dataloader:

```
from sslbio.dataloaders.dataloaders import ImageDataModule

data_module = ImageDataModule(experiment_name=experiment,
                                        technology=technology,
                                        resolution=resolution)

data_module.setup()
```

Then, choose a model with which you wish to extract representations. Available models are listed in the sslbio.models.LatenExtractor.Models class.
To see available models:

```
from sslbio.models.LatenExtractor import Models
print(list(Models.__members__.keys()))
```

With the model of your choice, create a feature extractor:

```
model = get_latent_extractor(model_name=Models['model_id'])
```

Alternatively:

```
model = get_latent_extractor(model_name=Models.model_id)
```

Then, run feature extraction:

```
model.extract_features(data_module)
```

Method 'extract_features' would extract latents fro each pair of (image, observation). The results will be then stored in the follwing way:

```
.
└── my_project
	├── extracted_features
	|	├── [technology]_[experiment_name]
	|	|	├── [technology]_[experiment_name]_[resolution]_[model_id]_feature_size-[latents_size]".npy	# HxW array, where H is #obs, W - size of latent representation of one image
	|	|	└── [technology]_[experiment_name]_[resolution]_[model_id]_feature_size-[latents_size]".pkl		# pairs of latent vectors and obs
	|	└── ..
	└── ..

```
