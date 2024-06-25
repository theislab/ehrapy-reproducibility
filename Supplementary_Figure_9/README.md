# Diabetic Retinopathy

Installation

```
conda create -n diabetic_retinopathy python=3.9
conda activate diabetic_retinopathy
pip install poetry
pip install ehrapy
pip install -U scvelo
pip install pytorch-lightning
conda install -c conda-forge jupyterlab
conda install tensorboard
conda install seaborn
pip install urllib3==1.26.15
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install networkx==2.8.8
conda install -c conda-forge -c plotly jupyter-dash
poetry install
```

To get the data:

1. Download tha data

```
kaggle competitions download -c diabetic-retinopathy-detection
```

2. Unzip the files:

```
cat test.zip.* >test.zip
unzip test.zip
```
