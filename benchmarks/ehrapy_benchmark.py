import ehrapy as ep
import os
import argparse
import sys
import logging
import warnings
warnings.filterwarnings("ignore")
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams



L = logging.getLogger()
L.setLevel(logging.INFO)
log_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s: %(levelname)s - %(message)s')
log_handler.setFormatter(formatter)
L.addHandler(log_handler)

# parse arguments
parser = argparse.ArgumentParser()

parser.add_argument('--input_anndata',
                    default='adata.h5ad',
                    help='')


#end parse

args, opt = parser.parse_known_args()


L.info("reading data")

adata = ep.io.read_h5ad(args.input_anndata)

L.info("adata shape:")
L.info(adata.shape)

L.info("start processing")
t = time.time()

obs_metric, var_metrics = ep.pp.qc_metrics(adata)
adata=adata[:,adata.var['missing_values_pct']<=70]

L.info("retaining these vars")
L.info(adata.shape[1])

ep.pp.winsorize(adata)
ep.pp.summarize_measurements(adata)
ep.anndata.infer_feature_types(adata) #new since fass imputation
ep.pp.knn_impute(adata, n_neighbours=20) #just leave low k should be fine
ep.pp.offset_negative_values(adata)
ep.pp.log_norm(adata)
ep.pp.pca(adata)
ep.pp.neighbors(adata)
ep.tl.leiden(adata)
ep.tl.umap(adata)

elapsed = time.time() - t
L.info("done processing")
L.info(elapsed)



