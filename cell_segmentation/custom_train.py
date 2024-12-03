# -*- coding: utf-8 -*-
# Running an Experiment Using CellViT cell segmentation network
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from training_util import get_training_args
import wandb

from base_ml.base_cli import ExperimentBaseParser
from cell_segmentation.experiments.experiment_cellvit_pannuke import (
    ExperimentCellVitPanNuke,
)
from cell_segmentation.experiments.experiment_cellvit_conic import (
    ExperimentCellViTCoNic,
)

from cell_segmentation.inference.inference_cellvit_experiment_pannuke import (
    InferenceCellViT,
)

if __name__ == "__main__":
    # Parse arguments
    parser = get_training_args()
    args = parser.parse_args()
    # Setup experiment 
    outdir = args.output
    inference = InferenceCellViT(
        run_dir=outdir,
        gpu=0,
        dataset_path=args.dataset,
        magnification=args.magnification,
        checkpoint_name=args.checkpoint,
    )
    (
        trained_model,
        inference_dataloader,
        dataset_config,
    ) = inference.setup_patch_inference()
    inference.run_patch_inference(
        trained_model, inference_dataloader, dataset_config, generate_plots=False
    )
    wandb.finish()
