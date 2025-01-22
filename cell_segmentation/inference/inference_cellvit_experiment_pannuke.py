# -*- coding: utf-8 -*-
# CellViT Inference Method for Patch-Wise Inference on a test set
# Without merging WSI
#
# Aim is to calculate metrics as defined for the PanNuke dataset
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import argparse
import inspect
import os
import sys
from dataloader_util import get_loader
import imageio.v3 as imageio
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)

from base_ml.base_experiment import BaseExperiment

BaseExperiment.seed_run(1232)

import json
from pathlib import Path
from typing import List, Tuple, Union
import micro_sam.training as sam_training
import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import yaml
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from skimage.color import rgba2rgb
from sklearn.metrics import accuracy_score
from tabulate import tabulate
from torch.utils.data import DataLoader
from torchmetrics.functional import dice
from torchmetrics.functional.classification import binary_jaccard_index
from torchvision import transforms
from utils.tools import unflatten_dict
from natsort import natsorted
from glob import glob

from cell_segmentation.datasets.dataset_coordinator import select_dataset
from models.segmentation.cell_segmentation.cellvit import DataclassHVStorage
from cell_segmentation.utils.metrics import (
    cell_detection_scores,
    cell_type_detection_scores,
    get_fast_pq,
    remap_label,
    binarize,
)
from cell_segmentation.utils.post_proc_cellvit import calculate_instances
from cell_segmentation.utils.tools import cropping_center, pair_coordinates
from models.segmentation.cell_segmentation.cellvit import (
    CellViT,
    CellViT256,
    CellViTSAM,
)
from models.segmentation.cell_segmentation.cellvit_shared import (
    CellViT256Shared,
    CellViTSAMShared,
    CellViTShared,
)
from utils.logger import Logger


class InferenceCellViT:
    def __init__(
        self,
        model_path,
        output_dir: Union[Path, str],
        input_dir: Union[Path, str],
        gpu: int,
        magnification: int = 40,
    ) -> None:
        """Inference for HoverNet

        Args:
            run_dir (Union[Path, str]): logging directory with checkpoints and configs
            gpu (int): CUDA GPU device to use for inference
            magnification (int, optional): Dataset magnification. Defaults to 40.
            checkpoint_name (str, optional): Select name of the model to load. Defaults to model_best.pth
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.run_dir = Path(os.path.join(output_dir, "logs"))
        self.inst_outdir = os.path.join(output_dir, "instance")
        self.class_outdir = os.path.join(output_dir, "semantic")
        os.makedirs(self.inst_outdir, exist_ok=True)
        os.makedirs(self.class_outdir, exist_ok=True)
        os.makedirs(self.run_dir, exist_ok=True)
        self.device = f"cuda:{gpu}"
        self.run_conf: dict = None
        self.logger: Logger = None
        self.magnification = magnification
        self.image_names = [os.path.basename(img) for img in natsorted(glob(os.path.join(input_dir, "eval_split", "test_images", "*.tiff")))]

        # self.__load_run_conf()

        self.input_dir = input_dir
        self.__instantiate_logger()
        self.__load_model()
        self.__check_eval_model()
        self.__setup_amp()


    def __instantiate_logger(self) -> None:
        """Instantiate logger

        Logger is using no formatters. Logs are stored in the run directory under the filename: inference.log
        """
        logger = Logger(
            level="INFO",
            log_dir=self.output_dir,
            comment="inference_pannuke",
            use_timestamp=False,
            formatter="%(message)s",
        )
        self.logger = logger.create_logger()

    def __check_eval_model(self) -> None:
        """Check if there is a best model pytorch file"""
        assert (self.model_path).is_file()

    def __setup_amp(self) -> None:
        """Setup automated mixed precision (amp) for inference."""
        self.mixed_precision = self.run_conf["training"].get("mixed_precision", False)

    
    def __load_model(self) -> None:
        """Load model and checkpoint and load the state_dict"""
        self.logger.info(f"Loading model: {self.model_path}")

        model_checkpoint = torch.load(self.model_path, map_location="cpu")

        # unpack checkpoint
        self.run_conf = unflatten_dict(model_checkpoint["config"], ".")
        self.model = self.get_model(model_type=model_checkpoint["arch"])
        self.logger.info(
            self.model.load_state_dict(model_checkpoint["model_state_dict"])
        )
        self.model.eval()
        self.model.to(self.device)


    def get_model(
        self, model_type: str
    ) -> Union[
        CellViT,
        CellViTShared,
        CellViT256,
        CellViT256Shared,
        CellViTSAM,
        CellViTSAMShared,
    ]:
        """Return the trained model for inference

        Args:
            model_type (str): Name of the model. Must either be one of:
                CellViT, CellViTShared, CellViT256, CellViT256Shared, CellViTSAM, CellViTSAMShared

        Returns:
            Union[CellViT, CellViTShared, CellViT256, CellViT256Shared, CellViTSAM, CellViTSAMShared]: Model
        """
        implemented_models = [
            "CellViT",
            "CellViTShared",
            "CellViT256",
            "CellViT256Shared",
            "CellViTSAM",
            "CellViTSAMShared",
        ]
        if model_type not in implemented_models:
            raise NotImplementedError(
                f"Unknown model type. Please select one of {implemented_models}"
            )
        if model_type in ["CellViT", "CellViTShared"]:
            if model_type == "CellViT":
                model_class = CellViT
            elif model_type == "CellViTShared":
                model_class = CellViTShared
            model = model_class(
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                embed_dim=self.run_conf["model"]["embed_dim"],
                input_channels=self.run_conf["model"].get("input_channels", 3),
                depth=self.run_conf["model"]["depth"],
                num_heads=self.run_conf["model"]["num_heads"],
                extract_layers=self.run_conf["model"]["extract_layers"],
                regression_loss=self.run_conf["model"].get("regression_loss", False),
            )

        elif model_type in ["CellViT256", "CellViT256Shared"]:
            if model_type == "CellViT256":
                model_class = CellViT256
            elif model_type == "CellViT256Shared":
                model_class = CellViT256Shared
            model = model_class(
                model256_path=None,
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                regression_loss=self.run_conf["model"].get("regression_loss", False),
            )
        elif model_type in ["CellViTSAM", "CellViTSAMShared"]:
            if model_type == "CellViTSAM":
                model_class = CellViTSAM
            elif model_type == "CellViTSAMShared":
                model_class = CellViTSAMShared
            model = model_class(
                model_path=None,
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                vit_structure=self.run_conf["model"]["backbone"],
                regression_loss=self.run_conf["model"].get("regression_loss", False),
            )
        return model

    def setup_patch_inference(
        self, test_folds: List[int] = None
    ) -> tuple[
        Union[
            CellViT,
            CellViTShared,
            CellViT256,
            CellViT256Shared,
            CellViTSAM,
            CellViTSAMShared,
        ],
        DataLoader,
        dict,
    ]:
        """Setup patch inference by defining a patch-wise datalaoder and loading the model checkpoint

        Args:
            test_folds (List[int], optional): Test fold to use. Otherwise defined folds from config.yaml (in run_dir) are loaded. Defaults to None.

        Returns:
            tuple[Union[CellViT, CellViTShared, CellViT256, CellViT256Shared, CellViTSAM, CellViTSAMShared], DataLoader, dict]:
                Union[CellViT, CellViTShared, CellViT256, CellViT256Shared, CellViTSAM, CellViTSAMShared]: Best model loaded form checkpoint
                DataLoader: Inference DataLoader
                dict: Dataset configuration. Keys are:
                    * "tissue_types": describing the present tissue types with corresponding integer
                    * "nuclei_types": describing the present nuclei types with corresponding integer

        """
        # get model for inference
        checkpoint = torch.load(
            self.model_path, map_location="cpu"
        )
        model = self.get_model(model_type=checkpoint["arch"])
        self.logger.info(
            f"Loading best model from {str(self.model_path)}"
        )
        self.logger.info(model.load_state_dict(checkpoint["model_state_dict"]))

        

        transform_settings = self.run_conf["transformations"]
        if "normalize" in transform_settings:
            mean = transform_settings["normalize"].get("mean", (0.5, 0.5, 0.5))
            std = transform_settings["normalize"].get("std", (0.5, 0.5, 0.5))
        else:
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)
        inference_dataloader = get_loader(
            path=self.input_dir,
            patch_shape=(512, 512),
            batch_size=1,
            num_workers=8,
            pin_memory=False,
            shuffle=False,
        )

        return model, inference_dataloader

    def run_patch_inference(
        self,
        model: Union[
            CellViT,
            CellViTShared,
            CellViT256,
            CellViT256Shared,
            CellViTSAM,
            CellViTSAMShared,
        ],
        inference_dataloader: DataLoader,
        generate_plots: bool = False,
    ) -> None:
        """Run Patch inference with given setup

        Args:
            model (Union[CellViT, CellViTShared, CellViT256, CellViT256Shared, CellViTSAM, CellViTSAMShared]): Model to use for inference
            inference_dataloader (DataLoader): Inference Dataloader. Must return a batch with the following structure:
                * Images (torch.Tensor)
                * Masks (dict)
                * Tissue types as str
                * Image name as str
            dataset_config (dict): Dataset configuration. Required keys are:
                    * "tissue_types": describing the present tissue types with corresponding integer
                    * "nuclei_types": describing the present nuclei types with corresponding integer
            generate_plots (bool, optional): If inference plots should be generated. Defaults to False.
        """
        # put model in eval mode
        model.to(device=self.device)
        model.eval()

        inference_loop = tqdm.tqdm(
            enumerate(inference_dataloader), total=len(inference_dataloader)
        )

        with torch.no_grad():
            for image_idx, batch in inference_loop:
                self.inference_step(
                    model, batch, generate_plots=generate_plots, image_name=self.image_names[image_idx]
                )
               




    def inference_step(
        self,
        model: Union[
            CellViT,
            CellViTShared,
            CellViT256,
            CellViT256Shared,
            CellViTSAM,
            CellViTSAMShared,
        ],
        batch: tuple,
        generate_plots: bool = False,
        image_name = None,
    ) -> None:
        """Inference step for a patch-wise batch

        Args:
            model (CellViT): Model to use for inference
            batch (tuple): Batch with the following structure:
                * Images (torch.Tensor)
                * Masks (dict)
                * Tissue types as str
                * Image name as str
            generate_plots (bool, optional):  If inference plots should be generated. Defaults to False.
        """
        # unpack batch, for shape compare train_step method
        img = batch[0].to(self.device)
        img = img.to(torch.float32)
        if torch.max(img) >= 5:
            img = img / 255
        if torch.max(img) >= 5:
            img = img / 255
        model.zero_grad()
        if self.mixed_precision:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                predictions = model.forward(img)
        else:
            predictions = model.forward(img)
        class_pred, inst_pred = self.unpack_predictions(predictions=predictions, model=model)


        int_class = class_pred.cpu().numpy().astype(np.uint16)
        squeezed_class = np.squeeze(int_class)
        final_class = np.argmax(squeezed_class, axis=0)
        class_output = os.path.join(str(self.class_outdir), image_name)
        imageio.imwrite(class_output, final_class)
        
        if inst_pred.dtype != np.int32:
            inst_pred = inst_pred.cpu().numpy()        
        int_inst = inst_pred.astype(np.uint16)
        squeezed_inst = np.squeeze(int_inst)
        inst_output = os.path.join(str(self.inst_outdir), image_name)
        imageio.imwrite(inst_output, squeezed_inst)



    def unpack_predictions(
        self, predictions: dict, model: CellViT,
    ) -> DataclassHVStorage:
        """Unpack the given predictions. Main focus lays on reshaping and postprocessing predictions, e.g. separating instances

        Args:
            predictions (dict): Dictionary with the following keys:
                * tissue_types: Logit tissue prediction output. Shape: (batch_size, num_tissue_classes)
                * nuclei_binary_map: Logit output for binary nuclei prediction branch. Shape: (batch_size, H, W, 2)
                * hv_map: Logit output for hv-prediction. Shape: (batch_size, H, W, 2)
                * nuclei_type_map: Logit output for nuclei instance-prediction. Shape: (batch_size, num_nuclei_classes, H, W)
            model (CellViT): Current model

        Returns:
            DataclassHVStorage: Processed network output

        """
        predictions["nuclei_binary_map"] = F.softmax(
            predictions["nuclei_binary_map"], dim=1
        )  # shape: (batch_size, 2, H, W)
        predictions["nuclei_type_map"] = F.softmax(
            predictions["nuclei_type_map"], dim=1
        )  # shape: (batch_size, num_nuclei_classes, H, W)
        (
            predictions["instance_map"],
            predictions["instance_types"],
        ) = model.calculate_instance_map(
            predictions, magnification=self.magnification
        )  # shape: (batch_size, H', W')
        predictions["instance_types_nuclei"] = model.generate_instance_nuclei_map(
            predictions["instance_map"], predictions["instance_types"]
        ).to(
            self.device
        )  

        return predictions["instance_types_nuclei"], remap_label(predictions["instance_map"]) 


class InferenceCellViTParser:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Perform CellViT inference for dataset",
        )
        parser.add_argument(
            "--model",
            type=str,
            help="Model checkpoint file that is used for inference",
            default=None,
        )

        parser.add_argument(
            "--dataset",
            type=str,
            help="Path to MoNuSeg dataset.",
            default="/projects/datashare/tio/histopathology/public-datasets/MoNuSeg/1024/testing",
        )
        parser.add_argument(
            "--outdir",
            type=str,
            help="Path to output directory to store results.",
            default="/homes/fhoerst/histo-projects/CellViT/results/PanNuke/Revision/CellViT/Common-Loss/SAM-H/x20/Fold-1-x20/MoNuSeg/x40/256_64",
        )
        parser.add_argument(
            "--gpu", type=int, help="Cuda-GPU ID for inference. Default: 0", default=0
        )
        parser.add_argument(
            "--magnification",
            type=int,
            help="Dataset Magnification. Either 20 or 40. Default: 40",
            choices=[20, 40],
            default=40,
        )
        self.parser = parser
    def parse_arguments(self) -> dict:
        opt = self.parser.parse_args()
        return vars(opt)
    
if __name__ == "__main__":
    configuration_parser = InferenceCellViTParser()
    configuration = configuration_parser.parse_arguments()
    print(configuration)

    inf = InferenceCellViT(
        model_path=configuration["model"],
        input_dir=configuration["dataset"],
        output_dir=configuration["outdir"],
        gpu=configuration["gpu"],
        magnification=configuration["magnification"],
    )
    model, dataloader = inf.setup_patch_inference()
    inf.run_patch_inference(model, dataloader, generate_plots=False)
