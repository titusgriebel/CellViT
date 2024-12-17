# -*- coding: utf-8 -*-
# PanNuke Dataset
#
# Dataset information: https://arxiv.org/abs/2003.10778
# Please Prepare Dataset as described here: docs/readmes/pannuke.md
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import logging
import sys  # remove
from pathlib import Path
from typing import Callable, Tuple, Union
import os
import numpy as np
import pandas as pd
import torch
import yaml
from numba import njit
from PIL import Image
from scipy.ndimage import center_of_mass, distance_transform_edt
from glob import glob
from natsort import natsorted
from torch_em.data.datasets.histopathology.monuseg import get_monuseg_loader
from torch_em.data.datasets.histopathology.lizard import get_lizard_loader
sys.path.append("/user/titus.griebel/u12649/CellViT/cell_segmentation")  # remove
from datasets.base_cell import CellDataset
import imageio


def fix_duplicates(inst_map: np.ndarray) -> np.ndarray:  # this and the following function could be imported 
    """Re-label duplicated instances in an instance labelled mask.

    Parameters
    ----------
        inst_map : np.ndarray
            Instance labelled mask. Shape (H, W).

    Returns
    -------
        np.ndarray:
            The instance labelled mask without duplicated indices.
            Shape (H, W).
    """
    current_max_id = np.amax(inst_map)
    inst_list = list(np.unique(inst_map))
    if 0 in inst_list:
        inst_list.remove(0)

    for inst_id in inst_list:
        inst = np.array(inst_map == inst_id, np.uint8)
        remapped_ids = ndimage.label(inst)[0]
        remapped_ids[remapped_ids > 1] += current_max_id
        inst_map[remapped_ids > 1] = remapped_ids[remapped_ids > 1]
        current_max_id = np.amax(inst_map)

    return inst_map

def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]

logger = logging.getLogger()
logger.addHandler(logging.NullHandler())

def remap_label(pred, by_size=False):
    """
    Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3]
    not [0, 2, 4, 6]. The ordering of instances (which one comes first)
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID

    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)
    """
    pred_id = list(np.unique(pred))
    if 0 in pred_id:
        pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred

class PanNukeDataset(CellDataset):
    """PanNuke dataset

    Args:
        dataset_path (Union[Path, str]): Path to PanNuke dataset. Structure is described under ./docs/readmes/cell_segmentation.md
        folds (Union[int, list[int]]): Folds to use for this dataset
        transforms (Callable, optional): PyTorch transformations. Defaults to None.
        stardist (bool, optional): Return StarDist labels. Defaults to False
        regression (bool, optional): Return Regression of cells in x and y direction. Defaults to False
        cache_dataset: If the dataset should be loaded to host memory in first epoch.
            Be careful, workers in DataLoader needs to be persistent to have speedup.
            Recommended to false, just use if you have enough RAM and your I/O operations might be limited.
            Defaults to False.
    """

    def __init__(
        self,
        dataset_path: Union[Path, str],
        split: int,
        transforms: Callable = None,
        stardist: bool = False,
        regression: bool = False,
        cache_dataset: bool = False,
    ) -> None:


        self.data_path = dataset_path
        self.transforms = transforms
        self.images = []
        self.masks = []
        self.img_names = []
        self.cache_dataset = cache_dataset
        self.stardist = stardist
        self.regression = regression
        raw_labels = []
        self.split = split
        
        if self.split == 'train':  # insert concat train loader
            loader = get_monuseg_loader(path=os.path.join(self.data_path, 'monuseg'), split=self.split, patch_shape=(512, 512), download=True, batch_size=1) #replace this with generalist_loader once running!, put raw transforms and consecutive label trafo
        else:  # insert concat val loader 
            loader = get_monuseg_loader(path=os.path.join(self.data_path, 'monuseg'), split=self.split, patch_shape=(512, 512), download=True, batch_size=1) #replace this with generalist_loader once running!, put raw transforms and consecutive label trafo
        count = 1
        os.makedirs(os.path.join(dataset_path, self.split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, self.split, 'labels'), exist_ok=True)
        for image, label in loader:
            img_array = image.numpy()
            label_array = label.numpy()   
            img_array = img_array.squeeze()
            label_array = label_array.squeeze()             
            img_uint8 = img_array.astype(np.uint8)
            uint_label = label_array.astype(np.uint16)
            if not img_uint8.shape == (512, 512, 3):
                img_uint8 = img_uint8.transpose(1, 2, 0)
            assert img_uint8.shape == (512, 512, 3), f'Shape error: unexpected shape of {img_uint8.shape}'
            imageio.imwrite(os.path.join(dataset_path, self.split, 'images', f'{count:04}.png'), img_uint8)
            np.save(os.path.join(dataset_path, self.split, 'labels', f'{count:04}.npy'), uint_label)
            self.images.append(os.path.join(dataset_path, self.split, 'images', f'{count:04}.png'))
            raw_labels.append(os.path.join(dataset_path, self.split, 'labels', f'{count:04}.npy'))  
            self.img_names.append(f'{count:04}.png')
            count += 1

        for raw_label in raw_labels:
            label = np.load(raw_label, allow_pickle=True)
            outname = f"{os.path.splitext(os.path.basename(raw_label))[0]}.npy"
            # need to create instance map and type map with shape 256x256
            inst_map = np.zeros((512, 512))
            num_nuc = 0
            for j in range(5):
                # copy value from new array if value is not equal 0
                layer_res = remap_label(label[:, :])
                # inst_map = np.where(mask[:,:,j] != 0, mask[:,:,j], inst_map)
                inst_map = np.where(layer_res != 0, layer_res + num_nuc, inst_map)
                num_nuc = num_nuc + np.max(layer_res)
            inst_map = remap_label(inst_map)
            label_outpath = os.path.join(os.path.dirname(raw_label), outname)
            outdict = {"inst_map": inst_map}
            np.save(label_outpath, outdict)
            self.masks.append(label_outpath)
        assert len(self.masks) == len(self.images), 'Label / Image mismatch'


        logger.info(f"Resulting dataset length: {self.__len__()}")

        if self.cache_dataset:
            self.cached_idx = []  # list of idx that should be cached
            self.cached_imgs = {}  # keys: idx, values: numpy array of imgs
            self.cached_masks = {}  # keys: idx, values: numpy array of masks
            logger.info("Using cached dataset. Cache is built up during first epoch.")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, dict, str, str]:
        """Get one dataset item consisting of transformed image,
        masks (instance_map, nuclei_type_map, nuclei_binary_map, hv_map) and tissue type as string

        Args:
            index (int): Index of element to retrieve

        Returns:
            Tuple[torch.Tensor, dict, str, str]:
                torch.Tensor: Image, with shape (3, H, W), in this case (3, 256, 256)
                dict:
                    "instance_map": Instance-Map, each instance is has one integer starting by 1 (zero is background), Shape (256, 256)
                    "nuclei_type_map": Nuclei-Type-Map, for each nucleus (instance) the class is indicated by an integer. Shape (256, 256)
                    "nuclei_binary_map": Binary Nuclei-Mask, Shape (256, 256)
                    "hv_map": Horizontal and vertical instance map.
                        Shape: (2 , H, W). First dimension is horizontal (horizontal gradient (-1 to 1)),
                        last is vertical (vertical gradient (-1 to 1)) Shape (2, 256, 256)
                    [Optional if stardist]
                    "dist_map": Probability distance map. Shape (256, 256)
                    "stardist_map": Stardist vector map. Shape (n_rays, 256, 256)
                    [Optional if regression]
                    "regression_map": Regression map. Shape (2, 256, 256). First is vertical, second horizontal.
                str: Tissue type
                str: Image Name
        """
        img_path = self.images[index]

        if self.cache_dataset:
            if index in self.cached_idx:
                img = self.cached_imgs[index]
                mask = self.cached_masks[index]
            else:
                # cache file
                img = self.load_imgfile(index)
                mask = self.load_maskfile(index)
                self.cached_imgs[index] = img
                self.cached_masks[index] = mask
                self.cached_idx.append(index)

        else:
            img = self.load_imgfile(index)
            mask = self.load_maskfile(index)

        if self.transforms is not None:
            transformed = self.transforms(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        inst_map = mask.copy()
        np_map = mask.copy()
        np_map[np_map > 0] = 1
        hv_map = PanNukeDataset.gen_instance_hv_map(inst_map)

        # torch convert
        img = torch.Tensor(img).type(torch.float32)
        img = img.permute(2, 0, 1)
        if torch.max(img) >= 5:
            img = img / 255

        masks = {
            "instance_map": torch.Tensor(inst_map).type(torch.int64),
            "nuclei_binary_map": torch.Tensor(np_map).type(torch.int64),
            "hv_map": torch.Tensor(hv_map).type(torch.float32),
        }

        # load stardist transforms if neccessary
        if self.stardist:
            dist_map = PanNukeDataset.gen_distance_prob_maps(inst_map)
            stardist_map = PanNukeDataset.gen_stardist_maps(inst_map)
            masks["dist_map"] = torch.Tensor(dist_map).type(torch.float32)
            masks["stardist_map"] = torch.Tensor(stardist_map).type(torch.float32)
        if self.regression:
            masks["regression_map"] = PanNukeDataset.gen_regression_map(inst_map)

        return img, masks, Path(img_path).name

    def __len__(self) -> int:
        """Length of Dataset

        Returns:
            int: Length of Dataset
        """
        return len(self.images)

    def set_transforms(self, transforms: Callable) -> None:
        """Set the transformations, can be used tp exchange transformations

        Args:
            transforms (Callable): PyTorch transformations
        """
        self.transforms = transforms

    def load_imgfile(self, index: int) -> np.ndarray:
        """Load image from file (disk)

        Args:
            index (int): Index of file

        Returns:
            np.ndarray: Image as array with shape (H, W, 3)
        """
        img_path = self.images[index]
        return np.array(Image.open(img_path)).astype(np.uint8)

    def load_maskfile(self, index: int) -> np.ndarray:
        """Load mask from file (disk)

        Args:
            index (int): Index of file

        Returns:
            np.ndarray: Mask as array with shape (H, W, 2)
        """
        mask_path = self.masks[index]
        mask = np.load(mask_path, allow_pickle=True)
        inst_map = mask[()]["inst_map"].astype(np.int32)
        mask = inst_map
        return mask

    

    @staticmethod
    def gen_instance_hv_map(inst_map: np.ndarray) -> np.ndarray:
        """Obtain the horizontal and vertical distance maps for each
        nuclear instance.

        Args:
            inst_map (np.ndarray): Instance map with each instance labelled as a unique integer
                Shape: (H, W)
        Returns:
            np.ndarray: Horizontal and vertical instance map.
                Shape: (2, H, W). First dimension is horizontal (horizontal gradient (-1 to 1)),
                last is vertical (vertical gradient (-1 to 1))
        """
        orig_inst_map = inst_map.copy()  # instance ID map

        x_map = np.zeros(orig_inst_map.shape[:2], dtype=np.float32)
        y_map = np.zeros(orig_inst_map.shape[:2], dtype=np.float32)

        inst_list = list(np.unique(orig_inst_map))
        inst_list.remove(0)  # 0 is background
        for inst_id in inst_list:
            inst_map = np.array(orig_inst_map == inst_id, np.uint8)
            inst_box = get_bounding_box(inst_map)

            # expand the box by 2px
            # Because we first pad the ann at line 207, the bboxes
            # will remain valid after expansion
            if inst_box[0] >= 2:
                inst_box[0] -= 2
            if inst_box[2] >= 2:
                inst_box[2] -= 2
            if inst_box[1] <= orig_inst_map.shape[0] - 2:
                inst_box[1] += 2
            if inst_box[3] <= orig_inst_map.shape[0] - 2:
                inst_box[3] += 2

            # improvement
            inst_map = inst_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]

            if inst_map.shape[0] < 2 or inst_map.shape[1] < 2:
                continue

            # instance center of mass, rounded to nearest pixel
            inst_com = list(center_of_mass(inst_map))

            inst_com[0] = int(inst_com[0] + 0.5)
            inst_com[1] = int(inst_com[1] + 0.5)

            inst_x_range = np.arange(1, inst_map.shape[1] + 1)
            inst_y_range = np.arange(1, inst_map.shape[0] + 1)
            # shifting center of pixels grid to instance center of mass
            inst_x_range -= inst_com[1]
            inst_y_range -= inst_com[0]

            inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

            # remove coord outside of instance
            inst_x[inst_map == 0] = 0
            inst_y[inst_map == 0] = 0
            inst_x = inst_x.astype("float32")
            inst_y = inst_y.astype("float32")

            # normalize min into -1 scale
            if np.min(inst_x) < 0:
                inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])
            if np.min(inst_y) < 0:
                inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0])
            # normalize max into +1 scale
            if np.max(inst_x) > 0:
                inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
            if np.max(inst_y) > 0:
                inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])

            ####
            x_map_box = x_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
            x_map_box[inst_map > 0] = inst_x[inst_map > 0]

            y_map_box = y_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
            y_map_box[inst_map > 0] = inst_y[inst_map > 0]

        hv_map = np.stack([x_map, y_map])
        return hv_map

    @staticmethod
    def gen_distance_prob_maps(inst_map: np.ndarray) -> np.ndarray:
        """Generate distance probability maps

        Args:
            inst_map (np.ndarray): Instance-Map, each instance is has one integer starting by 1 (zero is background), Shape (H, W)

        Returns:
            np.ndarray: Distance probability map, shape (H, W)
        """
        inst_map = fix_duplicates(inst_map)
        dist = np.zeros_like(inst_map, dtype=np.float64)
        inst_list = list(np.unique(inst_map))
        if 0 in inst_list:
            inst_list.remove(0)

        for inst_id in inst_list:
            inst = np.array(inst_map == inst_id, np.uint8)

            y1, y2, x1, x2 = get_bounding_box(inst)
            y1 = y1 - 2 if y1 - 2 >= 0 else y1
            x1 = x1 - 2 if x1 - 2 >= 0 else x1
            x2 = x2 + 2 if x2 + 2 <= inst_map.shape[1] - 1 else x2
            y2 = y2 + 2 if y2 + 2 <= inst_map.shape[0] - 1 else y2

            inst = inst[y1:y2, x1:x2]

            if inst.shape[0] < 2 or inst.shape[1] < 2:
                continue

            # chessboard distance map generation
            # normalize distance to 0-1
            inst_dist = distance_transform_edt(inst)
            inst_dist = inst_dist.astype("float64")

            max_value = np.amax(inst_dist)
            if max_value <= 0:
                continue
            inst_dist = inst_dist / (np.max(inst_dist) + 1e-10)

            dist_map_box = dist[y1:y2, x1:x2]
            dist_map_box[inst > 0] = inst_dist[inst > 0]

        return dist

    @staticmethod
    @njit
    def gen_stardist_maps(inst_map: np.ndarray) -> np.ndarray:
        """Generate StarDist map with 32 nrays

        Args:
            inst_map (np.ndarray): Instance-Map, each instance is has one integer starting by 1 (zero is background), Shape (H, W)

        Returns:
            np.ndarray: Stardist vector map, shape (n_rays, H, W)
        """
        n_rays = 32
        # inst_map = fix_duplicates(inst_map)
        dist = np.empty(inst_map.shape + (n_rays,), np.float32)

        st_rays = np.float32((2 * np.pi) / n_rays)
        for i in range(inst_map.shape[0]):
            for j in range(inst_map.shape[1]):
                value = inst_map[i, j]
                if value == 0:
                    dist[i, j] = 0
                else:
                    for k in range(n_rays):
                        phi = np.float32(k * st_rays)
                        dy = np.cos(phi)
                        dx = np.sin(phi)
                        x, y = np.float32(0), np.float32(0)
                        while True:
                            x += dx
                            y += dy
                            ii = int(round(i + x))
                            jj = int(round(j + y))
                            if (
                                ii < 0
                                or ii >= inst_map.shape[0]
                                or jj < 0
                                or jj >= inst_map.shape[1]
                                or value != inst_map[ii, jj]
                            ):
                                # small correction as we overshoot the boundary
                                t_corr = 1 - 0.5 / max(np.abs(dx), np.abs(dy))
                                x -= t_corr * dx
                                y -= t_corr * dy
                                dst = np.sqrt(x**2 + y**2)
                                dist[i, j, k] = dst
                                break

        return dist.transpose(2, 0, 1)

    @staticmethod
    def gen_regression_map(inst_map: np.ndarray):
        n_directions = 2
        dist = np.zeros(inst_map.shape + (n_directions,), np.float32).transpose(2, 0, 1)
        inst_map = fix_duplicates(inst_map)
        inst_list = list(np.unique(inst_map))
        if 0 in inst_list:
            inst_list.remove(0)
        for inst_id in inst_list:
            inst = np.array(inst_map == inst_id, np.uint8)
            y1, y2, x1, x2 = get_bounding_box(inst)
            y1 = y1 - 2 if y1 - 2 >= 0 else y1
            x1 = x1 - 2 if x1 - 2 >= 0 else x1
            x2 = x2 + 2 if x2 + 2 <= inst_map.shape[1] - 1 else x2
            y2 = y2 + 2 if y2 + 2 <= inst_map.shape[0] - 1 else y2

            inst = inst[y1:y2, x1:x2]
            y_mass, x_mass = center_of_mass(inst)
            x_map = np.repeat(np.arange(1, x2 - x1 + 1)[None, :], y2 - y1, axis=0)
            y_map = np.repeat(np.arange(1, y2 - y1 + 1)[:, None], x2 - x1, axis=1)
            # we use a transposed coordinate system to align to HV-map, correct would be -1*x_dist_map and -1*y_dist_map
            x_dist_map = (x_map - x_mass) * np.clip(inst, 0, 1)
            y_dist_map = (y_map - y_mass) * np.clip(inst, 0, 1)
            dist[0, y1:y2, x1:x2] = x_dist_map
            dist[1, y1:y2, x1:x2] = y_dist_map

        return dist
    


dataset = PanNukeDataset(dataset_path='/mnt/lustre-grete/usr/u12649/scratch/data/cellvit', split='test')