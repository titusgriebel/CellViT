import os
from glob import glob
from natsort import natsorted
from torch_em.data.datasets import util
import torch_em

def custom_transform(x, y):
    return x, y


def get_loader(path, patch_shape, batch_size, **kwargs):
    image_paths = natsorted(glob(os.path.join(path, "eval_split", "test_images", "*.tiff")))
    label_paths = natsorted(glob(os.path.join(path, "eval_split", "test_labels", "*.tiff")))
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        transform=custom_transform,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)