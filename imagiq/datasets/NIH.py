import os
import sys
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from monai.data import (
    CacheDataset,
)
from monai.apps.utils import download_and_extract, download_url
from monai.transforms import Randomizable
from imagiq.utils.file_systems import mkdir, remove
from imagiq.common import CACHE_DIR
from imagiq.datasets.logger import DownloadLog
import pandas as pd
from operator import itemgetter


class NIHDataset(Randomizable, CacheDataset):
    """
    Dataset to automatically download the NIH Chest X-ray dataset and generate
    items for training, validation, or test. It uses monai.data.CacheDataset
    for efficient use of data during training.

    `monai.apps.datasets.MedNISTDataset` was used as a template.

    Args:
        section: expected data section, can be: `training`, `validation` or
                 `test`.
        transform: transforms to execute operations on input data.
                   the default transform is `LoadPNGd`, which can load data
                   into numpy array with [H, W] shape. for further usage, use
                   `AddChanneld` to convert the shape to [C, H, W, D].
        download: whether to download and extract the MedNIST from resource
                  link, default is False. if expected file already exists,
                  skip downloading even set it to True. user can manually copy
                  `MedNIST.tar.gz` file or `MedNIST` folder to root directory.
        seed: random seed to randomly split training, validation and test
              datasets, default is 0.
        val_frac: percentage of of validation fraction in the whole dataset,
                  default is 0.1.
        test_frac: percentage of of test fraction in the whole dataset,
                   default is 0.1.
        cache_num: number of items to be cached. Default is `sys.maxsize`.
                   will take the minimum of
                   (cache_num, data_length x cache_rate, data_length).
        cache_rate: percentage of cached data in total, default is 1.0 (cache
                    all). will take the minimum of
                    (cache_num, data_length x cache_rate, data_length).
        num_workers: the number of worker threads to use.
            if 0 a single thread will be used. Default is 0.
    Raises:
        ValueError: When ``root_dir`` is not a directory.
        RuntimeError: When ``dataset_dir`` doesn't exist and downloading is not
                      selected (``download=False``).
    """

    dataset_folder_name = "NIHCXR"
    _nih_base_url = "https://nihcc.box.com/shared/static/"
    resources = [
        _nih_base_url + "vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz",
        _nih_base_url + "i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz",
        _nih_base_url + "f1t00wrtdk94satdfb9olcolqx20z2jp.gz",
        _nih_base_url + "0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz",
        _nih_base_url + "v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz",
        _nih_base_url + "asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz",
        _nih_base_url + "jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz",
        _nih_base_url + "tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz",
        _nih_base_url + "upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz",
        _nih_base_url + "l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz",
        _nih_base_url + "hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz",
        _nih_base_url + "ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz",
    ]
    compressed_file_name = [
        "images_{:03d}.tar.gz".format(i + 1) for i in range(len(resources))
    ]
    class_names = [
        "No Finding",
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pneumonia",
        "Pneumothorax",
        "Consolidation",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Pleural_Thickening",
        "Hernia",
    ]

    def __init__(
        self,
        section: str,
        transforms: Union[Sequence[Callable], Callable] = (),
        download: Union[int, List[int]] = [],
        seed: int = 0,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        cache_num: int = sys.maxsize,
        cache_rate: float = 1.0,
        num_workers: int = 0,
    ) -> None:

        # Setup root directory
        root_dir = str(CACHE_DIR / "datasets")
        dataset_dir = os.path.join(root_dir, self.dataset_folder_name)
        if not os.path.exists(dataset_dir):
            mkdir(dataset_dir)

        # Train-validation-test split
        self.section = section
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.set_random_state(seed=seed)

        # Download labels
        if not os.path.exists(os.path.join(dataset_dir, "Data_Entry_2017_v2020.csv")):
            download_url(
                "https://www.dropbox.com/s/f3xy66rfqayw197/"
                + "Data_Entry_2017_v2020.csv?dl=1",
                os.path.join(dataset_dir, "Data_Entry_2017_v2020.csv"),
            )

        # Download images and extract
        if not download:  # if the download list is empty, download everything.
            download = [i for i in range(len(self.resources))]

        self.logger = DownloadLog("NIHCXR", dataset_dir)
        for i in download:
            if not (0 <= i < 12):
                raise ValueError("Download index must be between 0 and 11.")
            if self.logger.exists(i):
                continue
            print(
                "Downloading {}. This may take several minutes.".format(
                    self.compressed_file_name[i]
                )
            )
            tarfile_name = os.path.join(dataset_dir, self.compressed_file_name[i])
            download_and_extract(self.resources[i], tarfile_name, dataset_dir)
            self.logger.insert(i)
            self.logger.save()
            remove(tarfile_name)

        if not os.path.exists(dataset_dir):
            raise RuntimeError(
                f"Cannot find dataset directory: {dataset_dir}, "
                + "please use download=True to download it."
            )

        self.class_count = None
        data = self._generate_data_list(dataset_dir)
        super().__init__(
            data,
            transforms,
            cache_num=cache_num,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )

    def randomize(self, data: Optional[Any] = None) -> None:
        self.rann = self.R.random()

    def getPositiveWeights(self):
        positiveWeights = [None] * len(self.class_count)
        for idx in range(len(self.class_count)):
            positiveWeights[idx] = (
                len(self.data) - self.class_count[idx]
            ) / self.class_count[idx]
        return positiveWeights

    def _generate_data_list(self, dataset_dir: str) -> List[Dict]:
        """
        Raises:
            ValueError: When ``section`` is not one of
                        ["training", "validation", "test"].
        """
        # Parse the label csv file first
        df = pd.read_csv(
            os.path.join(dataset_dir, "Data_Entry_2017_v2020.csv"),
        )
        image_to_label = pd.Series(
            df["Finding Labels"].values, index=df["Image Index"]
        ).to_dict()

        # Image files
        image_files_list = [
            os.path.join(dataset_dir, "images", x)
            for x in os.listdir(os.path.join(dataset_dir, "images"))
        ]
        image_files_list.sort()
        num_total = len(image_files_list)

        # Labels
        num_classes = len(self.class_names)
        class_to_index = {
            class_name: i for i, class_name in enumerate(self.class_names)
        }

        label_strings = [image_to_label[x.split("/")[-1]] for x in image_files_list]
        labels = []
        for label_string in label_strings:
            findings = label_string.split("|")
            label = [0] * num_classes
            for finding in findings:
                label[class_to_index[finding]] = 1
            labels.append(label)

        # TODO: Randomize based on Patient ID
        data = list()
        self.class_count = []
        for i in range(num_total):
            self.randomize()
            if self.section == "training":
                if self.rann < self.val_frac + self.test_frac:
                    continue
            elif self.section == "validation":
                if self.rann >= self.val_frac:
                    continue
            elif self.section == "test":
                if (
                    self.rann < self.val_frac
                    or self.rann >= self.val_frac + self.test_frac
                ):
                    continue
            else:
                raise ValueError(
                    f"Unsupported section: {self.section}, "
                    + 'available options are ["training", "validation", "test"].'
                )
            data.append({"image": image_files_list[i], "label": labels[i]})
            self.class_count.append(labels[i])
        self.class_count = np.sum(np.array(self.class_count), axis=0)
        return data

    def __repr__(self):
        return "NIH Chest X-ray Dataset (" + self.section + ")"

    def __str__(self):
        N = len(self)
        retVal = "\n" + "=" * 40 + "\n"
        retVal += self.__repr__() + "\n"
        retVal += f"N = {N}:\n"
        retVal += "-" * 40 + "\n"
        for i, class_name in enumerate(self.class_names):
            retVal += f"  {i}. {class_name}: {self.class_count[i]} ({self.class_count[i]/N:.2f}%)\n"
        retVal += "=" * 40
        return retVal

    # convert one-hot encoded label into raw label
    def to_raw_label(self, one_hot_labels, sep=" | "):
        if len(one_hot_labels.shape) == 1:
            raise ValueError("Expects one-hot encoded label, but got:", one_hot_labels)
        elif len(one_hot_labels.shape) != 2:
            raise ValueError("Expects a batch, but got:", one_hot_labels)

        raw_labels = [None] * one_hot_labels.shape[0]
        for idx, one_hot_label in enumerate(one_hot_labels):
            one_hot_label = list(one_hot_label.cpu().numpy())
            raw_label = itemgetter(*list(np.nonzero(one_hot_label)[0]))(
                self.class_names
            )
            raw_labels[idx] = (
                raw_label if isinstance(raw_label, str) else sep.join(list(raw_label))
            )
        return raw_labels
