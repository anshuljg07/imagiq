import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, Tuple
from monai.data import CacheDataset
from monai.data.image_reader import ITKReader
from monai.apps.utils import download_and_extract, download_url
from monai.transforms import Randomizable
from monai.transforms.compose import Transform, MapTransform
from monai.config import KeysCollection
from imagiq.utils.file_systems import mkdir, remove, move, path_split
from imagiq.common import CACHE_DIR
from imagiq.datasets.logger import DownloadLog
import pandas as pd
import glob
import sys
import numpy as np
import pydicom
from medpy.io.load import load
from copy import copy
import cv2
import time
from monai.data import Dataset as _MonaiDataset
from monai.data import DataLoader
import random


class CBISDDSMDataset(Randomizable, CacheDataset):
    dataset_folder_name = "CBISDDSM"
    _cbisddsm_base_url = "https://www.dropbox.com/sh/luzkvwfzddz2i69/"
    resources = [
        _cbisddsm_base_url + "AADCo9-oj8J5HSXMCWc0Aif5a/subFolder0.zip?dl=1",
        _cbisddsm_base_url + "AAA7LY-G4V-2ws1ubtmTi0jAa/subFolder1.zip?dl=1",
        _cbisddsm_base_url + "AACuzXvD_krnEzuSwabYwFNxa/subFolder2.zip?dl=1",
        _cbisddsm_base_url + "AAAbFCQIWcyyRSA-y64cOEvwa/subFolder3.zip?dl=1",
        _cbisddsm_base_url + "AAA0mgk8uq4xLh1MjBkYsiOwa/subFolder4.zip?dl=1",
        _cbisddsm_base_url + "AAAQEsrd3mQALhJiYVS119Upa/subFolder5.zip?dl=1",
        _cbisddsm_base_url + "AAAQGJ7F2yfofmU4mQWZjX8ca/subFolder6.zip?dl=1",
        _cbisddsm_base_url + "AABMTZvlm_7kEsvsOA6ph0kHa/subFolder7.zip?dl=1",
        _cbisddsm_base_url + "AAByBmJeLYmIJp9nFoj5XzrIa/subFolder8.zip?dl=1",
        _cbisddsm_base_url + "AACdFFmY0utDqUSmcU66epC0a/subFolder9.zip?dl=1",
    ]
    compressed_file_name = [
        "images_{:03d}.zip".format(i + 1) for i in range(len(resources))
    ]
    class_names = [1, 2, 3, 4]

    def __init__(
        self,
        section: str,
        transforms: Union[Sequence[Callable], callable] = (),
        download: Union[int, List[int]] = [],
        seed: int = 0,
        val_frac: float = 0.1,
        cache_num: int = sys.maxsize,
        cache_rate: float = 1.0,
        num_workers: int = 0,
    ):
        # setup root directory
        root_dir = str(CACHE_DIR / "datasets")
        dataset_dir = os.path.join(root_dir, self.dataset_folder_name)
        if not os.path.exists(dataset_dir):
            mkdir(dataset_dir)

        # Train-validation-test aplit
        self.section = section
        self.val_frac = val_frac
        self.set_random_state(seed=seed)

        # Download labels
        if not os.path.exists(os.path.join(dataset_dir, "cbisddsm.csv")):
            download_url(
                "https://www.dropbox.com/s/3zt5j3b1yl61dk3/" + "combined.csv?dl=1",
                os.path.join(dataset_dir, "cbisddsm.csv"),
            )

        # Download images and extract
        if not download:
            download = [i for i in range(len(self.resources))]

        self.logger = DownloadLog("CBISDDSM", dataset_dir)
        to_cleanse = False
        for i in download:
            if not (0 <= i < 10):
                raise ValueError("Download index must be between 0 and 9")
            if self.logger.exists(i):
                continue
            print(
                "Downloading {}. This may take several minutes.".format(
                    self.compressed_file_name[i]
                )
            )
            to_cleanse = True
            tarfile_name = os.path.join(dataset_dir, self.compressed_file_name[i])
            download_and_extract(self.resources[i], tarfile_name, dataset_dir)

            mkdir(os.path.join(dataset_dir, "cases"))
            cases = glob.glob(os.path.join(dataset_dir, "subFolder" + str(i), "*"))
            for case in cases:
                move(case, os.path.join(dataset_dir, "cases", path_split(case)[-1]))
            remove(os.path.join(dataset_dir, "subFolder" + str(i)))

            self.logger.insert(i)
            self.logger.save()
            remove(tarfile_name)

        if not os.path.exists(dataset_dir):
            raise RuntimeError(
                f"Cannot find dataset directory: {dataset_dir}, "
                + "please use download=True to download it"
            )

        self.class_count = None
        if to_cleanse:
            start_time = time.time()
            self.cleanse(dataset_dir)
            print("Cleanse:", time.time() - start_time, "s")
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

    def cleanse(self, dataset_dir: str) -> None:
        image_file_list = [
            image
            for case in os.listdir(os.path.join(dataset_dir, "cases"))
            for image in os.listdir(os.path.join(dataset_dir, "cases", case))
        ]

        image_file_path = [
            os.path.join(dataset_dir, "cases", "_".join(image.split("_")[:3]), image)
            for image in image_file_list
        ]

        reader = ITKReader()
        for data in image_file_path:
            img = reader.read(data)
            img_arr, metadata = reader.get_data(img)
            ds = pydicom.dcmread(data)
            update_required = False

            try:
                metadata["0008|103e"]
            except KeyError:
                print("Update [0008|103e]:", data.split("/")[-1])
                ds.add_new([0x0008, 0x103E], "LO", "no info given")
                update_required = True

            try:
                metadata["0020|0060"]
            except KeyError:
                print("Update [0020|0060]:", data.split("/")[-1])
                laterality = data.split("/")[-1].split("_")[3]

                if laterality == "RIGHT":
                    laterality = "R"
                elif laterality == "LEFT":
                    laterality = "L"
                else:
                    raise ValueError("Cleanse failure. Can not infer laterality")

                ds.add_new([0x0020, 0x0060], "CS", laterality)
                update_required = True

            if update_required:
                ds.save_as(data)

    def _generate_data_list(self, dataset_dir: str) -> List[Dict]:
        # parse label file
        df = pd.read_csv(
            os.path.join(dataset_dir, "cbisddsm.csv"),
            converters={"patient_id": lambda x: str(x)},
        )
        images_to_label = pd.Series(df["breast density"].values, index=df["patient_id"])

        # image files
        image_files_list = [
            image
            for case in os.listdir(os.path.join(dataset_dir, "cases"))
            for image in os.listdir(os.path.join(dataset_dir, "cases", case))
        ]
        image_files_list.sort()
        num_total = len(image_files_list)

        # prepare label
        num_classes = len(self.class_names)
        class_to_index = {
            class_name: i for i, class_name in enumerate(self.class_names)
        }
        label_strings = [images_to_label[x.split("_")[2]] for x in image_files_list]
        labels = []
        invalidLabels = []  # p1443 (wrong label)
        for idx, label_string in enumerate(label_strings):
            if int(label_string) not in [1, 2, 3, 4]:
                invalidLabels.append(idx)
                num_total -= 1
                continue
            label = [0] * num_classes
            label[class_to_index[label_string]] = 1
            labels.append(label)
        for idx in invalidLabels:
            image_files_list.pop(idx)

        # Respect CBISDDSM data split on train vs test.
        data = list()
        self.class_count = []
        for i in range(num_total):
            self.randomize()
            img_dir = os.path.join(
                dataset_dir,
                "cases",
                "_".join(image_files_list[i].split("_")[:3]),
                image_files_list[i],
            )
            section = image_files_list[i].split("_")[0].split("-")[1]
            section = section.lower()
            if self.section == "test":
                if section == "test":
                    data.append({"image": img_dir, "label": labels[i]})
                    self.class_count.append(labels[i])
            else:  # todo : split by patients
                if section == "test":
                    continue
                if self.section == "training":
                    if self.rann < self.val_frac:
                        continue
                elif self.section == "validation":
                    if self.rann >= self.val_frac:
                        continue
                data.append({"image": img_dir, "label": labels[i]})
                self.class_count.append(labels[i])
        self.class_count = np.sum(np.array(self.class_count), axis=0)
        return data

    def __repr__(self):
        return (
            "Curated Breast Imagig Subset (CBIS) of Digital Database for Screening Mammography (DDSM) Dataset {"
            + self.section
            + "}"
        )

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
    def to_raw_label(self, one_hot_label):
        if len(one_hot_label.shape) == 1:
            raise ValueError("Expects one-hot encoded label, but got:", one_hot_label)
        elif len(one_hot_label.shape) != 2:
            raise ValueError("Expects a batch, but got:", one_hot_label)

        label_raw = one_hot_label.argmax(dim=-1).cpu().detach().numpy()
        return [self.class_names[x] for x in label_raw]


# Rudimentary implementation of Breast Density dataset reader
# Checks dataset modality, monochrome. Process and normalize respectively.
class LoadBreastDensity(Transform):
    def __init__(
        self,
        dtype: np.dtype = np.float32,
        *args,
        **kwargs,
    ) -> None:

        self.dtype = dtype

    def __call__(
        self,
        filename: Union[Sequence[str], str],
    ):
        ds = pydicom.dcmread(filename, stop_before_pixels=True)

        if not (ds.Modality == "MG" or ds.Modality == "DR"):
            raise ValueError("Modality has to be MG or DR")

        img, _ = load(filename)
        img = np.squeeze(img).T.astype(np.float)

        # invert intensities if monochrome 1
        if "1" in ds.PhotometricInterpretation:
            img = copy(img) * -1
        else:
            img = copy(img)

        # normalize
        img = img / 65535.0

        # resize and replicate into 3 channels
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, 0)

        return img


class LoadBreastDensityd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        dtype: np.dtype = np.float32,
        meta_key_postfix: str = "meta_dict",
        overwriting: bool = False,
        image_only: bool = False,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ) -> None:

        super().__init__(keys)
        self._loader = LoadBreastDensity()

    def __call__(self, data):
        if isinstance(data, list):
            return_d = [None] * len(data)
            for idx, datum in enumerate(data):
                for key in self.keys:
                    d = dict(datum)
                    d[key] = self._loader(datum[key])
                return_d[idx] = d
        else:
            return_d = dict(data)
            for key in self.keys:
                return_d[key] = self._loader(data[key])
        return return_d


def sanity_check(ds_train, ds_val, ds_test):
    check_pass = True

    # check dataset is an extension of pytorch dataset
    print("SanityCheck: extension of monai.data.Dataset", end="\t")
    if not (
        isinstance(ds_train, _MonaiDataset)
        and isinstance(ds_val, _MonaiDataset)
        and isinstance(ds_test, _MonaiDataset)
    ):
        print("Error: Your dataset does not extend monai.data.Dataset")
        check_pass = False
    print("pass")

    # check 1) compatibility with monai's data loader and 2) data are store with 'immge' and 'label'
    print(
        "SanityCheck: compatibility with monai's data loader and key values", end="\t"
    )
    tn_loader = DataLoader(ds_train, batch_size=1)
    try:
        for idx, data in enumerate(tn_loader):
            data["image"].shape
            data["label"]
            break
    except:
        print(
            "Error: Train dataset. make sure your data has two keys: 'image', and 'label' for each observation"
        )
        check_pass = False

    val_loader = DataLoader(ds_val, batch_size=1)
    try:
        for idx, data in enumerate(val_loader):
            data["image"].shape
            data["label"]
            break
    except:
        print(
            "Error: Validation dataset. make sure your data has two keys: 'image', and 'label' for each observation"
        )
        check_pass = False

    test_loader = DataLoader(ds_test, batch_size=1)
    try:
        for idx, data in enumerate(test_loader):
            data["image"].shape
            data["label"]
            break
    except:
        print(
            "Error: Test dataset. make sure your data has two keys: 'image', and 'label' for each observation"
        )
        check_pass = False
    print("pass")

    # check if labels are prepared with one hot encoding
    print("SanityCheck: check labels are one-hot encoded and 4 labels", end="\t")
    random_idxes = random.sample(range(len(ds_train)), 10)
    for random_idx in random_idxes:
        label = ds_train[random_idx]["label"]
        if 4 != len(label):
            print(
                "Error: breast density (train) dataset is expecting to have 4 labels in one-hot encoded"
            )
            print(label)
            check_pass = False
            break

    random_idxes = random.sample(range(len(ds_val)), 10)
    for random_idx in random_idxes:
        label = ds_val[random_idx]["label"]
        if 4 != len(label):
            print(
                "Error: breast density (validation) dataset is expecting to have 4 labels in one-hot encoded"
            )
            print(label)
            check_pass = False
            break

    random_idxes = random.sample(range(len(ds_test)), 10)
    for random_idx in random_idxes:
        label = ds_test[random_idx]["label"]
        if 4 != len(label):
            print(
                "Error: breast density (test) dataset is expecting to have 4 labels in one-hot encoded"
            )
            print(label)
            check_pass = False
            break
    print("pass")

    # check overlap among data partition
    print(
        "SanityCheck: check overlap among training, validation, and test dataset",
        end="\t",
    )
    random_idxes = random.sample(range(len(ds_train)), 20)
    val_imgs = [data["image"] for data in ds_val.data]
    test_imgs = [data["image"] for data in ds_test.data]
    for random_idx in random_idxes:
        if ds_train.data[random_idx]["image"] in val_imgs:
            check_pass = False
            print(
                "Error: train image(",
                ds_train.data[random_idx]["image"],
                ") is in validation data set",
            )
            break
        if ds_train.data[random_idx]["image"] in test_imgs:
            check_pass = False
            print(
                "Error: train image(",
                ds_train.data[random_idx]["image"],
                ") is in test data set",
            )
            break

    random_idxes = random.sample(range(len(ds_val)), 20)
    for random_idx in random_idxes:
        if ds_val.data[random_idx]["image"] in test_imgs:
            check_pass = False
            print(
                "Error: train image(",
                ds_val.data[random_idx]["image"],
                ") is in test data set",
            )
            break
    print("pass")

    if check_pass:
        print("Congratulations! Your implementation passed all sanity check")
        return True
    else:
        print("Please make required changes stated")
        return False
