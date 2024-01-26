from imagiq.datasets import CBISDDSMDataset
from monai.transforms import Compose, LoadImaged, ScaleIntensityd
import pytest


@pytest.mark.skip("takes too long to download.")
def test_create():
    transforms = Compose([LoadImaged("image"), ScaleIntensityd("image")])
    ds = CBISDDSMDataset(section="test", transforms=transforms, download=[0])
    print(ds[0])
    print(ds[1])
