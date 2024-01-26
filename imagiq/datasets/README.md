# Datasets Module
`imagiq.datasets` module provides convenient data loaders for rapidly prototyping your code. This module is comprised of a number of classes that inherits `monai.data.CacheDataset` class (see [here](https://github.com/Project-MONAI/MONAI/blob/master/monai/data/dataset.py)), which again inherits `torch.utils.data.Dataset` class (see [here](https://pytorch.org/docs/stable/data.html)). Hence, if you are already familiar with either the MONAI dataset class or the PyTorch dataset class, the classes below should pretty much be straightforward.

Note that each of the below classes download a raw dataset from a public repository and caches it under a temporary folder (`.imagiq/datasets` under the user home directory). 

## NIH Chest X-Ray Dataset
`imagiq.datasets.NIHDataset` provides an easy access to the chest x-ray images from the National Institutes of Health (NIH), or so-called [NIH Chest X-ray Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC).

### Usage
```python
    ds = NIHDataset(section="test", download=3)
    print(ds[0])        # prints the zero-th entry
    print(ds[0].image)  # prints the image
    print(ds[0].label)  # prints the label
```
Here, `section` can be either `"training"`, `"validation"`, or `"test"`, depending on which a different subset of the original dataset will be returned. The `download` flag can be an integer or a list of integers specifying which files to be downloaded. Note that the NIH CXR dataset is comes with 12 archive files (*.tar.gz), each has the file size around 3-4 GB. In case you want to save some space on your hard drive, `NIHDataset` allows you to download only a few selected archive files that are specified in the download flag. ImagiQ keeps track of the downloads and thus the same archive files won't be downloaded again.

`NIHDataset` allows users to have a constant split of `training`, `validation`, and `test` by providing a `seed` parameter, with a default value of 0. To test with different sets of `training`, `validation`, and `test` data set, it can be simply done by passing a specific seed value, like `NIHDataset(section='training', download=0, seed=123)`.

The `NIHDataset` allows caching deterministic transforms' result so that it can reduce computation time at the training process. To be most effective, make sure to include as many deterministic transforms before random transforms. There are two parameters to control caching: `cache_num` and `cache_rate`. The `cache_num` specifies the number of samples to be cached and `cache_rate` specifies the percentages of the total samples to be cached. By default, `cache_num` is set to the `sys.maxsize` and `cache_rate` is set to `1.0`. The `NIHDataset` will cache the number of samples determined from the minimum of `cache_num`, `cache_rate * total samples`, and `total samples`. For example, it takes 74s, 122s, and 172s per epoch of training on average by caching 100%, 50%, and 0% of 4,480 samples under 1080 Ti for the DenseNet121 model. 

The simple demo of training DNN on NIH Chest X-ray dataset using `NIHDataset` module is provided [here](../../demos/NIHDataset.ipynb)