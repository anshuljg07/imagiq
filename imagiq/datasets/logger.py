import os


class DownloadLog:
    """Logger for the download status. Some datasets are large in size and may
    contain multiple subfiles (e.g. NIH Chest X-ray Dataset). This logger
    creates a log file (`download.log`) under the dataset folder such that
    the download status (which file has/has not been downloaded) can be traced.

    Args:
        dataset_name: name of the dataset.
        dataset_dir: path to the directory where the dataset is cached.
    """

    def __init__(self, dataset_name: str, dataset_dir: str) -> None:
        self.dataset_name = dataset_name
        self.log_path = os.path.join(dataset_dir, "download.log")
        self.indices = []  # indices for successfully downloaded files.
        self.load()

    def __repr__(self):
        return "Download log for: " + self.dataset_name

    def __str__(self):
        return self.__repr__(self)

    def insert(self, indices):
        """Inserts an index or indices to the log."""
        if type(indices) is list:
            self.indices += indices
        else:
            self.indices.append(indices)
        self.indices = list(set(self.indices))  # unique
        self.indices.sort()
        self.save()

    def exists(self, index):
        """Checks if an index exists in the log."""
        return index in self.indices

    def load(self):
        """Loads the log file from `self.log_path`"""
        if os.path.exists(self.log_path):
            with open(self.log_path, "r") as file:
                lines = file.readlines()
                # strip tailing line breaks
                lines = [x.rstrip() for x in lines]
                # add to indices
                self.indices = list(set([int(x) for x in lines if x != ""]))
        else:  # if the path doesn't exist, just return an empty array.
            self.indices = []

    def save(self):
        """Saves the log file to `self.log_path`"""
        with open(self.log_path, "w") as file:
            file.writelines([str(x) + "\n" for x in self.indices])
