from imagiq.models import Model
import torch
from torch.utils.data import Dataset
from monai.networks.nets import densenet121
import numpy as np


class DummyDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return {"image": self.inputs[index], "label": self.labels[index]}


def test_create_model():
    model = Model()
    print(model)


def test_create_model_from_pretrained():
    print("")
    model = Model(densenet121(spatial_dims=2, in_channels=1, out_channels=10))
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.net.parameters(), 1e-5)

    # create a dummy dataset
    N = 10
    inputs = torch.from_numpy(np.zeros((N, 1, 224, 224))).float()
    labels = torch.from_numpy(np.zeros((N))).long()
    dataset = DummyDataset(inputs, labels)

    # train with the dummy data
    model.train(dataset, loss, optimizer, epochs=2, device="cpu")
