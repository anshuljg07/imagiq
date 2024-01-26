from imagiq.models import Model, load_model, compare_nets
import torch.nn as nn
import torch.nn.functional as F
import os
from imagiq.common import CACHE_DIR, UID_LEN
from copy import deepcopy
from pathlib import Path


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 60)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def test_init_commit():
    # create a model
    model = Model()

    # first commit
    commit_uid = model.commit("Initial commit")
    assert len(model.history) == 1
    assert model.HEAD == commit_uid
    model_dir = CACHE_DIR / "models" / model.uid

    # reference to header pointer copntainer
    with open(str(model_dir / "history" / "HEAD"), "r") as f:
        lines = f.readlines()
        assert lines[0] == "ref: refs/heads/main"

    # header pointer pointing the most recent commit
    with open(str(model_dir / "history" / "refs" / "heads" / "main"), "r") as f:
        lines = f.readlines()
        assert lines[0] == model.history[0]["uid"]

    # directory containing files for the current commit
    object_dir = model_dir / "history" / "objects" / model.history[0]["uid"]
    assert os.path.exists(str(object_dir / "info.json"))
    assert os.path.exists(str(object_dir / "net.pt"))

    # commit logs
    with open(str(model_dir / "history" / "logs" / "HEAD"), "r") as f:
        lines = f.readlines()
        line = lines[0]
        assert line[:UID_LEN] == "0" * UID_LEN
        assert line[UID_LEN + 1 : 2 * UID_LEN + 1] == model.history[0]["uid"]
        tokens = line[2 * UID_LEN + 2 :].split("<")
        assert tokens[0][:-1] == model.history[0]["author"]
        tokens = tokens[1].split(">")
        assert tokens[0] == model.history[0]["email"]
        tokens = tokens[1].split("\t")
        assert float(tokens[0]) == model.history[0]["time"]
        assert tokens[1][:-1] == model.history[0]["desc"]

    # destroy the model
    model.destroy()
    assert not os.path.exists(str(model_dir))


def test_new_commit():
    model = Model()
    model.commit("Initial commit")
    model.net = Net()
    model.commit("Second commit")

    assert len(model.history) == 2

    model_dir = CACHE_DIR / "models" / model.uid

    # header pointer pointing the most recent commit
    with open(str(model_dir / "history" / "refs" / "heads" / "main"), "r") as f:
        lines = f.readlines()
        assert lines[0] == model.history[1]["uid"]

    # directory containing files for the current commit
    object_dir = model_dir / "history" / "objects" / model.history[1]["uid"]
    assert os.path.exists(str(object_dir / "info.json"))
    assert os.path.exists(str(object_dir / "net.pt"))

    # commit logs
    with open(str(model_dir / "history" / "logs" / "HEAD"), "r") as f:
        lines = f.readlines()
        line = lines[1]
        assert line[:UID_LEN] == model.history[0]["uid"]
        assert line[UID_LEN + 1 : 2 * UID_LEN + 1] == model.history[1]["uid"]
        tokens = line[2 * UID_LEN + 2 :].split("<")
        assert tokens[0][:-1] == model.history[1]["author"]
        tokens = tokens[1].split(">")
        assert tokens[0] == model.history[1]["email"]
        tokens = tokens[1].split("\t")
        assert float(tokens[0]) == model.history[1]["time"]
        assert tokens[1][:-1] == model.history[1]["desc"]

    # destroy the model
    model.destroy()
    assert not os.path.exists(str(model_dir))


def test_load_model():
    model = Model()
    model.commit("Initial commit")
    model.net = Net()
    model.commit("Second commit")

    recovered = load_model(str(model.model_dir))
    assert model.HEAD == recovered.HEAD
    assert model.uid == recovered.uid
    assert compare_nets(model.net, recovered.net)
    assert model.minValLoss == recovered.minValLoss
    assert model.license == recovered.license
    assert model.anatomy == recovered.anatomy
    assert model.scope == recovered.scope
    assert model.modality == recovered.modality
    assert len(model.history) == len(recovered.history)
    for i in range(len(model.history)):
        assert model.history[i] == recovered.history[i]
    assert model.name == recovered.name
    assert model.model_dir == recovered.model_dir


def test_checkout():
    net1 = Net()
    model = Model(net=net1)
    uid1 = model.commit("Initial commit")

    net2 = Net()
    model.net = net2
    uid2 = model.commit("Second commit")

    # revert back to the initial commit
    model.checkout(uid1)
    assert compare_nets(model.net, net1)
    assert model.HEAD == uid1

    # revert back to the second commit
    model.checkout(uid2)
    assert compare_nets(model.net, net2)
    assert model.HEAD == uid2


def test_load_model_after_checkout():
    net1 = Net()
    model = Model(net=net1)
    uid1 = model.commit("Initial commit")

    net2 = Net()
    model.net = net2
    model.commit("Second commit")

    model.checkout(uid1)

    recovered = load_model(str(model.model_dir))
    assert model.HEAD == uid1
    assert model.uid == recovered.uid
    assert compare_nets(model.net, recovered.net)
    assert model.minValLoss == recovered.minValLoss
    assert model.license == recovered.license
    assert model.anatomy == recovered.anatomy
    assert model.scope == recovered.scope
    assert model.modality == recovered.modality
    assert len(model.history) == len(recovered.history)
    for i in range(len(model.history)):
        assert model.history[i] == recovered.history[i]
    assert model.name == recovered.name
    assert model.model_dir == recovered.model_dir


def test_commit_path():
    model = Model()
    uid1 = model.commit("Initial commit")

    model.net = Net()
    uid2 = model.commit("Second commit")

    model.net = None
    uid3 = model.commit("Third commit")

    model.checkout(uid2)
    model.net = None
    uid4 = model.commit("Another commit")

    assert model._commit_path(uid1) == [uid1]
    assert model._commit_path(uid2) == [uid2, uid1]
    assert model._commit_path(uid3) == [uid3, uid2, uid1]
    assert model._commit_path(uid4) == [uid4, uid2, uid1]


def test_merge():
    # Create a model and make an initial commit
    net1 = Net()
    model = Model(net=net1)
    model.commit("Initial commit")

    # Create a copy of the model and make changes to the copy
    net2 = Net()
    model_copy = deepcopy(model)
    model_copy.model_dir = Path(str(model_copy.model_dir) + "_copy")
    model_copy.net = net2
    model_copy.commit("Second commit")

    # it should be mergeable
    assert model.is_mergeable(model_copy, True)

    # merge the copy to the main
    model.merge(model_copy, True)
    assert model == model_copy


# def test_illegal_merge():
# TODO: make sure illegal merges (e.g. merging different models) raise exceptions


# def test_list_cached():
#     # TODO: list all the models that are cached
#     model1 = Model()
#     model2 = Model()
#     model1.commit("Committed")
#     model2.commit("Committed")

#     models = iq.models.list_all_cached()
