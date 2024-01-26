from imagiq.common import uid_generator, CACHE_DIR, USER_NAME, USER_EMAIL, UID_LEN
from imagiq.utils import file_systems as fs
import sys
from monai.data import DataLoader
from monai.metrics import compute_roc_auc
import torch
import os
import json
import time
import copy
import logging
from pathlib import Path
from torch.optim.lr_scheduler import *
import pandas as pd
import numpy as np


class Model:
    """Model class"""

    def __init__(self, net=None, name=None):
        """Initialize a model.

        Args:
            net: PyTorch neural network
            name: Name of the model
        """
        self.net = net
        self.minValLoss = None

        self.license = ""  # TODO: License conditions
        self.anatomy = ""  # TODO: anatomy this model is trained for
        self.scope = ""  # TODO: body area this model is trained for
        self.modality = ""  # TODO: Imaging modality
        self.history = []
        self.HEAD = "0" * UID_LEN

        self.name = name
        self.uid = uid_generator()
        if self.name is None:
            self.name = "model_" + self.uid[:5]

        self.model_dir = CACHE_DIR / "models" / self.uid
        self.quality = -1

        self.train_record = None
        self.validation_record = None
        self.test_record = None

    def __call__(self, inputs):
        """Infer the model with given inputs.

        Args.
            inputs: Torch tensor
        """
        # PyTorch documentation: Remember that you must call model.eval() to
        # set dropout and batch normalization layers to evaluation mode
        # before running inference. Failing to do this will yield inconsistent
        # inference results.
        self.net.eval()
        return self.net(inputs)

    def parameters(self):
        return self.net.parameters()

    def train(
        self,
        dataset,
        loss_function,
        optimizer,
        batch_size=16,
        epochs=1,
        metrics=None,
        validation_dataset=None,
        device=None,
        dirpath=None,
        scheduler=None,
        earlystop=None,
    ):
        # validate early stopping parameters
        if earlystop is not None:
            if not (
                "patience" in list(earlystop.keys())
                or "delta" in list(earlystop.keys())
            ):  # earlystop parameters not found
                raise ValueError(
                    "earlystop parameter name not found. Specify either 'patience' or 'delta'. Given"
                    + " ".join(list(earlystop.keys()))
                )
            es_patience = (
                earlystop["patience"] if "patience" in list(earlystop.keys()) else 10
            )
            es_delta = earlystop["delta"] if "delta" in list(earlystop.keys()) else 0
            es_val_loss = None
            es_patience_ct = 0

        # setup training history
        history = (
            {"loss": [None] * epochs, "val_loss": [None] * epochs}
            if validation_dataset is not None
            else {"loss": [None] * epochs}
        )
        if metrics is not None:
            for metric in metrics:
                metric = metric.lower()
                # TODO: replace with Metric module
                if metric == "auc":
                    history[metric] = [None] * epochs
                    if validation_dataset is not None:
                        history["val_" + metric] = [None] * epochs

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        print("Using device:", device)
        self.net.to(device)

        # check scheduler operates on training
        tn_schedule = False
        if scheduler is not None:
            if isinstance(
                scheduler, (CyclicLR, OneCycleLR, CosineAnnealingWarmRestarts)
            ):
                tn_schedule = True

        train_loader = DataLoader(dataset, batch_size=batch_size)
        iters = len(train_loader)
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")

            self.net.train()

            for step, batch_data in enumerate(train_loader):
                # TODO: what if data do not follow this format?
                inputs, labels = (
                    batch_data["image"].to(device),
                    batch_data["label"].to(device),
                )
                optimizer.zero_grad()
                outputs = self.net(inputs)
                try:
                    loss = loss_function(outputs, labels)
                except RuntimeError:
                    logging.exception(
                        "Please typecast labels as required by the loss function"
                    )
                    return None

                loss.backward()
                optimizer.step()

                # Compute metrics for the current epoch
                if step == 0:
                    cum_outputs = outputs.detach()
                    cum_labels = labels.detach()
                else:
                    cum_outputs = torch.cat((cum_outputs, outputs.detach()), dim=0)
                    cum_labels = torch.cat((cum_labels, labels.detach()), dim=0)

                cum_loss = loss_function(cum_outputs, cum_labels)
                disp_str = "[%-30s] %d%% - loss: %.4f" % (
                    "=" * int(30 * (step + 1) / len(train_loader) + 0.5),
                    100 * (step + 1) / len(train_loader),
                    cum_loss.item(),
                )

                if scheduler is not None and tn_schedule:
                    if isinstance(scheduler, CosineAnnealingWarmRestarts):
                        scheduler.step(epoch + step / iters)
                    else:
                        scheduler.step()

                if metrics is not None:
                    for metric in metrics:
                        metric = metric.lower()
                        if metric == "auc":
                            rawLabel = True if len(cum_labels.shape) == 1 else False
                            rawInvalid = (
                                True
                                if rawLabel and len(cum_labels.unique()) == 1
                                else False
                            )
                            if not rawLabel:
                                validLabels = [
                                    idx if len(label.unique()) == 2 else -1
                                    for idx, label in enumerate(
                                        cum_labels.transpose(1, 0)
                                    )
                                ]
                                validLabels = list(
                                    filter(lambda x: x != -1, validLabels)
                                )

                            if rawLabel:
                                if rawInvalid:
                                    metric_val = 0
                                else:
                                    metric_val = compute_roc_auc(
                                        cum_outputs,
                                        cum_labels,
                                        to_onehot_y=True,
                                        softmax=True,
                                    )
                            else:
                                if (
                                    len(validLabels) != cum_labels.shape[-1]
                                    and not rawLabel
                                ):
                                    if len(validLabels) == 0:
                                        metric_val = 0
                                    else:
                                        metric_val = compute_roc_auc(
                                            cum_outputs[:, validLabels],
                                            cum_labels[:, validLabels],
                                            to_onehot_y=False,
                                            softmax=True,
                                        )
                                else:
                                    metric_val = compute_roc_auc(
                                        cum_outputs,
                                        cum_labels,
                                        to_onehot_y=False,
                                        softmax=True,
                                    )
                            disp_str += ", auc: %.4f" % metric_val
                            history[metric][epoch] = metric_val
                        # TODO: implement accuracy, binary crossentropy, categorical crossentropy, precision, recall, true positive rate,
                        # true negative rate, false positive rate, false negative rate, precision at recall, sensitivity at specificity, specificity at sensitivity,
                        # MeanIoU, Dice, MSE, RMSE, MAE, MAPE, MSLE, cosine similarity, logcosh error
                history["loss"][epoch] = cum_loss.item()
                sys.stdout.write("\r")
                sys.stdout.write(disp_str)
                sys.stdout.flush()

            self.net.eval()
            disp_str = ""
            if validation_dataset is not None:
                val_loader = DataLoader(validation_dataset, batch_size=batch_size)
                for step, batch_data in enumerate(val_loader):
                    inputs, labels = (
                        batch_data["image"].to(device),
                        batch_data["label"].to(device),
                    )
                    outputs = self.net(inputs)

                    # Compute metrics for the current epoch
                    if step == 0:
                        cum_outputs = outputs.detach()
                        cum_labels = labels.detach()
                    else:
                        cum_outputs = torch.cat((cum_outputs, outputs.detach()), dim=0)
                        cum_labels = torch.cat((cum_labels, labels.detach()), dim=0)
                try:
                    val_loss = loss_function(cum_outputs, cum_labels)
                except RuntimeError:
                    logging.exception(
                        "Please typecast labels as required by the loss function"
                    )
                    return None

                disp_str = " - val_loss: %.4f" % (val_loss.item())

                history["val_loss"][epoch] = val_loss.item()
                if scheduler is not None and not tn_schedule:
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(history["val_loss"][epoch])
                    else:
                        scheduler.step()

                if metrics is not None:
                    for metric in metrics:
                        metric = metric.lower()
                        if metric == "auc":
                            # check label is given as raw or onehot
                            rawLabel = True if len(cum_labels.shape) == 1 else False
                            rawInvalid = (
                                True
                                if rawLabel and len(cum_labels.unique()) == 1
                                else False
                            )
                            if not rawLabel:  # remove all 0s or all 1s label
                                validLabels = [
                                    idx if len(label.unique()) == 2 else -1
                                    for idx, label in enumerate(
                                        cum_labels.transpose(1, 0)
                                    )
                                ]
                                validLabels = list(
                                    filter(lambda x: x != -1, validLabels)
                                )

                            if rawLabel:
                                if rawInvalid:
                                    metric_val = 0
                                else:
                                    metric_val = compute_roc_auc(
                                        cum_outputs,
                                        cum_labels,
                                        to_onehot_y=True,
                                        softmax=True,
                                    )
                            else:
                                if (
                                    len(validLabels) != cum_labels.shape[-1]
                                    and not rawLabel
                                ):
                                    if len(validLabels) == 0:
                                        metric_val = 0
                                    else:
                                        metric_val = compute_roc_auc(
                                            cum_outputs[:, validLabels],
                                            cum_labels[:, validLabels],
                                            to_onehot_y=False,
                                            softmax=True,
                                        )
                                else:
                                    metric_val = compute_roc_auc(
                                        cum_outputs,
                                        cum_labels,
                                        to_onehot_y=False,
                                        softmax=True,
                                    )
                            disp_str += ", val_auc: %.4f" % metric_val
                            history["val_" + metric][epoch] = metric_val
            print(disp_str)
            # checkpoint
            if validation_dataset is not None:
                if self.minValLoss is None:
                    self.minValLoss = val_loss.item()
                elif val_loss.item() < self.minValLoss:
                    if dirpath is not None:
                        print(
                            f"Epoch {epoch+1} "
                            + "validation loss improved from {:.4} to {:.4}, saving model info to {}net.pt".format(
                                self.minValLoss, val_loss, os.path.join(dirpath)
                            )
                        )
                        self.save_snapshot(dirpath)
                    self.minValLoss = val_loss.item()

            # reset performance record as network has been changed
            self.train_record = None
            self.validation_record = None
            self.test_record = None

            if earlystop is not None:
                if epoch == 0:
                    es_val_loss = history["val_loss"][epoch]
                else:
                    if history["val_loss"][epoch] < es_val_loss or (
                        abs(history["val_loss"][epoch] - es_val_loss) < es_delta
                    ):
                        es_val_loss = history["val_loss"][epoch]
                        es_patience_ct = 0
                    else:
                        es_patience_ct += 1
                if es_patience_ct > es_patience:
                    print("Epoch", epoch, "Early stop.")
                    break

        return history
        # loader = DataLoader(dataset, batch_size=batch_size)
        # for batch in loader:
        #     outputs = self.predict(batch[0])
        #     loss = self.loss_function(outputs, batch[1])
        #     loss.backward()
        #     self.optimizer.step()
        #     print(outputs)
        #     break
        # assert 0

    def predict(self, dataset, batch_size=32, device=None, section=None):
        self.net.eval()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        print("Using device:", device)
        self.net.to(device)

        loader = DataLoader(dataset, batch_size=batch_size)

        # Compute the model's Accuracy and AUC on test data to see the model performance
        for step, batch_data in enumerate(loader):
            # TODO: what if data do not follow this format?
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            outputs = self.net(inputs)

            if step == 0:
                cum_outputs = outputs.detach()
                cum_labels = labels.detach()
            else:
                cum_outputs = torch.cat((cum_outputs, outputs.detach()), dim=0)
                cum_labels = torch.cat((cum_labels, labels.detach()), dim=0)

        # Save model's performance
        fileNames = [x["image"].split("/")[-1] for x in dataset.data]
        if "to_raw_label" in dir(dataset):
            cum_labels_raw = dataset.to_raw_label(cum_labels)
            predictions_raw = dataset.to_raw_label(cum_outputs)
        else:
            cum_labels_raw = [str(x) for x in cum_labels.cpu().numpy()]
            predictions_raw = [
                str(list(np.round(x, 3))) for x in cum_outputs.cpu().numpy()
            ]

        np.set_printoptions(precision=3, suppress=True)
        dfRecord = pd.DataFrame(
            {
                "filename": fileNames,
                "prediction": [
                    str(list(np.round(x, 3))) for x in cum_outputs.cpu().numpy()
                ],
                "predictionLabel": predictions_raw,
                "label": [str(x) for x in cum_labels.cpu().numpy()],
                "rawLabel": cum_labels_raw,
            }
        )

        if section is not None:
            if section.lower() in ["tn", "train", "training"]:
                self.train_record = dfRecord.copy()
            elif section.lower() in ["val", "validation"]:
                self.validation_record = dfRecord.copy()
            elif section.lower() in ["tt", "test", "testing"]:
                self.test_record = dfRecord.copy()
            else:
                raise ValueError("section has to be 'train', 'validation', or 'test'")

        # Return values: label and predict output data is needed to build an ensemble model
        return cum_labels, cum_outputs

    def has_commit(self):
        return len(self.history) > 0

    def _append_commit_log(self, path, snapshot):
        with open(path, "a") as f:
            if len(self.history) == 0:
                log = "0" * len(snapshot["uid"])
            else:
                log = self.history[-1]["uid"]
            log += " " + snapshot["uid"] + " "
            log += snapshot["author"]
            log += " <" + snapshot["email"] + "> "
            log += str(snapshot["time"])
            log += "\t" + snapshot["desc"] + "\n"
            f.write(log)

    def compare_model_info(self, model_info):
        if self.name != model_info["name"]:
            return True
        if self.license != model_info["license"]:
            return True
        if self.anatomy != model_info["anatomy"]:
            return True
        if self.scope != model_info["scope"]:
            return True
        if self.modality != model_info["modality"]:
            return True

    def has_things_to_commit(self):
        head_id = self.find_commit(self.HEAD)
        if head_id is None:
            return True

        dirpath = str(self.model_dir / "history" / "objects" / self.HEAD)
        if not os.path.exists(dirpath):
            return True

        with open(os.path.join(dirpath, "info.json")) as file:
            model_info = json.load(file)

        if not self.compare_model_info(model_info):
            return True

        if not compare_nets(self.net, torch.load(os.path.join(dirpath, "net.pt"))):
            return True

        return False

    def commit(self, description):
        """Takes a snapshot of the model and store it in a history.

        Args:
            description: a description of the changes made.
        """
        if not self.has_things_to_commit():
            print("No changes detected. Nothing to commit.")
            return

        snapshot = {
            "prev": self.HEAD,
            "uid": uid_generator(),
            "author": USER_NAME,
            "email": USER_EMAIL,
            "time": time.time(),
            "desc": description,
        }

        # create a model directory if it doesn't exists already
        history_dir = self.model_dir / "history"
        fs.mkdir(str(history_dir / "refs" / "heads"))
        fs.mkdir(str(history_dir / "logs"))

        # reference to header pointer
        # TODO: Branches
        with open(str(history_dir / "HEAD"), "w") as f:
            f.write("ref: refs/heads/main")

        # header pointer
        with open(str(history_dir / "refs" / "heads" / "main"), "w") as f:
            f.write(snapshot["uid"])

        # the actual snapshot data
        self.save_snapshot(str(history_dir / "objects" / snapshot["uid"]))

        # commit logs
        self._append_commit_log(str(history_dir / "logs" / "HEAD"), snapshot)

        # append it to the history
        self.history.append(snapshot)
        self.HEAD = snapshot["uid"]

        return snapshot["uid"]

    def save_snapshot(self, dirpath):
        """Saves a snapshot (commit) of a model

        Args:
            dirpath: path to the directory where the snapshot is to be saved
        """
        model_info = {
            "name": self.name,
            "uid": self.uid,
            "license": self.license,
            "anatomy": self.anatomy,
            "scope": self.scope,
            "modality": self.modality,
        }
        fs.mkdir(dirpath)
        with open(os.path.join(dirpath, "info.json"), "w") as file:
            json.dump(model_info, file)
        torch.save(self.net, os.path.join(dirpath, "net.pt"))
        np.set_printoptions(suppress=True)
        if self.train_record is not None:
            self.train_record.to_csv(
                path_or_buf=os.path.join(dirpath, "train_record.csv"), index=False
            )
        if self.validation_record is not None:
            self.validation_record.to_csv(
                path_or_buf=os.path.join(dirpath, "validation_record.csv"), index=False
            )
        if self.test_record is not None:
            self.test_record.to_csv(
                path_or_buf=os.path.join(dirpath, "test_record.csv"), index=False
            )

    def load_snapshot(self, dirpath):
        """Loads a snapshot (commit) of a model

        Args:
            dirpath: path to the directory where the snapshot is stored
        """
        with open(os.path.join(dirpath, "info.json")) as file:
            model_info = json.load(file)
        self.name = model_info["name"]
        self.uid = model_info["uid"]
        self.license = model_info["license"]
        self.anatomy = model_info["anatomy"]
        self.scope = model_info["scope"]
        self.modality = model_info["modality"]
        self.net = torch.load(os.path.join(dirpath, "net.pt"))

        if os.path.exists(os.path.join(dirpath, "train_record.csv")):
            self.train_record = pd.read_csv(os.path.join(dirpath, "train_record.csv"))
        if os.path.exists(os.path.join(dirpath, "validation_record.csv")):
            self.validation_record = pd.read_csv(
                os.path.join(dirpath, "validation_record.csv")
            )
        if os.path.exists(os.path.join(dirpath, "test_record.csv")):
            self.test_record = pd.read_csv(os.path.join(dirpath, "test_record.csv"))

    def load(self, dirpath=None):
        if dirpath is not None:
            self.model_dir = Path(dirpath)
        history_dir = self.model_dir / "history"

        # pointer to the branch
        with open(str(history_dir / "HEAD"), "r") as f:
            line = f.readline()
            # ref: refs/heads/main
            branch_head_dir = line[5:]

        # pointer to the head
        with open(str(history_dir / branch_head_dir), "r") as f:
            self.HEAD = f.readline()

        # read logs
        with open(str(history_dir / "logs" / "HEAD"), "r") as f:
            lines = f.readlines()
            self.history = []
            for line in lines:
                commit = {
                    "prev": None,
                    "uid": None,
                    "author": None,
                    "email": None,
                    "time": None,
                    "desc": None,
                }
                commit["prev"] = line[:UID_LEN]
                line = line[UID_LEN + 1 :]
                commit["uid"] = line[:UID_LEN]
                line = line[UID_LEN + 1 :]
                tokens = line.split("<")
                commit["author"] = tokens[0][:-1]
                tokens = tokens[1].split(">")
                commit["email"] = tokens[0]
                tokens = tokens[1].split("\t")
                commit["time"] = float(tokens[0])
                commit["desc"] = tokens[1][:-1]

                self.history.append(commit)

        # read objects
        self.load_snapshot(str(history_dir / "objects" / self.HEAD))

    def checkout(self, uid):
        self.HEAD = uid
        history_dir = self.model_dir / "history"
        self.load_snapshot(str(history_dir / "objects" / self.HEAD))

        # header pointer
        with open(str(history_dir / "refs" / "heads" / "main"), "w") as f:
            f.write(self.HEAD)

    def is_mergeable(self, model, force=False):
        """Check if an incoming model is mergeable to the current model.

        Args:
            model: incoming model
            force: if True, it will ignore model uid mismatch and force merge it.
        Returns:
            True if mergeable. False otherwise.
            TODO: return more detailed diagnostics than just true and false?
        """
        # Models with different UIDs cannot be merged.
        if not force:
            if self.uid != model.uid:
                return False

        # for each incoming commit, see if it already exists.
        # if it does, make sure the "prev" pointer is pointing
        # the same commit object.
        for incoming_commit in model.history:
            for commit in self.history:
                if commit["uid"] == incoming_commit["uid"]:
                    if commit["prev"] != incoming_commit["prev"]:
                        return False

        # passed all test. should be mergeable
        return True

    def merge(self, model, force=False):
        """Merge an incoming model to the current model.

        Args:
            model: incoming model
            force: if True, it will ignore model uid mismatch and force merge it.
        """
        if not self.is_mergeable(model, force):
            raise Exception(f"Model {model.name} cannot be merged to {self.name}.")

        # all possible commmit paths in the current model.
        all_paths = []
        all_commit_uids = []
        for commit in self.history:
            all_paths.append(self._commit_path(commit["uid"]))
            all_commit_uids.append(commit["uid"])

        # for each incoming commit, check if it exists.
        # if not, find the relevant commit path and add the path.
        paths_to_add = []
        for incoming_commit in model.history:
            if incoming_commit["uid"] in all_commit_uids:
                continue

            # incoming commit doesn't exist in the current model.
            # find the commit path that led to the incoming commit.
            incoming_path = model._commit_path(incoming_commit["uid"])

            # prune out duplicated part in the path
            for i, incoming_uid in enumerate(incoming_path):
                if incoming_uid in all_commit_uids:
                    break

            # add the pruned path to the queue
            paths_to_add.append(incoming_path[: i + 1])

        for path_to_add in paths_to_add:
            for incoming_uid in path_to_add[::-1]:
                if incoming_uid in all_commit_uids:
                    continue

                i = model.find_commit(incoming_uid)

                all_commit_uids.append(incoming_uid)
                self._append_commit_log(
                    str(self.model_dir / "history" / "logs" / "HEAD"), model.history[i]
                )

                self.history.append(model.history[i])

                # Copy files
                fs.copy(
                    str(model.model_dir / "history" / "objects" / incoming_uid),
                    str(self.model_dir / "history" / "objects" / incoming_uid),
                )

        # print("HEAD:", model.HEAD)
        self.checkout(model.HEAD)
        # print("HEAD:", self.HEAD)

    def find_commit(self, uid):
        """Finds a commit by UID.

        Args:
            uid: UID of the commit.
        Returns:
            list index of the commit. None if not found.
        """
        for i, commit in enumerate(self.history):
            if commit["uid"] == uid:
                return i
        return None

    def _commit_path(self, uid):
        """Paths from a particular commit to crawl up to the genesis commit.

        Args:
            uid: uid of a commit.
        """
        i = self.find_commit(uid)
        if i is None:
            raise ValueError(f"Cannot find {uid} in the model history.")

        prev_uid = self.history[i]["prev"]
        ret_val = [uid]
        if prev_uid != "0" * UID_LEN:
            ret_val.extend(self._commit_path(prev_uid))
        return ret_val

    def destroy(self):
        fs.remove(str(self.model_dir))

    def getTrajectory(self):
        trajectories = []
        for hist in self.history:
            trajectories.append(hist["author"])
        return trajectories

    def __repr__(self):
        # TODO: Implement
        return self.name

    def __str__(self):
        # TODO: Implement
        return "test"

    def __eq__(self, model):
        """Overides equal-to operator '=='

        model_dir does not need to be the same
        HEAD node must be the same.
        """
        for commit in self.history:
            i = model.find_commit(commit["uid"])
            if i is None:
                # print(f"{self.name} and {model.name} have different commit histories.")
                return False

        for commit in model.history:
            i = self.find_commit(commit["uid"])
            if i is None:
                # print(f"{self.name} and {model.name} have different commit histories.")
                return False

        if self.net is not None and model.net is None:
            # print(f"{self.name} and {model.name} have different networks.")
            return False
        if self.net is None and model.net is not None:
            # print(f"{self.name} and {model.name} have different networks.")
            return False

        if self.net is None and model.net is None:
            same_nets = True
        else:
            same_nets = compare_nets(self.net, model.net)

        return (
            same_nets
            and self.uid == model.uid
            and self.name == model.name
            and self.minValLoss == model.minValLoss
            and self.license == model.license
            and self.anatomy == model.anatomy
            and self.scope == model.scope
            and self.modality == model.modality
        )

    # assert len(model.history) == len(recovered.history)
    # for i in range(len(model.history)):
    #     assert model.history[i] == recovered.history[i]
    # assert model.name == recovered.name
    # assert model.model_dir == recovered.model_dir

    def compare_architecture(self, model):
        """Check if an incoming model has the same net structure as that of
            the current model
        Args:
            model: the incoming model which would be comparied with the current model
        Returns:
            True if coincides. False otherwise.
        """
        # first step, check if the incoming model has a parameter generator of
        # the same length as that of the current model
        model_param_generator = model.net.parameters()
        self_model_param_generator = self.net.parameters()

        if len(list(model_param_generator)) != len(list(self_model_param_generator)):
            # if not equal, returns False
            return False
        else:
            # second step, check if each parameter in the generator of the incoming
            # model has the same torch size as that of the current model
            model_param_generator = model.net.parameters()
            self_model_param_generator = self.net.parameters()

            for (idx, param) in enumerate(self.net.parameters()):
                param_1 = next(model_param_generator)
                param_2 = next(self_model_param_generator)

                # if not the same, returns False
                if param_1.data.shape != param_1.data.shape:
                    return False

            # if passes two-step checking, returns True
            return True

    def __add__(self, model):
        """Gives '+' operator in the fed_avg_algorithm.
            Returns a model whose every net parameter is the sum of net paramters of
            the current model and an incoming model in respective place.
        Args:
            model: the incoming model which would be added to the current model
        Returns:
            result_model: the result model after addition, which has the exact same
                          model.history as the current model
        """

        # check if the nets of the current model and the incoming model coincide
        if self.compare_architecture(model) == False:

            raise TypeError(
                "Two model nets do not coincide, thus could not do the addition."
            )

        else:
            # prepare result_model and its parameter generator
            result_model = copy.deepcopy(self)
            result_model_param_generator = result_model.net.parameters()

            # get the parameter generators of the current model and the incoming model
            self_model_param_generator = self.net.parameters()
            other_model_param_generator = model.net.parameters()

            # do the addition by parameter in generator
            for (idx, param) in enumerate(self.net.parameters()):
                param_1 = next(self_model_param_generator)
                param_2 = next(other_model_param_generator)

                # assign the result of addition
                result_param = next(result_model_param_generator)
                result_param.data = param_1.data + param_2.data

            return result_model

    def __mul__(self, scalar):
        """Gives '*' operator in the fed_avg_algorithm.
            Returns a model whose every net parameter is the product of the scalar and
            the net paramter of the current model in respective place.
        Args:
            scalar: the number that would time every net parameter of the current model
        Returns:
            result_model: the result model after multiplication, which has the exact same
                          model.history as the current model
        """
        # prepare result_model and its parameter generator
        result_model = copy.deepcopy(self)
        result_model_param_generator = result_model.net.parameters()

        # get the parameter generator of the current model
        self_model_param_generator = self.net.parameters()

        # do the multiplication by parameter in generator
        for (idx, param) in enumerate(self.net.parameters()):
            param = next(self_model_param_generator)

            # assign the result of multiplication
            result_param = next(result_model_param_generator)
            result_param.data = param.data * float(scalar)

        return result_model

    def __truediv__(self, scalar):
        """Gives '/' operator in the fed_avg_algorithm.
            Returns a model whose every net parameter is the quotient of the net paramter
            of the current model divided by the scalar in respective place.
        Args:
            scalar: the number that would divide every net parameter of the current model
        Returns:
            result_model: the result model after division, which has the exact same
                          model.history as the current model
        """
        # check if the scalar is nonzero
        if scalar == 0:
            raise ValueError("Net parameters could not be divided by 0.")
        else:
            # prepare result_model and its parameter generator
            result_model = copy.deepcopy(self)
            result_model_param_generator = result_model.net.parameters()

            # get the parameter generator of the current model
            self_model_param_generator = self.net.parameters()

            # do the division by parameter in generator
            for (idx, param) in enumerate(self.net.parameters()):
                param = next(self_model_param_generator)

                # assign the result of division
                result_param = next(result_model_param_generator)
                result_param.data = param.data / float(scalar)

            return result_model


def load_model(filepath):
    model = Model()
    model.load(filepath)
    return model


def compare_nets(net1, net2):
    """Compares if two torch.nn.Module objects are the same.

    Returns:
        bool: True if the nets have the same parameters.
    """
    for p1, p2 in zip(net1.parameters(), net2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
        return True
