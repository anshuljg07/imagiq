import os
import torch
import numpy as np
import math
from collections import Counter
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from itertools import combinations
from imagiq.common import uid_generator
import matplotlib.pyplot as plt
from datetime import datetime
import json
import csv
import copy
from imagiq.utils.file_systems import mkdir
import pandas as pd


class Ensemble:
    """Ensemble class"""

    def __init__(
        self,
        size,
        models,
        val_dataset,  # used for ensemble creation
        name=None,
        diversity_measure="auc",
        vote_method="majority",
        test_dataset=None,  # used for ensemble evaluation
        description=None,
    ):
        self.uid = uid_generator()
        self.ensemble_size = size
        self.model_bench = models  # list of available models

        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.name = "ensemble_" + self.uid[:5] if name is None else name
        self.description = description

        self.diversity_measure = diversity_measure
        self.vote_method = vote_method
        self.ensemble_models = None

        self.history = []
        self.model_preds = dict()
        self.model_preds_test = dict()
        """
        Generated variable
            output = {
                "uid":uid,
                "time":time stamp,
                "size":number of models in final selection,
                "method":"normal","hill_climbing"
                "diversity_measure":measurement to evaluate esenemble models.,
                "voted_measure":aggregating measurement. "majority" and "probability" available,
                "model_bench":[model_uids]
                "ensemble_models":{'name':model names,'uid':model uids},
                "pred":ensemble predictions on val dataset
                "diversity_score":,
                "ensemble_val_auc":,
                "ensemble_val_acc":,}
            model_preds: a dictionary with predictions according the validation dataset.
                Keys: 
                    y -- labels
                    model.uids_init / model.uids_commit.uids -- predictions according to the model and version
        """

    def dataset_update(self, dataset, is_test=False):
        """Change a dataset. This function can help you to make the changes clean. Since once you change the dataset,
            you would want a re-evaluation on the model with the new dataset. Thus, this function cleans the record when
            you change a new dataset.

        Args:
            dataset:the dataset you want to import
            is_test: True means you are changing the test dataset but not the validation dataset. To change the validation dataset, set into False
        """
        if not is_test:  # update val dataset
            self.val_dataset = dataset

            # change a dataset will empty the saved model_preds and the outputs
            self.model_preds = dict()
            self.outputs = []
        else:  # update test dataset
            self.test_dataset = dataset

    def create(
        self, percentage=0, plot_chart=False, method="normal"
    ):  # percentage 75/25
        """find k number of model out of n which with the highest diversity_measure score of ensemble model.
        Args:
            model_uids: a list of uids from the model pools
            diversity_measure: measurement to evaluate esenemble models.
            vote_method: votes aggregating measurement. "majority" and "probability" available
            percentage: when the diversity_measure is auc, percentage is useless. But when you choose a "gd",
                the percentage represents the percentage that the diversity takes. For example, if percentage is 0.2,
                then the calculation would be 0.2*diversity + (1-0.2) * auc
            method: can be normal or hill_climbing where normal tries all combinations and hill climbing saves
                some time by optimization the comparsion time
        """
        # alert them about a useless setting
        if plot_chart and method == "normal":
            raise Exception("plot chart is not applicable for normal ensemble method")

        if method == "normal":
            best_combination = self.__normal_ensemble(
                self.diversity_measure, percentage
            )
        elif method == "hill_climbing":
            best_combination = self.__hill_climbing(
                self.diversity_measure,
                percentage,
                plot_chart=plot_chart,
            )

        self.evaluate(self.model_bench, test=True)
        _, test_predictions, _ = self.evaluate(self.ensemble_models, test=True)

        # save result to history
        output = {
            "uid": uid_generator(),
            "time": datetime.timestamp(datetime.now()),
            "size": self.ensemble_size,
            "method": method,
            "diversity_measure": self.diversity_measure,
            "voted_measure": self.vote_method,
            "diversity_percentage": percentage,
            "model_bench": [model.uid for model in self.model_bench],
            "ensemble_models": {
                "uid": best_combination["model_uids"],
                "name": best_combination["model_names"],
            },
            "pred": best_combination["pred"],
            "pred_test": test_predictions,
            "diversity_score": best_combination["diversity_score"],
            "ensemble_val_auc": best_combination["auc"],
            "ensemble_val_acc": best_combination["acc"],
        }
        self.history.append(output)
        return best_combination

    def __normal_ensemble(self, diversity_measure="auc", percentage=0):
        comb = list(combinations(self.model_bench, self.ensemble_size))
        best_combination = {}
        auc_sum = auc_sqr_sum = count = 0
        for models in comb:
            # models = [model for model in self.node.model_bench if model.uid in uids]
            data = {}
            data["model_names"] = [model.name for model in models]
            data["model_uids"] = [model.uid for model in models]
            y, e_pred, checks = self.evaluate(models, check=True)

            # format check
            rawLabel = (
                True if len(y) == 1 else False
            )  # bug: y is tensor, len(y) == 1 could be evoked for batch size 1
            data["pred"] = e_pred
            y = self.__data_check_and_convert(y)
            e_pred = self.__data_check_and_convert(e_pred)

            auc = (
                # self.__get_auc(y, self.__convert_possibility_to_label(e_pred))
                roc_auc_score(
                    y, self.__convert_possibility_to_label(e_pred), multi_class="ovo"
                )
                if self.vote_method == "majority"
                else roc_auc_score(
                    y, e_pred, multi_class="ovo"
                )  # self.__gen_auc(y, e_pred)
            )

            data["auc"] = auc
            data["acc"] = (
                (y.argmax(axis=1) == e_pred.argmax(axis=1)).astype(float).mean()
            )
            auc_sum += data["auc"]
            auc_sqr_sum += data["auc"] ** 2
            count += 1

            # Set diversity score as the AUC
            if diversity_measure == "auc":
                data["diversity_score"] = data["auc"]
            else:
                avg_auc = sum([model.quality for model in models]) / self.ensemble_size
                # TODO: model.quality shoudl be repalced with the AUC score since `model.quality` can be computed with other measurements.

                # Calculate diversity score with entropy measurement
                """
                if diversity_measure == "entropy": #no more entropy
                    weights = sum(checks)
                    diversity = 0
                    for i in range(len(weights)):
                        diversity += min(weights[i], k - weights[i])
                    diversity = diversity * 1 / len(y) * 1 / (k - math.ceil(k / 2))
                    data["diversity_score"] = avg_auc + diversity
                """
                # Calculate diversity score with generalized diversity measurement
                if diversity_measure == "gd":  # generated diversity for multi-class
                    data["diversity_score"] = self.__generalized_diversity(
                        self.model_bench,
                        checks,
                        avg_auc,
                        self.ensemble_size,
                        y,
                        percentage,
                    )
                # Calculate diversity score with double fault measurement
                elif diversity_measure == "df":
                    data["diversity_score"] = self.__double_fault_measurement(
                        self.model_bench, checks, avg_auc, y, percentage
                    )
                else:
                    raise ValueError(
                        "diversity_measure parameter does not have '"
                        + diversity_measure
                        + "' value"
                    )

            if best_combination.get("model_names"):
                best_combination = max(
                    best_combination, data, key=lambda x: x["diversity_score"]
                )
            else:
                best_combination = data

        self.ensemble_models = [
            model
            for model in self.model_bench
            if model.uid in best_combination["model_uids"]
        ]

        return best_combination

    def __hill_climbing(
        self,
        diversity_measure="auc",
        percentage=0,
        testdataset=False,
        plot_chart=False,
    ):
        """find k number of model out of n which with the highest diversity_measure score of ensemble model.
        Args:
            model_uids: a list of uids from the model pools
            diversity_measure: measurement to evaluate esenemble models.
            vote_method: votes aggregating measurement. "majority" and "probability" available
            percentage: when the diversity_measure is auc, percentage is useless. But when you choose a "gd",
                the percentage represents the percentage that the diversity takes. For example, if percentage is 0.2,
                then the calculation would be 0.2*diversity + (1-0.2) * auc
            testdataset: True to see the relationship between the testdataset and the validation dataset.
                No need to fill this if you just want to run on validations
            plot_chart: True if you want to see the chart that shows comparsion and auc
        """
        comparison_number = 0
        results = []

        sorted_bench = self.__select_top_k(
            self.model_bench, k=len(self.model_bench)
        )  # sort bench model by performance
        best_result = (
            {}
        )  # it contains "model_names", "auc","acc","diversity_score","comparison_number"
        ensemble_set = sorted_bench[
            : self.ensemble_size
        ]  # select first k model as init value

        # Sorted the ensemble set from the worst performance to the best
        ensemble_set = sorted(
            ensemble_set, key=lambda model: model.quality, reverse=False
        )
        best_result["model_names"] = [model.name for model in ensemble_set]
        best_result["model_uids"] = [model.uid for model in ensemble_set]

        y, e_pred, checks = self.evaluate(ensemble_set, check=True)
        best_result["pred"] = e_pred
        """
        if testdataset : 
            y_test, e_pred_test, checks = self.evaluate(ensemble_set, check=True, ensemble_measure=vote_method, debug = False,test=True)
            best_result['test_auc']= compute_roc_auc(e_pred_test, y_test, to_onehot_y=True, softmax=True)
            best_result['test_acc'] = (y_test == e_pred_test.argmax(dim=-1)).float().mean()
        """
        y = self.__data_check_and_convert(y)
        e_pred = self.__data_check_and_convert(e_pred)
        if self.vote_method == "majority":
            # auc = self.__gen_auc(y, self.__convert_possibility_to_label(e_pred))
            auc = roc_auc_score(
                y, self.__convert_possibility_to_label(e_pred), multi_class="ovo"
            )
        elif self.vote_method == "probability":
            # auc = self.__gen_auc(y, e_pred)
            auc = roc_auc_score(y, e_pred, multi_class="ovo")
        best_result["auc"] = auc
        best_result["acc"] = (
            (y.argmax(axis=1) == e_pred.argmax(axis=1)).astype(float).mean()
        )
        avg_auc = sum([model.quality for model in ensemble_set]) / self.ensemble_size

        if diversity_measure == "auc":
            best_result["diversity_score"] = best_result["auc"]
        elif diversity_measure == "gd":
            best_result["diversity_score"] = self.__generalized_diversity(
                self.model_bench, checks, avg_auc, self.ensemble_size, y, percentage
            )
        elif diversity_measure == "df":
            best_result["diversity_score"] = self.__double_fault_measurement(
                self.model_bench, checks, avg_auc, y, percentage
            )
        else:
            raise ValueError(
                "diversity_measure parameter does not have '"
                + diversity_measure
                + "' value"
            )

        best_result["comparison_number"] = comparison_number
        print(
            "comparsion number = ",
            str(comparison_number),
            ", and init model is",
            best_result,
        )

        if plot_chart:
            results.append(dict(best_result))

        # replace model in ensemble_set one by one
        for pivot in range(self.ensemble_size):  # loop in ensemble set
            for model in sorted_bench:
                # print("current model name is", model.name," and pivot is ", str(pivot))
                if model.name not in [
                    m.name for m in ensemble_set
                ]:  # if model already selected, no need to evaluate
                    comparison_number += 1
                    # print("comparsion number = ", str(comparison_number))
                    tmp_ensemble_set = (
                        ensemble_set[:pivot] + [model] + ensemble_set[(pivot + 1) :]
                    )
                    # print([model.name for model in tmp_ensemble_set])
                    y, e_pred, checks = self.evaluate(tmp_ensemble_set, check=True)
                    y = self.__data_check_and_convert(y)
                    e_pred = self.__data_check_and_convert(e_pred)

                    if self.vote_method == "majority":
                        # auc = self.__gen_auc(
                        #     y, self.__convert_possibility_to_label(e_pred)
                        # )
                        auc = roc_auc_score(
                            y,
                            self.__convert_possibility_to_label(e_pred),
                            multi_class="ovo",
                        )
                    elif self.vote_method == "probability":
                        # auc = self.__gen_auc(y, e_pred)
                        auc = roc_auc_score(y, e_pred, multi_class="ovo")
                    else:
                        raise ValueError("vote_method does not exists")
                    tmp_auc = auc
                    tmp_acc = (
                        (y.argmax(axis=1) == e_pred.argmax(axis=1)).astype(float).mean()
                    )
                    tmp_avg_auc = (
                        sum([model.quality for model in tmp_ensemble_set])
                        / self.ensemble_size
                    )
                    # compute evaluate objective value
                    if diversity_measure == "auc":
                        tmp_diversity_score = tmp_auc
                    elif diversity_measure == "gd":
                        tmp_diversity_score = self.__generalized_diversity(
                            self.model_bench,
                            checks,
                            tmp_avg_auc,
                            self.ensemble_size,
                            y,
                            percentage,
                        )
                    elif diversity_measure == "df":
                        tmp_diversity_score = self.__double_fault_measurement(
                            self.model_bench, checks, tmp_avg_auc, y, percentage
                        )
                    else:
                        raise ValueError("Indicated method not exsist")
                    """ -------------------Measure standard-----------------------
                    gd larger = better
                    df lower = better
                    """

                    if tmp_diversity_score > best_result[
                        "diversity_score"
                    ] and diversity_measure in ["auc", "gd"]:

                        ensemble_set = tmp_ensemble_set
                        best_result["diversity_score"] = tmp_diversity_score
                        best_result["pred"] = e_pred
                        best_result["auc"] = tmp_auc
                        best_result["acc"] = tmp_acc
                        best_result["model_names"] = [
                            model.name for model in ensemble_set
                        ]
                        best_result["model_uids"] = [
                            model.uid for model in ensemble_set
                        ]
                        best_result["comparison_number"] = comparison_number
                        """
                        if testdataset !=None: 
                            tmp_y_test, tmp_e_pred_test, checks = self.evaluate(ensemble_set, check=True, ensemble_measure=vote_method, debug = False,test=True)
                            best_result['test_auc']= compute_roc_auc(tmp_e_pred_test, tmp_y_test, to_onehot_y=True, softmax=True)
                            best_result['test_acc'] = (tmp_y_test == tmp_e_pred_test.argmax(dim=-1)).float().mean()
                        """
                        print(
                            "Comparison Number = ",
                            str(comparison_number),
                            "Better result is found: ",
                            best_result,
                        )

                        # results.append(dict(best_result))
                    elif tmp_diversity_score < best_result[
                        "diversity_score"
                    ] and diversity_measure in ["df"]:
                        ensemble_set = tmp_ensemble_set
                        best_result["diversity_score"] = tmp_diversity_score
                        best_result["pred"] = e_pred
                        best_result["auc"] = tmp_auc
                        best_result["acc"] = tmp_acc
                        best_result["model_names"] = [
                            model.name for model in ensemble_set
                        ]
                        best_result["model_uids"] = [
                            model.uid for model in ensemble_set
                        ]
                        best_result["comparison_number"] = comparison_number
                        """
                        if testdataset !=None: 
                            tmp_y_test, tmp_e_pred_test, checks = self.evaluate(ensemble_set, check=True, ensemble_measure=vote_method, debug = False,test=True)
                            best_result['test_auc']= compute_roc_auc(tmp_e_pred_test, tmp_y_test, to_onehot_y=True, softmax=True)
                            best_result['test_acc'] = (tmp_y_test == tmp_e_pred_test.argmax(dim=-1)).float().mean()
                        """
                        print(
                            "Comparison Number = ",
                            str(comparison_number),
                            "Better result is found: ",
                            best_result,
                        )
                        # results.append(dict(best_result))

                    if plot_chart:
                        results.append(dict(best_result))

        if plot_chart:
            print("Chart Showing")
            # chart for comparison_number vs auc
            x1 = [element for element in range(comparison_number + 1)]
            y1 = [element["auc"] for element in results]
            plt.plot(x1, y1, "b-", label="Attemps vs Validation AUC")

            """
            if testdataset !=None: 
                x2 = [element for element in range(comparison_number+1) ]
                y2 = [element['test_auc'] for element in results ]
                plt.plot(x2,y2,'k-',label='Attemps vs Test AUC')
            """
            plt.xlabel("Number of Attemps")
            plt.ylabel("AUC")
            plt.title("Comparison History")
            plt.legend()
            plt.show()
            # print("Final Results:",results)
            # return results#results[-1] is the best performance

        self.ensemble_models = [
            model
            for model in self.model_bench
            if model.uid in best_result["model_uids"]
        ]

        return best_result

    def __generalized_diversity(self, model_bench, checks, avg_auc, k, y, percentage=0):
        """
        private function to calculate the generalized diversity.
        Should not be called outside of this class
        """
        weights = sum(checks)
        xy = Counter(weight.item() for weight in weights)

        d = []
        for i in range(k):
            j = xy.get(i)
            if j:
                d.append(j)
            else:
                d.append(0)
        p = d[::-1]
        pi = [i / len(y) for i in p]

        p1 = sum(
            [
                (a + 1) * (1 / len(model_bench)) * b  # model_bench pool?
                for a, b in zip(range(len(model_bench)), pi)
            ]
        )
        p2 = sum(
            [
                (a + 1)
                * (a)
                * (1 / len(model_bench))
                * (1 / (len(model_bench) - 1))
                * b
                for a, b in zip(range(len(model_bench)), pi)
            ]
        )

        diversity = 1 - (p2 / p1)
        diversity_score = (1 - percentage) * avg_auc + percentage * diversity
        return diversity_score

    def __double_fault_measurement(self, model_bench, checks, avg_auc, y, percentage=0):
        """
        private function to calculate the double fault measurement.
        Should not be called outside of this class
        """
        d = 0
        for i in range(len(checks)):
            for j in range(i + 1, len(checks)):
                N = [0, 0, 0, 0]
                for k in range(len(y)):
                    a = y[k]
                    c1, c2 = checks[i][k], checks[j][k]
                    if c1 == True:
                        if c2 == True:
                            N[0] += 1
                        else:
                            N[1] += 1
                    else:
                        if c2 == True:
                            N[2] += 1
                        else:
                            N[3] += 1
                d += N[3] / sum(N)
        diversity = d / ((len(model_bench) * (len(model_bench) - 1)) / 2)
        diversity_score = (1 - percentage) * avg_auc - percentage * diversity
        return diversity_score

    def __str__(self):
        """printing Ensembles"""

        return f"Ensemble {self.name} - {self.uid}"

    def __select_top_k(self, model_bench, k=3):
        """private function to Selects k best models in model_bench based on the assigned quality score among the individual models.
        Should not be called out side of this class
        Args:
            model_bench: model pools
            k: number of model to be selected.
        """
        for model in model_bench:
            commit_uid = model.history[-1]["uid"] if len(model.history) != 0 else "init"
            uid = f"{model.uid}_{commit_uid}"
            if uid not in self.model_preds.keys():
                self.model_preds["y"], self.model_preds[uid] = model.predict(
                    self.val_dataset, batch_size=32, section="validation"
                )
                if (
                    len(self.model_preds["y"].shape) == 1
                ):  # not in one-hot encoded format
                    tmp_ys = torch.zeros(
                        (self.model_preds["y"].shape[0], 2)
                    )  # pbug: only tyring to handle binary case
                    for idx in range(self.model_preds["y"].shape[0]):
                        if self.model_preds["y"][idx] == 0:
                            tmp_ys[idx, 0] = 1
                        else:
                            tmp_ys[idx, 1] = 1
                    self.model_preds["y"] = tmp_ys.to(self.model_preds[uid].device)
                auc_value = ""
                if torch.unique(self.model_preds["y"]).shape[0] == 1:
                    metric_auc = 0
                else:
                    y = self.__data_check_and_convert(
                        self.model_preds["y"]
                    )  # It should be one-hot encoded
                    e_pred = self.__data_check_and_convert(
                        self.model_preds[uid]
                    )  # It should be one-hot encoded

                    # metric_auc = self.__gen_auc(y, e_pred)
                    metric_auc = roc_auc_score(y, e_pred, multi_class="ovo")
                    # self.node.model_update[model.uid] = True
                    auc_value += "%.4f" % metric_auc
                    model.quality = float(auc_value)
            if uid not in self.model_preds_test.keys():
                self.model_preds_test["y"], self.model_preds_test[uid] = model.predict(
                    self.test_dataset, batch_size=32, section="test"
                )

                if (
                    len(self.model_preds_test["y"].shape) == 1
                ):  # not in one-hot encoded format
                    tmp_ys = torch.zeros(
                        (self.model_preds_test["y"].shape[0], 2)
                    )  # pbug:
                    for idx in range(self.model_preds_test["y"].shape[0]):
                        if self.model_preds_test["y"][idx] == 0:
                            tmp_ys[idx, 0] = 1
                        else:
                            tmp_ys[idx, 1] = 1
                    self.model_preds_test["y"] = tmp_ys.to(
                        self.model_preds_test[uid].device
                    )
                auc_value = ""
        return sorted(self.model_bench, key=lambda model: model.quality, reverse=True)[
            : self.ensemble_size
        ]

    def evaluate(
        self,
        models,
        batch_size=32,
        check=False,
        test=False,
    ):
        """Aggregate votes from models for ensemble model.
        Args:
            models: selected models for ensemble.
            batch_size: batch size for evaluate.
            ensemble_measure: votes aggregating measurement. "majority" and "probability" available
            check: flag for saving check table. Internal variable for other function to calculate diversity score.
                Please do not modify on this value if you only wants to evaluate the ensemble result
            test: True if predict on test dataset
        """

        predictions = []
        checks = []
        for model in models:
            commit_uid = model.history[-1]["uid"] if len(model.history) != 0 else "init"
            uid = f"{model.uid}_{commit_uid}"
            if not test:
                if uid not in self.model_preds.keys():
                    (self.model_preds["y"], self.model_preds[uid],) = model.predict(
                        self.val_dataset, batch_size, section="validation"
                    )
                    if (
                        len(self.model_preds["y"].shape) == 1
                    ):  # not in one-hot encoded format
                        tmp_ys = torch.zeros(
                            (self.model_preds["y"].shape[0], 2)
                        )  # pbug:
                        for idx in range(self.model_preds["y"].shape[0]):
                            if self.model_preds["y"][idx] == 0:
                                tmp_ys[idx, 0] = 1
                            else:
                                tmp_ys[idx, 1] = 1
                        self.model_preds["y"] = tmp_ys.to(self.model_preds[uid].device)
                y = self.model_preds["y"]
                y_pred = self.model_preds[uid]
            else:
                if self.test_dataset == None:
                    raise ValueError(
                        "There is no test dataset uploaded in this Ensemble"
                    )
                else:
                    if uid not in self.model_preds_test.keys():
                        (
                            self.model_preds_test["y"],
                            self.model_preds_test[uid],
                        ) = model.predict(self.test_dataset, batch_size, section="test")
                        if (
                            len(self.model_preds_test["y"].shape) == 1
                        ):  # not in one-hot encoded format
                            tmp_ys = torch.zeros(
                                (self.model_preds_test["y"].shape[0], 2)
                            )  # pbug:
                            for idx in range(self.model_preds_test["y"].shape[0]):
                                if self.model_preds_test["y"][idx] == 0:
                                    tmp_ys[idx, 0] = 1
                                else:
                                    tmp_ys[idx, 1] = 1
                            self.model_preds_test["y"] = tmp_ys.to(
                                self.model_preds_test[uid].device
                            )
                    # self.node.model_update[model.uid] = True
                    y = self.model_preds_test["y"]
                    y_pred = self.model_preds_test[uid]
            predictions.append(y_pred)

            if check:
                checks.append(y.argmax(dim=-1) == y_pred.argmax(dim=-1))

        e_pred = self.aggregate_predictions(predictions)

        y = self.__data_check_and_convert(y)  # It should be one-hot encoded
        e_pred = self.__data_check_and_convert(e_pred)  # It should be one-hot encoded

        if self.vote_method == "majority":
            # auc = self.__gen_auc(y, self.__convert_possibility_to_label(e_pred))
            auc = roc_auc_score(
                y, self.__convert_possibility_to_label(e_pred), multi_class="ovo"
            )
        else:
            # auc = self.__gen_auc(y, e_pred)
            auc = roc_auc_score(y, e_pred, multi_class="ovo")
        acc = (y.argmax(axis=1) == e_pred.argmax(axis=1)).astype(float).mean()

        return y, e_pred, checks

    def set_best_ensemble(self):
        """to get the best result from history list based on diversity score"""
        attemps = [h for h in self.history]
        best_result = max(attemps, key=lambda x: x["diversity_score"])

        models = [
            model
            for model in self.model_bench
            if model.uid in best_result["ensemble_models"]["uid"]
        ]
        self.ensemble_models = models

    def aggregate_predictions(self, predictions, vote_method=None):
        if self.vote_method is not None and vote_method is None:
            vote_method = self.vote_method

        if vote_method == "majority":
            ensemble_prediction = sum(
                self.__convert_possibility_to_label(prediction)
                for prediction in predictions
            )
        elif self.vote_method == "probability":
            ensemble_prediction = sum(predictions) / len(predictions)
        else:
            raise ValueError("Indicated vote_method does not exists")

        return ensemble_prediction

    def save(
        self,
        ensemble_save_root_dir,
        output=True,
        ens_pred=True,
        model_pred=True,
    ):
        mkdir(ensemble_save_root_dir)

        # output in json to txt -- history
        history_filename = f"{ensemble_save_root_dir}/ensemble_{self.uid}_history.json"
        hist = copy.deepcopy(self.history)
        for i in hist:
            i["pred"] = i["pred"].tolist()
            i["pred_test"] = i["pred_test"].tolist()
        json_content = json.dumps(hist)
        f = open(history_filename, "w")
        f.write(json_content)
        f.close()

        # ensemble predict to csv
        for history in self.history:
            direct_path = os.path.join(ensemble_save_root_dir, history["uid"])
            mkdir(direct_path)

            val_pred_filename = (
                f"{direct_path}/ensemble_{self.uid}_{history['uid']}_validation.csv"
            )
            np.set_printoptions(precision=3, suppress=True)
            pd.DataFrame(
                {
                    "filename": list(self.model_bench[0].validation_record["filename"]),
                    "prediction": [str(list(np.round(x, 3))) for x in history["pred"]],
                    "predictionLabel": history["pred"].argmax(axis=1) + 1,
                    "label": self.model_bench[0].validation_record["label"],
                    "rawLabel": self.model_bench[0].validation_record["rawLabel"],
                }
            ).to_csv(path_or_buf=val_pred_filename, index=False)

            val_eval_filename = f"{direct_path}/ensemble_{self.uid}_{history['uid']}_validationEvaluation.csv"

            # tag
            y_label = (
                self.model_preds["y"].argmax(dim=1).cpu().numpy()
            )  # one-hot   => index
            y_pred = torch.Tensor(history["pred"]).argmax(dim=1).cpu().numpy()

            target_names = [str(i) for i in range(max(y_label) + 1)]

            matrix = classification_report(
                y_label,
                y_pred,
                target_names=target_names,
                output_dict=True,
                zero_division=0,
            )
            with open(val_eval_filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "label",
                        "auc",
                        "acc",
                        "precision",
                        "recall",
                        "f1-score",
                        "support",
                    ]
                )  # header

                for key in matrix.keys():
                    if key in target_names:
                        # model pred convert to binary
                        s_label = []
                        s_pred = []
                        for j in y_label:
                            s_l = 1 if j == int(key) else 0
                            s_label.append(s_l)
                        for j in y_pred:
                            s_p = 1 if j == int(key) else 0
                            s_pred.append(s_p)

                        acc = accuracy_score(s_label, s_pred)
                        auc = roc_auc_score(s_label, s_pred)
                        writer.writerow(
                            [
                                key,
                                str(auc),
                                str(acc),
                                str(matrix[key]["precision"]),
                                str(matrix[key]["recall"]),
                                str(matrix[key]["f1-score"]),
                                str(matrix[key]["support"]),
                            ]
                        )
                    elif key == "accuracy":
                        writer.writerow([key, "", "", "", "", str(matrix[key]), ""])
                    else:
                        writer.writerow(
                            [
                                key,
                                "",
                                "",
                                str(matrix[key]["precision"]),
                                str(matrix[key]["recall"]),
                                str(matrix[key]["f1-score"]),
                                str(matrix[key]["support"]),
                            ]
                        )

            if self.test_dataset is not None:
                test_pred_filename = (
                    f"{direct_path}/ensemble_{self.uid}_{history['uid']}_test.csv"
                )
                np.set_printoptions(precision=3, suppress=True)
                pd.DataFrame(
                    {
                        "filename": list(self.model_bench[0].test_record["filename"]),
                        "prediction": [
                            str(list(np.round(x, 3))) for x in history["pred_test"]
                        ],
                        "predictionLabel": history["pred_test"].argmax(axis=1) + 1,
                        "label": self.model_bench[0].test_record["label"],
                        "rawLabel": self.model_bench[0].test_record["rawLabel"],
                    }
                ).to_csv(path_or_buf=test_pred_filename, index=False)

                test_eval_filename = f"{direct_path}/ensemble_{self.uid}_{history['uid']}_testEvaluation.csv"
                y_label = (
                    self.model_preds_test["y"].argmax(dim=1).cpu().numpy()
                )  # one-hot   => index
                y_pred = torch.Tensor(history["pred_test"]).argmax(dim=1).cpu().numpy()

                target_names = [str(i) for i in range(max(y_label) + 1)]

                matrix = classification_report(
                    y_label,
                    y_pred,
                    target_names=target_names,
                    output_dict=True,
                    zero_division=0,
                )
                with open(test_eval_filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "label",
                            "auc",
                            "acc",
                            "precision",
                            "recall",
                            "f1-score",
                            "support",
                        ]
                    )  # header

                    for key in matrix.keys():
                        if key in target_names:
                            # model pred convert to binary
                            s_label = []
                            s_pred = []
                            for j in y_label:
                                s_l = 1 if j == int(key) else 0
                                s_label.append(s_l)
                            for j in y_pred:
                                s_p = 1 if j == int(key) else 0
                                s_pred.append(s_p)

                            acc = accuracy_score(s_label, s_pred)
                            auc = roc_auc_score(s_label, s_pred)
                            writer.writerow(
                                [
                                    key,
                                    str(auc),
                                    str(acc),
                                    str(matrix[key]["precision"]),
                                    str(matrix[key]["recall"]),
                                    str(matrix[key]["f1-score"]),
                                    str(matrix[key]["support"]),
                                ]
                            )
                        elif key == "accuracy":
                            writer.writerow([key, "", "", "", "", str(matrix[key]), ""])
                        else:
                            writer.writerow(
                                [
                                    key,
                                    "",
                                    "",
                                    str(matrix[key]["precision"]),
                                    str(matrix[key]["recall"]),
                                    str(matrix[key]["f1-score"]),
                                    str(matrix[key]["support"]),
                                ]
                            )

        # model prediction and performance to csv
        for model in self.model_bench:
            model.validation_record.to_csv(
                path_or_buf=os.path.join(
                    direct_path,
                    f"ensemble_{self.uid}_model_{model.uid}_validationPerformance.csv",
                ),
                index=False,
            )
            if self.test_dataset is not None:
                model.test_record.to_csv(
                    path_or_buf=os.path.join(
                        direct_path,
                        f"ensemble_{self.uid}_model_{model.uid}_testPerformance.csv",
                    ),
                    index=False,
                )

        for model_uid in self.model_preds.keys():
            if model_uid == "y":
                continue

            # model performance
            val_model_eval_filename = f"{direct_path}/ensemble_{self.uid}_model_{model_uid}_validationEvaluation.csv"

            y_label = (
                self.model_preds["y"].argmax(dim=1).cpu().numpy()
            )  # one-hot   => index
            y_pred = (
                self.model_preds[model_uid].argmax(dim=1).cpu().numpy()
            )  # one-hot   => index

            target_names = [str(i) for i in range(max(y_label) + 1)]

            matrix = classification_report(
                y_label,
                y_pred,
                target_names=target_names,
                output_dict=True,
                zero_division=0,
            )
            with open(val_model_eval_filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "label",
                        "auc",
                        "acc",
                        "precision",
                        "recall",
                        "f1-score",
                        "support",
                    ]
                )  # header

                for key in matrix.keys():
                    if key in target_names:
                        # model pred convert to binary
                        s_label = []
                        s_pred = []
                        for j in y_label:
                            s_l = 1 if j == int(key) else 0
                            s_label.append(s_l)
                        for j in y_pred:
                            s_p = 1 if j == int(key) else 0
                            s_pred.append(s_p)

                        acc = accuracy_score(s_label, s_pred)
                        auc = roc_auc_score(s_label, s_pred)
                        writer.writerow(
                            [
                                key,
                                str(auc),
                                str(acc),
                                str(matrix[key]["precision"]),
                                str(matrix[key]["recall"]),
                                str(matrix[key]["f1-score"]),
                                str(matrix[key]["support"]),
                            ]
                        )
                    elif key == "accuracy":
                        writer.writerow([key, "", "", "", "", str(matrix[key]), ""])
                    else:
                        writer.writerow(
                            [
                                key,
                                "",
                                "",
                                str(matrix[key]["precision"]),
                                str(matrix[key]["recall"]),
                                str(matrix[key]["f1-score"]),
                                str(matrix[key]["support"]),
                            ]
                        )

        for model_uid in self.model_preds_test.keys():
            if model_uid != "y":
                # model performance
                test_model_eval_filename = f"{direct_path}/ensemble_{self.uid}_model_{model_uid}_testEvaluation.csv"

                y_label = self.model_preds_test["y"].argmax(dim=1).cpu().numpy()
                y_pred = self.model_preds_test[model_uid].argmax(dim=1).cpu().numpy()

                target_names = [str(i) for i in range(max(y_label) + 1)]
                matrix = classification_report(
                    y_label,
                    y_pred,
                    target_names=target_names,
                    output_dict=True,
                    zero_division=0,
                )
                with open(test_model_eval_filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "label",
                            "auc",
                            "acc",
                            "precision",
                            "recall",
                            "f1-score",
                            "support",
                        ]
                    )  # header

                    for key in matrix.keys():
                        if key in target_names:
                            # model pred convert to binary
                            s_label = []
                            s_pred = []
                            for j in y_label:
                                s_l = 1 if j == int(key) else 0
                                s_label.append(s_l)
                            for j in y_pred:
                                s_p = 1 if j == int(key) else 0
                                s_pred.append(s_p)

                            acc = accuracy_score(s_label, s_pred)
                            auc = roc_auc_score(s_label, s_pred)
                            writer.writerow(
                                [
                                    key,
                                    str(auc),
                                    str(acc),
                                    str(matrix[key]["precision"]),
                                    str(matrix[key]["recall"]),
                                    str(matrix[key]["f1-score"]),
                                    str(matrix[key]["support"]),
                                ]
                            )
                        elif key == "accuracy":
                            writer.writerow([key, "", "", "", "", str(matrix[key]), ""])
                        else:
                            writer.writerow(
                                [
                                    key,
                                    "",
                                    "",
                                    str(matrix[key]["precision"]),
                                    str(matrix[key]["recall"]),
                                    str(matrix[key]["f1-score"]),
                                    str(matrix[key]["support"]),
                                ]
                            )

        # data statistics
        data_filename = f"{direct_path}/ensemble_{self.uid}_data_analysis.csv"
        with open(data_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset", "size", "label composition"])  # header

            # val_dataset
            # init
            label_composition = []
            if len(self.model_preds["y"].shape) == 1:
                y_label = self.model_preds["y"].cpu().numpy()
            else:
                y_label = self.model_preds["y"].argmax(dim=1).cpu().numpy()

            for i in range(max(y_label) + 1):
                label_composition.append(0)

            # sum
            for label in y_label:
                label_composition[label] += 1

            writer.writerow(["val_dataset", len(y_label), label_composition])

            # test_dataset
            # init
            label_composition_test = []
            if len(self.model_preds_test["y"].shape) == 1:
                y_label = self.model_preds_test["y"].cpu().numpy()
            else:
                y_label = self.model_preds_test["y"].argmax(dim=1).cpu().numpy()

            for i in range(max(y_label) + 1):
                label_composition_test.append(0)

            # sum
            for label in y_label:
                label_composition_test[label] += 1

            writer.writerow(["test_dataset", len(y_label), label_composition_test])

    def __data_check_and_convert(self, data):
        # This may has an issue if the data does not have all categories.For example,
        # if we should have 4 categories, but no sample is labeled in class 4.
        # Then it will be considered as 3 categories

        rawLabel = True if len(data.shape) == 1 else False

        if rawLabel:
            print("data is converting to one-hot-encoded")
            label_encoder = LabelEncoder()
            integer_encoded = label_encoder.fit_transform(data.cpu())
            onehot_encoder = OneHotEncoder(sparse=False)
            integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
            onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
            return onehot_encoded
        if torch.is_tensor(data):
            return data.cpu().numpy()

        return data

    def __data_check_and_convert_new(self, data):
        rawLabel = True if len(data.shape) == 1 else False

        if rawLabel:
            print("data is converting to one-hot-encoded")

            # assume the raw label is having the same context as classnames
            # assume the val dataset and test dataset will have the same number of class
            num_class = self.val_dataset.class_names
            lst = []
            for i in data:
                # row
                row = []
                for j in range(len(num_class)):
                    if j == i:
                        row.append(1)
                    else:
                        row.append(0)
                lst.append(row)

            return np.asarray(lst)

        if torch.is_tensor(data):
            return data.cpu().numpy()
        return data

    def __convert_possibility_to_label(self, data):
        lst = []
        num_class = len(data[0])
        if torch.is_tensor(data):
            label = data.argmax(dim=1)
        else:
            label = data.argmax(axis=1)
        for i in label:
            row = []
            for j in range(num_class):
                if i == j:
                    row.append(1)
                else:
                    row.append(0)
            lst.append(row)
        return np.asarray(lst)

    def __gen_auc(self, label, class_prediction):
        # if len(label.shape) == 1:
        #     tmp_labels = torch.zeros((label.shape[0], class_prediction.shape[-1]))
        #     for idx in range(labels.shape[0]):
        #         tmp_labels[idx][labels[idx]] = 1.0
        #     label = tmp_labels.to(label.device)

        # label and class_prediction are assume to be local
        c = len(class_prediction[0])
        A = []
        for i in range(c):
            A_i = []
            for j in range(c):
                A_i.append(0)
            A.append(A_i)

        # Check if the sum of the probabilities of each class is 1
        if not all(x >= 0.99 for x in list(map(sum, class_prediction))):
            class_prediction = torch.nn.Softmax()(torch.Tensor(class_prediction))

        if all(x >= 0.99 for x in list(map(sum, class_prediction))):
            for class_i in range(c):
                for class_j in range(c):
                    if class_i != class_j:
                        f = []
                        for i, e in enumerate(label):
                            label_idx = (e == 1).nonzero()[0]
                            if label_idx == class_i:
                                f.append(class_prediction[i][class_i])
                        g = []
                        for j, e in enumerate(label):
                            label_idx = (e == 1).nonzero()[0]
                            if label_idx == class_j:
                                g.append(class_prediction[j][class_i])
                        si = sum([sorted(g + f).index(x) + 1 for x in f])
                        n = len(class_prediction)
                        n_i = len(f)
                        n_j = len(g)
                        if n_i * n_j != 0:
                            A[class_i][class_j] = (si - (n_i * (n_i + 1)) / 2) / (
                                n_i * n_j
                            )
            M = 0
            for class_i in range(c):
                for class_j in range(class_i + 1, c):
                    M += (A[class_i][class_j] + A[class_j][class_i]) / (c * (c - 1))
            return M
        else:
            raise Exception("Check the result is correct")

    def predict(
        self,
        dataset,
        batch_size=32,
    ):
        predictions = []
        checks = []
        for model in self.ensemble_models:
            (
                y,
                y_pred,
            ) = model.predict(dataset, batch_size)
            predictions.append(y_pred)

        ensemble_prediction = self.aggregate_predictions(predictions)
        return y, ensemble_prediction
