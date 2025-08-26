import torch
from prettytable import PrettyTable
from torch_scatter import scatter


class AccuracyAnalysis:
    def __init__(self):

        self.values_dict = {"ground_truth": [], "prediction": [], "scatter_idx": []}

    def append_values(self, value_dict):

        for key, value in value_dict.items():
            if key in self.values_dict:

                if key == "scatter_idx":
                    self.values_dict[key].append(
                        value.expand(value_dict["prediction"].size(0))
                    )

                else:
                    self.values_dict[key].append(value)

    def lists_to_tensors(self):
        self.values_dict = {
            key: torch.cat(value, dim=0)
            for key, value in self.values_dict.items()
            if value
        }

    def get_nmae(self):

        mae = scatter(
            torch.linalg.norm(
                self.values_dict["ground_truth"] - self.values_dict["prediction"],
                dim=-1,
            ),
            self.values_dict["scatter_idx"],
            dim=0,
            reduce="mean",
        )

        return mae / torch.max(
            torch.linalg.norm(self.values_dict["ground_truth"], dim=-1)
        )

    def get_approximation_error(self):

        approximation_error = torch.sqrt(
            scatter(
                torch.linalg.norm(
                    self.values_dict["ground_truth"] - self.values_dict["prediction"],
                    dim=-1,
                )
                ** 2,
                self.values_dict["scatter_idx"],
                dim=0,
                reduce="sum",
            )
            / scatter(
                torch.linalg.norm(self.values_dict["ground_truth"], dim=-1) ** 2,
                self.values_dict["scatter_idx"],
                dim=0,
                reduce="sum",
            )
        )

        return approximation_error

    def get_mean_cosine_similarity(self):

        cosine_similarity = torch.nn.CosineSimilarity(dim=-1).forward(
            self.values_dict["ground_truth"], self.values_dict["prediction"]
        )

        mean_cosine_similarity = scatter(
            cosine_similarity, self.values_dict["scatter_idx"], dim=0, reduce="mean"
        )

        return mean_cosine_similarity

    def get_residual_mae(self):

        mae = scatter(
            torch.linalg.norm(self.values_dict["prediction"], dim=-1),
            self.values_dict["scatter_idx"],
            dim=0,
            reduce="mean",
        )

        return mae

    def accuracy_table(self):

        self.lists_to_tensors()

        nmae = self.get_nmae()
        approximation_error = self.get_approximation_error()
        mean_cosine_similarity = self.get_mean_cosine_similarity()

        """
        This is really slow for scalar prediction targets due to the PyTorch implementation of norm.
        """

        table = PrettyTable(["Metric", "Mean", "Standard Deviation"])

        table.add_row(
            [
                "NMAE",
                "{0:.1%}".format(torch.mean(nmae).item()),
                "{0:.1%}".format(torch.std(nmae).item()),
            ]
        )

        table.add_row(
            [
                "Approximation Error",
                "{0:.1%}".format(torch.mean(approximation_error).item()),
                "{0:.1%}".format(torch.std(approximation_error).item()),
            ]
        )

        table.add_row(
            [
                "Mean Cosine Similarity",
                "{:.2f}".format(torch.mean(mean_cosine_similarity).item()),
                "{:.2f}".format(torch.std(mean_cosine_similarity).item()),
            ]
        )

        return table

    def residual_table(self):

        self.lists_to_tensors()

        mae = self.get_residual_mae()

        table = PrettyTable(["Metric", "Mean", "Standard Deviation"])

        table.add_row(
            [
                "MAE",
                "{0:.1f}".format(torch.mean(mae).item()),
                "{0:.1f}".format(torch.std(mae).item()),
            ]
        )

        return table
