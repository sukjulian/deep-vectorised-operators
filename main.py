import os
import statistics
import sys
import warnings
from argparse import ArgumentParser
from time import asctime

import potpourri3d as pp3d
import torch
import torch_geometric as pyg
from gatr.interface import (
    embed_oriented_plane,
    embed_point,
    extract_oriented_plane,
    extract_scalar,
)
from lab_gatr.models import LaBGATr, LaBVaTr
from lab_gatr.transforms import PointCloudPoolingScales
from torch.fft import rfft
from torch.nn.functional import interpolate
from torch.nn.parallel import DistributedDataParallel
from torch_cluster import knn
from torch_dvf.models import PointNet
from torch_dvf.transforms import RadiusPointCloudHierarchy
from torch_geometric.transforms import Compose
from tqdm import tqdm

import wandb_impostor as wandb
from src.datasets import CoronaryDeepOperatorLearning
from src.nn.models import MLP
from src.transforms import PointCloudSampling, compute_gradient
from src.utils import AccuracyAnalysis

parser = ArgumentParser()

parser.add_argument("--model_id", type=str, required=True)

parser.add_argument("--num_epochs", type=int, default=0)
parser.add_argument("--run_id", type=str, default=None)
parser.add_argument("--fold_idx", type=int, default=0)
parser.add_argument("--num_gpus", type=int, default=1)

args = parser.parse_args()


# All aboard the hype train (Weights & Biases)
per_gpu_batch_size = {
    "mlp": 4,
    "pointnet": 4,
    "lab_vatr": 4,
    "lab_gatr": 1,
}
wandb_config = {
    "batch_size": {"accumulated": 12, "per_gpu": per_gpu_batch_size[args.model_id]},
    "learning_rate": 3e-4,  # best learning rate for Adam, hands down
    "num_epochs": args.num_epochs,
    "loss_term_factors": {"velocity": 1.0, "pressure": 3e-1, "flux": None},
    "lr_decay_gamma": 0.9955,  # currently unused
}

model_is_neural_operator = (
    False  # misnomer: "neural operator" refers to a time-stepping scheme here
)

len_waveform_fft = 0  # can be zero

pressure_drop = True if not model_is_neural_operator else False

use_steady_state_solution = True

warnings.simplefilter(action="ignore", category=UserWarning)


def main(rank, num_gpus):
    ddp_setup(
        rank,
        num_gpus,
        project_name="coronary_deep_operator_learning",
        wandb_config=wandb_config,
        run_id=f"codol{'-' if args.run_id else ''}{args.run_id or ''}",
    )

    dataset = CoronaryDeepOperatorLearning
    if "pointnet" in args.model_id:
        positional_transform = RadiusPointCloudHierarchy(
            (0.0033, 0.33443333, 0.66556667, 0.9967), interp_simplex="tetrahedron"
        )
    elif "lab" in args.model_id:
        positional_transform = PointCloudPoolingScales(
            (0.0033,), interp_simplex="tetrahedron"
        )
    else:
        positional_transform = torch.nn.Identity()
    dataset = dataset(
        transform=input_transform,
        positional_transform=Compose(
            (
                # PointCloudSampling(ratio_volume_samples=1.),
                positional_transform,
                positional_encoding,
            )
        ),
    )

    training_data_loader = pyg.loader.DataLoader(
        # split_dataset_for_gpus(dataset[dataset.fold_(args.fold_idx)['training_index']], num_gpus)[rank],
        split_dataset_for_gpus(dataset[dataset.splits["training"]], num_gpus)[rank],
        batch_size=wandb.config["batch_size"]["per_gpu"],
        shuffle=True,
    )
    validation_data_loader = pyg.loader.DataLoader(
        # split_dataset_for_gpus(dataset[~dataset.fold_(args.fold_idx)['training_index']], num_gpus)[rank],
        split_dataset_for_gpus(dataset[dataset.splits["validation"]], num_gpus)[rank],
        batch_size=wandb.config["batch_size"]["per_gpu"],
        shuffle=True,
    )
    # test_data_slice = slice(*dataset.fold_(args.fold_idx)['evaluation'])
    test_data_slice = dataset.splits["test"]
    # visualisation_data_range = range(*dataset.fold_(args.fold_idx)['evaluation'])[:2]
    visualisation_data_range = dataset.splits["test"].nonzero().squeeze()[:2]

    match args.model_id:
        case "mlp":
            neural_network = MLP(
                num_input_channels=4
                * (model_is_neural_operator + use_steady_state_solution)
                + 31
                + len_waveform_fft,
                num_output_channels=4,
                num_layers=8,
                num_latent_channels=323,
            )
        case "pointnet":
            neural_network = PointNet(
                num_input_channels=4
                * (model_is_neural_operator + use_steady_state_solution)
                + 31
                + len_waveform_fft,
                num_output_channels=4,
                num_hierarchies=len(positional_transform.rel_sampling_ratios),
                num_latent_channels=172,
                use_running_stats_in_norm=True,
            )
        case "lab_vatr":
            neural_network = LaBVaTr(
                num_input_channels=4
                * (model_is_neural_operator + use_steady_state_solution)
                + 31
                + len_waveform_fft,
                num_output_channels=4,
                d_model=98,  # 112
                num_blocks=6,
                num_attn_heads=4,
                pooling_mode="message_passing",  # 'cross_attention'
            )
        case "lab_gatr":
            neural_network = LaBGATr(
                GeometricAlgebraInterface,
                d_model=16,
                num_blocks=6,
                num_attn_heads=4,
                pooling_mode="cross_attention",
            )

    training_device = torch.device(f"cuda:{rank}")
    neural_network.to(training_device)

    working_directory = working_directory = (
        f"codol{'-' if args.run_id else ''}{args.run_id or ''}"
    )
    load_neural_network_weights(neural_network, working_directory)

    # Distributed data parallel (multi-GPU training)
    neural_network = ddp_module(neural_network, rank)

    loss_function = torch.nn.L1Loss()
    optimiser = torch.optim.Adam(
        neural_network.parameters(), lr=wandb.config["learning_rate"]
    )

    load_optimiser_state(rank, optimiser, working_directory)

    # Optimisation loop
    wandb.watch(neural_network)
    optimisation_loop(
        rank,
        {
            "neural_network": neural_network,
            "training_device": training_device,
            "optimiser": optimiser,
            "loss_function": loss_function,
            "training_data_loader": training_data_loader,
            "validation_data_loader": validation_data_loader,
            "working_directory": working_directory,
        },
    )

    ddp_rank_zero(
        assessment_loop,
        {
            "neural_network": neural_network,
            "training_device": training_device,
            "dataset": dataset,
            "test_data_slice": test_data_slice,
            "visualisation_data_range": visualisation_data_range,
            "working_directory": working_directory,
        },
    )

    ddp_cleanup()


@torch.no_grad()
def input_transform(data):
    scale_factors = torch.tensor((*[1e-1] * 3, 1e-5))  # (velocity, pressure)

    data.x *= scale_factors.to(data.x.device)
    data.y *= scale_factors.to(data.y.device)

    data.waveform *= 1e-1
    data.heart_rate *= 1e-2

    if model_is_neural_operator:
        data.y -= data.x  # estimation target
        x = data.x

    if hasattr(data, "pos_deprecated"):
        data.inlet_index = data.inlet_index_deprecated  # for BCT encoding

    data.x = torch.cat(
        (
            bct_encoding(data),
            temporal_encoding(data) * 1e1,
            data.positional_encoding.to(data.x.device),
            waveform_encoding(data),
        ),
        dim=-1,
    )

    if hasattr(data, "pos_deprecated"):
        data.y = PointCloudSampling.interpolate(data.pos_deprecated, data.pos, data.y)
        data.steady_state_solution = PointCloudSampling.interpolate(
            data.pos_deprecated, data.pos, data.steady_state_solution
        )

    if model_is_neural_operator:
        data.x = torch.cat((x, data.x), dim=1)  # "x" must be interpolated as well

    elif pressure_drop:
        data.y[:, 3] -= data.x[:, 4]

    if use_steady_state_solution:
        data.steady_state_solution *= scale_factors.to(
            data.steady_state_solution.device
        )
        data.steady_state_solution[:, 3] -= data.x[:, 4] if pressure_drop else 0.0

        data.x = torch.cat((data.steady_state_solution, data.x), dim=1)

    return data


def bct_encoding(data):
    velocity, pressure = data.y.split((3, 1), dim=1)

    # Inlet velocity profile and pressure can be easily measured in the clinic
    mean_velocity_inlet = velocity[data.inlet_index].mean(dim=0)

    bct_encoding = torch.cat(
        (
            (mean_velocity_inlet / mean_velocity_inlet.norm()).expand(
                data.pos.size(0), 3
            ),
            mean_velocity_inlet.norm().expand(data.pos.size(0), 1),
            pressure[data.inlet_index].mean().expand(data.pos.size(0), 1),
        ),
        dim=1,
    )

    return bct_encoding


def temporal_encoding(data):

    if model_is_neural_operator:
        temporal_encoding = data.t[1] - data.t[0]
    else:
        temporal_encoding = data.t[1] % (6e-1 / data.heart_rate)  # periodicity

    return temporal_encoding.expand(data.pos.size(0), 1)


def waveform_encoding(data):

    waveform_encoding = torch.stack(
        (
            data.waveform.mean(),
            data.waveform.std(),
            data.waveform.min(),
            data.waveform.max(),
            data.heart_rate,
        )
    )

    if len_waveform_fft:
        waveform_encoding = torch.cat(
            (
                waveform_encoding,
                rfft(
                    interpolate(
                        data.waveform.view(1, 1, -1),
                        size=len_waveform_fft * 2 - 1,
                        mode="linear",
                        align_corners=True,
                    ).squeeze()
                ).real,
            )
        )

    return waveform_encoding.expand(data.pos.size(0), waveform_encoding.numel())


def positional_encoding(data):

    vectors_to = {
        key: data.pos[value.long()] - data.pos
        for key, value in compute_nearest_boundary_vertex(data).items()
    }
    distances_to = {
        key: torch.linalg.norm(value, dim=-1, keepdim=True)
        for key, value in vectors_to.items()
    }

    diffusion_distances_to = {
        key: value.view(-1, 1)
        for key, value in compute_diffusion_distances(data).items()
    }

    diffusion_vectors_to = {}
    for key, value in diffusion_distances_to.items():
        diffusion_vectors_to[key] = compute_gradient(
            data.pos, value.squeeze(), num_neighbours=64
        )

    data.positional_encoding = torch.cat(
        (
            vectors_to["inlet"] / torch.clamp(distances_to["inlet"], min=1e-16),
            vectors_to["lumen_wall"]
            / torch.clamp(distances_to["lumen_wall"], min=1e-16),
            vectors_to["outlets"] / torch.clamp(distances_to["outlets"], min=1e-16),
            distances_to["inlet"],
            distances_to["lumen_wall"],
            distances_to["outlets"],
            diffusion_vectors_to["inlet"],
            diffusion_vectors_to["outlets"],
            diffusion_distances_to["inlet"],
            diffusion_distances_to["outlets"],
        ),
        dim=1,
    )

    delattr(data, "tets")

    return data


def compute_nearest_boundary_vertex(data):
    index_dict = {}

    for key in ("inlet", "lumen_wall", "outlets"):
        index_dict[key] = data[f"{key}_index"][
            knn(data.pos[data[f"{key}_index"].long()], data.pos, k=1)[1].long()
        ]

    return index_dict


def compute_diffusion_distances(data):
    solver = pp3d.PointCloudHeatSolver(data.pos)

    diffusion_distances_to = {}
    for key in ("inlet", "outlets"):
        diffusion_distances_to[key] = torch.from_numpy(
            solver.compute_distance_multisource(data[f"{key}_index"]).astype("f4")
        )

    return diffusion_distances_to


def split_dataset_for_gpus(dataset, num_gpus):
    num_samples = len(dataset)

    per_gpu = int(num_samples / num_gpus)
    first_and_last_idx_per_gpu = tuple(
        zip(
            range(0, num_samples - per_gpu + 1, per_gpu),
            range(per_gpu, num_samples + 1, per_gpu),
        )
    )

    return [dataset[slice(*idcs)] for idcs in first_and_last_idx_per_gpu]


class GeometricAlgebraInterface:
    num_input_channels = 1 + model_is_neural_operator + use_steady_state_solution + 6
    num_output_channels = 2

    num_input_scalars = (
        model_is_neural_operator + use_steady_state_solution + 13 + len_waveform_fft
    )
    num_output_scalars = None

    @staticmethod
    @torch.no_grad()
    def embed(data):

        idx = 4 * (model_is_neural_operator + use_steady_state_solution)
        multivectors = torch.cat(
            (
                embed_point(data.pos).view(-1, 1, 16),
                embed_oriented_plane(data.x[:, slice(idx, idx + 3)], data.pos).view(
                    -1, 1, 16
                ),  # mean inlet velocity direction
                embed_oriented_plane(data.x[:, slice(idx + 6, idx + 9)], data.pos).view(
                    -1, 1, 16
                ),  # direction to inlet
                embed_oriented_plane(
                    data.x[:, slice(idx + 9, idx + 12)], data.pos
                ).view(
                    -1, 1, 16
                ),  # direction to lumen wall
                embed_oriented_plane(
                    data.x[:, slice(idx + 12, idx + 15)], data.pos
                ).view(
                    -1, 1, 16
                ),  # direction to outlets
                embed_oriented_plane(
                    data.x[:, slice(idx + 18, idx + 21)], data.pos
                ).view(
                    -1, 1, 16
                ),  # diffusion direction to inlet
                embed_oriented_plane(
                    data.x[:, slice(idx + 21, idx + 24)], data.pos
                ).view(
                    -1, 1, 16
                ),  # diffusion direction to outlets
            ),
            dim=1,
        )
        scalars = torch.cat(
            (
                data.x[
                    :, slice(idx + 3, idx + 6)
                ],  # mean inlet velocity magnitude, mean inlet pressure, (relative) time
                data.x[
                    :, slice(idx + 15, idx + 18)
                ],  # distance to inlet, lumen wall and outlets
                data.x[
                    :, slice(idx + 24, idx + 26)
                ],  # diffusion distance to inlet and outlets
                data.x[
                    :, slice(idx + 26, idx + 31 + len_waveform_fft)
                ],  # mean, st. dev., min. and max. waveform, heart rate(, FFT)
            ),
            dim=1,
        )

        if model_is_neural_operator:
            idx = 4 * use_steady_state_solution
            multivectors = torch.cat(
                (
                    multivectors[:, :1],
                    embed_oriented_plane(data.x[:, slice(idx, idx + 3)], data.pos).view(
                        -1, 1, 16
                    ),  # input hemodynamics
                    multivectors[:, 1:],
                ),
                dim=1,
            )
            scalars = torch.cat((data.x[:, slice(idx + 3, idx + 4)], scalars), dim=1)

        if use_steady_state_solution:
            multivectors = torch.cat(
                (
                    multivectors[:, :1],
                    embed_oriented_plane(data.x[:, :3], data.pos).view(
                        -1, 1, 16
                    ),  # steady-state solution
                    multivectors[:, 1:],
                ),
                dim=1,
            )
            scalars = torch.cat((data.x[:, 3:4], scalars), dim=1)

        return multivectors, scalars

    @staticmethod
    def dislodge(multivectors, scalars):
        velocity = (
            extract_scalar(multivectors[:, 0]).view(-1, 1)
            * extract_oriented_plane(multivectors[:, 0]).squeeze()
        )
        pressure = extract_scalar(multivectors[:, 1]).view(-1, 1)

        return torch.cat((velocity, pressure), dim=1)


def load_neural_network_weights(neural_network, working_directory=""):
    if os.path.exists(os.path.join(working_directory, "trained_parameters.pt")):

        neural_network.load_state_dict(
            torch.load(os.path.join(working_directory, "trained_parameters.pt"))
        )
        print("Resuming from pre-trained parameters.")


def load_optimiser_state(rank, optimiser, working_directory=""):
    if os.path.exists(
        os.path.join(working_directory, f"rank_{rank}_optimiser_state.pt")
    ):

        optimiser.load_state_dict(
            torch.load(
                os.path.join(working_directory, f"rank_{rank}_optimiser_state.pt")
            )
        )
        print("Resuming from previous optimiser state.")


def optimisation_loop(rank, config):
    assert (
        wandb.config["batch_size"]["accumulated"]
        % wandb.config["batch_size"]["per_gpu"]
        == 0
    ), "Gradient accumulation mismatch."

    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=config['optimiser'], gamma=wandb.config['lr_decay_gamma'])

    for epoch in tqdm(
        range(wandb.config["num_epochs"]), desc="Epochs", position=0, leave=True
    ):

        loss_data = {
            key: {
                "velocity": [],
                "pressure": [],
            }
            for key in ("training", "validation")
        }

        # Objective convergence
        config["neural_network"].train()

        for i, batch in enumerate(
            tqdm(
                config["training_data_loader"],
                desc="Training split",
                position=1,
                leave=False,
            ),
            start=1,
        ):

            batch = batch.to(config["training_device"])
            with warnings.catch_warnings():
                prediction = config["neural_network"](batch)

            loss_terms = {
                "velocity": config["loss_function"](prediction[:, :3], batch.y[:, :3]),
                "pressure": config["loss_function"](prediction[:, -1], batch.y[:, -1]),
            }
            active_loss_terms = {
                key: value
                for key, value in loss_terms.items()
                if wandb.config["loss_term_factors"][key]
            }
            loss_value = sum(
                (
                    wandb.config["loss_term_factors"][key] * value
                    for key, value in active_loss_terms.items()
                )
            )

            for key, value in loss_terms.items():
                loss_data["training"][key].append(value.item())

            loss_value /= (
                wandb.config["batch_size"]["accumulated"]
                // wandb.config["batch_size"]["per_gpu"]
            )

            loss_value.backward()  # "autograd" hook fires and triggers gradient synchronisation across processes
            # torch.nn.utils.clip_grad_norm_(config['neural_network'].parameters(), max_norm=1.0, error_if_nonfinite=True)

            # Gradient accumulation
            gradient_is_complete = (
                i
                * wandb.config["batch_size"]["per_gpu"]
                % wandb.config["batch_size"]["accumulated"]
                == 0
            )
            if gradient_is_complete or i == len(config["training_data_loader"]):

                config["optimiser"].step()
                config["optimiser"].zero_grad()

            del batch, prediction

        # scheduler.step()

        ddp_rank_zero(
            save_neural_network_weights,
            config["neural_network"],
            config["working_directory"],
        )
        torch.save(
            config["optimiser"].state_dict(),
            os.path.join(
                config["working_directory"], f"rank_{rank}_optimiser_state.pt"
            ),
        )

        # Learning task
        config["neural_network"].eval()

        with torch.no_grad():
            for batch in tqdm(
                config["validation_data_loader"],
                desc="Validation split",
                position=1,
                leave=False,
            ):

                batch = batch.to(config["training_device"])
                prediction = config["neural_network"](batch)

                loss_terms = {
                    "velocity": config["loss_function"](
                        prediction[:, :3], batch.y[:, :3]
                    ),
                    "pressure": config["loss_function"](
                        prediction[:, -1], batch.y[:, -1]
                    ),
                }

                for key, value in loss_terms.items():
                    loss_data["validation"][key].append(value.item())

                del batch, prediction

        for phase_name in loss_data.keys():
            wandb.log(
                {
                    phase_name: {
                        key: statistics.mean(value)
                        for key, value in loss_data[phase_name].items()
                    }
                }
            )


def save_neural_network_weights(neural_network, working_directory="", file_name=None):

    if isinstance(neural_network, DistributedDataParallel):
        neural_network_weights = neural_network.module.state_dict()
    else:
        neural_network_weights = neural_network.state_dict()

    if working_directory and not os.path.exists(working_directory):
        os.makedirs(working_directory)

    torch.save(
        neural_network_weights,
        os.path.join(working_directory, file_name or "trained_parameters.pt"),
    )


def assessment_loop(config):
    accuracy_analysis = {
        "hemodynamics": AccuracyAnalysis(),
        "velocity": AccuracyAnalysis(),
        "pressure": AccuracyAnalysis(),
    }

    config["neural_network"].eval()

    with torch.no_grad():

        # Quantitative
        for i, data in enumerate(
            tqdm(
                config["dataset"][config["test_data_slice"]],
                desc="Test split",
                position=0,
                leave=False,
            )
        ):

            data = data.to(config["training_device"])
            prediction = config["neural_network"](data)

            # Estimation target
            if model_is_neural_operator:
                prediction += data.x[:, :4]
                data.y += data.x[:, :4]

            elif pressure_drop:
                prediction[:, 3] += data.x[:, 4 * use_steady_state_solution + 4]
                data.y[:, 3] += data.x[:, 4 * use_steady_state_solution + 4]

            accuracy_analysis["hemodynamics"].append_values(
                {
                    "ground_truth": data.y.cpu(),
                    "prediction": prediction.cpu(),
                    "scatter_idx": torch.tensor(i),
                }
            )
            accuracy_analysis["velocity"].append_values(
                {
                    "ground_truth": data.y[:, :3].cpu(),
                    "prediction": prediction[:, :3].cpu(),
                    "scatter_idx": torch.tensor(i),
                }
            )
            accuracy_analysis["pressure"].append_values(
                {
                    "ground_truth": data.y[:, 3:4].cpu(),
                    "prediction": prediction[:, 3:4].cpu(),
                    "scatter_idx": torch.tensor(i),
                }
            )

            del data

        print(f"Hemodynamics\n{accuracy_analysis['hemodynamics'].accuracy_table()}")
        print(f"Velocity\n{accuracy_analysis['velocity'].accuracy_table()}")
        print(f"Pressure\n{accuracy_analysis['pressure'].accuracy_table()}")


def ddp_setup(rank, num_gpus, project_name, wandb_config, run_id=None):

    if num_gpus > 1:

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        sys.stderr = open(
            f"{project_name}{'_' if run_id else ''}{run_id or ''}_{rank}.out", "w"
        )  # used by "tqdm"

        torch.distributed.init_process_group("nccl", rank=rank, world_size=num_gpus)
        wandb.init(
            project=project_name,
            config=wandb_config,
            group=f"{run_id or 'DDP'} ({asctime()})",
        )

    else:
        wandb.init(project=project_name, config=wandb_config, name=run_id)


def ddp_module(torch_module, rank):

    if torch.distributed.is_initialized():
        torch_module = DistributedDataParallel(
            torch_module,
            device_ids=[rank],
            find_unused_parameters=isinstance(torch_module, LaBGATr),
        )

    return torch_module


def ddp_rank_zero(fun, *args, **kwargs):

    if torch.distributed.is_initialized():
        fun(*args, **kwargs) if torch.distributed.get_rank() == 0 else None

        torch.distributed.barrier()  # synchronises all processes

    else:
        fun(*args, **kwargs)


def ddp_cleanup():

    wandb.finish()
    (
        torch.distributed.destroy_process_group()
        if torch.distributed.is_initialized()
        else None
    )

    (
        sys.stderr.close() if torch.distributed.is_initialized() else None
    )  # last executed statement


def ddp(fun, num_gpus):
    (
        torch.multiprocessing.spawn(fun, args=(num_gpus,), nprocs=num_gpus, join=True)
        if num_gpus > 1
        else fun(rank=0, num_gpus=num_gpus)
    )


if __name__ == "__main__":
    ddp(main, args.num_gpus)
