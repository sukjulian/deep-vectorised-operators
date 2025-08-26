import os
import re
from glob import glob
from pathlib import Path

import h5py
import meshio
import numpy as np
import torch
import torch_geometric as pyg

from src.transforms import compute_mesh_vertex_normals, scale_to_centered_hypercube


class LeftMainCoronaryBifurcation(pyg.data.Dataset):
    def __init__(self, transform=None, positional_transform=None):
        super().__init__(transform=transform)

        self.num_simulations = 585
        self.num_time_points_per_simulation = 26

        self.num_time_steps_per_simulation = self.num_time_points_per_simulation - 1
        self.num_data_points = self.num_simulations * self.num_time_steps_per_simulation

        self.path_to_root = os.path.abspath("lmcb-dataset")
        self.path_to_hdf5 = glob(os.path.join(self.path_to_root, "raw", "*.hdf5"))[0]

        with h5py.File(self.path_to_hdf5, "r") as hdf5_file:
            self.simulation_names = list(hdf5_file)

        self.fold_size = 12  # cross-validation

        self.pos_cache = {"tets": {}, "face": {}}

        self.simplices_cache = {"tets": {}, "face": {}}

        self.shape_id_cache = {}

        self.tensor_transform_dict = {
            "compute_mesh_vertex_normals": compute_mesh_vertex_normals,
            "scale_to_centered_hypercube": scale_to_centered_hypercube,
            "positional_transform": positional_transform,
        }
        self.transform_cache = {
            "compute_mesh_vertex_normals": {},
            "scale_to_centered_hypercube": {},
            "positional_transform": {},
        }

        self.boundary_idcs_cache = {}

        self.waveform_cache = {}

    @property
    def raw_file_names(self):
        return

    @property
    def processed_file_names(self):
        return

    def len(self):
        return self.num_data_points

    def get(self, idx):
        memory_location = self.idx_to_memory_location(idx)

        with h5py.File(self.path_to_hdf5, "r") as hdf5_file:
            data = self.read_hdf5_data(hdf5_file, *memory_location)

        return data

    def idx_to_memory_location(self, idx):

        memory_location = (
            self.simulation_names[idx // self.num_time_steps_per_simulation],
            idx % self.num_time_steps_per_simulation,
        )

        return memory_location

    def read_hdf5_data(self, hdf5_file, simulation_name, time_step_id):

        time_step_slice = slice(time_step_id, time_step_id + 2)
        velocity_field = hdf5_file[simulation_name]["velocity"][time_step_slice]
        pressure_field = hdf5_file[simulation_name]["pressure"][
            time_step_slice
        ]  # / 1333.22 [mmHg]

        t = hdf5_file[simulation_name]["t"][time_step_slice].astype("f4")

        pos = self.get_pos(hdf5_file, simulation_name, "tets")

        # tets = self.get_simplices(hdf5_file, simulation_name, 'tets')

        shape_id = self.get_shape_id(hdf5_file, simulation_name)

        boundary_idcs = self.get_boundary_idcs(hdf5_file, simulation_name)

        waveform = self.get_waveform(hdf5_file, simulation_name)

        data = pyg.data.Data(
            x=torch.from_numpy(
                np.concatenate((velocity_field[0], pressure_field[0][:, None]), axis=-1)
            ),
            y=torch.from_numpy(
                np.concatenate((velocity_field[1], pressure_field[1][:, None]), axis=-1)
            ),
            t=torch.from_numpy(t),
            pos=torch.from_numpy(pos),
            inlet_index=torch.from_numpy(boundary_idcs["inlet"]),
            lumen_wall_index=torch.from_numpy(boundary_idcs["lumen_wall"]),
            outlets_index=torch.from_numpy(boundary_idcs["outlets"]),
            waveform=torch.from_numpy(waveform),
            heart_rate=torch.tensor(
                60.0 / (len(waveform) * (t[1] - t[0])), dtype=torch.float
            ),  # [bpm]
        )

        if self.tensor_transform_dict["positional_transform"]:
            positional_data = self.disk_cached_transform(
                shape_id,
                "positional_transform",
                pyg.data.Data(
                    pos=data.pos,
                    **{key: value for key, value in data.items() if "index" in key},
                ),
            ).clone()

            data = positional_data.update(data)

        return data

    def get_pos(self, hdf5_file, simulation_name, simplices_name):
        shape_id = self.get_shape_id(hdf5_file, simulation_name)

        if shape_id in self.pos_cache[simplices_name]:
            pos = self.pos_cache[simplices_name][shape_id]

        else:
            pos = hdf5_file[simulation_name][f"pos_{simplices_name}"][()]
            self.pos_cache[simplices_name][shape_id] = pos

        return pos

    def get_simplices(self, hdf5_file, simulation_name, simplices_name):
        shape_id = self.get_shape_id(hdf5_file, simulation_name)

        if shape_id in self.simplices_cache[simplices_name]:
            simplices = self.simplices_cache[simplices_name][shape_id]

        else:
            simplices = hdf5_file[simulation_name][simplices_name][
                ()
            ].T  # transpose to match PyG convention
            self.simplices_cache[simplices_name][shape_id] = simplices

        return simplices

    def get_shape_id(self, hdf5_file, simulation_name):

        if simulation_name in self.shape_id_cache:
            shape_id = self.shape_id_cache[simulation_name]

        else:
            shape_id = hdf5_file[simulation_name].attrs["shape id"]
            self.shape_id_cache[simulation_name] = shape_id

        return shape_id

    def cached_transform(self, shape_id, transform_name, *tensors):

        if shape_id in self.transform_cache[transform_name]:
            transformed_tensor = self.transform_cache[transform_name][shape_id]

        else:
            transformed_tensor = self.tensor_transform_dict[transform_name](*tensors)
            self.transform_cache[transform_name][shape_id] = transformed_tensor

        return transformed_tensor

    def get_boundary_idcs(self, hdf5_file, simulation_name):
        shape_id = self.get_shape_id(hdf5_file, simulation_name)

        if shape_id in self.boundary_idcs_cache:
            boundary_idcs = self.boundary_idcs_cache[shape_id]

        else:
            boundary_idcs = {
                "surface": hdf5_file[simulation_name]["surface_idcs"][()],
                "inlet": hdf5_file[simulation_name]["inlet_idcs"][()],
                "lumen_wall": hdf5_file[simulation_name]["lumen_wall_idcs"][()],
                "outlets": hdf5_file[simulation_name]["outlets_idcs"][()],
            }
            self.boundary_idcs_cache[shape_id] = boundary_idcs

        return boundary_idcs

    def get_waveform(self, hdf5_file, simulation_name):

        if simulation_name in self.waveform_cache:
            waveform = self.waveform_cache[simulation_name]

        else:
            inlet_idcs = self.get_boundary_idcs(hdf5_file, simulation_name)["inlet"]
            waveform = np.linalg.norm(
                hdf5_file[simulation_name]["velocity"][()][:, inlet_idcs].mean(axis=1),
                axis=1,
            )

            self.waveform_cache[simulation_name] = waveform

        return waveform

    def disk_cached_transform(self, shape_id, transform_name, *tensors):
        hash_id = re.sub(
            "<.*?>",
            "function",
            self.tensor_transform_dict["positional_transform"].__repr__(),
        )
        hash_id = "".join([str(ord(char)) for char in hash_id])[
            ::4
        ]  # Linux maximum file name length

        disk_location = os.path.join(
            self.path_to_root, "processed", f"{hash_id}_{shape_id}.pt"
        )

        if shape_id in self.transform_cache[transform_name]:
            transformed_tensor = self.transform_cache[transform_name][shape_id]

        elif os.path.exists(disk_location):
            transformed_tensor = torch.load(disk_location, weights_only=False)
            self.transform_cache[transform_name][shape_id] = transformed_tensor

        else:
            transformed_tensor = self.tensor_transform_dict[transform_name](*tensors)
            self.transform_cache[transform_name][shape_id] = transformed_tensor

            if not os.path.exists(os.path.join(self.path_to_root, "processed")):
                os.makedirs(os.path.join(self.path_to_root, "processed"))

            torch.save(transformed_tensor, disk_location)

        return transformed_tensor

    def get_(self, start, stop):
        return (
            start * 5 * self.num_time_steps_per_simulation,
            stop * 5 * self.num_time_steps_per_simulation,
        )

    def fold_(self, fold_idx):

        evaluation = self.get_(
            fold_idx * self.fold_size, (fold_idx + 1) * self.fold_size
        )

        training_index = torch.ones(self.num_data_points, dtype=torch.bool)
        training_index[slice(*evaluation)] = False

        return {"training_index": training_index, "evaluation": evaluation}

    def get_geometry_for_visualisation(self, idx):
        simulation_name, _ = self.idx_to_memory_location(idx)

        with h5py.File(self.path_to_hdf5, "r") as hdf5_file:
            pos = self.get_pos(hdf5_file, simulation_name, "tets")
            tets = self.get_simplices(hdf5_file, simulation_name, "tets")

        data = pyg.data.Data(pos=torch.from_numpy(pos), tets=torch.from_numpy(tets))

        return data


class CoronaryDeepOperatorLearning(LeftMainCoronaryBifurcation):
    def __init__(self, transform=None, positional_transform=None):
        super().__init__(transform=transform)

        self.path_to_root = os.path.abspath("codol-dataset")
        self.path_to_raw = os.path.join(self.path_to_root, "raw")

        self.simulation_names = sorted(glob(os.path.join(self.path_to_raw, "*")))

        self.time_point_names = [
            sorted(glob(os.path.join(key, "sol_*.pt"))) for key in self.simulation_names
        ]
        self.time_step_names = [
            time_point_names[:-1] for time_point_names in self.time_point_names
        ]

        self._idx_to_memory_location = [
            name for names in self.time_step_names for name in names
        ]

        self.num_data_points = len(self._idx_to_memory_location)

        self.fold_size = 8  # cross-validation

        # Cache
        self.mesh_cache = {}

        self.tensor_transform_dict = {"positional_transform": positional_transform}
        self.transform_cache = {"positional_transform": {}}

        self.steady_state_solution_cache = {}

    def get(self, idx):
        return self.read_data(self.idx_to_memory_location(idx))

    def idx_to_memory_location(self, idx):
        return self._idx_to_memory_location[idx]

    def read_data(self, time_step_name):
        simulation_name = Path(time_step_name).parents[0]
        time_step_id = Path(time_step_name).stem[4:7]

        x = torch.load(time_step_name).float().roll(3, dims=1)
        y = (
            torch.load(
                time_step_name.replace(
                    f"sol_{time_step_id}", f"sol_{int(time_step_id) + 1:03d}"
                )
            )
            .float()
            .roll(3, dims=1)
        )

        dt = 0.025  # [s]

        pos, tets = self.get_mesh(simulation_name)

        boundary_idcs = self.get_boundary_idcs(simulation_name)

        scale_factors = torch.tensor((*[1e-1] * 3, 1e1))  # (velocity, pressure)
        waveform = self.get_waveform(simulation_name) * scale_factors[:3].mean()

        data = pyg.data.Data(
            x=x * scale_factors,
            y=y * scale_factors,
            t=torch.tensor(((int(time_step_id) * dt, (int(time_step_id) + 1) * dt))),
            pos=pos,
            inlet_index=boundary_idcs["inlet"],
            lumen_wall_index=boundary_idcs["lumen_wall"],
            outlets_index=boundary_idcs["outlets"],
            waveform=waveform * 1e-1,
            heart_rate=torch.tensor(60.0 / (len(waveform) * dt)),  # [bpm]
            steady_state_solution=self.get_steady_state_solution(simulation_name)
            * scale_factors,
        )

        if self.tensor_transform_dict["positional_transform"]:
            positional_data = self.disk_cached_transform(
                self.get_shape_id(simulation_name),
                "positional_transform",
                pyg.data.Data(
                    pos=data.pos,
                    tets=tets.T,
                    **{key: value for key, value in data.items() if "index" in key},
                ),
            ).clone()

            data = data.update(positional_data)

        return data

    def get_mesh(self, simulation_name):
        shape_id = self.get_shape_id(simulation_name)

        if shape_id in self.mesh_cache:
            pos, tets = self.mesh_cache[shape_id]

        else:
            meshio_mesh = meshio.read(
                os.path.join(self.path_to_raw, simulation_name, "mesh.vtu")
            )
            pos = torch.from_numpy(meshio_mesh.points.astype("f4"))
            tets = torch.from_numpy(meshio_mesh.cells_dict["tetra"].astype("i4"))

            pos *= 1e-1  # [mm] to [cm]
            pos -= pos.mean(dim=0)

            self.mesh_cache[shape_id] = (pos, tets)

        return pos, tets

    def get_shape_id(self, simulation_name):
        return Path(simulation_name).stem[:6]

    def get_boundary_idcs(self, simulation_name):
        shape_id = self.get_shape_id(simulation_name)

        if shape_id in self.boundary_idcs_cache:
            boundary_idcs = self.boundary_idcs_cache[shape_id]

        else:
            boundary_id = torch.load(
                os.path.join(self.path_to_raw, simulation_name, "mesh_ids.pt")
            )

            boundary_idcs = {
                "inlet": torch.where(boundary_id == 1)[0].int(),
                "lumen_wall": torch.where(boundary_id == 0)[0].int(),
                "outlets": torch.where(boundary_id > 1)[0].int(),
            }

            self.boundary_idcs_cache[shape_id] = boundary_idcs

        return boundary_idcs

    def get_waveform(self, simulation_name):

        if simulation_name in self.waveform_cache:
            waveform = self.waveform_cache[simulation_name]

        else:
            inlet_idcs = self.get_boundary_idcs(simulation_name)["inlet"]

            velocity = []
            for path_to_pt in glob(os.path.join(simulation_name, "sol_*.pt")):
                velocity.append(torch.load(path_to_pt)[:, 1:])

            waveform = torch.stack(velocity)[:, inlet_idcs].mean(dim=1).norm(dim=1)

            self.waveform_cache[simulation_name] = waveform

        return waveform

    def get_steady_state_solution(self, simulation_name):

        if simulation_name in self.steady_state_solution_cache:
            steady_state_solution = self.steady_state_solution_cache[simulation_name]

        else:
            steady_state_solution = (
                torch.load(os.path.join(simulation_name, "steady_state_sol.pt"))
                .float()
                .roll(3, dims=1)
            )
            self.steady_state_solution_cache[simulation_name] = steady_state_solution

        return steady_state_solution

    def get_(self, start, stop):

        start = len([name for names in self.time_step_names[:start] for name in names])
        stop = len([name for names in self.time_step_names[:stop] for name in names])

        return start, stop

    @property
    def splits(self):

        left = [name for name in self.simulation_names if "SX_" in name]
        right = [name for name in self.simulation_names if "DX_" in name]

        splits = {
            "training": [*left[:36], *right[:19]],
            "validation": [*left[36:38], *right[19:21]],
            "test": [*left[38:], *right[21:]],
        }

        # Cross-check patient identifiers
        ids = {
            key: set([name[-7:-4] for name in value]) for key, value in splits.items()
        }
        assert len(set((*ids["training"], *ids["test"]))) == len(ids["training"]) + len(
            ids["test"]
        ), "Data leakage detected."

        for key, value in splits.items():
            splits[key] = torch.repeat_interleave(
                torch.from_numpy(
                    (
                        np.array(value)[:, None]
                        == np.array(self.simulation_names)[None, :]
                    ).any(axis=0)
                ),
                torch.tensor([len(names) for names in self.time_step_names]),
            )

        return splits

    def get_geometry_for_visualisation(self, idx):
        simulation_name = Path(self.idx_to_memory_location(idx)).parents[0]

        pos, tets = self.get_mesh(simulation_name)

        data = pyg.data.Data(pos=pos, tets=tets.T)

        return data
