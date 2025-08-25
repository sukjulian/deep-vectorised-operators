import numpy as np
import torch
import torch_geometric as pyg
import trimesh
from torch.nn.functional import normalize
from torch_cluster import knn, radius_graph
from torch_scatter import scatter_add, scatter_mean
from trimesh import Trimesh


def tets_to_face(tets):

    # External and internal triangles
    face = torch.cat(
        (tets[[0, 1, 2]], tets[[0, 1, 3]], tets[[0, 2, 3]], tets[[1, 2, 3]]), dim=-1
    )

    # Internal triangles have duplicates (opposite winding)
    _, idcs, counts = torch.unique(
        torch.sort(face, dim=0)[0], return_inverse=True, return_counts=True, dim=-1
    )
    face = face[:, counts[idcs] == 1]

    # Coalesce vertices and produce volume-to-surface mapping
    surface_idcs, unique_inverse = torch.unique(face, return_inverse=True)
    face = unique_inverse.reshape(face.shape).int()

    return face, surface_idcs.int()


def volume_to_surface(volume_idcs, surface_idcs):
    return (
        torch.nonzero(torch.any(surface_idcs[:, None] == volume_idcs[None, :], dim=-1))
        .squeeze()
        .int()
    )


def compute_mesh_vertex_normals(pos, face):
    long_face = face.long()

    face_normals = normalize(
        torch.cross(
            pos[long_face[1]] - pos[long_face[0]], pos[long_face[2]] - pos[long_face[0]]
        ),
        p=2,
        dim=-1,
    )

    norm = scatter_add(
        face_normals.repeat(3, 1),
        torch.cat([long_face[0], long_face[1], long_face[2]], dim=0),
        dim=0,
        dim_size=pos.size(0),
    )

    return normalize(norm, p=2, dim=-1)


def scale_to_centered_hypercube(pos):

    pos = pos - torch.mean(pos, dim=0, keepdim=True)
    pos = pos / torch.max(torch.abs(pos))

    return pos


def compute_gradient(pos, field, num_neighbours=14):

    # Radius for average number of neighbours
    target_idcs, source_idcs = knn(pos, pos, k=num_neighbours)
    # radius = (pos[source_idcs] - pos[target_idcs]).norm(dim=1).quantile(0.75).item()  # limited to ca. 16 mio. edges
    radius = np.quantile(
        (pos[source_idcs] - pos[target_idcs]).norm(dim=1).numpy(), q=0.75
    ).item()

    edge_index = radius_graph(pos, radius, max_num_neighbors=pos.size(0))

    field_diff = field[edge_index[1]] - field[edge_index[0]]
    pos_diff = pos[edge_index[1]] - pos[edge_index[0]]

    pos_distance = pos_diff.norm(dim=1, keepdim=True).clamp(min=1e-16)

    return scatter_mean(
        (field_diff.view(-1, 1) / pos_distance) * (pos_diff / pos_distance),
        edge_index[0],
        dim=0,
    )


class PointCloudSampling:
    def __init__(self, ratio_volume_samples: int):
        self.ratio_volume_samples = ratio_volume_samples

    def __call__(self, data: pyg.data.Data) -> pyg.data.Data:
        face, surface_idcs = tets_to_face(data.tets)
        n = data.num_nodes

        mesh = Trimesh(data.pos[surface_idcs], face.T)
        mesh.ray = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh, scale_to_box=False)

        pos_surface = trimesh.sample.sample_surface_even(mesh, surface_idcs.numel())[
            0
        ].astype("f4")
        pos_volume = trimesh.sample.volume_mesh(
            mesh, int((self.ratio_volume_samples * n - surface_idcs.numel()) * 1e2)
        ).astype("f4")

        data = self.project_idcs(data, torch.from_numpy(pos_surface))
        pos = torch.from_numpy(
            np.concatenate((pos_surface, pos_volume), axis=0)
        )  # indices refer to "pos_surface"

        data.pos_deprecated = data.pos
        data.pos = pos

        return data

    @staticmethod
    def project_idcs(data: pyg.data.Data, pos: torch.Tensor) -> pyg.data.Data:
        _, source_idcs = knn(data.pos, pos, k=1)

        for key in data.keys():
            if "_index" in key:

                idcs_mask = torch.any(
                    source_idcs.view(-1, 1) == data[key].view(1, -1), dim=-1
                )
                data[f"{key}_deprecated"] = data[key]
                data[key] = torch.nonzero(idcs_mask).squeeze().int()

        return data

    @staticmethod
    def interpolate(
        pos_source: torch.Tensor, pos_target: torch.Tensor, field: torch.Tensor
    ) -> torch.Tensor:
        interp_target, interp_source = knn(pos_source, pos_target, k=4)

        pos_diff = pos_source[interp_source] - pos_target[interp_target]
        squared_pos_dist = torch.clamp(
            torch.sum(pos_diff**2, dim=-1, keepdim=True), min=1e-16
        )

        numerator = scatter_add(
            field[interp_source] / squared_pos_dist, interp_target, dim=0
        )
        denominator = scatter_add(1.0 / squared_pos_dist, interp_target, dim=0)

        return numerator / denominator

    def __repr__(self):
        return f"{self.__class__.__name__}(ratio_volume_samples={self.ratio_volume_samples})"
