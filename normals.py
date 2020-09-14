import numba as nb
import numpy as np
from sklearn.neighbors import NearestNeighbors, KDTree
import torch

'''
If we can compute the normals of the LIDAR point cloud, we can use
them during the SIREN fitting process. The normals represent the
gradient of the signed distance function, so a point cloud with
normals represents a boundary value problem (where the point cloud
represents the boundary condition and the normals are the derivative).

Normally, we would compute normals by first computing nearest
neighbors for every point (using a K-D tree or a ball tree). Then for
each neighborhood, subtract out the centroid and use SVD to find the
smallest singular vector (i.e should be pointing orthogonal to the
surface plane). Then we flip all the normals that are backwards
relative to ego. The tricky bit is handling the low vertical
resolution of LIDAR scans. Because there is such a large gap between
each ring, it is easy for all `n_neighbors` neighbors of a point to
lie within the same ring. As a result, we end up fitting a plane to a
distribution of points which looks like a line (the LIDAR ring).

To overcome this problem, we instead compute a K-D tree for each LIDAR
ring and then find `k` nearest neighbors on the rings above and
below. Instead of using a 3D K-D tree, we use a 2D K-D tree, ignoring
the z-component. (TODO: Add explanation for why 2D is probably
sufficient)
'''

def get_neighbors_naive(pc, n_neighbors=5, max_similarity_dist=1):
    """
    Returns a tensor that contains the indices of the neighbors of each point in
    the array. A single K-D tree is used to find neighbors.

    :param pc: input point cloud
    :n_neighbors: number of neighbors to find for each point
    :max_similarity_dist: max dist of a neighbor, farther neighbors will be 0
    :return: [N, n_neighbors] tensor where each row `r` contains the indices
    of the neighbors of point `r`
    """
    pc_np = pc[:, :3].detach().numpy()

    # this chunk of code just gets the n_neighbors nearest neighbors
    # given that they're all within `max_similarity_dist`
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree").fit(pc_np)

    distances, indices = nbrs.kneighbors(pc_np)

    indices[distances > max_similarity_dist] = 0

    indices = torch.from_numpy(indices).cpu()

    return indices


@nb.njit
def get_ring_endpoints(pc):
    """
    Returns the end of each ring within a point cloud array (assumes ring
    is last channel). This function is JIT compiled with numba because it
    would be very slow with a Python for loop.

    >>> a = np.array([[12, 0], [3, 0], [5, 0], [2, 1], [4, 1], [0, 2], [9, 2])
    >>> get_ring_endpoints(a)
    array([3, 5, 7])

    :param pc: input point cloud
    :return: array that stores the end index of every ring
    """
    ring_endpoints = np.zeros(int(pc[:, -1].max()) + 1, dtype=np.int64)
    ri = 0
    curr = pc[0, -1]
    l = 0
    for i in range(pc.shape[0]):
        p = pc[i, -1]
        if p == curr:
            l += 1
        else:
            curr = p
            ring_endpoints[ri] = l
            ri += 1
    ring_endpoints[-1] = l

    return ring_endpoints


def get_neighbors_rings(pc, k=3):
    """
    Returns a tensor that contains the indices of the neighbors of each point in
    the array. `k` samples are taken from the ring above, `k` samples from the
    current ring, and `k` samples from the ring below. A 2D K-D tree is used to
    get the closest samples in each ring.

    :param pc: input point cloud
    :k: number of neighbors to sample in each ring
    :return: [N, 3 * k] tensor where each row `r` contains the indices of the
    neighbors of point `r`
    """
    num_rings = int(pc[:, -1].max().item()) + 1
    ring_endpoints = get_ring_endpoints(pc.numpy())

    ring_starts = [
        ring_endpoints[i - 1] if i > 0 else 0 for i in range(len(ring_endpoints))
    ]

    rings = [
        pc[ring_starts[i] : ring_endpoints[i]].numpy()
        for i in range(len(ring_endpoints))
    ]
    kdtrees = [KDTree(rings[i][:, :2]) for i in range(len(ring_endpoints))]

    # offset by the starting index
    query_against = (
        lambda r, x: kdtrees[r].query(
            x[:, :2], k=k, return_distance=False, sort_results=True
        )
        + ring_starts[r]
    )

    n_neighbors = k * 3  # 3 rings
    pc_neighbors = (
        pc.expand(n_neighbors, pc.shape[0], pc.shape[-1]).permute(1, 0, 2).contiguous()
    )
    indices = torch.zeros((pc_neighbors.shape[0], n_neighbors), dtype=torch.int64)

    for ring_id in range(num_rings):
        start = ring_starts[ring_id]
        end = ring_endpoints[ring_id]

        # one would assume that the closest point in each ring is just the adjacent
        # points in the pointcloud array, but that's not always the case...
        current_ring_pts = query_against(ring_id, rings[ring_id])

        below_ring_pts = (
            query_against(ring_id - 1, rings[ring_id])
            if ring_id > 0
            else current_ring_pts.copy()
        )
        above_ring_pts = (
            query_against(ring_id + 1, rings[ring_id])
            if ring_id < num_rings - 1
            else current_ring_pts.copy()
        )

        indices[start:end, :k] = torch.from_numpy(current_ring_pts)
        indices[start:end, k : 2 * k] = torch.from_numpy(below_ring_pts)
        indices[start:end, 2 * k : 3 * k] = torch.from_numpy(above_ring_pts)

    return indices


def compute_normals(pc, ego, use_rings=True):
    """
    Computes normals for a point cloud given that we know ego. This function
    flips normals that are facing the wrong way.

    :param pc: point cloud tensor where the rings are the last channel [N, 5]
    :param ego: ego vector
    :return: normals tensor [N, 3]
    """
    pc_cpu = pc.detach().cpu()
    ego_cpu = ego.detach().cpu()

    # if we don't have ring info then we can't use our ring-aware
    # neighbor-finding approach
    if pc.shape[1] > 3 and use_rings:
        indices = get_neighbors_rings(pc_cpu)
    else:
        indices = get_neighbors_naive(pc_cpu)

    n_neighbors = indices.shape[-1]

    indices = (
        indices.expand(pc.shape[-1], pc.shape[0], n_neighbors)
        .permute(1, 2, 0)
        .contiguous()
    )

    pc_neighbors = (
        pc_cpu.expand(n_neighbors, pc.shape[0], pc.shape[-1])
        .permute(1, 0, 2)
        .contiguous()
    )

    pc_neighbors = torch.gather(pc_neighbors, 0, indices)
    pc_neighbors = pc_neighbors[:, :, :3]

    # subtract out centroid, transpose from `[BS, N, 3]` to `[BS, 3, N]`
    pc_neighbors = (pc_neighbors - pc_cpu[:, :3].unsqueeze(1)).transpose(1, 2)

    # pick singular vector corresponding to least singular value -- hopefully
    # a good approximation of the normal?
    normals = torch.svd(pc_neighbors)[0].transpose(1, 2)[:, -1, :]

    # find the normals that are flipped relative to the vector from ego
    flip = torch.bmm(normals.unsqueeze(1), (pc_cpu[:, :3] - ego_cpu).unsqueeze(-1))

    # and unflip them
    flip_mask = flip.view(flip.shape[0], 1).repeat(1, normals.shape[-1]) > 0
    normals[flip_mask] = -normals[flip_mask]

    return normals
