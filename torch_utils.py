import torch

float_t = torch.float32

def tf(arr, dtype=float_t, torch_device='cuda'):
    return torch.tensor(arr, dtype=dtype, device=torch_device)

def tz(dims, dtype=float_t, torch_device='cuda'):
    return torch.zeros(dims, dtype=dtype, device=torch_device)


def trand(dims, dtype=float_t, torch_device='cuda'):
    return torch.rand(dims, dtype=dtype, device=torch_device)

def normalize(a):
    return a / torch.norm(a, p=2, dim=-1)

def look_at(eye, center, up, torch_device='cuda'):
    n = normalize(eye - center)
    u = normalize(torch.cross(up, n, dim=-1))
    v = torch.cross(n, u)

    out = torch.zeros((4, 4), dtype=float_t, device=torch_device)

    out[0, :3] = u
    out[1, :3] = v
    out[2, :3] = n

    out[0, 3] = torch.dot(-1.0 * u, eye)
    out[1, 3] = torch.dot(-1.0 * v, eye)
    out[2, 3] = torch.dot(-1.0 * n, eye)
    out[3, 3] = 1.0

    return out

def projection_gen(proj_matrix, view_matrix, width, height, torch_device='cuda'):
    '''
    generates matrix that has ray positions on the near plane and their directions
    '''
    rays = torch.zeros((2, height, width, 3),
                       dtype=float_t, device=torch_device)
    inv = torch.inverse(proj_matrix.cpu() @ view_matrix.cpu()).to(
        device=torch_device)
    origin = (torch.inverse(view_matrix.cpu()).to(device=torch_device) @ torch.tensor(
        (0.0, 0.0, 0.0, 1.0), dtype=float_t, device=torch_device))[:3]
    near = 0.1
    grid = torch.meshgrid(torch.linspace(-1.0, 1.0, height, dtype=float_t, device=torch_device),
                          torch.linspace(-1.0, 1.0, width, dtype=float_t, device=torch_device))
    clip_space = torch.stack(
        (grid[0], grid[1],
         torch.ones((height, width), dtype=float_t, device=torch_device),
         torch.ones((height, width), dtype=float_t, device=torch_device)), dim=-1)
    tmp = torch.matmul(inv, clip_space.view(height, width, 4, 1)).squeeze()
    tmp = tmp / tmp[:, :, 3:]
    tmp = tmp[:, :, :3]
    tmp = tmp - torch.tensor((origin[0], origin[1], origin[2]),
                             dtype=float_t, device=torch_device)
    ray_vec = tmp / torch.norm(tmp, p=2, dim=2).unsqueeze(-1)
    rays[0, :, :] = origin + ray_vec * near
    rays[1, :, :] = ray_vec

    return rays, origin
