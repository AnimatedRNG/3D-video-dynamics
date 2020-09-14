import torch
import torch.nn as nn
import numpy as np
from torch_utils import *

class SphereTracerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sdf_model, ray_matrix, iterations, *args):
        # ray_matrix[BS, 0] is the starting position of each ray
        orig_ray_pos = ray_matrix[:, 0, :, :]
        ray_pos = orig_ray_pos.clone()

        # ray_matrix[BS, 1] is the normalized direction of each ray
        # (the equivalent of w in Niemeyer et al.)
        ray_vec = ray_matrix[:, 1, :, :]
        surface_depth = tz((ray_pos.shape[0], ray_pos.shape[1], ray_pos.shape[2], 1))

        # performs sphere tracing in-place,
        # ray-marching steps are not memorized
        for i in range(iterations):
            ray_pos_shape = ray_pos.shape

            # flatten all other dimensions
            pos_permuted = torch.flatten(ray_pos, 0, 2)

            # to make the ray marching less unstable, we truncate the SDF
            # surface_depth += sdf_model(ray_pos).permute(0, 2, 3, 1)
            # surface_depth += torch.clamp(sdf_model(pos_permuted), -
            #                             1.0, 1.0).permute(0, 2, 3, 1)
            surface_depth += torch.clamp(sdf_model(pos_permuted)[0], -1.0, 1.0).view(
                ray_pos_shape[0], ray_pos_shape[1], ray_pos_shape[2], 1
            )

            # advance ray. note that the dimensions of ray_vec are
            # [BS, width, height, 3] and the dimensions of surface_depth
            # are [BS, width, height, 1] (originally
            # [BS, 1, width, height] before permute)
            ray_pos = orig_ray_pos + ray_vec * surface_depth

        # we just need the ray matrix, the final pos, and the sdf_model
        ctx.save_for_backward(ray_matrix, ray_pos)
        ctx.sdf_model = sdf_model
        ctx.args = tuple(args)
        # ctx.mark_non_differentiable(ray_matrix)

        return surface_depth

    # TODO: Come up with better tests for why or why this
    # doesn't work!
    @staticmethod
    def backward(ctx, dloss_ddepth):
        ray_matrix, ray_pos = ctx.saved_tensors
        ray_pos.requires_grad_()
        sdf_model = ctx.sdf_model

        ray_vec = ray_matrix[:, 1, :, :]

        bs = ray_pos.shape[0]
        w = ray_pos.shape[1]
        h = ray_pos.shape[2]

        # incoming adjoint (ignored)
        musk = torch.ones((bs, 1, w, h), dtype=float_t, device=torch_device)

        with torch.enable_grad():
            # recompute the final signed distance value,
            # and track the graph -- JUST for this operation
            # final_pos_permuted = ray_pos.permute(0, 3, 1, 2)
            final_pos_permuted = torch.flatten(ray_pos, 0, 2)
            final_sdf = sdf_model(final_pos_permuted)

        # sphere tracing convergence mask
        # mask = torch.where(torch.abs(final_sdf) < 1.0, tf(1.0), tf(0.0))

        # compute the gradient w.r.t the ray position
        dsdf_dpos = torch.autograd.grad(
            final_sdf, ray_pos, grad_outputs=musk, retain_graph=True, create_graph=True
        )[0]

        # We don't want to compute gradients for `ray_pos` again!
        ray_pos.requires_grad_(False)

        bs_w_h = bs * w * h

        # see equation 11; note that we need to perform a dot product
        mu = (-1.0 * dloss_ddepth) / (
            torch.bmm(dsdf_dpos.view(bs_w_h, 1, 3), ray_vec.view(bs_w_h, 3, 1))
        ).view(bs, 1, w, h)

        # mask contributions to the derivative where there shouldn't be any?
        # mu = mu * mask

        # compute the gradient w.r.t theta (neural net parameters)
        torch.autograd.backward(
            final_sdf, grad_tensors=mu, retain_graph=True, create_graph=False
        )

        # we could put this in grad_tensors if needed
        return (None, tz(ray_matrix.shape), None, *tuple(arg.grad for arg in ctx.args))


class SphereTracer(nn.Module):
    def __init__(self):
        super(SphereTracer, self).__init__()

    def forward(self, sdf_model, ray_matrix, iterations=32):
        return SphereTracerFunction.apply(
            sdf_model,
            ray_matrix,
            iterations,
            # *sdf_model.parameters()
        )


def render(sdf_model):
    import matplotlib.pyplot as plt
    torch_device = "cuda"

    projection = tf(
        [
            [0.75, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0002, 1.0],
            [0.0, 0.0, -0.2, 0.0],
        ]
    )

    view = look_at(
        tf([0.0, 0.0, 8.0]), tf([0.0, 0.0, 0.0]), tf([0.0, 1.0, 0.0]), torch_device
    )

    ray_matrix = projection_gen(projection, view, 512, 512)[0].unsqueeze(0)

    sdf_model.eval()

    # def sdf_model(pos):
    #    return (torch.norm(pos, p=2, dim=1).unsqueeze(-1) - 1.0, None)

    tracer = SphereTracer().to(torch_device)

    render = tracer(sdf_model, ray_matrix).squeeze()

    # render = torch.clamp(render, 0.0, 8.0)
    # render /= 16.0
    render = render.detach().cpu().numpy()

    # plt.imshow(render * 255, cmap='gray', vmin=0, vmax=255)
    # plt.imshow(render)
    plt.imshow(render, cmap="gray")
    plt.show()
