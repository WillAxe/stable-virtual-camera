"""Standalone SEVA inference entry point — runs inside the submodule's venv."""

import argparse
import json
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from seva.geometry import DEFAULT_FOV_RAD, get_lookat_w2cs, get_default_intrinsics
from seva.model import SGMWrapper
from seva.modules.autoencoder import AutoEncoder
from seva.modules.conditioner import CLIPConditioner
from seva.sampling import DiscreteDenoiser
from seva.utils import load_model as seva_load_weights
from seva.eval import create_samplers, do_sample, get_value_dict

# Constants
H, W = 576, 576
T = 5
C, F_FACTOR = 4, 8
CFG, STEPS = 2.0, 50
LOOK_AT = torch.tensor([0.0, 0.0, 10.0])
UP = torch.tensor([0.0, -1.0, 0.0])


def build_trajectory(yaw: float, pitch: float, zoom: float, num_frames: int = T):
    source_pos = torch.zeros(3)
    yaw_rad = -math.radians(yaw)
    pitch_rad = -math.radians(pitch)
    offset = source_pos - LOOK_AT

    cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
    Ry = torch.tensor([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])

    cos_p, sin_p = math.cos(pitch_rad), math.sin(pitch_rad)
    Rx = torch.tensor([[1, 0, 0], [0, cos_p, -sin_p], [0, sin_p, cos_p]])

    target_pos = LOOK_AT + Ry @ Rx @ offset
    t_vals = torch.linspace(0, 1, num_frames)
    positions = source_pos[None] + t_vals[:, None] * (target_pos - source_pos)[None]

    w2cs = get_lookat_w2cs(positions, LOOK_AT, UP)
    c2ws = torch.linalg.inv(w2cs)

    source_fov = DEFAULT_FOV_RAD
    target_fov = DEFAULT_FOV_RAD / max(zoom, 0.1)
    fovs = torch.linspace(source_fov, target_fov, num_frames)
    return c2ws, fovs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--params", required=True, help="JSON: {yaw, pitch, zoom}")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    params = json.loads(args.params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load frame
    frame_bgr = cv2.imread(args.frame)
    if frame_bgr is None:
        raise FileNotFoundError(f"Cannot read: {args.frame}")
    h_orig, w_orig = frame_bgr.shape[:2]

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().div_(255).mul_(2).sub_(1)
    img = F.interpolate(img.unsqueeze(0), (H, W), mode="area").squeeze(0)

    # Load model
    ae = AutoEncoder(chunk_size=1).to(device)
    conditioner = CLIPConditioner().to(device)
    denoiser = DiscreteDenoiser(num_idx=1000, device=device)
    raw_model = seva_load_weights(
        model_version=1.1,
        pretrained_model_name_or_path="stabilityai/stable-virtual-camera",
        weight_name="model.safetensors",
        device="cpu",
        verbose=False,
    ).eval()
    model = SGMWrapper(raw_model).to(device)

    # Camera trajectory
    c2ws, fovs = build_trajectory(params["yaw"], params["pitch"], params["zoom"])
    Ks = get_default_intrinsics(fovs, aspect_ratio=W / H)

    imgs = img.new_zeros(T, 3, H, W)
    imgs[0] = img

    value_dict = get_value_dict(
        curr_imgs=imgs.to(device),
        curr_input_frame_indices=[0],
        curr_c2ws=c2ws[:, :3].float(),
        curr_Ks=Ks.float(),
        curr_input_camera_indices=[0],
        all_c2ws=c2ws.float(),
        camera_scale=2.0,
    )

    samplers = create_samplers(
        guider_types=1,
        discretization=denoiser.discretization,
        num_frames=[T],
        num_steps=STEPS,
        cfg_min=1.2,
        device=device,
    )

    samples = do_sample(
        model=model, ae=ae, conditioner=conditioner, denoiser=denoiser,
        sampler=samplers[0], value_dict=value_dict,
        H=H, W=W, C=C, F=F_FACTOR, T=T, cfg=CFG,
        encoding_t=1, decoding_t=1, verbose=False,
    )

    # Extract last frame
    if isinstance(samples, dict):
        output = next((v for k, v in samples.items() if "rgb" in k), next(iter(samples.values())))
    else:
        output = samples

    target = output[-1].detach().cpu()
    target = ((target + 1) / 2 * 255).clamp(0, 255).byte()
    result_rgb = target.permute(1, 2, 0).numpy()
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

    if (h_orig, w_orig) != (H, W):
        result_bgr = cv2.resize(result_bgr, (w_orig, h_orig))

    cv2.imwrite(args.output, result_bgr)
    print(f"OK: {args.output}")


if __name__ == "__main__":
    main()