"""PI0.5 inference test — Libero / DROID.

Usage:
    python test/test_pi05.py [--dataset libero droid]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from openworldlib.pipelines.pi0.pipeline_pi0 import PI0Pipeline

TOKENIZER = 'google/paligemma-3b-mix-224'
DEVICE    = 'cuda:0' if torch.cuda.is_available() else 'cpu'

CONFIGS = {
    'libero': dict(
        model_path       = 'lerobot/pi05_base',
        norm_stats_path  = './data/test_case/test_vla_case1/libero/pi0_5_norm_stats.json',
        robot_type       = 'libero',
        action_dim       = 7,
        use_delta_actions= False,
        out_path         = 'outputs/pi05_libero_demo.png',
        img_keys         = ['observation.images.cam_high',
                            'observation.images.cam_left_wrist'],
        img_files        = {
            'observation.images.cam_high':       Image.open('./data/test_case/test_vla_case1/libero/main_view.png').convert('RGB'),
            'observation.images.cam_left_wrist': Image.open('./data/test_case/test_vla_case1/libero/wrist_view.png').convert('RGB'),
        },
    ),
    'droid': dict(
        model_path       = 'lerobot/pi05_base',  # replace with pi05_droid checkpoint in practice
        norm_stats_path  = './data/test_case/test_vla_case1/droid/pi05_norm_states_droid_joint.json',
        robot_type       = 'droid',
        action_dim       = 8,
        use_delta_actions= True,
        out_path         = 'outputs/pi05_droid_demo.png',
        img_keys         = ['observation.images.cam_high',
                            'observation.images.cam_left_wrist',
                            'observation.images.cam_right_wrist'],
        img_files        = {
            'observation.images.cam_high':        Image.open('./data/test_case/test_vla_case1/droid/exterior_image_1_left.png').convert('RGB'),
            'observation.images.cam_left_wrist':  Image.open('./data/test_case/test_vla_case1/droid/wrist_image_left.png').convert('RGB'),
            'observation.images.cam_right_wrist': Image.open('./data/test_case/test_vla_case1/droid/exterior_image_2_left.png').convert('RGB'),
        },
    ),
}


def load_data(name):
    if name == 'libero':
        d = json.load(open('./data/test_case/test_vla_case1/libero/meta.json'))
        return torch.tensor(d['observation']['state'], dtype=torch.float32), d['task']
    if name == 'droid':
        d = json.load(open('./data/test_case/test_vla_case1/droid/step_data.json'))
        state = torch.tensor(d['observation_numeric']['joint_position'] +
                             d['observation_numeric']['gripper_position'], dtype=torch.float32)
        return state, d['language_instruction']


def visualize_action(action: np.ndarray, out_path: str, action_dim: int) -> None:
    action = action[:, :action_dim]
    num_ts, num_dim = action.shape
    fig, axs = plt.subplots(num_dim, 1, figsize=(10, 2 * num_dim))
    if num_dim == 1:
        axs = [axs]
    for i, ax in enumerate(axs):
        ax.plot(np.arange(num_ts) / 30.0, action[:, i], linewidth=2)
        ax.set_title(f'Joint {i}')
        ax.set_xlabel('Time (s)')
        ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  -> Saved to {out_path}')


def run(name: str, cfg: dict) -> None:
    state, task = load_data(name)
    norm = json.load(open(cfg['norm_stats_path']))['norm_stats']
    state_norm  = norm.get('observation.state', norm.get('state'))
    action_norm = norm.get('action', norm.get('actions'))

    pipe = PI0Pipeline.from_pretrained(
        model_path=cfg['model_path'],
        tokenizer_model_path=TOKENIZER,
        state_norm_stats=state_norm,
        action_norm_stats=action_norm,
        original_action_dim=cfg['action_dim'],
        discrete_state_input=True,
        device=DEVICE,
        present_img_keys=cfg['img_keys'],
        robot_type=cfg['robot_type'],
        use_delta_actions=cfg['use_delta_actions'],
    )
    pipe.compile()

    pred = pipe(cfg['img_files'], task, state)
    print(f'  [{name}] pred shape: {pred.shape},  sample: {pred[0, :cfg["action_dim"]].tolist()}')
    visualize_action(pred.detach().cpu().numpy(), cfg['out_path'], cfg['action_dim'])
    del pipe
    torch.cuda.empty_cache()


if __name__ == '__main__':
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', nargs='+', choices=list(CONFIGS), default=list(CONFIGS))
    args = parser.parse_args()

    for name in args.dataset:
        run(name, CONFIGS[name])
