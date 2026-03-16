import json
import os

import matplotlib
import numpy as np
from sympy import use
import torch
from PIL import Image as PILImage

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from openworldlib.pipelines.giga_brain_0.pipeline_giga_brain_0 import GigaBrain0Pipeline, ImageInput


MODEL_PATH = 'open-gigaai/GigaBrain-0-3.5B-Base'
NORM_STATS_PATH = 'data/test_case/test_vla_case1/libero/pi0_norm_stats.json'
MAIN_VIEW_PATH = 'data/test_case/test_vla_case1/libero/main_view.png'
WRIST_VIEW_PATH = 'data/test_case/test_vla_case1/libero/wrist_view.png'
META_PATH = 'data/test_case/test_vla_case1/libero/meta.json'
OUTPUT_PATH = 'outputs/giga_brain_0_demo.png'

ORIGINAL_ACTION_DIM = 7
DELTA_MASK = [True] * ORIGINAL_ACTION_DIM
EMBODIMENT_ID = 0
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
WEIGHT_DTYPE = torch.bfloat16  # can be set to torch.float16 or None to use the model's original dtype

# Additional components required beyond the main model
REQUIRED_COMPONENTS = {
    'tokenizer': 'google/paligemma-3b-pt-224',
    'fast_tokenizer': 'physical-intelligence/fast',
}


def visualize_action(pred_action: np.ndarray, out_path: str, action_names: list[str] | None = None) -> None:
    """Visualize the predicted action trajectory."""
    if (pred_action.ndim == 1):
        pred_action = pred_action[None, :]
    pred_action = pred_action[:, :ORIGINAL_ACTION_DIM]
    num_ts, num_dim = pred_action.shape
    fig, axs = plt.subplots(num_dim, 1, figsize=(10, 2 * num_dim))
    time_axis = np.arange(num_ts) / 30.0
    colors = plt.colormaps['viridis'](np.linspace(0, 1, num_dim))
    action_names = action_names or [str(i) for i in range(num_dim)]

    for ax_idx in range(num_dim):
        ax = axs[ax_idx]
        ax.plot(time_axis, pred_action[:, ax_idx], label='Pred', color=colors[ax_idx], linewidth=2, linestyle='-')
        ax.set_title(f'Joint {ax_idx}: {action_names[ax_idx]}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (rad)')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    with open(NORM_STATS_PATH, 'r') as f:
        norm_stats_data = json.load(f)['norm_stats']

    # Support both key naming conventions
    state_norm = norm_stats_data.get('observation.state', norm_stats_data.get('state'))
    action_norm = norm_stats_data.get('action', norm_stats_data.get('actions'))

    pipe = GigaBrain0Pipeline.from_pretrained(
        model_path=MODEL_PATH,
        required_components=REQUIRED_COMPONENTS,
        embodiment_id=EMBODIMENT_ID,
        state_norm_stats=state_norm,
        action_norm_stats=action_norm,
        delta_mask=DELTA_MASK,
        original_action_dim=ORIGINAL_ACTION_DIM,
        depth_img_prefix_name=None,
        device=DEVICE,
        weight_dtype=WEIGHT_DTYPE,
        present_img_keys=['observation.images.cam_high', 'observation.images.cam_wrist'],
        use_quantiles=False,  
    )
    pipe.compile()

    # Use PIL.Image as image input
    images: dict[str, ImageInput] = {
        'observation.images.cam_high': PILImage.open(MAIN_VIEW_PATH).convert('RGB'),
        'observation.images.cam_wrist': PILImage.open(WRIST_VIEW_PATH).convert('RGB'),
    }

    with open(META_PATH, 'r') as f:
        meta_data = json.load(f)
    task = meta_data['task']
    state = torch.tensor(meta_data['observation']['state'], dtype=torch.float32)

    result = pipe(images, prompt=task, state=state)
    pred_action = result if isinstance(result, torch.Tensor) else result[0]
    visualize_action(
        pred_action.detach().cpu().numpy(),
        OUTPUT_PATH,
        action_names=None,
    )
