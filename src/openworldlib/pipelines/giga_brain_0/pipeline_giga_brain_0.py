from pathlib import Path
from typing import Any, Union, cast

import torch
from PIL import Image as PILImage
from torchvision.transforms import functional as TF

from ...operators.giga_brain_0_operator import GigaBrain0Operator
from ...synthesis.vla_generation.giga_brain_0.giga_brain_0_synthesis import GigaBrain0Synthesis

# Supported image input types: torch.Tensor, PIL.Image, file path string or Path object
ImageInput = Union[torch.Tensor, PILImage.Image, str, Path]


def _to_tensor(img: ImageInput) -> torch.Tensor:
    """Convert an image path, PIL.Image, or torch.Tensor into a float32 CHW Tensor."""
    if isinstance(img, (str, Path)):
        img = PILImage.open(img).convert('RGB')
    if isinstance(img, PILImage.Image):
        img = img.convert('RGB')
        return TF.to_tensor(img)
    if isinstance(img, torch.Tensor):
        return img
    raise TypeError(f'Unsupported image type: {type(img)}')


def _convert_images(images: dict[str, ImageInput]) -> dict[str, torch.Tensor]:
    """Convert each value in the images dict to a torch.Tensor."""
    return {k: _to_tensor(v) for k, v in images.items()}


class GigaBrain0Pipeline:
    """Pipeline wrapper for GigaBrain0 policy inference using a dedicated operator."""

    def __init__(
        self,
        synthesis: GigaBrain0Synthesis,
        operator: GigaBrain0Operator,
        embodiment_id: int,
        original_action_dim: int,
        device: str | torch.device | None = None,
        weight_dtype: torch.dtype | None = None,
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight_dtype = weight_dtype
        self.synthesis = synthesis.to(self.device)
        self.operator = operator.to(self.device)
        self.operator.set_action_dim(self.synthesis.max_action_dim)
        self.embodiment_id = embodiment_id
        self.original_action_dim = original_action_dim
        self.resize_imgs_with_padding = (224, 224)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        required_components: dict[str, str] | None = None,
        embodiment_id: int = 0,
        state_norm_stats: dict | None = None,
        action_norm_stats: dict | None = None,
        delta_mask: list[bool] | None = None,
        original_action_dim: int = 7,
        discrete_state_input: bool = True,
        autoregressive_inference_mode: bool = False,
        depth_img_prefix_name: str | None = None,
        device: str | torch.device | None = None,
        weight_dtype: torch.dtype | None = None,
        present_img_keys: list[str] | None = None,
        use_quantiles: bool = True,
        **policy_kwargs: Any,
    ) -> 'GigaBrain0Pipeline':
        """Load the pipeline from pretrained weights.

        Args:
            model_path: HuggingFace model ID or local path to the main model.
            required_components: Additional components to load beyond the main model,
                as a dict mapping component name to HuggingFace ID or local path.
                e.g. ``{"tokenizer": "google/paligemma-3b-pt-224", "fast_tokenizer": "physical-intelligence/fast"}``.
            device: Device to load all models onto, e.g. ``"cuda:0"`` or ``"cpu"``.
            weight_dtype: Weight dtype for all models in the pipeline, e.g. ``torch.float16``
                or ``torch.bfloat16``. Defaults to ``None`` (uses the model's original dtype).
            embodiment_id: Robot embodiment ID.
            state_norm_stats: State normalization statistics.
            action_norm_stats: Action normalization statistics.
            delta_mask: Boolean mask indicating which action dimensions are deltas.
            original_action_dim: Original action dimension.
            discrete_state_input: Whether to use discrete state input.
            autoregressive_inference_mode: Whether to use autoregressive inference mode.
            depth_img_prefix_name: Prefix name for depth image keys.
            present_img_keys: List of image keys present in the current observation.
            **policy_kwargs: Extra keyword arguments forwarded to the underlying policy.
        """
        required_components = required_components or {}

        tokenizer_model_path = required_components.get('tokenizer', 'google/paligemma-3b-pt-224')
        fast_tokenizer_path = required_components.get('fast_tokenizer', 'physical-intelligence/fast')

        # Forward weight_dtype to the underlying model loader via torch_dtype
        if weight_dtype is not None:
            policy_kwargs.setdefault('torch_dtype', weight_dtype)

        synthesis = GigaBrain0Synthesis.from_pretrained(model_path, device=device, **policy_kwargs)

        # Cast model weights to the requested dtype after loading
        if weight_dtype is not None:
            synthesis.policy.to(weight_dtype)

        operator = GigaBrain0Operator(
            embodiment_id=embodiment_id,
            state_norm_stats=state_norm_stats or {},
            action_norm_stats=action_norm_stats or {},
            delta_mask=delta_mask or [],
            tokenizer_model_path=tokenizer_model_path,
            fast_tokenizer_path=fast_tokenizer_path,
            resize_imgs_with_padding=(224, 224),
            enable_depth_img=synthesis.vision_in_channels == 4,
            depth_img_prefix_name=depth_img_prefix_name,
            discrete_state_input=discrete_state_input,
            autoregressive_inference_mode=autoregressive_inference_mode,
            text_max_length=200,
            present_img_keys=present_img_keys,
            use_quantiles=use_quantiles,
        )
        return cls(
            synthesis=synthesis,
            operator=operator,
            embodiment_id=embodiment_id,
            original_action_dim=original_action_dim,
            device=device,
            weight_dtype=weight_dtype,
        )

    def to(self, device: str | torch.device):
        self.device = device
        self.synthesis.to(device)
        self.operator.to(device)
        return self

    def quantize(self) -> None:
        """Quantize via synthesis wrapper."""
        self.synthesis.quantize()

    def compile(self, **kwargs: Any) -> None:
        """Compile the `sample_actions` method using `torch.compile` for improved runtime speed."""
        self.synthesis.compile(**kwargs)

    def process(
        self,
        images: dict[str, ImageInput],
        prompt: str,
        state: torch.Tensor,
        pad_state: bool = True,
        add_batch_dim: bool = True,
    ):
        """Preprocess inputs (perception + interaction) to build model-ready tensors."""
        ori_device = state.device if state is not None else self.device
        tensor_images: dict[str, torch.Tensor] = {k: _to_tensor(v).to(self.device) for k, v in images.items()}
        state = state.to(self.device)

        proc_images, img_masks, image_transform_params, proc_state = self.operator.process_perception(tensor_images, state, pad_state=pad_state)
        state = cast(torch.Tensor, proc_state)
        lang_tokens, lang_masks, _, _, _, _ = self.operator.process_interaction(task=prompt, state=state)

        if add_batch_dim:
            proc_images = [img.unsqueeze(0) for img in proc_images]
            img_masks = [mask.unsqueeze(0) for mask in img_masks]
            lang_tokens = lang_tokens.unsqueeze(0)
            lang_masks = lang_masks.unsqueeze(0)
            emb_ids = torch.tensor([self.embodiment_id], dtype=torch.long, device=self.device)
        else:
            emb_ids = torch.tensor(self.embodiment_id, dtype=torch.long, device=self.device)

        # Cast float tensors to match model weight dtype to avoid dtype mismatch with torch.compile
        if self.weight_dtype is not None:
            proc_images = [img.to(self.weight_dtype) for img in proc_images]
            state = state.to(self.weight_dtype)

        return {
            'images': proc_images,
            'img_masks': img_masks,
            'lang_tokens': lang_tokens,
            'lang_masks': lang_masks,
            'state': state,
            'image_transform_params': image_transform_params,
            'emb_ids': emb_ids,
            'ori_device': ori_device,
        }

    @torch.no_grad()
    def __call__(
        self,
        images: dict[str, ImageInput],
        prompt: str,
        state: torch.Tensor,
        enable_2d_traj_output: bool = False,
        autoregressive_mode_only: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if autoregressive_mode_only:
            return self.predict_autoregressive_actions(images, prompt, state)

        processed = self.process(images, prompt, state, pad_state=True, add_batch_dim=True)

        outputs = self.synthesis.predict(
            images=processed['images'],
            img_masks=processed['img_masks'],
            lang_tokens=processed['lang_tokens'],
            lang_masks=processed['lang_masks'],
            emb_ids=processed['emb_ids'],
            enable_2d_traj_output=enable_2d_traj_output,
        )
        traj_pred: torch.Tensor | None = None
        if enable_2d_traj_output:
            pred_action, traj_pred = outputs
        else:
            pred_action = outputs

        pred_action = self.operator.process_output(
            pred_action[0],
            processed['state'],
            self.original_action_dim,
            image_transform_params=processed['image_transform_params'],
            traj_pred=None,
        )
        if isinstance(pred_action, tuple):
            pred_action = pred_action[0]
        pred_action = pred_action.to(processed['ori_device'])

        if enable_2d_traj_output and traj_pred is not None:
            traj_pred = traj_pred[0]
            if 'resize_with_pad' in processed['image_transform_params']:
                ratio = processed['image_transform_params']['resize_with_pad']['ratio']
                pad_x, pad_y = processed['image_transform_params']['resize_with_pad']['padding']
                traj_pred[:, ::2] = (traj_pred[:, ::2] * self.resize_imgs_with_padding[0] - pad_x) * ratio
                traj_pred[:, 1::2] = (traj_pred[:, 1::2] * self.resize_imgs_with_padding[1] - pad_y) * ratio
            traj_pred = traj_pred.to(processed['ori_device'])
            return pred_action, traj_pred

        return pred_action

    @torch.no_grad()
    def predict_current_subtask(self, images: dict[str, ImageInput], prompt: str) -> list[str]:
        tokenizer = self.operator.tokenizer

        tensor_images: dict[str, torch.Tensor] = {k: _to_tensor(v).to(self.device) for k, v in images.items()}
        proc_images, img_masks, _, _ = self.operator.process_perception(tensor_images, state=torch.empty(0), pad_state=False)
        lang_tokens, lang_masks, _, _, _, _ = self.operator.process_interaction(task=prompt)

        for i in range(len(proc_images)):
            proc_images[i] = proc_images[i][None, ...]
            img_masks[i] = img_masks[i][None, ...]
        lang_tokens = lang_tokens[None, ...]
        lang_masks = lang_masks[None, ...]

        generated = self.generate_autoregressive_tokens(proc_images, img_masks, lang_tokens, lang_masks)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        return decoded

    @torch.no_grad()
    def predict_autoregressive_actions(
        self, images: dict[str, ImageInput], prompt: str, state: torch.Tensor, max_new_tokens: int = 200
    ) -> torch.Tensor:
        processed = self.process(images, prompt, state, pad_state=False, add_batch_dim=False)
        proc_images: list[torch.Tensor] = processed['images']
        img_masks: list[torch.Tensor] = processed['img_masks']
        lang_tokens: torch.Tensor = processed['lang_tokens']
        lang_masks: torch.Tensor = processed['lang_masks']
        proc_state: torch.Tensor = processed['state']
        ori_device = processed['ori_device']

        generated = self.generate_autoregressive_tokens(proc_images, img_masks, lang_tokens, lang_masks, max_new_tokens=max_new_tokens)

        pred_action = self.operator.extract_actions(generated, self.synthesis.n_action_steps, self.original_action_dim)
        pred_action = pred_action.to(self.device)
        pred_action = self.operator.process_output(pred_action, proc_state, self.original_action_dim)
        if isinstance(pred_action, tuple):
            pred_action = pred_action[0]
        pred_action = pred_action.to(ori_device)
        return pred_action

    @torch.no_grad()
    def generate_autoregressive_tokens(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        max_new_tokens: int = 64,
    ) -> list[list[int]]:
        for i in range(len(images)):
            if images[i].ndim == 3:
                images[i] = images[i][None, ...]
            if img_masks[i].ndim == 1:
                img_masks[i] = img_masks[i][None, ...]
        if lang_tokens.ndim == 1:
            lang_tokens = lang_tokens[None, ...]
        if lang_masks.ndim == 1:
            lang_masks = lang_masks[None, ...]

        next_logits, gen_state = self.synthesis.init_lang_generation(images, img_masks, lang_tokens, lang_masks)

        tokenizer = self.operator.tokenizer
        eos_id = tokenizer.eos_token_id
        generated: list[list[int]] = [[] for _ in range(lang_tokens.shape[0])]
        finished = torch.zeros(lang_tokens.shape[0], dtype=torch.bool, device=self.device)

        for _ in range(max_new_tokens):
            step_token = torch.argmax(next_logits, dim=-1).to(torch.long)
            step_token = torch.where(finished, torch.tensor(eos_id, device=step_token.device), step_token)
            for i in range(len(generated)):
                if not finished[i].item():
                    generated[i].append(int(step_token[i].item()))
            finished = finished | (step_token == eos_id)
            if torch.all(finished):
                break
            input_token = step_token.view(lang_tokens.shape[0], 1)
            next_logits, gen_state = self.synthesis.next_lang_logits(gen_state, input_token)

        return generated
