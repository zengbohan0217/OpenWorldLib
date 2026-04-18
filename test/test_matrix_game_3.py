import os
import json

from PIL import Image

from openworldlib.pipelines.matrix_game.pipeline_matrix_game_3 import MatrixGame3Pipeline


def main():
    # Keep test style close to MatrixGame2: a default pretrained model id.
    pretrained_model_path = "Skywork/Matrix-Game-3.0"

    image = Image.open("./data/test_case/test_image_case1/ref_image.png").convert("RGB")

    pipeline = MatrixGame3Pipeline.from_pretrained(model_path=pretrained_model_path, device="cuda")

    test_list = [
        {
            "name": "nav_left_right",
            "prompt": "A first-person interactive scene.",
            "interactions": ["forward", "left", "right"],
            "save_name": "matrix_game_3_demo_0",
        },
        {
            "name": "nav_camera_turn",
            "prompt": "A first-person interactive scene.",
            "interactions": ["forward", "camera_l", "forward", "camera_r"],
            "save_name": "matrix_game_3_demo_1",
        },
    ]

    for case in test_list:
        custom_video_path = os.path.join("./output", "custom", f"{case['save_name']}.mp4")
        result = pipeline(
            images=image,
            prompt=case["prompt"],
            interactions=case["interactions"],
            output_dir="./output",
            save_name=case["save_name"],
            size="704*1280",
            num_iterations=2,
            num_inference_steps=3,
            fa_version="0",
            save_video=False,
            return_result=True,
        )
        video_tensor = result.get("video_tensor")
        if video_tensor is not None:
            saved_path = pipeline.save_video_tensor(video_tensor, custom_video_path)
        else:
            saved_path = None
        case["video_path"] = saved_path
        case["video_tensor_shape"] = list(video_tensor.shape) if video_tensor is not None else None
        case["custom_video_exists"] = bool(saved_path and os.path.exists(saved_path))

    print(json.dumps(test_list, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
