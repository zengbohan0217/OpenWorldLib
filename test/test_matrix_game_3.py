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
        video_path = pipeline(
            images=image,
            prompt=case["prompt"],
            interactions=case["interactions"],
            output_dir="./output",
            save_name=case["save_name"],
            size="704*1280",
            num_iterations=2,
            num_inference_steps=3,
            fa_version="0",
        )
        case["video_path"] = video_path

    print(json.dumps(test_list, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
