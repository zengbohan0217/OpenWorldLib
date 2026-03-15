import json
from PIL import Image

from openworldlib.pipelines.spirit_ai.pipeline_spirit_v1p5 import SpiritV1p5Pipeline
from openworldlib.pipelines.libero.pipeline_libero import LiberoPipeline


# Path to a specific LIBERO BDDL task file
BDDL_FILE = "./data/test_case/test_vla_case1/libero/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.bddl"


def main():
    # ---- Spirit-v1.5 pipeline setup (same as test_spirit_v1p5.py) ----
    ckpt_path = "Spirit-AI-robotics/Spirit-v1.5"
    norm_stats_path = "./data/test_case/test_vla_case1/norm_stats.json"
    meta_path = "./data/test_case/test_vla_case1/libero/meta.json"

    spirit_pipeline = SpiritV1p5Pipeline.from_pretrained(
        pretrained_model_path=ckpt_path,
        norm_stats_path=norm_stats_path,
        device="cuda",
        use_bf16=True,
    )

    with open(meta_path, "r") as f:
        obs_data = json.load(f)

    images = {
        "cam_high": Image.open(obs_data["paths"]["main_view"]).convert("RGB"),
        "cam_left_wrist": Image.open(obs_data["paths"]["wrist_view"]).convert("RGB"),
    }
    raw_state = obs_data["observation"]["state"]
    task = obs_data.get("task", "Pick up the object")
    robot_type = obs_data.get("robot_type", "Franka")

    # ---- Run Spirit pipeline ----
    actions = spirit_pipeline(
        images=images,
        raw_state=raw_state,
        task=task,
        robot_type=robot_type,
        return_all_steps=True,
    )

    print(f"Number of action steps: {len(actions)}")
    print(f"First action: {actions[0]}")

    # ---- Ask user whether to save video ----
    save_video = input("Save video? [y/N] ").strip().lower() == "y"
    if save_video:
        default_path = obs_data.get("libero_video_path")
        custom_path = input(f"Output path [{default_path}]: ").strip()
        video_path = custom_path if custom_path else default_path
    else:
        video_path = None

    # ---- Build LIBERO pipeline and visualize ----
    libero_pipeline = LiberoPipeline.from_pretrained(
        bddl_file=obs_data.get("bddl_file", BDDL_FILE),
        benchmark_name="libero_10",
        task_id=obs_data.get("task_id", 0),
        # norm_stats=norm_stats,  # pass if actions need unnormalization
        action_dim=7,
        device=0,
    )

    result = libero_pipeline(
        actions=actions,
        video_path=video_path,
        fps=30,
    )

    print(f"Success: {result['success']}")
    if result["video_path"]:
        print(f"Video saved to: {result['video_path']}")
    else:
        print("Video not saved.")


if __name__ == "__main__":
    main()
