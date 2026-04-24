#!/bin/bash

# Function to display the help message and available methods
show_help() {
    echo "Usage: bash scripts/test_inference/test_nav_video_gen.sh [method_name]"
    echo ""
    echo "Available methods:"
    echo "  - matrix-game-2        : Run test_matrix_game_2.py"
    echo "  - infinite-world       : Run test_infinite_world.py"
    echo "  - matrix-game-3        : Run test_matrix_game_3.py (uses default HF repo id by default)"
    echo "  - hunyuan-gamecraft    : Run test_hunyuan_gamecraft.py"
    echo "  - hunyuanworld-voyager : Run test_hunyuan_world_voyager.py"
    echo "  - astra                : Run test_astra.py"
    echo "  - yume-1p5             : Run test_yume_1p5.py"
    echo "  - lingbot-world        : Run test_lingbot_world.py"
    echo ""
}

PYTHON_BIN=${PYTHON_BIN:-python}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN=python3
    else
        echo "Error: neither 'python' nor 'python3' is available in PATH."
        exit 1
    fi
fi

# Check if an argument is provided
if [ -z "$1" ]; then
    echo "Error: Please provide a method name to execute."
    show_help
    exit 1
fi

METHOD_NAME=$1

# Execute the corresponding command based on the input method name
case $METHOD_NAME in
    "matrix-game-2")
        echo "Executing: matrix_game_2..."
        CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" test/test_matrix_game_2.py
        ;;
    "matrix-game-3"|"matrix-game3")
        echo "Executing: matrix_game_3..."
        CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" test/test_matrix_game_3.py
        ;;
    "infinite-world")
        echo "Executing: infinite_world..."
        CUDA_VISIBLE_DEVICES=0 python test/test_infinite_world.py
        ;;
    "hunyuan-gamecraft")
        echo "Executing: hunyuan_gamecraft..."
        torchrun --nproc_per_node=1 test/test_hunyuan_gamecraft.py
        ;;
    "hunyuanworld-voyager")
        echo "Executing: hunyuan_world_voyager..."
        CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" test/test_hunyuan_world_voyager.py
        ;;
    "astra")
        echo "Executing: astra..."
        CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" test/test_astra.py
        ;;
    "yume-1p5")
        echo "Executing: yume..."
        CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" test/test_yume_1p5.py
        ;;
    "lingbot-world")
        echo "Executing: lingbot_world..."
        torchrun --nproc_per_node=2 test/test_lingbot_world.py
        ;;
    *)
        # If the input does not match any method, show an error message
        echo "Error: Unknown method name '$METHOD_NAME'"
        show_help
        exit 1
        ;;
esac
