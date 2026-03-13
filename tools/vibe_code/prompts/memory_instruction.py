memory_prompt = """
The world model requires the implementation of memory management capabilities for interactive and multi-turn scenarios. Our framework needs to maintain interaction history and state across multiple rounds of generation or reasoning; therefore, a Memory class must be defined.

The Memory class is primarily invoked within the Pipeline class's stream() function. It serves as the state management backbone for multi-turn interactive pipelines, enabling the system to:
- Record interaction data (images, videos, text, audio, actions) after each turn
- Retrieve relevant context for the current turn based on history
- Compress and consolidate long-term memories to maintain efficiency
- Convert stored memories into model-ready formats (e.g., KV cache, latent tokens)
- Manage lifecycle events such as reset, eviction, and STM-to-LTM transfer

It should follow the structure below:
```python
class BaseMemory(object):
    ###### Generic Multimodal Memory System Template
    ###### Designed for VLM, VLA, and Generative/Reasoning tasks
    ######
    ###### Key interfaces used by the Pipeline:
    ######   - record(): called after each generation/reasoning turn
    ######   - select(): called before each turn to retrieve context
    ######   - manage(): called to reset or consolidate memory
    ######
    ###### Internal processing functions:
    ######   - compress(): reduce memory footprint
    ######   - process(): convert to model-ready format

    def __init__(self, capacity=None, **kwargs):
        #### Initialize storage structures and resource constraints
        #### self.storage should be a list of dicts following this template:
        #### [
        ####   {
        ####     'content': <data>,          # The actual content (PIL.Image, str, tensor, etc.)
        ####     'type': <type>,              # One of: 'image', 'video', 'text', 'audio', 'action', 'other'
        ####     'timestamp': <int>,          # Sequential index or frame count
        ####     'metadata': <dict>           # Additional info (e.g., interaction signal, model config)
        ####   },
        ####   ...
        #### ]
        self.storage = []
        self.capacity = capacity

    def check_template(self, **kwargs):
        #### Validate that self.storage entries follow the required template format
        #### Allowed types: ['image', 'video', 'text', 'audio', 'action', 'other']
        pass

    def record(self, data, metadata=None, **kwargs):
        #### 1. Recording (Ingestion)
        #### Purpose:
        ####   Ingest raw interaction data (images, generated frames, text responses, etc.)
        ####   after each turn of generation or reasoning
        #### Logic:
        ####   - Determine the data type (PIL.Image, list of frames, text, etc.)
        ####   - Assign metadata tags (timestamp, interaction signal, etc.)
        ####   - Append to self.storage in the template format
        #### This is called by Pipeline.stream() after each __call__ invocation
        pass

    def select(self, context_query=None, **kwargs):
        #### 2. Selection (Retrieval)
        #### Purpose:
        ####   Retrieve relevant memory entries to provide context for the current turn
        #### Logic:
        ####   - For simple cases: return the most recent entry (e.g., last generated frame)
        ####   - For complex cases: similarity matching, temporal correlation, or
        ####     importance-based filtering using context_query
        #### This is called by Pipeline.stream() before each __call__ invocation
        pass

    def compress(self, memory_items, **kwargs):
        #### 3. Compression (Refinement)
        #### Purpose:
        ####   Reduce memory size or distill key information from stored entries
        #### Logic:
        ####   - Text: summarization of conversation history
        ####   - Visual: extract key frames or feature vectors
        ####   - General: downsample, quantize, or merge similar entries
        pass

    def process(self, refined_data, target_format="kv_cache", **kwargs):
        #### 4. Processing (Adaptation)
        #### Purpose:
        ####   Convert refined memories into model-ready representations
        #### Logic:
        ####   - "kv_cache": convert to key-value cache for transformer models
        ####   - "latent": encode into latent token representations
        ####   - "embedding": convert to embedding vectors for retrieval
        pass

    def manage(self, action="reset", **kwargs):
        #### 5. Management (Lifecycle & Consolidation)
        #### Purpose:
        ####   Maintain long-term memory health and handle lifecycle events
        #### Logic:
        ####   - "reset": clear all storage and accumulated state
        ####   - "evict": remove oldest or least relevant entries
        ####   - "consolidate": merge Short-Term Memory (STM) into Long-Term Memory (LTM)
        pass
```
"""

example_memory_code = """
Here are the organized code results for matrix-game-2: https://github.com/SkyworkAI/Matrix-Game.
The Memory implementation is as follows:

```python
from ...base_memory import BaseMemory
import numpy as np
from PIL import Image
from typing import Optional


def tensor_to_pil(tensor: np.ndarray) -> Image.Image:
    # Convert a numpy array (float [0,1]) to PIL Image (uint8)
    last_frame = (tensor * 255).astype(np.uint8)
    return Image.fromarray(last_frame)


class MatrixGame2Memory(BaseMemory):
    # Memory module for MatrixGame2 interactive video generation.
    # Stores generated frames across multiple interaction turns,
    # enabling continuous video generation from the last frame.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.storage = []       # List of dicts following BaseMemory template
        self.all_frames = []    # Accumulated video frames across all turns

    def record(self, data, **kwargs):
        # Record interaction data after each generation turn.
        # Accepts either:
        #   - PIL.Image: initial input image (first turn)
        #   - list of numpy arrays: generated video frames (subsequent turns)
        if isinstance(data, Image.Image):
            current_image = data
        elif isinstance(data, list):
            last_frame = data[-1]
            current_image = tensor_to_pil(last_frame)
            self.all_frames.extend(data)
        self.storage.append({
            'content': current_image,
            'type': 'image',
            'timestamp': len(self.all_frames),
            'metadata': {}
        })

    def select(self, **kwargs) -> Optional[Image.Image]:
        # Retrieve the most recent frame as input for the next generation turn.
        # Returns None if no data has been recorded yet.
        if len(self.storage) == 0:
            return None
        return self.storage[-1]['content']

    def manage(self, action: str = "reset", **kwargs):
        # Manage memory lifecycle.
        # "reset": clear all storage and accumulated frames (used at stream start)
        if action == "reset":
            self.storage = []
            self.all_frames = []
```

The Memory module is used in the Pipeline's stream() function as follows.
Note how record() and select() work together to chain multi-turn interactions:

```python
class MatrixGame2Pipeline:

    def stream(self,
               interaction_signal: List[str],
               initial_image: Optional[Image.Image] = None,
               num_output_frames: int = 15,
               resize_H: int = 352,
               resize_W: int = 640,
               operation_visualization: bool = False,
               **kwds) -> torch.Tensor:
        # Multi-turn interactive generation using Memory for state management.
        #
        # Turn 1: initial_image is provided, recorded into memory
        # Turn 2+: initial_image is None, last frame is retrieved from memory
        #
        # After each generation, the output video is recorded back into memory,
        # so the next turn can continue from the last generated frame.

        if initial_image is not None:
            print("--- Stream Started ---")
            self.memory_module.record(initial_image)

        # Retrieve the most recent frame as input for this turn
        current_image = self.memory_module.select()
        if current_image is None:
            raise ValueError("No image in storage. Provide 'initial_image' first.")

        # Run generation using the retrieved frame
        video_output = self.__call__(
            input_image=current_image,
            num_output_frames=num_output_frames,
            interaction_signal=interaction_signal,
            resize_H=resize_H,
            resize_W=resize_W,
            operation_visualization=operation_visualization,
            **kwds
        )

        # Record generated output for the next turn
        self.memory_module.record(video_output)
        return video_output
```

The corresponding test_stream file demonstrates multi-turn usage:

```python
from openworldlib.pipelines.matrix_game.pipeline_matrix_game_2 import MatrixGame2Pipeline
from PIL import Image

pipeline = MatrixGame2Pipeline.from_pretrained(
    synthesis_model_path="Skywork/Matrix-Game-2.0",
    mode="universal",
    device="cuda"
)

input_image = Image.open("./data/test_case1/ref_image.png").convert('RGB')
turn_idx = 0

while True:
    interaction_input = input(f"[Turn {turn_idx}] Enter interaction(s) (or 'q' to stop): ")
    if interaction_input in ['q', 'n']:
        break

    current_signal = [s.strip() for s in interaction_input.split(',')]
    start_img = input_image if turn_idx == 0 else None

    video_output = pipeline.stream(
        interaction_signal=current_signal,
        initial_image=start_img,
        num_output_frames=15,
    )
    turn_idx += 1

# Export accumulated video
from diffusers.utils import export_to_video
export_to_video(pipeline.memory_module.all_frames, "output.mp4", fps=12)
```
"""
