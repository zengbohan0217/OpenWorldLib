# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""
LTX-2 sink-bidirectional refiner base model.

Wraps ``diffusers LTX2VideoTransformer3DModel`` + ``LTX2TextConnectors`` to provide
a clean interface for cross-model refiner reuse.
"""

from .ltx2_refiner import DiffusersLTX2Refiner, STAGE_2_DISTILLED_SIGMA_VALUES

__all__ = ["DiffusersLTX2Refiner", "STAGE_2_DISTILLED_SIGMA_VALUES"]
