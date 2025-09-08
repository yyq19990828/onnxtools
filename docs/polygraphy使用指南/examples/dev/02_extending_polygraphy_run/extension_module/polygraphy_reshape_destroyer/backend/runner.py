#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
此文件定义了 `IdentityOnlyRunner` 运行器，它接受一个仅包含 Identity 节点的
 ONNX-GraphSurgeon 图并运行推理。

该运行器实现了标准的 `BaseRunner` 接口。
"""

import copy
import time
from collections import OrderedDict

from polygraphy import mod, util
from polygraphy.backend.base import BaseRunner
from polygraphy.common import TensorMetadata
from polygraphy.logger import G_LOGGER


@mod.export()
class IdentityOnlyRunner(BaseRunner):
    """
    使用自定义 Python 代码运行推理。
    仅支持仅包含 Identity 节点的模型。
    """

    def __init__(self, graph, name=None, speed: str = None):
        """
        参数:
            graph (Union[onnx_graphsurgeon.Graph, Callable() -> onnx_graphsurgeon.Graph]):
                    ONNX-GraphSurgeon 图或返回图的可调用对象。
            name (str):
                    此运行器使用的人类可读名称前缀。
                    运行器计数和时间戳将附加到此前缀。
            speed (str):
                    运行推理的速度。应该是 [“slow”, “medium”, “fast”] 中的一个。
                    默认为 “fast”。
        """
        super().__init__(name=name, prefix="pluginref-runner")
        self._graph = graph

        self.speed = util.default(speed, "fast")

        VALID_SPEEDS = ["slow", "medium", "fast"]
        if self.speed not in VALID_SPEEDS:
            # 与 Polygraphy 一样，扩展模块应该对任何不可恢复的错误使用 `G_LOGGER.critical()`。
            G_LOGGER.critical(
                f"Invalid speed: {self.speed}. Note: Valid speeds are: {VALID_SPEEDS}"
            )

    @util.check_called_by("activate")
    def activate_impl(self):
        # 与加载器一样，`graph` 参数可能是 `gs.Graph` 或返回图的可调用对象，
        # 例如加载器，所以我们尝试调用它。
        self.graph, _ = util.invoke_if_callable(self._graph)

    #
    # 从这个点往后的所有方法都保证只在 `activate()` 之后调用，
    # 所以我们可以假设 `self.graph` 将可用。
    #

    @util.check_called_by("get_input_metadata")
    def get_input_metadata_impl(self):
        # 输入元数据被 Polygraphy 的默认数据加载器用来确定
        # 输入缓冲区的所需形状和数据类型。
        meta = TensorMetadata()
        for tensor in self.graph.inputs:
            meta.add(tensor.name, tensor.dtype, tensor.shape)
        return meta

    @util.check_called_by("infer")
    def infer_impl(self, feed_dict):
        start = time.time()

        # 由于我们的运行器仅支持 Identity，我们在推理时只需要将节点输出绑定到它们的输入。
        # 我们将从输入张量的副本开始：
        tensor_values = copy.copy(feed_dict)

        for node in self.graph.nodes:
            # 我们不支持非 Identity 节点，所以如果我们看到一个就会报告错误
            if node.op != "Identity":
                G_LOGGER.critical(
                    f"Encountered an unsupported type of node: {node.op}."
                    "Note: This runner only supports Identity nodes!"
                )

            inp_tensor = node.inputs[0]
            out_tensor = node.outputs[0]
            # Identity 节点的输出应该与其输入相同。
            tensor_values[out_tensor.name] = tensor_values[inp_tensor.name]

        # 根据 `self.graph.outputs` 查找输出张量并创建一个我们可以返回的字典：
        outputs = OrderedDict()
        for out in self.graph.outputs:
            outputs[out.name] = tensor_values[out.name]

        # 接下来我们将实现我们的人工延迟，这样我们就可以在“fast”模式下看到惊人的性能提升！
        delay = {"slow": 1.0, "medium": 0.5, "fast": 0.0}[self.speed]
        time.sleep(delay)

        end = time.time()

        # 为了允许 Polygraphy 准确报告推理时间，运行器负责报告
        # 它们自己的推理时间。这通过设置 `self.inference_time` 属性来完成。
        self.inference_time = end - start
        return outputs

    @util.check_called_by("deactivate")
    def deactivate_impl(self):
        del self.graph
