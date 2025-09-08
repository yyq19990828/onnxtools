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
此文件定义了 `ReplaceReshapes` 加载器，它接受一个 ONNX-GraphSurgeon 图
并将任何无操作的 Reshape 节点替换为 Identity 节点。

该加载器实现了标准的 `BaseLoader` 接口。
"""

from typing import Callable, Union

from polygraphy import mod, util
from polygraphy.backend.base import BaseLoader
from polygraphy.logger import G_LOGGER

# 对于外部依赖项或任何 Polygraphy 后端
# （除了 `polygraphy.backend.base` 之外），您应该使用 `mod.lazy_import`。
#
# 这将使 Polygraphy 能够在需要时在运行时自动安装依赖项，并且
# 将避免对外部包创建硬依赖。
#
# 注意：顾名思义，`lazy_import` 在第一次访问之前*不会*导入模块。
#       因此，您应该小心避免以下反模式：
#
#   my_module = mod.lazy_import("my_module")
#   submodule = my_module.submodule
#
# 第二行将触发 `my_module` 的立即导入。
# 相反，使用类似以下的方式：
#
#   submodule = mod.lazy_import("my_module.submodule")
#
gs = mod.lazy_import("onnx_graphsurgeon")


# `mod.export()` 将装饰的类或函数添加到此模块的 __all__ 属性中。
# 当我们从此子模块中的 `__init__.py` 文件执行 `import *` 时，这将确保
# 仅导出装饰的对象。
#
# 注意：我们使用 `funcify=True`，以便为我们自动生成一个立即求值的函数式加载器（称为 `replace_reshapes`）。
#       这不会被命令行工具使用，但如果此模块通过 Python API 使用，则可能有用。
#
@mod.export(funcify=True)
class ReplaceReshapes(BaseLoader):
    """
    函数对象，在 ONNX-GraphSurgeon 图中用 Identity 替换无操作的 Reshape 节点。
    """

    def __init__(
        self, graph: Union[gs.Graph, Callable[[], gs.Graph]], rename_nodes: bool = None
    ):
        """
        在 ONNX-GraphSurgeon 图中用 Identity 替换无操作的 Reshape 节点。

        参数:
            graph (Union[gs.Graph, Callable() -> gs.Graph]):
                    ONNX-GraphSurgeon 图或返回图的可调用对象。
            rename_nodes (bool):
                    当我们将 Reshape 节点转换为 Identity 时是否重命名节点。
                    默认为 False。
        """
        # 除了直接接受 `gs.Graph` 之外，我们还将支持可调用对象，例如 Polygraphy 加载器。
        # 这将允许我们的加载器与其他 Polygraphy 加载器组合使用。
        #
        # 由于 `graph` 参数可能是一个可调用对象，我们将其分配给一个“私有”成员，即以 '_' 为前缀，
        # 以避免与实际的 `gs.Graph` 混淆。
        #
        self._graph = graph

        # 有关为什么我们使用这种方法而不是标准 Python 默认参数的详细信息，请参阅 `util.default` 中的注释。
        self.rename_nodes = util.default(rename_nodes, False)

    # `call_impl` 方法负责执行加载器的实际工作。
    @util.check_called_by("__call__")
    def call_impl(self):
        """
        返回:
            gs.Graph: 将无操作 Reshape 节点替换为 Identity 的图。
        """
        # 如前所述，`self._graph` 可能是一个可调用对象，所以在这里如果需要我们就调用它。
        #
        # 提示：`invoke_if_callable` 返回的第二个值（这里未使用）是一个布尔值，指示
        #      参数是否确实是一个可调用对象。
        #
        graph, _ = util.invoke_if_callable(self._graph)

        for node in graph.nodes:
            if node.op != "Reshape":
                continue

            # 除非新形状在推理时间之前已知，即一个常数，否则我们无法确定 Reshape 是否为无操作。
            if not isinstance(node.inputs[1], gs.Constant):
                continue

            # 只有当新形状与旧形状相同时，Reshape 才是无操作的。
            new_shape = node.inputs[1].values
            if list(node.inputs[0].shape) != list(new_shape):
                continue

            # 用 Identity 替换无操作的 reshape。我们可以简单地编辑操作符名称，
            # 清除任何属性，然后删除第二个输入。
            G_LOGGER.info(f"Replacing no-op reshape: {node.name} with an Identity node")
            if self.rename_nodes:
                node.name += "_destroyed"
                G_LOGGER.info(f"Renamed Identity node to: {node.name}")

            node.op = "Identity"
            node.attrs.clear()
            del node.inputs[1]

        # 最后，清理图以移除任何悬空的张量并返回它。
        graph.cleanup()
        return graph
