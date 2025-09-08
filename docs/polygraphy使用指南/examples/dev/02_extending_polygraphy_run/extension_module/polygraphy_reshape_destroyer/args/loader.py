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
此文件定义了 `ReplaceReshapeArgs` 参数组，它管理
控制 `ReplaceReshape` 加载器的命令行选项。

该参数组实现了标准的 `BaseArgs` 接口。
"""


from polygraphy import mod
from polygraphy.tools.args import OnnxLoadArgs
from polygraphy.tools.args import util as args_util
from polygraphy.tools.args.base import BaseArgs
from polygraphy.tools.script import make_invocable


# 注意：我们的参数组可以依赖于 `polygraphy run` 订阅的任何参数组。
#       有关完整列表，请参阅 `polygraphy/tools/run/run.py`。
#       在这种情况下，我们将利用 OnnxLoadArgs 为我们加载 ONNX 模型。
@mod.export()
class ReplaceReshapeArgs(BaseArgs):
    # 参数组采用其文档字符串的标准化格式：
    #
    #  - 第一行必须包含用冒号（':'）分隔的标题和描述。
    #   描述应回答问题：“这个参数组负责什么？”。
    #
    # - 如果我们的参数组依赖于其他参数组，我们还必须添加一个 `Depends on:` 部分
    #   列出我们的依赖项。
    #
    # 有关预期格式的更多详细信息，请参阅 `BaseArgs` 文档字符串。
    #
    """
    ONNX Reshape 替换：在 ONNX 模型中用 Identity 替换无操作的 Reshape 节点

    依赖于:

        - OnnxLoadArgs
    """

    # 添加我们为加载器想要的任何命令行选项。
    def add_parser_args_impl(self):
        # `BaseArgs` 构造函数将自动将 `self.group` 设置为一个 `argparse`
        # 参数组，我们可以向其中添加我们的命令行选项。
        #
        # 注意：为了防止与其他 Polygraphy 选项冲突，我们将我们添加的所有选项
        #       都以 `--res-des` 作为前缀，即 `REShape DEStroyer` 的简称。
        self.group.add_argument(
            "--res-des-rename-nodes",
            help="如果节点被替换是否重命名",
            action="store_true",
            default=None,
        )

    # 接下来，我们将为我们添加的参数实现解析代码。
    # 这将允许我们的参数组被其他参数组使用。
    def parse_impl(self, args):
        # `parse_impl` 的文档字符串必须记录它填充的属性。
        # 这些属性被认为是参数组公共接口的一部分，
        # 并且可能被其他参数组和/或命令行工具使用。
        """
        解析命令行参数并填充以下属性:

        属性:
            rename_nodes (bool): 如果节点被替换是否重命名。
        """
        # 我们将使用 `args_util.get` 从 `args` 中检索属性，如果找不到属性，将返回 `None`。
        # 这将确保我们的参数组在代码中禁用命令行选项时仍能继续工作。
        self.rename_nodes = args_util.get(args, "res_des_rename_nodes")

    # 最后，我们可以实现将代码添加到脚本的逻辑。
    def add_to_script_impl(self, script):
        # 首先，我们将使用 `OnnxLoadArgs` 添加代码来加载 ONNX 模型。
        # 这将确保我们的参数组遵守与 ONNX 模型加载相关的任何选项。
        # `OnnxLoadArgs` 的 `add_to_script` 方法将返回加载 ONNX 模型的加载器的名称。
        loader_name = self.arg_groups[OnnxLoadArgs].add_to_script(script)

        # 接下来，我们将添加 Polygraphy 的 `GsFromOnnx` 加载器，以便我们可以将 ONNX 模型转换为
        # 可以传递给我们自定义加载器的 ONNX-GraphSurgeon 图。
        #
        # 首先，从 Polygraphy 导入加载器：
        script.add_import(imports=["GsFromOnnx"], frm="polygraphy.backend.onnx")
        # 接下来，使用参数（在这种情况下是 ONNX 模型加载器名称）调用加载器，并将其添加到脚本中。
        loader_name = script.add_loader(
            make_invocable("GsFromOnnx", loader_name), loader_id="gs_from_onnx"
        )

        # 最后，添加 ReplaceReshapeArgs 加载器。
        # 与 Polygraphy 加载器不同，我们需要从扩展模块导入我们的加载器。
        script.add_import(
            imports=["ReplaceReshapes"], frm="polygraphy_reshape_destroyer.backend"
        )
        # 添加加载器并返回 ID，以便后续的加载器或运行器可以使用它。
        # 注意：我们可以向 `make_invocable` 提供额外的位置参数和关键字参数，以将它们传递给加载器。
        return script.add_loader(
            make_invocable(
                "ReplaceReshapes", loader_name, rename_nodes=self.rename_nodes
            ),
            loader_id="replace_reshapes",
        )
