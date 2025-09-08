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
此文件定义了 `IdentityOnlyRunnerArgs` 参数组，它管理
控制 `IdentityOnlyRunner` 运行器的命令行选项。

该参数组实现了标准的 `BaseRunnerArgs` 接口，该接口继承自 `BaseArgs`。
"""

from polygraphy import mod
from polygraphy.tools.args import util as args_util
from polygraphy.tools.args.base import BaseRunnerArgs
from polygraphy.tools.script import make_invocable
from polygraphy_reshape_destroyer.args.loader import ReplaceReshapeArgs


# 注意：与加载器参数组很相似，运行器参数组可能依赖于其他参数组。
@mod.export()
class IdentityOnlyRunnerArgs(BaseRunnerArgs):
    """
    仅 Identity 运行器推理：使用仅 Identity 运行器运行推理。

    依赖于:

        - ReplaceReshapeArgs
    """

    def get_name_opt_impl(self):
        # 与常规的 `BaseArgs` 参数组不同，运行器参数组还需要
        # 为运行器提供人类可读的名称以及用于
        # 切换运行器的选项名称，不包括前导的破折号。
        #
        # 我们将使用 "res-des" 作为选项，这将允许我们通过设置 `--res-des` 来使用运行器。
        return "仅 Identity 运行器", "res-des"

    def add_parser_args_impl(self):
        # 再一次，为了防止与其他 Polygraphy 选项冲突，我们在选项前加上 `res-des` 前缀。
        self.group.add_argument(
            "--res-des-speed",
            help="运行推理的速度",
            choices=["slow", "medium", "fast"],
            # 由于我们的运行器使用 `util.default`，我们可以使用 `None` 作为通用默认值。
            default=None,
        )

    def parse_impl(self, args):
        """
        解析命令行参数并填充以下属性:

        属性:
            speed (str): 运行推理的速度。
        """
        self.speed = args_util.get(args, "res_des_speed")

    def add_to_script_impl(self, script):
        # 我们将依赖我们的 ReplaceReshapeArgs 参数组为我们创建 ONNX-GraphSurgeon 图：
        loader_name = self.arg_groups[ReplaceReshapeArgs].add_to_script(script)

        # 接下来，我们将为我们的运行器添加一个导入。
        script.add_import(
            imports=["IdentityOnlyRunner"], frm="polygraphy_reshape_destroyer.backend"
        )
        # 最后，我们可以使用 `Script.add_runner()` API 添加我们的运行器。
        # 与加载器实现一样，可以直接向 `make_invocable` 提供额外的参数。
        script.add_runner(
            make_invocable("IdentityOnlyRunner", loader_name, speed=self.speed)
        )

        # 注意：与常规 `BaseArgs` 的 `add_to_script_impl` 方法不同，`BaseRunnerArgs` 的该方法
        #       不需要返回任何内容。
