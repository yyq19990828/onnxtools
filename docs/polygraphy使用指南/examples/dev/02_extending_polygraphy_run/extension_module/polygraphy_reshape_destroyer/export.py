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
此文件定义了将由我们的扩展模块导出的入口点。
`polygraphy run` 将使用此入口点来添加我们的自定义参数组。
"""

from polygraphy_reshape_destroyer.args import IdentityOnlyRunnerArgs, ReplaceReshapeArgs


# 入口点应该不接受参数并返回参数组实例的列表。
#
# 注意：参数组将按照提供的顺序进行解析，
#       并且在所有 Polygraphy 的内置参数组之后解析。
def export_argument_groups():
    return [
        ReplaceReshapeArgs(),
        IdentityOnlyRunnerArgs(),
    ]
