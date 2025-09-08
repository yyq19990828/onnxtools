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


from setuptools import find_packages, setup

import polygraphy_reshape_destroyer

import os


def main():
    # 我们切换到项目根目录，以便 `setup.py` 可以从任何目录使用。
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(ROOT_DIR)

    setup(
        # 注意：对于您的自定义扩展模块，您需要编辑大部分这些字段。
        name="polygraphy_reshape_destroyer",
        version=polygraphy_reshape_destroyer.__version__,
        description="Polygraphy Reshape Destroyer: Destroyer Of Reshapes",
        long_description=open("README.md", "r", encoding="utf-8").read(),
        url="https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy",
        author="NVIDIA",
        author_email="svc_tensorrt@nvidia.com",
        classifiers=[
            "Intended Audience :: Developers",
            "Programming Language :: Python :: 3",
        ],
        license="Apache 2.0",
        install_requires=[
            "polygraphy",
            # 我们包含的加载器需要 ONNX-GraphSurgeon 来修改模型。
            "onnx_graphsurgeon",
            "numpy<2",
        ],
        packages=find_packages(exclude=("tests", "tests.*")),
        # entry_points 的格式是：
        # {
        #   "polygraphy.run.plugins"    # Polygraphy run 入口点（这个不应该更改）
        #      : ["<plugin-name>=module.<submodule...>:object"]
        # }
        entry_points={
            "polygraphy.run.plugins": [
                "reshape-destroyer=polygraphy_reshape_destroyer.export:export_argument_groups",
            ]
        },
        zip_safe=True,
        python_requires=">=3.6",
    )


if __name__ == "__main__":
    main()
