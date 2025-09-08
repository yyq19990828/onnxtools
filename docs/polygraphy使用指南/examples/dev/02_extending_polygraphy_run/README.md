# 扩展 `polygraphy run`

## 简介

`polygraphy run` 允许您使用多个后端（包括 TensorRT 和 ONNX-Runtime）运行推理，并比较输出结果。
尽管它确实提供了从不受支持的后端加载和比较自定义输出的机制，
通过扩展模块添加对后端的支持可以使其更无缝地集成，
提供更好的用户体验。

在此示例中，我们将为 `polygraphy run` 创建一个名为 `polygraphy_reshape_destroyer` 的扩展模块，
它将包括以下内容：

- 一个特殊的加载器，将在 ONNX 模型中用 `Identity` 节点替换无操作的 `Reshape` 节点。

- 一个支持仅包含 `Identity` 节点的 ONNX 模型的自定义运行器。

- 命令行选项：
    - 在加载器应用转换时启用或禁用节点重命名。
    - 以 `slow`、`medium` 或 `fast` 模式运行模型。
        在 `slow` 和 `medium` 模式下，我们将在推理期间注入 `time.sleep()`
        （这将在 `fast` 模式下实现巨大的性能提升！）。

## 背景

尽管此示例是自包含的，并且概念将在您遇到它们时得到解释，但仍然
建议您首先熟悉
[Polygraphy 的 `Loader` 和 `Runner` API](../../../polygraphy/README.md)、
[`Argument Group` 接口](../../../polygraphy/tools/args/README.md)，
以及 [`Script` 接口](../../../polygraphy/tools/script.py)。

之后，为 `polygraphy run` 创建扩展模块只需要定义您的
自定义 `Loader`/`Runner` 和 `Argument Group`，并通过
`setuptools` 的 `entry_points` API 使它们对 Polygraphy 可见。

*注意：严格来说并不需要定义自定义 `Loader`，但为了完整性，本示例中会涉及此内容。*

按照惯例，Polygraphy 扩展模块名称以 `polygraphy_` 作为前缀。

## 阅读示例代码

我们已经结构化了我们的示例扩展模块，使其在某种程度上反映了 Polygraphy 仓库的结构。
这应该使您更容易看到扩展模块中的功能与 Polygraphy 原生提供的功能之间的平行关系。
结构如下：
<!-- Polygraphy Test: Ignore Start -->
```bash
- extension_module/
    - polygraphy_reshape_destroyer/
        - backend/
            - __init__.py   # Controls submodule-level exports
            - loader.py     # Defines our custom loader.
            - runner.py     # Defines our custom runner.
        - args/
            - __init__.py   # Controls submodule-level exports
            - loader.py     # Defines command-line argument group for our custom loader.
            - runner.py     # Defines command-line argument group for our custom runner.
        - __init__.py       # Controls module-level exports
        - export.py         # Defines the entry-point for `polygraphy run`.
    - setup.py              # Builds our module
```
<!-- Polygraphy Test: Ignore End -->

建议您按以下顺序阅读这些文件：

1. [backend/loader.py](./extension_module/polygraphy_reshape_destroyer/backend/loader.py)
2. [backend/runner.py](./extension_module/polygraphy_reshape_destroyer/backend/runner.py)
3. [backend/\_\_init\_\_.py](./extension_module/polygraphy_reshape_destroyer/backend/__init__.py)
4. [args/loader.py](./extension_module/polygraphy_reshape_destroyer/args/loader.py)
5. [args/runner.py](./extension_module/polygraphy_reshape_destroyer/args/runner.py)
6. [args/\_\_init\_\_.py](./extension_module/polygraphy_reshape_destroyer/args/__init__.py)
7. [\_\_init\_\_.py](./extension_module/polygraphy_reshape_destroyer/__init__.py)
8. [export.py](./extension_module/polygraphy_reshape_destroyer/export.py)
9. [setup.py](./extension_module/setup.py)


## 运行示例

1. 构建并安装扩展模块：

    使用 `setup.py` 构建：

    ```bash
    python3 extension_module/setup.py bdist_wheel
    ```

    安装 wheel 包：

    ```bash
    python3 -m pip install extension_module/dist/polygraphy_reshape_destroyer-0.0.1-py3-none-any.whl \
        --extra-index-url https://pypi.ngc.nvidia.com
    ```

    *提示：如果您对示例扩展模块进行了更改，可以通过*
    *重新构建（按照步骤 1）然后运行以下命令来更新已安装的版本：*

    ```bash
    python3 -m pip install extension_module/dist/polygraphy_reshape_destroyer-0.0.1-py3-none-any.whl \
        --force-reinstall --no-deps
    ```

2. 一旦安装了扩展模块，您应该在 `polygraphy run` 的帮助输出中看到您添加的选项：

    ```bash
    polygraphy run -h
    ```

3. 接下来，我们可以使用包含无操作 Reshape 的 ONNX 模型测试我们的自定义运行器：

    ```bash
    polygraphy run no_op_reshape.onnx --res-des
    ```

4. 我们还可以试用我们添加的一些其他命令行选项：

    - 重命名被替换的节点：

        ```bash
        polygraphy run no_op_reshape.onnx --res-des --res-des-rename-nodes
        ```

    - 不同的推理速度：

        ```bash
        polygraphy run no_op_reshape.onnx --res-des --res-des-speed=slow
        ```

        ```bash
        polygraphy run no_op_reshape.onnx --res-des --res-des-speed=medium
        ```

        ```bash
        polygraphy run no_op_reshape.onnx --res-des --res-des-speed=fast
        ```

5. 最后，让我们将我们的实现与 ONNX-Runtime 进行比较，以确保其功能正确：

    ```bash
    polygraphy run no_op_reshape.onnx --res-des --onnxrt
    ```
