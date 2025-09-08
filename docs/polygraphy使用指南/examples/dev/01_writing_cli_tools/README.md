# 编写自定义命令行工具


## 简介

Polygraphy 包含各种辅助工具，使从头编写新的命令行工具变得更容易。

在此示例中，我们将编写一个名为 `gen-data` 的全新工具，该工具将使用 Polygraphy 的默认数据加载器生成随机数据，然后将其写入输出文件。用户可以指定要生成的值的数量以及输出路径。

为此，我们将创建 `Tool` 的子类，并使用 Polygraphy 提供的 `DataLoaderArgs` 参数组。


## 运行示例

1. 您可以从此目录运行示例工具。例如：

    ```bash
    ./gen-data -o data.json --num-values 25
    ```

2. 我们甚至可以使用 `inspect data` 检查生成的数据：

    ```bash
    polygraphy inspect data data.json -s
    ```

要查看示例工具中可用的其他命令行选项，
请运行：
```bash
./gen-data -h
```
