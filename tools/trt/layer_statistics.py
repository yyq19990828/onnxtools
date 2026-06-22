#!/usr/bin/env python3
"""
层统计工具 - 使用 Polygraphy 现成函数

该脚本使用 Polygraphy 库的现成函数来分析 ONNX 模型和 TensorRT 网络中的所有层和张量信息，
模拟 --onnx-outputs mark all 和 --trt-outputs mark all 的行为。

使用方法:
    python layer_statistics.py --model model.onnx
    python layer_statistics.py --model model.onnx --build-trt
    python layer_statistics.py --model model.onnx --save-json stats.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from polygraphy.backend.onnx import ModifyOutputs, OnnxFromPath
    from polygraphy.backend.onnx.util import all_tensor_names, get_num_nodes
    from polygraphy.constants import MARK_ALL

    print("✅ Polygraphy ONNX 后端导入成功")
except ImportError as e:
    print(f"❌ Polygraphy ONNX 后端导入失败: {e}")
    sys.exit(1)

try:
    import tensorrt as trt
    from polygraphy.backend.trt import ModifyNetworkOutputs, NetworkFromOnnxPath
    from polygraphy.backend.trt.util import get_all_tensors

    TRT_AVAILABLE = True
    print("✅ Polygraphy TensorRT 后端导入成功")
except ImportError as e:
    print(f"⚠️  Polygraphy TensorRT 后端不可用: {e}")
    TRT_AVAILABLE = False

try:
    import onnx

    print("✅ ONNX 库导入成功")
except ImportError:
    print("❌ 需要安装 onnx 库: pip install onnx")
    sys.exit(1)


class PolygraphyONNXAnalyzer:
    """使用 Polygraphy ONNX 后端的分析器"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.onnx_model = None
        self.load_model()

    def load_model(self):
        """使用 Polygraphy 加载 ONNX 模型"""
        try:
            # 使用 Polygraphy 的 OnnxFromPath 加载模型
            loader = OnnxFromPath(self.model_path)
            self.onnx_model = loader()
            print(f"✅ 使用 Polygraphy 成功加载 ONNX 模型: {self.model_path}")
        except Exception as e:
            print(f"❌ 使用 Polygraphy 加载 ONNX 模型失败: {e}")
            sys.exit(1)

    def get_all_tensor_names_with_polygraphy(self) -> dict[str, list[str]]:
        """使用 Polygraphy 的 all_tensor_names 函数获取张量名称"""
        # 获取所有非常量张量（不包含输入）
        all_outputs = all_tensor_names(self.onnx_model, include_inputs=False)

        # 获取包含输入的所有张量
        all_outputs_with_inputs = all_tensor_names(self.onnx_model, include_inputs=True)

        # 获取输入张量名称
        input_names = [inp.name for inp in self.onnx_model.graph.input]

        # 获取输出张量名称
        output_names = [out.name for out in self.onnx_model.graph.output]

        return {
            "non_constant_tensors": all_outputs,  # mark all 标记的张量
            "all_tensors_with_inputs": all_outputs_with_inputs,
            "input_tensors": input_names,
            "output_tensors": output_names,
        }

    def analyze_with_mark_all(self) -> dict[str, Any]:
        """模拟 --onnx-outputs mark all 的行为"""
        try:
            # 使用 ModifyOutputs 和 MARK_ALL 来模拟 mark all 行为
            modify_loader = ModifyOutputs(OnnxFromPath(self.model_path), outputs=MARK_ALL)
            modified_model = modify_loader()

            print("✅ 使用 ModifyOutputs(outputs=MARK_ALL) 成功修改模型")

            # 获取修改后的输出张量
            modified_output_names = [out.name for out in modified_model.graph.output]

            return {
                "original_outputs_count": len([out.name for out in self.onnx_model.graph.output]),
                "mark_all_outputs_count": len(modified_output_names),
                "mark_all_output_names": modified_output_names,
            }

        except Exception as e:
            print(f"❌ ModifyOutputs 操作失败: {e}")
            return {}

    def analyze_model_info(self) -> dict[str, Any]:
        """分析模型基本信息"""
        stats = {
            "model_info": {
                "model_path": self.model_path,
                "ir_version": self.onnx_model.ir_version,
                "producer_name": self.onnx_model.producer_name,
                "producer_version": self.onnx_model.producer_version,
            },
            "graph_info": {
                "name": self.onnx_model.graph.name,
                "total_nodes": len(self.onnx_model.graph.node),
                "total_inputs": len(self.onnx_model.graph.input),
                "total_outputs": len(self.onnx_model.graph.output),
                "total_initializers": len(self.onnx_model.graph.initializer),
            },
        }

        # 使用 Polygraphy 的 get_num_nodes 函数
        try:
            polygraphy_node_count = get_num_nodes(self.onnx_model)
            stats["polygraphy_node_count"] = polygraphy_node_count
        except Exception as e:
            print(f"⚠️  get_num_nodes 调用失败: {e}")

        # 张量分析
        tensor_info = self.get_all_tensor_names_with_polygraphy()
        stats["tensor_analysis"] = {
            "mark_all_tensor_count": len(tensor_info["non_constant_tensors"]),
            "mark_all_tensor_names": tensor_info["non_constant_tensors"],
            "all_tensors_count": len(tensor_info["all_tensors_with_inputs"]),
            "input_tensor_names": tensor_info["input_tensors"],
            "output_tensor_names": tensor_info["output_tensors"],
        }

        # mark all 行为分析
        mark_all_info = self.analyze_with_mark_all()
        stats["mark_all_analysis"] = mark_all_info

        # 层类型统计
        layer_types = defaultdict(int)
        for node in self.onnx_model.graph.node:
            layer_types[node.op_type] += 1

        stats["layer_types"] = dict(layer_types)
        stats["most_common_layers"] = sorted(layer_types.items(), key=lambda x: x[1], reverse=True)[:10]

        return stats


class PolygraphyTensorRTAnalyzer:
    """使用 Polygraphy TensorRT 后端的分析器"""

    def __init__(self, model_path: str):
        if not TRT_AVAILABLE:
            raise ImportError("TensorRT 不可用")

        self.model_path = model_path
        self.network_info = None
        self.setup_network()

    def setup_network(self):
        """使用 Polygraphy 构建 TensorRT 网络"""
        try:
            # 使用 Polygraphy 的 NetworkFromOnnxPath 构建网络
            network_loader = NetworkFromOnnxPath(self.model_path)
            builder, network, parser = network_loader()

            self.network_info = {"builder": builder, "network": network, "parser": parser}

            print("✅ 使用 Polygraphy NetworkFromOnnxPath 成功构建 TensorRT 网络")

        except Exception as e:
            print(f"❌ 使用 Polygraphy 构建 TensorRT 网络失败: {e}")
            raise

    def get_all_tensors_with_polygraphy(self) -> dict[str, Any]:
        """使用 Polygraphy 的 get_all_tensors 函数获取张量信息"""
        network = self.network_info["network"]

        # 使用 Polygraphy 的 get_all_tensors 函数
        all_tensors = get_all_tensors(network)

        # 按照层的顺序收集张量信息
        ordered_tensor_names = []
        tensor_info = {}

        # 遍历每一层，按顺序收集张量
        for layer_idx in range(network.num_layers):
            layer = network.get_layer(layer_idx)

            # 收集这一层的输入张量
            for i in range(layer.num_inputs):
                tensor = layer.get_input(i)
                if tensor is not None and tensor.name in all_tensors:
                    if tensor.name not in tensor_info:  # 避免重复
                        ordered_tensor_names.append(tensor.name)
                        tensor_info[tensor.name] = {
                            "name": tensor.name,
                            "shape": tuple(tensor.shape) if hasattr(tensor, "shape") else None,
                            "dtype": str(tensor.dtype) if hasattr(tensor, "dtype") else None,
                            "layer_index": layer_idx,
                            "tensor_type": "input",
                        }

            # 收集这一层的输出张量
            for i in range(layer.num_outputs):
                tensor = layer.get_output(i)
                if tensor is not None and tensor.name in all_tensors:
                    if tensor.name not in tensor_info:  # 避免重复
                        ordered_tensor_names.append(tensor.name)
                        tensor_info[tensor.name] = {
                            "name": tensor.name,
                            "shape": tuple(tensor.shape) if hasattr(tensor, "shape") else None,
                            "dtype": str(tensor.dtype) if hasattr(tensor, "dtype") else None,
                            "layer_index": layer_idx,
                            "tensor_type": "output",
                        }

        # 添加任何遗漏的张量（如果有的话）
        for name, tensor in all_tensors.items():
            if name not in tensor_info:
                ordered_tensor_names.append(name)
                tensor_info[name] = {
                    "name": name,
                    "shape": tuple(tensor.shape) if hasattr(tensor, "shape") else None,
                    "dtype": str(tensor.dtype) if hasattr(tensor, "dtype") else None,
                    "layer_index": -1,  # 未知层
                    "tensor_type": "unknown",
                }

        return {
            "tensor_count": len(all_tensors),
            "tensor_names": ordered_tensor_names,  # 现在是有序的
            "tensor_details": tensor_info,
        }

    def analyze_with_mark_all(self) -> dict[str, Any]:
        """模拟 --trt-outputs mark all 的行为"""
        try:
            network = self.network_info["network"]

            # 使用 ModifyNetworkOutputs 和 MARK_ALL
            modify_network_loader = ModifyNetworkOutputs(NetworkFromOnnxPath(self.model_path), outputs=MARK_ALL)

            builder, modified_network, parser = modify_network_loader()

            # 获取修改后的输出信息
            original_output_count = network.num_outputs
            modified_output_count = modified_network.num_outputs

            modified_output_names = []
            for i in range(modified_output_count):
                output_tensor = modified_network.get_output(i)
                modified_output_names.append(output_tensor.name)

            print("✅ 使用 ModifyNetworkOutputs(outputs=MARK_ALL) 成功修改 TensorRT 网络")

            return {
                "original_outputs_count": original_output_count,
                "mark_all_outputs_count": modified_output_count,
                "mark_all_output_names": modified_output_names,
            }

        except Exception as e:
            print(f"❌ ModifyNetworkOutputs 操作失败: {e}")
            return {}

    def analyze_network_info(self) -> dict[str, Any]:
        """分析 TensorRT 网络信息"""
        network = self.network_info["network"]

        stats = {
            "network_info": {
                "name": network.name,
                "num_layers": network.num_layers,
                "num_inputs": network.num_inputs,
                "num_outputs": network.num_outputs,
            }
        }

        # 张量分析
        tensor_info = self.get_all_tensors_with_polygraphy()
        stats["tensor_analysis"] = tensor_info

        # mark all 行为分析
        mark_all_info = self.analyze_with_mark_all()
        stats["mark_all_analysis"] = mark_all_info

        # 层类型统计
        layer_types = defaultdict(int)
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            layer_type = str(layer.type).split(".")[-1]
            layer_types[layer_type] += 1

        stats["layer_types"] = dict(layer_types)
        stats["most_common_layers"] = sorted(layer_types.items(), key=lambda x: x[1], reverse=True)[:10]

        # 输入输出信息
        input_info = []
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            input_info.append(
                {"name": input_tensor.name, "shape": tuple(input_tensor.shape), "dtype": str(input_tensor.dtype)}
            )

        output_info = []
        for i in range(network.num_outputs):
            output_tensor = network.get_output(i)
            output_info.append(
                {"name": output_tensor.name, "shape": tuple(output_tensor.shape), "dtype": str(output_tensor.dtype)}
            )

        stats["input_info"] = input_info
        stats["output_info"] = output_info

        return stats


def print_analysis_summary(onnx_stats: dict[str, Any], trt_stats: dict[str, Any] | None = None):
    """打印分析摘要"""
    print("\n" + "=" * 80)
    print("📊 Polygraphy 层统计分析报告")
    print("=" * 80)

    # ONNX 分析结果
    print("\n🔹 ONNX 模型分析 (使用 Polygraphy ONNX 后端):")
    print(f"   模型路径: {onnx_stats['model_info']['model_path']}")
    print(f"   总节点数: {onnx_stats['graph_info']['total_nodes']}")

    if "polygraphy_node_count" in onnx_stats:
        print(f"   Polygraphy 节点计数: {onnx_stats['polygraphy_node_count']}")

    print(f"   输入数量: {onnx_stats['graph_info']['total_inputs']}")
    print(f"   原始输出数量: {onnx_stats['graph_info']['total_outputs']}")

    # mark all 行为分析
    if onnx_stats.get("mark_all_analysis"):
        mark_all = onnx_stats["mark_all_analysis"]
        if mark_all:
            print("   📌 --onnx-outputs mark all 效果:")
            print(f"      原始输出数量: {mark_all.get('original_outputs_count', 'N/A')}")
            print(f"      mark all 后输出数量: {mark_all.get('mark_all_outputs_count', 'N/A')}")

    # 张量分析
    tensor_analysis = onnx_stats["tensor_analysis"]
    print("\n   📌 all_tensor_names() 函数结果:")
    print(f"      mark all 标记的张量数: {tensor_analysis['mark_all_tensor_count']}")
    print(f"      总张量数(含输入): {tensor_analysis['all_tensors_count']}")

    print("\n   前10个 mark all 张量:")
    for i, name in enumerate(tensor_analysis["mark_all_tensor_names"][:10]):
        print(f"      {i + 1:2d}. {name}")
    if len(tensor_analysis["mark_all_tensor_names"]) > 10:
        print(f"      ... 还有 {len(tensor_analysis['mark_all_tensor_names']) - 10} 个张量")

    print("\n   最常见层类型:")
    for layer_type, count in onnx_stats["most_common_layers"][:5]:
        print(f"      • {layer_type}: {count}")

    # TensorRT 分析结果
    if trt_stats:
        print("\n🔹 TensorRT 网络分析 (使用 Polygraphy TensorRT 后端):")
        print(f"   网络名称: {trt_stats['network_info']['name']}")
        print(f"   总层数: {trt_stats['network_info']['num_layers']}")
        print(f"   输入数量: {trt_stats['network_info']['num_inputs']}")
        print(f"   原始输出数量: {trt_stats['network_info']['num_outputs']}")

        # mark all 行为分析
        if trt_stats.get("mark_all_analysis"):
            mark_all = trt_stats["mark_all_analysis"]
            if mark_all:
                print("   📌 --trt-outputs mark all 效果:")
                print(f"      原始输出数量: {mark_all.get('original_outputs_count', 'N/A')}")
                print(f"      mark all 后输出数量: {mark_all.get('mark_all_outputs_count', 'N/A')}")

        # 张量分析
        tensor_analysis = trt_stats["tensor_analysis"]
        print("\n   📌 get_all_tensors() 函数结果:")
        print(f"      总张量数: {tensor_analysis['tensor_count']}")

        print("\n   前10个张量:")
        for i, name in enumerate(tensor_analysis["tensor_names"][:10]):
            print(f"      {i + 1:2d}. {name}")
        if len(tensor_analysis["tensor_names"]) > 10:
            print(f"      ... 还有 {len(tensor_analysis['tensor_names']) - 10} 个张量")

        print("\n   最常见层类型:")
        for layer_type, count in trt_stats["most_common_layers"][:5]:
            print(f"      • {layer_type}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="使用 Polygraphy 现成函数分析模型层统计信息",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python layer_statistics.py --model model.onnx
  python layer_statistics.py --model model.onnx --build-trt
  python layer_statistics.py --model model.onnx --save-json
  python layer_statistics.py --model model.onnx --build-trt --save-json
        """,
    )

    parser.add_argument("--model", "-m", required=True, help="ONNX 模型文件路径")

    parser.add_argument("--build-trt", action="store_true", help="同时构建 TensorRT 网络进行分析")

    parser.add_argument("--save-json", action="store_true", help="将统计结果保存为 JSON 文件到 runs/{model_name}/ 目录")

    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"❌ 模型文件不存在: {args.model}")
        sys.exit(1)

    try:
        # ONNX 分析
        print("🔄 正在使用 Polygraphy 分析 ONNX 模型...")
        onnx_analyzer = PolygraphyONNXAnalyzer(args.model)
        onnx_stats = onnx_analyzer.analyze_model_info()

        # TensorRT 分析
        trt_stats = None
        if args.build_trt:
            if not TRT_AVAILABLE:
                print("⚠️  TensorRT 不可用，跳过 TensorRT 分析")
            else:
                print("🔄 正在使用 Polygraphy 构建和分析 TensorRT 网络...")
                try:
                    trt_analyzer = PolygraphyTensorRTAnalyzer(args.model)
                    trt_stats = trt_analyzer.analyze_network_info()
                except Exception as e:
                    print(f"⚠️  TensorRT 分析失败: {e}")

        # 打印分析结果
        print_analysis_summary(onnx_stats, trt_stats)

        # 保存 JSON 文件
        if args.save_json:
            # 获取模型名称（不含扩展名）
            model_name = Path(args.model).stem
            output_dir = Path("runs") / model_name
            output_dir.mkdir(parents=True, exist_ok=True)

            # 保存 ONNX 分析结果
            onnx_file = output_dir / "onnx_layers.json"
            onnx_result = {"analysis_method": "Polygraphy ONNX 后端", "model_path": args.model, "analysis": onnx_stats}

            with open(onnx_file, "w", encoding="utf-8") as f:
                json.dump(onnx_result, f, ensure_ascii=False, indent=2, default=str)

            print(f"\n💾 ONNX 分析结果已保存到: {onnx_file}")

            # 保存 TensorRT 分析结果
            if trt_stats:
                trt_file = output_dir / "trt_layers.json"
                trt_result = {
                    "analysis_method": "Polygraphy TensorRT 后端",
                    "model_path": args.model,
                    "analysis": trt_stats,
                }

                with open(trt_file, "w", encoding="utf-8") as f:
                    json.dump(trt_result, f, ensure_ascii=False, indent=2, default=str)

                print(f"💾 TensorRT 分析结果已保存到: {trt_file}")

            print(f"📁 输出目录: {output_dir.absolute()}")

        print("\n✅ 分析完成!")
        print("\n💡 关键发现:")
        print(f"   • ONNX mark all 会标记 {onnx_stats['tensor_analysis']['mark_all_tensor_count']} 个张量为输出")
        if trt_stats:
            print(f"   • TensorRT mark all 会标记 {trt_stats['tensor_analysis']['tensor_count']} 个张量为输出")

    except KeyboardInterrupt:
        print("\n⚠️  用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
