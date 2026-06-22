"""MOTChallenge 格式数据解析层。

负责把磁盘上的 MOTChallenge 风格目录（``seqinfo.ini`` / ``seqmap.txt`` /
``classes.txt`` / ``gt/gt.txt`` 以及跟踪器输出的 ``<seq>.txt``）解析为统一的内存
表示，供 :mod:`onnxtools.eval.eval_mot` 中的评估器消费。

数据格式参考 ``data/track/MOT_dataset/README.md``：每行 9 列
``frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility``。
跟踪器输出（result）通常为 MOTChallenge result 格式
``frame, id, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1``（无 class 列）。

内部统一用每帧一个 ``ndarray`` 表示，列定义见 :data:`COLUMNS`：
``[id, x, y, w, h, conf, class]``（``x, y, w, h`` 为左上角坐标与宽高，像素）。
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# 统一内部列定义：每帧 ndarray 的列含义。
COLUMNS = ("id", "x", "y", "w", "h", "conf", "class")
COL_ID, COL_X, COL_Y, COL_W, COL_H, COL_CONF, COL_CLASS = range(7)


@dataclass
class SeqInfo:
    """单个序列的元信息（来自 ``seqinfo.ini``）。"""

    name: str
    seq_length: int = 0
    frame_rate: int = 30
    im_width: int = 0
    im_height: int = 0
    im_dir: str = "img1"
    im_ext: str = ".jpg"
    orig_name: str | None = None


@dataclass
class MOTSequence:
    """一个序列的标注/预测：帧号到框数组的映射。

    Attributes:
        name: 序列名（如 ``"MOT-01"``）。
        frames: ``{frame(1-based int): ndarray(N, 7)}``，列见 :data:`COLUMNS`。
        info: 对应的 :class:`SeqInfo`，可能为 ``None``（预测侧通常无 seqinfo）。
    """

    name: str
    frames: dict[int, np.ndarray] = field(default_factory=dict)
    info: SeqInfo | None = None

    @property
    def frame_ids(self) -> list[int]:
        """升序排列的帧号列表。"""
        return sorted(self.frames)

    def get(self, frame: int) -> np.ndarray:
        """取某帧的框数组；不存在时返回形状 ``(0, 7)`` 的空数组。"""
        return self.frames.get(frame, np.empty((0, len(COLUMNS)), dtype=float))


def parse_seqinfo(path: str | Path) -> SeqInfo:
    """解析 ``seqinfo.ini``。

    Args:
        path: ``seqinfo.ini`` 文件路径。

    Returns:
        :class:`SeqInfo` 实例。缺失字段回落到默认值。
    """
    path = Path(path)
    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    sec = parser["Sequence"] if parser.has_section("Sequence") else {}

    def _get(key: str, default: str = "") -> str:
        return sec.get(key, default) if hasattr(sec, "get") else default

    return SeqInfo(
        name=_get("name", path.parent.name),
        seq_length=int(_get("seqLength", "0") or 0),
        frame_rate=int(_get("frameRate", "30") or 30),
        im_width=int(_get("imWidth", "0") or 0),
        im_height=int(_get("imHeight", "0") or 0),
        im_dir=_get("imDir", "img1"),
        im_ext=_get("imExt", ".jpg"),
        orig_name=_get("origName") or None,
    )


def read_seqmap(path: str | Path) -> list[str]:
    """读取 ``seqmap.txt``，返回序列名列表。

    兼容两种风格：纯序列名逐行（首行可为表头 ``name``）以及 MOTChallenge 官方
    ``c10-...`` CSV 风格（取每行第一段）。空行与表头被跳过。

    Args:
        path: ``seqmap.txt`` 路径。

    Returns:
        序列名列表，保持文件内顺序。
    """
    names: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        token = line.split(",")[0].strip()
        if token.lower() in {"name", "seqmap"}:
            continue
        names.append(token)
    return names


def read_classes(path: str | Path) -> dict[int, str]:
    """读取 ``classes.txt``，返回 ``{class_id: name}``。

    每行格式 ``<id> <中文名> <english>``，取 id 与最后一个英文名（无则用中文名）。

    Args:
        path: ``classes.txt`` 路径。

    Returns:
        类别 id 到名称的映射；文件不存在时返回空字典。
    """
    path = Path(path)
    if not path.exists():
        return {}
    mapping: dict[int, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if not parts:
            continue
        try:
            cid = int(parts[0])
        except ValueError:
            continue
        mapping[cid] = parts[-1] if len(parts) > 1 else str(cid)
    return mapping


def read_mot_file(
    path: str | Path,
    *,
    is_gt: bool = False,
    default_class: int = -1,
) -> dict[int, np.ndarray]:
    """解析一个 MOTChallenge 逗号分隔的标注/预测文件。

    Args:
        path: ``gt.txt`` 或跟踪器输出 ``<seq>.txt`` 路径。
        is_gt: 为 ``True`` 时按 GT 语义处理——第 7 列 ``conf`` 作为 "consider"
            标志，``conf <= 0`` 的行（被标注忽略的目标）会被丢弃，第 8 列作为
            class。为 ``False`` 时按 MOTChallenge **result** 语义——第 7 列为检测
            置信度（保留不过滤），第 8 列及之后是 ``-1`` 占位（3D 坐标），**不含
            class**，故 class 一律填 ``default_class``。需携带类别的预测请改用内存
            ``dict`` 形态（见 :func:`load_predictions`）。
        default_class: result 文件（``is_gt=False``）或 GT 缺 class 列时填充的类别 id。

    Returns:
        ``{frame(1-based int): ndarray(N, 7)}``，列见 :data:`COLUMNS`。空文件返回 ``{}``。
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {}

    rows = np.atleast_2d(np.genfromtxt(path, delimiter=",", dtype=float))
    # genfromtxt 对单行可能返回一维；atleast_2d 已规整。过滤全 NaN 行。
    rows = rows[~np.all(np.isnan(rows), axis=1)]
    if rows.size == 0:
        return {}

    ncols = rows.shape[1]
    frame = rows[:, 0].astype(int)
    track_id = rows[:, 1].astype(int)
    xywh = rows[:, 2:6]
    conf = rows[:, 6] if ncols > 6 else np.ones(len(rows))
    # class 仅存在于 GT 格式的第 8 列；result 格式第 8 列是 -1 占位，非类别。
    if is_gt and ncols > 7:
        cls = rows[:, 7].astype(float)
    else:
        cls = np.full(len(rows), float(default_class))

    keep = np.ones(len(rows), dtype=bool)
    if is_gt:
        keep = conf > 0  # consider 标志：0 表示该框不参与评估

    out: dict[int, list[list[float]]] = {}
    for i in np.nonzero(keep)[0]:
        out.setdefault(int(frame[i]), []).append([track_id[i], *xywh[i].tolist(), float(conf[i]), float(cls[i])])
    return {f: np.asarray(v, dtype=float) for f, v in out.items()}


def load_gt(
    gt_root: str | Path,
    seqmap: list[str] | None = None,
) -> dict[str, MOTSequence]:
    """加载整个 MOTChallenge 数据集的 ground truth。

    目录结构::

        gt_root/
        ├── seqmap.txt          # 可选，未提供时自动发现 MOT-* 子目录
        ├── classes.txt         # 可选
        ├── <SEQ>/seqinfo.ini
        └── <SEQ>/gt/gt.txt

    Args:
        gt_root: 数据集根目录。
        seqmap: 显式序列名列表；为 ``None`` 时优先读 ``seqmap.txt``，再回落到
            自动发现包含 ``gt/gt.txt`` 的子目录。

    Returns:
        ``{seq_name: MOTSequence}``，按序列名排序。

    Raises:
        FileNotFoundError: 根目录不存在，或最终未发现任何含 ``gt/gt.txt`` 的序列。
    """
    gt_root = Path(gt_root)
    if not gt_root.is_dir():
        raise FileNotFoundError(f"GT 根目录不存在: {gt_root}")

    if seqmap is None:
        seqmap_file = gt_root / "seqmap.txt"
        if seqmap_file.exists():
            seqmap = read_seqmap(seqmap_file)
        else:
            seqmap = sorted(p.name for p in gt_root.iterdir() if (p / "gt" / "gt.txt").exists())

    sequences: dict[str, MOTSequence] = {}
    for name in seqmap:
        seq_dir = gt_root / name
        gt_file = seq_dir / "gt" / "gt.txt"
        if not gt_file.exists():
            continue
        info_file = seq_dir / "seqinfo.ini"
        info = parse_seqinfo(info_file) if info_file.exists() else None
        sequences[name] = MOTSequence(
            name=name,
            frames=read_mot_file(gt_file, is_gt=True),
            info=info,
        )

    if not sequences:
        raise FileNotFoundError(f"在 {gt_root} 下未发现任何含 gt/gt.txt 的序列")
    return dict(sorted(sequences.items()))


def write_mot_file(path: str | Path, frames: dict[int, np.ndarray]) -> None:
    """把每帧框数组写为 MOTChallenge result 格式 ``<seq>.txt``。

    输出 10 列 ``frame,id,x,y,w,h,conf,-1,-1,-1``（class 不写入 result，以兼容
    通用 MOTChallenge 评测工具）。

    Args:
        path: 输出文件路径，父目录会被自动创建。
        frames: ``{frame: ndarray(N, 7)}``，列见 :data:`COLUMNS`。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for frame in sorted(frames):
        for row in frames[frame]:
            tid = int(row[COL_ID])
            x, y, w, h = row[COL_X], row[COL_Y], row[COL_W], row[COL_H]
            conf = row[COL_CONF]
            lines.append(f"{frame},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_predictions(
    predictions: str | Path | dict[str, MOTSequence | dict[int, np.ndarray]],
    seqmap: list[str] | None = None,
    default_class: int = -1,
) -> dict[str, MOTSequence]:
    """把多种形态的预测归一化为 ``{seq_name: MOTSequence}``。

    Args:
        predictions: 三种形态之一——

            * 目录路径：内含 ``<seq>.txt`` 的 MOTChallenge result 文件。
            * ``{seq: MOTSequence}``：已构造好的序列对象（如来自
              :func:`onnxtools.eval.eval_mot.run_tracker_on_gt`）。
            * ``{seq: {frame: ndarray(N, 7)}}``：每序列的帧字典。

        seqmap: 目录形态下限定读取的序列名；为 ``None`` 时读取目录内全部 ``*.txt``。
        default_class: 目录形态下文件无 class 列时填充的类别 id。

    Returns:
        ``{seq_name: MOTSequence}``。

    Raises:
        FileNotFoundError: 目录形态下路径不存在。
        TypeError: ``predictions`` 形态不被支持。
    """
    if isinstance(predictions, str | Path):
        pred_dir = Path(predictions)
        if not pred_dir.is_dir():
            raise FileNotFoundError(f"预测目录不存在: {pred_dir}")
        files = sorted(pred_dir.glob("*.txt"))
        out: dict[str, MOTSequence] = {}
        for f in files:
            name = f.stem
            if seqmap is not None and name not in seqmap:
                continue
            out[name] = MOTSequence(
                name=name,
                frames=read_mot_file(f, is_gt=False, default_class=default_class),
            )
        return out

    if isinstance(predictions, dict):
        out = {}
        for name, value in predictions.items():
            if isinstance(value, MOTSequence):
                out[name] = value
            elif isinstance(value, dict):
                out[name] = MOTSequence(name=name, frames=value)
            else:
                raise TypeError(
                    f"序列 {name!r} 的预测类型不支持: {type(value)!r}，应为 MOTSequence 或 {{frame: ndarray}}"
                )
        return out

    raise TypeError(f"predictions 类型不支持: {type(predictions)!r}，应为目录路径或 dict")
