"""``onnxtools.eval`` MOT 评估器单元测试。

解析层（:mod:`onnxtools.eval.mot_data`）与几何工具不依赖任何可选库，无条件测试；
指标计算用 ``pytest.importorskip`` 在缺 motmetrics / trackeval 时跳过；
``run_tracker_on_gt`` 依赖 supervision。

测试用合成的小型 MOTChallenge 数据集（写入 ``tmp_path``），不依赖
``data/track/MOT_dataset``。
"""

from __future__ import annotations

import numpy as np
import pytest

from onnxtools.eval.eval_mot import MOTEvaluator, iou_matrix_xywh, run_tracker_on_gt
from onnxtools.eval.mot_data import (
    COLUMNS,
    MOTSequence,
    load_gt,
    read_mot_file,
    read_seqmap,
    write_mot_file,
)

# ---------------------------------------------------------------------------
# 合成数据集
# ---------------------------------------------------------------------------


def _write_seq(seq_dir, gt_lines: list[str], seq_length: int) -> None:
    (seq_dir / "gt").mkdir(parents=True, exist_ok=True)
    (seq_dir / "gt" / "gt.txt").write_text("\n".join(gt_lines) + "\n", encoding="utf-8")
    (seq_dir / "seqinfo.ini").write_text(
        "[Sequence]\n"
        f"name={seq_dir.name}\n"
        "imDir=img1\n"
        "frameRate=5\n"
        f"seqLength={seq_length}\n"
        "imWidth=1920\nimHeight=1080\nimExt=.jpg\n",
        encoding="utf-8",
    )


@pytest.fixture
def tiny_dataset(tmp_path):
    """两序列、两类别的小型数据集。

    MOT-01: 3 帧，2 条轨迹（id 1=class1, id 2=class2），框稳定移动。
    MOT-02: 2 帧，1 条轨迹。含一行 conf=0（应被 GT 解析丢弃）。
    """
    root = tmp_path / "MOT"
    # frame,id,x,y,w,h,conf,class,vis
    seq1 = [
        "1,1,10,10,20,20,1,1,1",
        "1,2,100,100,30,30,1,2,1",
        "2,1,12,10,20,20,1,1,1",
        "2,2,102,100,30,30,1,2,1",
        "3,1,14,10,20,20,1,1,1",
        "3,2,104,100,30,30,1,2,1",
    ]
    seq2 = [
        "1,1,50,50,40,40,1,1,1",
        "1,9,0,0,5,5,0,2,1",  # conf=0 → consider 标志关闭，应被丢弃
        "2,1,52,50,40,40,1,1,1",
    ]
    _write_seq(root / "MOT-01", seq1, seq_length=3)
    _write_seq(root / "MOT-02", seq2, seq_length=2)
    (root / "seqmap.txt").write_text("name\nMOT-01\nMOT-02\n", encoding="utf-8")
    (root / "classes.txt").write_text("1 行人 pedestrian\n2 非机动车 non-motor\n", encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# 解析层
# ---------------------------------------------------------------------------


def test_read_seqmap_skips_header(tmp_path):
    f = tmp_path / "seqmap.txt"
    f.write_text("name\nMOT-01\n\nMOT-02\n", encoding="utf-8")
    assert read_seqmap(f) == ["MOT-01", "MOT-02"]


def test_read_mot_file_gt_drops_zero_conf(tiny_dataset):
    frames = read_mot_file(tiny_dataset / "MOT-02" / "gt" / "gt.txt", is_gt=True)
    # frame 1 原有 2 行，conf=0 的那行被丢弃 → 只剩 1 行
    assert len(frames[1]) == 1
    assert frames[1][0, 0] == 1  # id
    assert frames[1].shape[1] == len(COLUMNS)


def test_read_mot_file_result_default_class(tmp_path):
    # MOTChallenge result 格式无 class 列
    f = tmp_path / "MOT-01.txt"
    f.write_text("1,1,10,10,20,20,0.9,-1,-1,-1\n", encoding="utf-8")
    frames = read_mot_file(f, is_gt=False, default_class=7)
    assert frames[1][0, -1] == 7  # class 回落到 default_class
    assert frames[1][0, 5] == pytest.approx(0.9)  # conf 保留


def test_read_mot_file_empty(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("", encoding="utf-8")
    assert read_mot_file(f) == {}


def test_load_gt_counts_and_filtering(tiny_dataset):
    gt = load_gt(tiny_dataset)
    assert list(gt) == ["MOT-01", "MOT-02"]
    assert gt["MOT-01"].info.frame_rate == 5
    assert gt["MOT-01"].info.seq_length == 3
    # MOT-01 共 6 个框，MOT-02 丢弃 conf=0 后剩 2 个
    assert sum(len(a) for a in gt["MOT-01"].frames.values()) == 6
    assert sum(len(a) for a in gt["MOT-02"].frames.values()) == 2


def test_load_gt_autodiscover_without_seqmap(tiny_dataset):
    (tiny_dataset / "seqmap.txt").unlink()
    gt = load_gt(tiny_dataset)
    assert set(gt) == {"MOT-01", "MOT-02"}


def test_load_gt_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_gt(tmp_path / "does_not_exist")


def test_write_mot_file_roundtrip(tmp_path):
    frames = {
        1: np.array([[1, 10, 10, 20, 20, 1.0, 2]], dtype=float),
        2: np.array([[1, 12, 10, 20, 20, 1.0, 2]], dtype=float),
    }
    out = tmp_path / "MOT-01.txt"
    write_mot_file(out, frames)
    reloaded = read_mot_file(out, is_gt=False)
    assert set(reloaded) == {1, 2}
    assert reloaded[1][0, 1] == pytest.approx(10.0)  # x


# ---------------------------------------------------------------------------
# 几何
# ---------------------------------------------------------------------------


def test_iou_matrix_perfect_and_disjoint():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[0, 0, 10, 10], [100, 100, 10, 10]], dtype=float)
    iou = iou_matrix_xywh(a, b)
    assert iou.shape == (1, 2)
    assert iou[0, 0] == pytest.approx(1.0)
    assert iou[0, 1] == pytest.approx(0.0)


def test_iou_matrix_half_overlap():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[5, 0, 10, 10]], dtype=float)  # 交 50，并 150 → 1/3
    assert iou_matrix_xywh(a, b)[0, 0] == pytest.approx(1 / 3)


def test_iou_matrix_empty():
    assert iou_matrix_xywh(np.empty((0, 4)), np.array([[0, 0, 1, 1.0]])).shape == (0, 1)


# ---------------------------------------------------------------------------
# 评估器
# ---------------------------------------------------------------------------


def _gt_as_predictions(gt) -> dict[str, MOTSequence]:
    """把 GT 原样当作完美预测（id/框完全一致）。"""
    return {name: MOTSequence(name=name, frames=dict(seq.frames)) for name, seq in gt.items()}


def test_evaluate_unknown_metric_raises(tiny_dataset):
    ev = MOTEvaluator(tiny_dataset)
    with pytest.raises(ValueError, match="未知指标族"):
        ev.evaluate(_gt_as_predictions(ev.gt), metrics=("nope",))


def test_evaluate_perfect_clear_identity(tiny_dataset):
    pytest.importorskip("motmetrics")
    ev = MOTEvaluator(tiny_dataset)
    res = ev.evaluate(_gt_as_predictions(ev.gt), metrics=("clear", "identity"))
    assert res.overall["MOTA"] == pytest.approx(100.0)
    assert res.overall["IDF1"] == pytest.approx(100.0)
    assert int(res.overall["IDsw"]) == 0
    assert int(res.overall["FP"]) == 0
    assert int(res.overall["FN"]) == 0
    # GT_IDs: MOT-01 有 2，MOT-02 有 1 → 3
    assert int(res.overall["GT_IDs"]) == 3


def test_evaluate_perfect_hota(tiny_dataset):
    pytest.importorskip("trackeval")
    ev = MOTEvaluator(tiny_dataset)
    res = ev.evaluate(_gt_as_predictions(ev.gt), metrics=("hota",))
    assert res.overall["HOTA"] == pytest.approx(100.0, abs=1e-6)
    assert res.overall["LocA"] == pytest.approx(100.0, abs=1e-6)
    assert res.metric_names == ["HOTA", "DetA", "AssA", "LocA"]


def test_evaluate_id_switch_penalized(tiny_dataset):
    pytest.importorskip("motmetrics")
    ev = MOTEvaluator(tiny_dataset)
    preds = _gt_as_predictions(ev.gt)
    # 在 MOT-01 第 3 帧把轨迹 1 的 id 改成 99 → 制造一次 ID switch
    f3 = preds["MOT-01"].frames[3].copy()
    f3[f3[:, 0] == 1, 0] = 99
    preds["MOT-01"].frames[3] = f3
    res = ev.evaluate(preds, metrics=("clear",))
    assert int(res.overall["IDsw"]) >= 1


def test_evaluate_missing_prediction_is_all_fn(tiny_dataset):
    pytest.importorskip("motmetrics")
    ev = MOTEvaluator(tiny_dataset)
    preds = _gt_as_predictions(ev.gt)
    del preds["MOT-02"]  # 缺一个序列的预测
    res = ev.evaluate(preds, metrics=("clear",))
    # MOT-02 全漏检
    assert res.per_sequence["MOT-02"]["Rcll"] == pytest.approx(0.0)
    assert int(res.per_sequence["MOT-02"]["FN"]) == 2


def test_evaluate_class_filter(tiny_dataset):
    pytest.importorskip("motmetrics")
    ev = MOTEvaluator(tiny_dataset)
    preds = _gt_as_predictions(ev.gt)
    # class 2 (non-motor) 只在 MOT-01 出现 1 条轨迹
    res = ev.evaluate(preds, metrics=("clear",), classes=[2])
    assert int(res.overall["GT_IDs"]) == 1


def test_evaluate_from_directory(tiny_dataset, tmp_path):
    pytest.importorskip("trackeval")
    ev = MOTEvaluator(tiny_dataset)
    pred_dir = tmp_path / "preds"
    for name, seq in _gt_as_predictions(ev.gt).items():
        write_mot_file(pred_dir / f"{name}.txt", seq.frames)
    res = ev.evaluate(str(pred_dir), metrics=("hota",))
    assert res.overall["HOTA"] == pytest.approx(100.0, abs=1e-6)


def test_summary_table_renders(tiny_dataset):
    pytest.importorskip("trackeval")
    ev = MOTEvaluator(tiny_dataset)
    res = ev.evaluate(_gt_as_predictions(ev.gt), metrics=("hota",))
    table = res.summary_table()
    assert "OVERALL" in table
    assert "HOTA" in table
    assert "MOT-01" in table


# ---------------------------------------------------------------------------
# run_tracker_on_gt
# ---------------------------------------------------------------------------


def test_run_tracker_on_gt_produces_evaluable_predictions(tiny_dataset):
    pytest.importorskip("supervision")
    pytest.importorskip("trackeval")
    ev = MOTEvaluator(tiny_dataset)
    preds = run_tracker_on_gt(tiny_dataset, "bytetrack", frame_rate=5)
    assert set(preds) == {"MOT-01", "MOT-02"}
    # 喂的是理想检测，关联应接近完美 → HOTA 较高
    res = ev.evaluate(preds, metrics=("hota",))
    assert res.overall["HOTA"] > 50.0
