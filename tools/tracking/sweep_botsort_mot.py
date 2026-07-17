"""BoT-SORT 超参数搜索：MOT 指标 + 3Hz 代理指标 + ReID 特征缓存。

这个脚本仿照 ``sweep_bytetrack_mot.py``，但 tracker 换成 ``botsort``，并在检测缓存阶段
顺手把 ReID embedding 算好。后续每组超参数只重放缓存，不重复跑检测器和 ReID ONNX。

默认 ReID 模型使用已下载的小型行人 OSNet：

``models/reid/person/osnet/msmt17/osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0_Nx3x256x128.onnx``

用法::

    .venv/bin/python tools/tracking/sweep_botsort_mot.py
    .venv/bin/python tools/tracking/sweep_botsort_mot.py --skip-viz --max-mot-frames 30 --max-proxy 2
    .venv/bin/python tools/tracking/sweep_botsort_mot.py --disable-reid
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import defaultdict
from collections.abc import Sequence
from itertools import product
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import supervision as sv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from onnxtools import create_detector  # noqa: E402
from onnxtools.eval import MOTEvaluator  # noqa: E402
from onnxtools.eval.mot_data import MOTSequence, parse_seqinfo  # noqa: E402
from onnxtools.tracking import create_tracker  # noqa: E402
from onnxtools.tracking.class_voting import ClassVotingTracker  # noqa: E402

# ----------------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------------
MODEL = "models/rfdetr-medium_20260629_d_unified.onnx"
MODEL_TYPE = "rfdetr_unified"
FALLBACK_MODEL = "models/vehicle_det_detr_batch4.onnx"
FALLBACK_MODEL_TYPE = "rtdetr"
MOT_ROOT = "data/track/MOT_dataset"
PROXY_DIR = "data/track/proxy_videos"
DEFAULT_REID_MODEL = (
    "models/reid/person/osnet/msmt17/osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0_Nx3x256x128.onnx"
)

DET_CONF_FLOOR = 0.4
KEEP_CLASSES = (9, 5, 6, 7)  # pedestrian, bicycle, cyclist, tricycle
VRU_CLASS = 0

MOT_FRAME_RATE = 5
PROXY_FPS = 3.0

# 搜索空间。新检测模型(rfdetr-medium_20260629_d)误检跳变显著降低，
# 因此较旧模型扩大三个轴的范围（与 sweep_bytetrack_mot.py 对齐）：
#   - high/new 向下加 0.5(检测可信→可放更多高质量框)，向上加 0.8(更严格 stage-1)
#   - match_thresh 向下加 0.6、向上加 0.95(更紧/更松两端都试)
#   - track_buffer 向上加 30/60(检测更稳→可容忍更长真实遮挡)
# ReID 三个阈值暂保持单点，避免组合数爆炸(7×6×5×K^3)；需扩时改这里。
TRACK_LOW_THRESH = 0.4
HIGH_NEW_PAIRS = (
    (0.5, 0.5),
    (0.5, 0.6),
    (0.6, 0.6),
    (0.6, 0.7),
    (0.7, 0.7),
    (0.7, 0.8),
    (0.8, 0.8),
)
MATCH_THRESH = (0.6, 0.7, 0.8, 0.85, 0.9, 0.95)
TRACK_BUFFER = (3, 9, 15, 30, 60)
APPEARANCE_THRESH = (0.25,)
PROXIMITY_THRESH = (0.50,)
REID_ALPHA = (0.90,)

GATE_SHORT_ID = 0.30
GATE_NUMID_MULT = 3.0

FrameRecord = dict[str, Any]
LOGGER = logging.getLogger("botsort_sweep")
LOG_EVERY_MOT_FRAMES = 20
LOG_EVERY_PROXY_SAMPLES = 20
CACHE_VERSION = 2  # v2: 加入 cls 字段(原始检测类别 id，供 class_aware/voting 使用)


def configure_logging(level: str) -> None:
    """Configure real-time console logging for long sweep jobs."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def _path_signature(path: str | Path | None) -> dict[str, object] | None:
    """Return a stable-ish signature for invalidating cache files."""
    if path is None:
        return None
    p = Path(path)
    try:
        stat = p.stat()
    except FileNotFoundError:
        return {"path": str(p)}
    return {
        "path": str(p.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _cache_key(payload: dict[str, object]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _safe_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)[:80]


def build_cache_identity(detector_model: str, reid_model: str | None, use_reid: bool) -> dict[str, object]:
    """Build the shared cache identity for detection/ReID outputs."""
    return {
        "version": CACHE_VERSION,
        "detector_model": _path_signature(detector_model),
        "reid_model": _path_signature(reid_model) if use_reid else None,
        "use_reid": use_reid,
        "det_conf_floor": DET_CONF_FLOOR,
        "keep_classes": list(KEEP_CLASSES),
        "vru_class": VRU_CLASS,
    }


def _object_array(values: Sequence[object]) -> np.ndarray:
    arr = np.empty(len(values), dtype=object)
    for i, value in enumerate(values):
        arr[i] = value
    return arr


def _save_npz_atomic(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def _mot_cache_path(cache_dir: Path, name: str, meta: dict[str, object]) -> Path:
    return cache_dir / "mot" / f"{_safe_stem(name)}_{_cache_key(meta)}.npz"


def _proxy_cache_path(cache_dir: Path, video: Path, meta: dict[str, object]) -> Path:
    return cache_dir / "proxy" / f"{_safe_stem(video.stem)}_{_cache_key(meta)}.npz"


def _save_mot_cache(path: Path, seq: dict, meta: dict[str, object]) -> None:
    frame_ids = sorted(seq["frames"])
    records = [seq["frames"][frame_id] for frame_id in frame_ids]
    image_paths = [str(seq["image_paths"][frame_id]) for frame_id in frame_ids]
    _save_npz_atomic(
        path,
        meta_json=np.array(json.dumps(meta, sort_keys=True, ensure_ascii=False), dtype=object),
        seq_len=np.array(seq["len"], dtype=np.int32),
        frame_ids=np.array(frame_ids, dtype=np.int32),
        xyxy=_object_array([record["xyxy"] for record in records]),
        conf=_object_array([record["conf"] for record in records]),
        cls=_object_array([record.get("cls") for record in records]),
        features=_object_array([record.get("features") for record in records]),
        image_paths=np.array(image_paths, dtype=object),
    )


def _load_mot_cache(path: Path, meta: dict[str, object]) -> dict | None:
    try:
        with np.load(path, allow_pickle=True) as data:
            cached_meta = json.loads(str(data["meta_json"].item()))
            if cached_meta != meta:
                return None
            frame_ids = data["frame_ids"].astype(int)
            xyxy = data["xyxy"]
            conf = data["conf"]
            cls_arr = data["cls"] if "cls" in data.files else None
            features = data["features"]
            image_paths_raw = data["image_paths"]
            frames: dict[int, FrameRecord] = {}
            image_paths: dict[int, Path] = {}
            for i, frame_id in enumerate(frame_ids):
                feature = features[i]
                cls = None if cls_arr is None or cls_arr[i] is None else np.asarray(cls_arr[i], dtype=int)
                frames[int(frame_id)] = {
                    "xyxy": np.asarray(xyxy[i], dtype=np.float32),
                    "conf": np.asarray(conf[i], dtype=np.float32),
                    "cls": cls,
                    "features": None if feature is None else np.asarray(feature, dtype=np.float32),
                }
                image_paths[int(frame_id)] = Path(str(image_paths_raw[i]))
            return {"len": int(data["seq_len"]), "frames": frames, "image_paths": image_paths}
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        LOGGER.warning("[cache:MOT] ignore unreadable cache %s: %s", path, exc)
        return None


def _save_proxy_cache(path: Path, records: list[FrameRecord], meta: dict[str, object]) -> None:
    _save_npz_atomic(
        path,
        meta_json=np.array(json.dumps(meta, sort_keys=True, ensure_ascii=False), dtype=object),
        xyxy=_object_array([record["xyxy"] for record in records]),
        conf=_object_array([record["conf"] for record in records]),
        cls=_object_array([record.get("cls") for record in records]),
        features=_object_array([record.get("features") for record in records]),
    )


def _load_proxy_cache(path: Path, meta: dict[str, object]) -> list[FrameRecord] | None:
    try:
        with np.load(path, allow_pickle=True) as data:
            cached_meta = json.loads(str(data["meta_json"].item()))
            if cached_meta != meta:
                return None
            xyxy = data["xyxy"]
            conf = data["conf"]
            cls_arr = data["cls"] if "cls" in data.files else None
            features = data["features"]
            records: list[FrameRecord] = []
            for i in range(len(xyxy)):
                feature = features[i]
                cls = None if cls_arr is None or cls_arr[i] is None else np.asarray(cls_arr[i], dtype=int)
                records.append(
                    {
                        "xyxy": np.asarray(xyxy[i], dtype=np.float32),
                        "conf": np.asarray(conf[i], dtype=np.float32),
                        "cls": cls,
                        "features": None if feature is None else np.asarray(feature, dtype=np.float32),
                    }
                )
            return records
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        LOGGER.warning("[cache:proxy] ignore unreadable cache %s: %s", path, exc)
        return None


# ----------------------------------------------------------------------------
# ReID ONNX 编码器
# ----------------------------------------------------------------------------
class ReIDOnnxEncoder:
    """轻量 ONNX ReID 编码器，输出 L2-normalized embedding。

    预处理按 PINTO OSNet demo / TorchReID 常用配置执行：
    ``BGR crop -> RGB -> resize(128x256) -> /255 -> ImageNet mean/std -> NCHW``。
    """

    def __init__(
        self,
        model_path: str | Path,
        providers: Sequence[str],
        batch_size: int = 64,
        fallback_hw: tuple[int, int] = (256, 128),
    ) -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "启用 ReID 需要 onnxruntime。请用 inference 环境运行，或追加 "
                "`uv run --with onnxruntime-gpu ...`；临时关闭可加 `--disable-reid`。"
            ) from exc

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ReID 模型不存在: {self.model_path}\n"
                "可以先确认 models/reid/person/osnet/msmt17/ 下的小 OSNet ONNX 是否已下载，"
                "或通过 `--reid-model` 指向其他 ONNX。"
            )

        available = set(ort.get_available_providers())
        selected = [p for p in providers if p in available]
        if not selected:
            selected = ["CPUExecutionProvider"]

        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        self.session = ort.InferenceSession(str(self.model_path), sess_options=sess_options, providers=selected)
        self.providers = self.session.get_providers()
        self.input_meta = self.session.get_inputs()[0]
        self.output_name = self.session.get_outputs()[0].name
        self.input_name = self.input_meta.name
        self.batch_size = int(batch_size)
        self.fixed_batch = self._dim_to_int(self.input_meta.shape[0])

        h = self._dim_to_int(self.input_meta.shape[2]) or fallback_hw[0]
        w = self._dim_to_int(self.input_meta.shape[3]) or fallback_hw[1]
        self.input_hw = (h, w)

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    @staticmethod
    def _dim_to_int(value: object) -> int | None:
        return int(value) if isinstance(value, int) and value > 0 else None

    def __call__(self, frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        if len(xyxy) == 0:
            return np.empty((0, 0), dtype=np.float32)

        batch = np.stack([self._preprocess_crop(frame, box) for box in xyxy]).astype(np.float32)
        chunks: list[np.ndarray] = []
        step = self.fixed_batch or self.batch_size
        for start in range(0, len(batch), step):
            chunk = batch[start : start + step]
            chunks.append(self._run_chunk(chunk))
        features = np.vstack(chunks).astype(np.float32)
        features = features.reshape(features.shape[0], -1)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        return np.divide(features, np.maximum(norms, 1e-12)).astype(np.float32)

    def _preprocess_crop(self, frame: np.ndarray, box: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = np.round(box).astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)

        out_h, out_w = self.input_hw
        if x2 <= x1 or y2 <= y1:
            crop = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        else:
            crop = cv2.resize(frame[y1:y2, x1:x2], (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        crop = crop[..., ::-1].astype(np.float32) / 255.0
        chw = crop.transpose(2, 0, 1)
        return (chw - self.mean) / self.std

    def _run_chunk(self, chunk: np.ndarray) -> np.ndarray:
        original_n = len(chunk)
        if self.fixed_batch and original_n != self.fixed_batch:
            padded = np.zeros((self.fixed_batch, *chunk.shape[1:]), dtype=chunk.dtype)
            padded[:original_n] = chunk
            chunk = padded
        out = self.session.run([self.output_name], {self.input_name: chunk})[0]
        return np.asarray(out[:original_n], dtype=np.float32)


# ----------------------------------------------------------------------------
# 检测 + ReID 缓存
# ----------------------------------------------------------------------------
def detect_filtered(detector, image: np.ndarray, reid_encoder: ReIDOnnxEncoder | None) -> FrameRecord:
    """跑检测，保留目标类并按需计算 ReID 特征。

    保留 detector 的原始类别 id(不再池化为 VRU)，让 class_aware 关联 +
    ClassVotingTracker 在 sweep 中有效。评估时仍调用 ``classes=None`` 走类无关
    HOTA，所以 GT 的类体系与预测无须对齐(预测里的类列只是占位)。
    """
    r = detector(image)
    boxes = np.asarray(r.boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(r.scores, dtype=np.float32).reshape(-1)
    cls = np.asarray(r.class_ids, dtype=int).reshape(-1)

    keep = np.isin(cls, KEEP_CLASSES) & (scores >= DET_CONF_FLOOR)
    boxes = boxes[keep]
    scores = scores[keep]
    cls = cls[keep]
    features = reid_encoder(image, boxes) if reid_encoder is not None and len(boxes) else None
    return {"xyxy": boxes, "conf": scores, "cls": cls, "features": features}


def make_dets(record: FrameRecord, use_reid: bool) -> sv.Detections:
    xyxy = np.asarray(record["xyxy"], dtype=np.float32)
    conf = np.asarray(record["conf"], dtype=np.float32)
    if len(xyxy) == 0:
        return sv.Detections.empty()

    data = {}
    features = record.get("features")
    if use_reid and features is not None:
        data["features"] = np.asarray(features, dtype=np.float32)

    cls = record.get("cls")
    if cls is None:
        # 旧缓存(无 cls 字段)的兼容路径：回退到单一 VRU 类，class_aware/voting 退化为 no-op。
        class_ids = np.full(len(xyxy), VRU_CLASS, dtype=int)
    else:
        class_ids = np.asarray(cls, dtype=int)

    return sv.Detections(
        xyxy=xyxy.astype(float),
        confidence=conf.astype(float),
        class_id=class_ids,
        data=data,
    )


def make_tracker(params: dict, frame_rate: int, *, use_reid: bool, class_aware: bool, class_voting: bool):
    """构建 BoT-SORT(+ ClassVotingTracker 包装)。"""
    inner = create_tracker(
        "botsort",
        **tracker_kwargs(params, frame_rate, use_reid, class_aware=class_aware),
    )
    if class_voting:
        return ClassVotingTracker(inner, decay=1.0)
    return inner


def read_seqmap(mot_root: Path) -> list[str]:
    seqmap = mot_root / "seqmap.txt"
    names = []
    for line in seqmap.read_text().splitlines():
        s = line.strip()
        if s and s.lower() != "name":
            names.append(s)
    return names


def cache_mot_detections(
    detector,
    mot_root: Path,
    max_frames: int | None,
    reid_encoder: ReIDOnnxEncoder | None,
    cache_dir: Path | None,
    cache_identity: dict[str, object],
) -> dict[str, dict]:
    """``{seq: {"len": L, "frames": {frame: record}, "image_paths": {frame: path}}}``。"""
    cache: dict[str, dict] = {}
    for name in read_seqmap(mot_root):
        info = parse_seqinfo(mot_root / name / "seqinfo.ini")
        n = info.seq_length
        im_dir = mot_root / name / info.im_dir
        im_ext = info.im_ext
        limit = min(n, max_frames) if max_frames else n
        cache_meta = {
            **cache_identity,
            "kind": "mot",
            "mot_root": str(mot_root.resolve()),
            "seq": name,
            "seqinfo": _path_signature(mot_root / name / "seqinfo.ini"),
            "limit": limit,
        }
        cache_path = _mot_cache_path(cache_dir, name, cache_meta) if cache_dir is not None else None
        if cache_path is not None and cache_path.exists():
            loaded = _load_mot_cache(cache_path, cache_meta)
            if loaded is not None:
                cache[name] = loaded
                LOGGER.info("[cache:MOT] hit %s frames=%d path=%s", name, len(loaded["frames"]), cache_path)
                continue

        LOGGER.info("[cache:MOT] start %s frames=%d reid=%s", name, limit, reid_encoder is not None)
        frames: dict[int, FrameRecord] = {}
        image_paths: dict[int, Path] = {}
        for f in range(1, limit + 1):
            path = im_dir / f"{f:06d}{im_ext}"
            img = cv2.imread(str(path))
            if img is None:
                continue
            frames[f] = detect_filtered(detector, img, reid_encoder)
            image_paths[f] = path
            if f == limit or f % LOG_EVERY_MOT_FRAMES == 0:
                LOGGER.info("[cache:MOT] %s progress %d/%d cached=%d", name, f, limit, len(frames))
        cache[name] = {"len": limit, "frames": frames, "image_paths": image_paths}
        LOGGER.info("[cache:MOT] done %s cached=%d/%d", name, len(frames), limit)
        if cache_path is not None:
            _save_mot_cache(cache_path, cache[name], cache_meta)
            LOGGER.info("[cache:MOT] saved %s", cache_path)
    return cache


def select_proxy_videos(proxy_dir: Path) -> list[Path]:
    return sorted(p for p in proxy_dir.glob("*.mp4") if "heizhima" not in p.name.lower())


def cache_proxy_detections(
    detector,
    videos: list[Path],
    reid_encoder: ReIDOnnxEncoder | None,
    keep_frames: bool,
    cache_dir: Path | None,
    cache_identity: dict[str, object],
) -> dict[str, list[FrameRecord]]:
    """``{video_name: [record, ...]}``，按约 3Hz 采样。"""
    cache: dict[str, list[FrameRecord]] = {}
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, round(fps / PROXY_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cache_meta = {
            **cache_identity,
            "kind": "proxy",
            "video": _path_signature(v),
            "proxy_fps": PROXY_FPS,
            "source_fps": fps,
            "step": step,
            "keep_frames": keep_frames,
        }
        cache_path = _proxy_cache_path(cache_dir, v, cache_meta) if cache_dir is not None else None
        if cache_path is not None and cache_path.exists() and not keep_frames:
            loaded = _load_proxy_cache(cache_path, cache_meta)
            if loaded is not None:
                cache[v.name] = loaded
                cap.release()
                LOGGER.info("[cache:proxy] hit %s sampled=%d path=%s", v.name, len(loaded), cache_path)
                continue
        LOGGER.info(
            "[cache:proxy] start %s fps=%.2f step=%d frames=%d reid=%s",
            v.name,
            fps,
            step,
            total_frames,
            reid_encoder is not None,
        )
        records: list[FrameRecord] = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                record = detect_filtered(detector, frame, reid_encoder)
                if keep_frames:
                    record["frame"] = frame.copy()
                records.append(record)
                if len(records) % LOG_EVERY_PROXY_SAMPLES == 0:
                    LOGGER.info(
                        "[cache:proxy] %s sampled=%d src_frame=%d",
                        v.name,
                        len(records),
                        idx,
                    )
            idx += 1
        cap.release()
        cache[v.name] = records
        LOGGER.info("[cache:proxy] done %s sampled=%d step=%d", v.name, len(records), step)
        if cache_path is not None and not keep_frames:
            _save_proxy_cache(cache_path, records, cache_meta)
            LOGGER.info("[cache:proxy] saved %s", cache_path)
    return cache


def resolve_detector_model(model: str, model_type: str) -> tuple[str, str]:
    """Resolve the detector model path, with a local fallback for the default model.

    Returns ``(path, model_type)`` — fallback may change model_type (e.g. rfdetr_unified → rtdetr).
    """
    if Path(model).exists():
        return model, model_type
    if model == MODEL and Path(FALLBACK_MODEL).exists():
        LOGGER.info(
            "[model] 默认检测模型不存在，自动改用本机已有模型: %s (type=%s)",
            FALLBACK_MODEL,
            FALLBACK_MODEL_TYPE,
        )
        return FALLBACK_MODEL, FALLBACK_MODEL_TYPE
    raise FileNotFoundError(f"检测模型不存在: {model}")


# ----------------------------------------------------------------------------
# 在缓存上跑 tracker
# ----------------------------------------------------------------------------
def tracker_kwargs(params: dict, frame_rate: int, use_reid: bool, *, class_aware: bool = False) -> dict:
    return dict(
        track_high_thresh=params["track_high_thresh"],
        track_low_thresh=TRACK_LOW_THRESH,
        new_track_thresh=params["new_track_thresh"],
        match_thresh=params["match_thresh"],
        track_buffer=params["track_buffer"],
        frame_rate=frame_rate,
        class_aware=class_aware,
        camera_motion=bool(params["camera_motion"]),
        with_reid=use_reid,
        feature_key="features",
        appearance_thresh=params["appearance_thresh"],
        proximity_thresh=params["proximity_thresh"],
        reid_alpha=params["reid_alpha"],
    )


def _tracker_frame(record: FrameRecord, camera_motion: bool) -> np.ndarray | None:
    if not camera_motion:
        return None
    return record.get("frame")


def track_mot(
    mot_cache: dict, params: dict, use_reid: bool, *, class_aware: bool, class_voting: bool
) -> dict[str, MOTSequence]:
    """在缓存的 MOT 检测/ReID 特征上跑 BoT-SORT(+ class_aware/voting)，产出 ``{seq: MOTSequence}``。"""
    preds: dict[str, MOTSequence] = {}
    for name, seq in mot_cache.items():
        tracker = make_tracker(
            params,
            MOT_FRAME_RATE,
            use_reid=use_reid,
            class_aware=class_aware,
            class_voting=class_voting,
        )
        frames_out: dict[int, np.ndarray] = {}
        for f in range(1, seq["len"] + 1):
            if f not in seq["frames"]:
                continue
            record = seq["frames"][f]
            if params["camera_motion"]:
                record = dict(record)
                record["frame"] = cv2.imread(str(seq["image_paths"][f]))
            tracked = tracker.update(make_dets(record, use_reid), _tracker_frame(record, params["camera_motion"]))
            if tracked.tracker_id is None or len(tracked) == 0:
                continue
            txyxy = tracked.xyxy
            rows = np.column_stack(
                [
                    tracked.tracker_id.astype(float),
                    txyxy[:, 0],
                    txyxy[:, 1],
                    txyxy[:, 2] - txyxy[:, 0],
                    txyxy[:, 3] - txyxy[:, 1],
                    np.ones(len(tracked)),
                    np.full(len(tracked), VRU_CLASS, dtype=float),
                ]
            )
            frames_out[f] = rows
        preds[name] = MOTSequence(name=name, frames=frames_out)
    return preds


def proxy_metrics(
    proxy_cache: dict[str, list[FrameRecord]],
    params: dict,
    use_reid: bool,
    *,
    class_aware: bool,
    class_voting: bool,
) -> dict:
    """3Hz 无监督代理指标，跨视频平均。"""
    if not proxy_cache:
        return {"proxy_num_ids": 0.0, "proxy_mean_life": 0.0, "proxy_short_id_ratio": 0.0}

    per_video = []
    for records in proxy_cache.values():
        tracker = make_tracker(
            params,
            int(PROXY_FPS),
            use_reid=use_reid,
            class_aware=class_aware,
            class_voting=class_voting,
        )
        id_frames: dict[int, int] = defaultdict(int)
        for record in records:
            tracked = tracker.update(make_dets(record, use_reid), _tracker_frame(record, params["camera_motion"]))
            ids = tracked.tracker_id if tracked.tracker_id is not None else np.array([], dtype=int)
            for tid in ids:
                id_frames[int(tid)] += 1
        lifetimes = list(id_frames.values())
        if not lifetimes:
            per_video.append((0, 0.0, 0.0))
            continue
        short_ratio = sum(1 for v in lifetimes if v <= 2) / len(lifetimes)
        per_video.append((len(lifetimes), float(np.mean(lifetimes)), short_ratio))

    return {
        "proxy_num_ids": float(np.mean([x[0] for x in per_video])),
        "proxy_mean_life": float(np.mean([x[1] for x in per_video])),
        "proxy_short_id_ratio": float(np.mean([x[2] for x in per_video])),
    }


# ----------------------------------------------------------------------------
# 网格 / 排名
# ----------------------------------------------------------------------------
def build_grid(camera_motion_values: Sequence[bool]) -> list[dict]:
    grid = []
    for (high, new), mt, tb, app, prox, alpha, cmc in product(
        HIGH_NEW_PAIRS,
        MATCH_THRESH,
        TRACK_BUFFER,
        APPEARANCE_THRESH,
        PROXIMITY_THRESH,
        REID_ALPHA,
        camera_motion_values,
    ):
        grid.append(
            {
                "track_high_thresh": high,
                "new_track_thresh": new,
                "match_thresh": mt,
                "track_buffer": tb,
                "appearance_thresh": app,
                "proximity_thresh": prox,
                "reid_alpha": alpha,
                "camera_motion": bool(cmc),
            }
        )
    return grid


def rank_with_gate(rows: list[dict], top_k: int) -> tuple[list[dict], float]:
    """门控 + HOTA 排名。返回 ``(ranked_rows, used_gate_short_id)``。"""
    min_numid = min(r["proxy_num_ids"] for r in rows) or 1.0
    gate = GATE_SHORT_ID
    while True:
        for r in rows:
            r["pass_gate"] = r["proxy_short_id_ratio"] <= gate and r["proxy_num_ids"] <= GATE_NUMID_MULT * min_numid
        survivors = [r for r in rows if r["pass_gate"]]
        if len(survivors) >= top_k or gate >= 1.0:
            break
        gate = round(gate + 0.05, 2)

    survivors = sorted((r for r in rows if r["pass_gate"]), key=lambda r: r["HOTA"], reverse=True)
    rest = sorted((r for r in rows if not r["pass_gate"]), key=lambda r: r["HOTA"], reverse=True)
    ranked = survivors + rest
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
    return ranked, gate


# ----------------------------------------------------------------------------
# 可视化
# ----------------------------------------------------------------------------
def render_seq(
    mot_root: Path,
    mot_cache: dict,
    seq: str,
    params: dict,
    out_path: Path,
    use_reid: bool,
    *,
    class_aware: bool,
    class_voting: bool,
) -> None:
    """用给定参数重跑 tracker，并把带轨迹+ID 的标注写成 mp4。"""
    info = parse_seqinfo(mot_root / seq / "seqinfo.ini")
    w, h = info.im_width, info.im_height
    im_dir = mot_root / seq / info.im_dir
    im_ext = info.im_ext
    seqc = mot_cache[seq]

    box_a = sv.BoxAnnotator(thickness=2)
    label_a = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    trace_a = sv.TraceAnnotator(trace_length=30, thickness=2)

    tracker = make_tracker(
        params,
        MOT_FRAME_RATE,
        use_reid=use_reid,
        class_aware=class_aware,
        class_voting=class_voting,
    )
    info = sv.VideoInfo(width=w, height=h, fps=MOT_FRAME_RATE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with sv.VideoSink(str(out_path), info) as sink:
        for f in range(1, seqc["len"] + 1):
            img = cv2.imread(str(im_dir / f"{f:06d}{im_ext}"))
            if img is None:
                continue
            record = seqc["frames"].get(f)
            if record is None:
                record = {"xyxy": np.empty((0, 4)), "conf": np.empty((0,)), "cls": None, "features": None}
            record = dict(record)
            if params["camera_motion"]:
                record["frame"] = img
            tracked = tracker.update(make_dets(record, use_reid), img if params["camera_motion"] else None)
            frame = img.copy()
            if tracked.tracker_id is not None and len(tracked):
                labels = [f"#{int(t)}" for t in tracked.tracker_id]
                frame = trace_a.annotate(frame, tracked)
                frame = box_a.annotate(frame, tracked)
                frame = label_a.annotate(frame, tracked, labels=labels)
            sink.write_frame(frame)


def _tag_value(value: object) -> str:
    return str(value).replace(".", "p")


def param_tag(p: dict) -> str:
    return (
        f"h{_tag_value(p['track_high_thresh'])}_n{_tag_value(p['new_track_thresh'])}_"
        f"m{_tag_value(p['match_thresh'])}_b{p['track_buffer']}_"
        f"a{_tag_value(p['appearance_thresh'])}_p{_tag_value(p['proximity_thresh'])}_"
        f"cmc{int(p['camera_motion'])}"
    )


# ----------------------------------------------------------------------------
# 报告
# ----------------------------------------------------------------------------
def write_report(out_dir: Path, top5: list[dict], gate: float, ctx: dict) -> None:
    lines = []
    lines.append("# BoT-SORT 超参数搜索报告\n")
    lines.append("## 1. 搜索设置\n")
    lines.append(f"- 检测模型：`{ctx['model']}` (type={ctx.get('model_type', '?')}, conf floor={DET_CONF_FLOOR})")
    lines.append(
        f"- Tracker 配置：class_aware={ctx.get('class_aware', '?')}, "
        f"class_voting={ctx.get('class_voting', '?')} (投票含反向写回到 STrack.cls)"
    )
    lines.append(f"- ReID：{'启用' if ctx['use_reid'] else '关闭'}")
    if ctx["use_reid"]:
        lines.append(f"- ReID 模型：`{ctx['reid_model']}`")
        lines.append(f"- ReID providers：`{ctx['reid_providers']}`")
    lines.append("- 目标类(检测->VRU)：pedestrian(9) + bicycle(5)/cyclist(6)/tricycle(7)，其余过滤")
    lines.append(f"- MOT 评估：`{ctx['mot_root']}`，{ctx['n_seq']} 序列，原生 {MOT_FRAME_RATE}Hz")
    lines.append(f"- 3Hz 代理护栏：{ctx['n_proxy']} 个视频")
    lines.append(f"- 网格组合数：{ctx['n_grid']}")
    lines.append("")
    lines.append("## 2. 搜索空间\n")
    lines.append(f"- `track_low_thresh` = {TRACK_LOW_THRESH} (固定)")
    lines.append(f"- `(track_high_thresh, new_track_thresh)` in {list(HIGH_NEW_PAIRS)}")
    lines.append(f"- `match_thresh` in {list(MATCH_THRESH)}")
    lines.append(f"- `track_buffer` in {list(TRACK_BUFFER)}")
    lines.append(f"- `appearance_thresh` in {list(APPEARANCE_THRESH)}")
    lines.append(f"- `proximity_thresh` in {list(PROXIMITY_THRESH)}")
    lines.append(f"- `reid_alpha` in {list(REID_ALPHA)}")
    lines.append(f"- `camera_motion` in {ctx['camera_motion_values']}")
    lines.append("")
    lines.append("## 3. 目标函数\n")
    lines.append(
        f"先用 3Hz 代理 `short_id_ratio <= {gate}` 且 `num_ids <= {GATE_NUMID_MULT} x min` 做门，"
        "幸存者按 MOT HOTA 排序取 top5。"
    )
    lines.append("")
    lines.append("## 4. Top5 参数\n")
    lines.append(
        "| 排名 | high | new | match | buffer | app | prox | alpha | cmc | HOTA | MOTA | IDF1 | "
        "proxy_short_id | proxy_num_ids | proxy_mean_life | 过门 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in top5:
        lines.append(
            f"| {r['rank']} | {r['track_high_thresh']} | {r['new_track_thresh']} | "
            f"{r['match_thresh']} | {r['track_buffer']} | {r['appearance_thresh']} | "
            f"{r['proximity_thresh']} | {r['reid_alpha']} | {int(r['camera_motion'])} | "
            f"{r['HOTA']:.2f} | {r['MOTA']:.2f} | {r['IDF1']:.2f} | "
            f"{r['proxy_short_id_ratio']:.3f} | {r['proxy_num_ids']:.1f} | "
            f"{r['proxy_mean_life']:.1f} | {'PASS' if r['pass_gate'] else 'RELAX'} |"
        )
    lines.append("")
    lines.append("## 5. 推荐参数(rank 1)逐序列指标\n")
    best = top5[0]
    lines.append("| 序列 | HOTA | MOTA | IDF1 |")
    lines.append("|---|---|---|---|")
    for seq, mm in best.get("per_sequence", {}).items():
        lines.append(
            f"| {seq} | {mm.get('HOTA', float('nan')):.2f} | "
            f"{mm.get('MOTA', float('nan')):.2f} | {mm.get('IDF1', float('nan')):.2f} |"
        )
    lines.append("")
    lines.append("## 6. 结论\n")
    bp = {
        k: best[k]
        for k in (
            "track_high_thresh",
            "new_track_thresh",
            "match_thresh",
            "track_buffer",
            "appearance_thresh",
            "proximity_thresh",
            "reid_alpha",
            "camera_motion",
        )
    }
    lines.append(
        f"推荐参数 `{bp}` (+ `track_low_thresh={TRACK_LOW_THRESH}`)，MOT HOTA={best['HOTA']:.2f}、"
        f"IDF1={best['IDF1']:.2f}，3Hz 代理 short_id_ratio={best['proxy_short_id_ratio']:.3f}。"
    )
    lines.append("")
    lines.append("> 注意：默认 ReID 是小型行人 OSNet，只用于快速验证外观融合链路；骑行者整体框需要再做数据验证。")
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="BoT-SORT 超参搜索 (MOT + 3Hz 代理 + ReID)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--model-type", default=MODEL_TYPE, choices=["rtdetr", "yolo", "rfdetr", "rfdetr_unified"])
    ap.add_argument("--mot-root", default=MOT_ROOT)
    ap.add_argument("--proxy-dir", default=PROXY_DIR)
    ap.add_argument("--out-dir", default="runs/botsort_sweep")
    ap.add_argument("--cache-dir", default=None, help="检测/ReID 磁盘缓存目录；默认 <out-dir>/cache")
    ap.add_argument("--no-disk-cache", action="store_true", help="关闭检测/ReID 磁盘缓存")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-mot-frames", type=int, default=None, help="每序列最多检测帧数(调试用)")
    ap.add_argument("--max-proxy", type=int, default=None, help="最多用几个代理视频(调试用)")
    ap.add_argument("--skip-viz", action="store_true")
    ap.add_argument("--disable-reid", action="store_true", help="关闭 ReID，只搜索 BoT-SORT 运动/IoU 部分")
    ap.add_argument("--reid-model", default=DEFAULT_REID_MODEL)
    ap.add_argument("--reid-batch-size", type=int, default=64)
    ap.add_argument(
        "--reid-provider",
        nargs="+",
        default=["CPUExecutionProvider"],
        help="ONNX Runtime providers，例如: CUDAExecutionProvider CPUExecutionProvider",
    )
    ap.add_argument("--camera-motion", action="store_true", help="所有组合都启用 CMC")
    ap.add_argument("--sweep-camera-motion", action="store_true", help="同时搜索 camera_motion=False/True")
    ap.add_argument(
        "--no-class-aware", dest="class_aware", action="store_false", help="禁用 BoT-SORT 类别隔离匹配 (默认开启)"
    )
    ap.add_argument(
        "--no-class-voting",
        dest="class_voting",
        action="store_false",
        help="禁用 ClassVotingTracker 类别投票(含反向写回到 STrack.cls) (默认开启)",
    )
    ap.set_defaults(class_aware=True, class_voting=True)
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="日志级别",
    )
    args = ap.parse_args()
    configure_logging(args.log_level)

    mot_root = Path(args.mot_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = None if args.no_disk_cache else Path(args.cache_dir or out_dir / "cache")
    detector_model, detector_model_type = resolve_detector_model(args.model, args.model_type)

    use_reid = not args.disable_reid
    reid_encoder = None
    if use_reid:
        reid_encoder = ReIDOnnxEncoder(args.reid_model, args.reid_provider, batch_size=args.reid_batch_size)
        LOGGER.info("[reid] model=%s", args.reid_model)
        LOGGER.info("[reid] providers=%s input=%s", reid_encoder.providers, reid_encoder.input_meta.shape)

    if args.sweep_camera_motion:
        camera_motion_values = (False, True)
    elif args.camera_motion:
        camera_motion_values = (True,)
    else:
        camera_motion_values = (False,)
    keep_proxy_frames = any(camera_motion_values)
    LOGGER.info(
        "[config] model=%s resolved_model=%s mot_root=%s proxy_dir=%s out_dir=%s cache_dir=%s",
        args.model,
        detector_model,
        args.mot_root,
        args.proxy_dir,
        args.out_dir,
        cache_dir,
    )
    LOGGER.info("[config] use_reid=%s camera_motion_values=%s", use_reid, list(camera_motion_values))
    LOGGER.info(
        "[config] class_aware=%s class_voting=%s (投票含反向写回到 STrack.cls)",
        args.class_aware,
        args.class_voting,
    )
    LOGGER.info(
        "[config] grid axes: (high,new)=%s, match=%s, buffer=%s, app=%s, prox=%s, alpha=%s",
        list(HIGH_NEW_PAIRS),
        list(MATCH_THRESH),
        list(TRACK_BUFFER),
        list(APPEARANCE_THRESH),
        list(PROXIMITY_THRESH),
        list(REID_ALPHA),
    )

    LOGGER.info("=== 1/4 构建检测器并缓存检测/ReID ===")
    detector = create_detector(detector_model_type, detector_model, conf_thres=DET_CONF_FLOOR, iou_thres=0.5)
    cache_identity = build_cache_identity(detector_model, args.reid_model, use_reid)
    mot_cache = cache_mot_detections(
        detector,
        mot_root,
        args.max_mot_frames,
        reid_encoder,
        cache_dir,
        cache_identity,
    )
    proxy_videos = select_proxy_videos(Path(args.proxy_dir))
    if args.max_proxy is not None:
        proxy_videos = proxy_videos[: args.max_proxy]
    LOGGER.info("[cache:proxy] selected_videos=%d", len(proxy_videos))
    proxy_cache = cache_proxy_detections(
        detector,
        proxy_videos,
        reid_encoder,
        keep_frames=keep_proxy_frames,
        cache_dir=cache_dir,
        cache_identity=cache_identity,
    )

    LOGGER.info("=== 2/4 扫参数网格 ===")
    grid = build_grid(camera_motion_values)
    LOGGER.info("[grid] combinations=%d", len(grid))
    evaluator = MOTEvaluator(mot_root)
    rows: list[dict] = []
    for i, p in enumerate(grid, 1):
        mot_preds = track_mot(
            mot_cache,
            p,
            use_reid,
            class_aware=args.class_aware,
            class_voting=args.class_voting,
        )
        res = evaluator.evaluate(
            mot_preds,
            iou_threshold=0.5,
            metrics=("clear", "identity", "hota"),
            classes=None,
        )
        row = dict(p)
        row["HOTA"] = float(res.overall.get("HOTA", float("nan")))
        row["MOTA"] = float(res.overall.get("MOTA", float("nan")))
        row["IDF1"] = float(res.overall.get("IDF1", float("nan")))
        row["per_sequence"] = {s: dict(m) for s, m in res.per_sequence.items()}
        row.update(
            proxy_metrics(
                proxy_cache,
                p,
                use_reid,
                class_aware=args.class_aware,
                class_voting=args.class_voting,
            )
        )
        rows.append(row)
        LOGGER.info(
            "[grid] %3d/%d h%s n%s m%s b%s app%s cmc%d -> HOTA=%.2f IDF1=%.2f short_id=%.3f",
            i,
            len(grid),
            p["track_high_thresh"],
            p["new_track_thresh"],
            p["match_thresh"],
            p["track_buffer"],
            p["appearance_thresh"],
            int(p["camera_motion"]),
            row["HOTA"],
            row["IDF1"],
            row["proxy_short_id_ratio"],
        )

    LOGGER.info("=== 3/4 门控 + 排名 ===")
    ranked, gate = rank_with_gate(rows, args.top_k)
    top5 = ranked[: args.top_k]
    ctx = {
        "model": detector_model,
        "model_type": detector_model_type,
        "mot_root": args.mot_root,
        "reid_model": args.reid_model,
        "reid_providers": reid_encoder.providers if reid_encoder else [],
        "use_reid": use_reid,
        "camera_motion_values": list(camera_motion_values),
        "class_aware": args.class_aware,
        "class_voting": args.class_voting,
        "n_seq": len(mot_cache),
        "n_proxy": len(proxy_cache),
        "n_grid": len(grid),
    }

    result = {
        "config": {
            "model": args.model,
            "model_type": args.model_type,
            "resolved_model": detector_model,
            "resolved_model_type": detector_model_type,
            "reid_model": args.reid_model if use_reid else None,
            "reid_providers": reid_encoder.providers if reid_encoder else [],
            "use_reid": use_reid,
            "keep_classes": list(KEEP_CLASSES),
            "det_conf_floor": DET_CONF_FLOOR,
            "mot_frame_rate": MOT_FRAME_RATE,
            "proxy_fps": PROXY_FPS,
            "gate_short_id": gate,
            "gate_numid_mult": GATE_NUMID_MULT,
            "class_aware": args.class_aware,
            "class_voting": args.class_voting,
            **{k: v for k, v in ctx.items() if k.startswith("n_")},
        },
        "top5": [{k: v for k, v in r.items() if k != "per_sequence"} for r in top5],
        "all_ranked": [{k: v for k, v in r.items() if k != "per_sequence"} for r in ranked],
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(out_dir, top5, gate, ctx)
    LOGGER.info("已写 %s/result.json 和 REPORT.md", out_dir)
    LOGGER.info("Top%d:", len(top5))
    for r in top5:
        LOGGER.info(
            "  rank%d: h%s n%s m%s b%s app%s cmc%d | HOTA=%.2f IDF1=%.2f short_id=%.3f %s",
            r["rank"],
            r["track_high_thresh"],
            r["new_track_thresh"],
            r["match_thresh"],
            r["track_buffer"],
            r["appearance_thresh"],
            int(r["camera_motion"]),
            r["HOTA"],
            r["IDF1"],
            r["proxy_short_id_ratio"],
            "PASS" if r["pass_gate"] else "RELAX",
        )

    if args.skip_viz:
        LOGGER.info("=== 4/4 跳过可视化 ===")
        return
    LOGGER.info("=== 4/4 渲染 top%d x %d 序列可视化 ===", args.top_k, len(mot_cache))
    viz_dir = out_dir / "viz"
    for r in top5:
        for seq in mot_cache:
            out_path = viz_dir / seq / f"rank{r['rank']}_{param_tag(r)}.mp4"
            render_seq(
                mot_root,
                mot_cache,
                seq,
                r,
                out_path,
                use_reid,
                class_aware=args.class_aware,
                class_voting=args.class_voting,
            )
            LOGGER.info("  %s", out_path)
    LOGGER.info("完成。共 %d 个可视化 mp4 于 %s", len(top5) * len(mot_cache), viz_dir)


if __name__ == "__main__":
    main()
