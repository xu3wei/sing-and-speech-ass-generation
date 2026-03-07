import os
import json
import base64
import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

# ---------------- 配置类 ----------------
@dataclass
class OCRConfig:
    max_workers: int = 4  # 帧处理并行线程数
    breakpoint_save_interval: int = 60  # 断点保存间隔（帧）
    max_continuous_gap_frames: int = 30  # 字幕最大连续断帧
    yolo_conf: float = 0.75  # YOLO检测置信度
    glm_max_new_tokens: int = 128  # GLM最大生成token数
    pos_threshold: int = 100  # 位置聚合阈值
    frame_skip: int = 5  # 默认跳帧间隔
    min_frames: int = 10  # 字幕最小有效帧数
    segment_duration: int = 300  # 分段时长（秒）
    gpu_cache_clean_batch: int = 16  # GPU缓存清理间隔

# ---------------- 序列化与工具辅助 ----------------
def _serialize_text_key(text_key: Tuple[str, float, float]) -> str:
    text, x_key, y_key = text_key
    text_b64 = base64.b64encode(text.encode("utf-8")).decode("utf-8")
    return f"{text_b64}|{x_key}|{y_key}"

def _deserialize_text_key(key_str: str) -> Tuple[str, float, float]:
    parts = key_str.split("|")
    if len(parts) != 3:
        raise ValueError(f"无效的key格式: {key_str}")
    text_b64, x_key, y_key = parts
    try:
        text = base64.b64decode(text_b64).decode("utf-8")
    except Exception:
        text = ""
    return (text, float(x_key), float(y_key))

def frame_to_timestamp(frame_idx: int, fps: float) -> str:
    total_seconds = frame_idx / fps
    hours = int(total_seconds // 3600)
    minutes = int(total_seconds % 3600 // 60)
    seconds = total_seconds % 60
    return f"{hours}:{minutes:02d}:{seconds:05.2f}"

def generate_progress_bar(progress: float, bar_length: int = 10) -> str:
    progress = max(0.0, min(1.0, progress))
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + '▱' * (bar_length - filled_length)
    return f"[{bar}] {progress * 100:4.1f}%"

def update_progress_display(current_progress_line: str, segment_idx, total_segments, seg_progress, total_progress, current_frame, end_frame):
    progress_text = (
        f"分段{segment_idx + 1}/{total_segments} | "
        f"分段进度{generate_progress_bar(seg_progress)} | "
        f"总进度{generate_progress_bar(total_progress)} "
        f"(帧={current_frame}/{end_frame})"
    )
    erase_length = max(len(current_progress_line), len(progress_text))
    sys.stdout.write(f"\r{' ' * erase_length}\r{progress_text}")
    sys.stdout.flush()
    return progress_text

def ts_to_sec(ts):
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)

# ---------------- 断点续传文件管理 ----------------
def get_breakpoint_file_path(video_path: str) -> str:
    video_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(video_dir, f"{video_name}_breakpoint.json")

def save_breakpoint(video_path: str, total_segments: int, processed_segments: List[int],
                    segment_duration: int, frame_skip: int, min_frames: int, pos_threshold: int,
                    global_subtitle_stats: Dict, last_saved_stats: Dict, serialize_func):
    new_stats = {}
    for k, v in global_subtitle_stats.items():
        serialized_k = serialize_func(k) if isinstance(k, tuple) else k
        if serialized_k not in last_saved_stats:
            new_stats[serialized_k] = v
    last_saved_stats.update(new_stats)

    breakpoint_data = {
        "video_path": video_path,
        "total_segments": total_segments,
        "processed_segments": processed_segments,
        "parameters": {
            "segment_duration": segment_duration,
            "frame_skip": frame_skip,
            "min_frames": min_frames,
            "pos_threshold": pos_threshold
        },
        "global_subtitle_stats": last_saved_stats
    }

    try:
        breakpoint_path = get_breakpoint_file_path(video_path)
        temp_path = f"{breakpoint_path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(breakpoint_data, f, ensure_ascii=False, indent=2)

        if os.name == "nt" and os.path.exists(breakpoint_path):
            os.remove(breakpoint_path)
        os.replace(temp_path, breakpoint_path)
    except Exception:
        pass

def load_breakpoint(video_path: str, deserialize_func):
    breakpoint_path = get_breakpoint_file_path(video_path)
    if not os.path.exists(breakpoint_path):
        return None, 0, {}

    try:
        with open(breakpoint_path, "r", encoding="utf-8") as f:
            breakpoint_data = json.load(f)

        global_subtitle_stats = defaultdict(lambda: {"segments": []})
        for key_str, data in breakpoint_data.get("global_subtitle_stats", {}).items():
            try:
                text_key = deserialize_func(key_str)
                global_subtitle_stats[text_key] = data
            except Exception:
                continue

        breakpoint_data["global_subtitle_stats"] = global_subtitle_stats
        processed_frame_count = breakpoint_data.get("processed_frame_count", 0)
        print(f"加载断点成功，已处理{len(breakpoint_data['processed_segments'])}段")
        return breakpoint_data, processed_frame_count, global_subtitle_stats
    except Exception:
        os.remove(breakpoint_path)
        return None, 0, {}

def delete_breakpoint(video_path: str):
    breakpoint_path = get_breakpoint_file_path(video_path)
    if os.path.exists(breakpoint_path):
        try:
            os.remove(breakpoint_path)
        except Exception:
            pass