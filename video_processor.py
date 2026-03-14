import cv2
import math
import logging
from dataclasses import dataclass
import time
from collections import defaultdict
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from ocr_models import OCRModelEngine
from ass_generator import create_subtitle_structure
import os
import cv2
import json
import hashlib
from typing import Dict
from ocr_models import OCRConfig
logger = logging.getLogger(__name__)
from asr_processor import ts_to_sec

@dataclass
class VideoSegmentParams:
    cap: cv2.VideoCapture
    start_frame: int
    end_frame: int
    fps: float
    video_width: int
    video_height: int
    frame_skip: int
    video_path: str
    total_segments: int
    processed_segments: List[int]
    segment_idx: int


@dataclass
class TextFrameData:
    frames: List[int] = None
    x_mid_list: List[float] = None
    y_top_list: List[float] = None

    def __post_init__(self):
        self.frames = self.frames or []
        self.x_mid_list = self.x_mid_list or []
        self.y_top_list = self.y_top_list or []

def generate_video_breakpoint_key(
    video_path: str,
    fps: float,
    total_frames: int,
    video_width: int,
    video_height: int,
    frame_skip: int,
    segment_duration: int
) -> str:
    """生成视频+处理参数的唯一标识，避免断点错乱"""
    raw_str = f"{os.path.abspath(video_path)}_{fps}_{total_frames}_{video_width}_{video_height}_{frame_skip}_{segment_duration}"
    return hashlib.md5(raw_str.encode()).hexdigest()[:12]

def get_breakpoint_file_path(video_path: str, breakpoint_key: str) -> str:
    """生成断点文件路径，和视频文件同目录"""
    video_dir = os.path.dirname(os.path.abspath(video_path))
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(video_dir, f".{video_name}_{breakpoint_key}_ocr_breakpoint.json")

def load_breakpoint(breakpoint_file: str) -> Dict:
    """加载断点数据，不存在/损坏则返回空字典"""
    if not os.path.exists(breakpoint_file):
        return {}
    try:
        with open(breakpoint_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        logger.warning(f"断点文件损坏，将重新处理: {breakpoint_file}")
        return {}

def save_breakpoint_atomically(breakpoint_file: str, data: Dict) -> None:
    """原子化写入断点文件，避免写入中断导致文件损坏"""
    temp_file = breakpoint_file + ".tmp"
    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # 原子重命名，Windows/Linux均兼容
        os.replace(temp_file, breakpoint_file)
    except Exception as e:
        logger.error(f"断点保存失败: {str(e)}")
        if os.path.exists(temp_file):
            os.remove(temp_file)


def convert_tuple_keys_to_str(data: Dict) -> Dict:
    """将字典中的 tuple 类型 Key 转换为 JSON 字符串（用于保存）"""
    result = {}
    for key, value in data.items():
        # 如果 Key 是 tuple，转为 JSON 字符串
        if isinstance(key, tuple):
            str_key = json.dumps(key, ensure_ascii=False)
        else:
            str_key = key

        # 递归处理（如果 value 也是字典）
        if isinstance(value, dict):
            result[str_key] = convert_tuple_keys_to_str(value)
        else:
            result[str_key] = value
    return result


def convert_str_keys_back_to_tuple(data: Dict) -> Dict:
    """将 JSON 字符串 Key 还原回 tuple（用于加载）"""
    result = {}
    for str_key, value in data.items():
        # 尝试将字符串 Key 解析回 tuple
        try:
            key = json.loads(str_key)
            # 如果解析出来是 list，转为 tuple（因为 JSON 没有 tuple，只有 array）
            if isinstance(key, list):
                key = tuple(key)
        except (json.JSONDecodeError, ValueError):
            key = str_key

        # 递归处理
        if isinstance(value, dict):
            result[key] = convert_str_keys_back_to_tuple(value)
        else:
            result[key] = value
    return result
class VideoProcessor:
    """视频文字处理核心类（重构版：CPU/GPU流水线 + 大Batch推理）"""
    MIN_POS_THRESHOLD = 10
    # 【关键参数】每收集多少帧后，做一次GLM批量推理
    # 建议值：10-50（值越大，Batch越大，GPU利用率越高，但内存占用也越大）
    FRAMES_PER_GLM_BATCH = 30

    def __init__(self, model_engine: OCRModelEngine, config: OCRConfig):
        self.engine = model_engine
        self.config = config
        self.device = model_engine.device
        # 线程池仅用于CPU侧的YOLO检测和抠图
        self.max_workers = max(2, min(8, torch.cuda.device_count() * 4 if torch.cuda.is_available() else 4))
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        # 视频维度参数
        self._pos_threshold_x = None
        self._pos_threshold_y = None
        self._max_gap = None
        self._coord_dist_threshold = None
        self._coord_std_threshold = None

    def _init_video_dim_params(self, video_width: int, video_height: int, frame_skip: int):
        self._pos_threshold_x = video_width / 20
        self._pos_threshold_y = video_height / 10
        self._max_gap = max(10, frame_skip * 2)
        self._coord_dist_threshold = video_width / 20
        self._coord_std_threshold = video_width / 40

    def get_text_key(self, x_mid: float, y_top: float) -> Tuple[float, float]:
        x_key = math.floor(x_mid / self._pos_threshold_x) * self._pos_threshold_x
        y_key = math.floor(y_top / self._pos_threshold_y) * self._pos_threshold_y
        return (x_key, y_key)

    def _aggregate_raw_results(self, raw_results: List[dict], frame_skip: int, video_width: int) -> Dict:
        """将YOLO+GLM的原始结果聚合为分段统计数据"""
        temp_group = defaultdict(lambda: defaultdict(TextFrameData))

        for res in raw_results:
            text = res.get("text", "")
            if not text:
                continue
            pos_key = self.get_text_key(res["x_mid"], res["y_top"])
            data = temp_group[pos_key][text]
            data.frames.append(res["frame_idx"])
            data.x_mid_list.append(res["x_mid"])
            data.y_top_list.append(res["y_top"])

        segment_stats = defaultdict(lambda: {"segments": []})

        for pos_key, text_dict in temp_group.items():
            for text, data in text_dict.items():
                if not data.frames:
                    continue

                # 简单的连续帧合并逻辑（简化版，可根据需要扩展）
                sorted_pairs = sorted(zip(data.frames, data.x_mid_list, data.y_top_list), key=lambda x: x[0])
                if not sorted_pairs:
                    continue

                groups = [[sorted_pairs[0]]]
                for pair in sorted_pairs[1:]:
                    last_pair = groups[-1][-1]
                    frame_gap = pair[0] - last_pair[0]
                    coord_dist = math.hypot(pair[1] - last_pair[1], pair[2] - last_pair[2])

                    if frame_gap <= self._max_gap and coord_dist < self._coord_dist_threshold:
                        groups[-1].append(pair)
                    else:
                        groups.append([pair])

                for group in groups:
                    frames = [p[0] for p in group]
                    avg_x = sum(p[1] for p in group) / len(group)
                    avg_y = sum(p[2] for p in group) / len(group)
                    text_key = (text, pos_key[0], pos_key[1])
                    segment_stats[text_key]["segments"].append({
                        "frames": frames,
                        "x_mid": avg_x,
                        "y_top": avg_y
                    })
        return segment_stats

    def process_video_segment(self, params: VideoSegmentParams) -> Dict:
        """处理单个视频分段（核心重构：CPU预处理并行 -> GPU大Batch推理）"""
        cap = params.cap
        start_frame, end_frame = params.start_frame, params.end_frame
        video_width, video_height = params.video_width, params.video_height
        frame_skip = params.frame_skip
        segment_idx = params.segment_idx

        # 跳转到起始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        all_raw_results = []
        frame_buffer = []  # 缓存帧数据，凑够一批后处理

        logger.info(f"分段 {segment_idx} 开始处理 | 帧范围: {start_frame}-{end_frame}")

        # 1. 第一阶段：读帧 + 多线程YOLO检测 + 抠图
        while current_frame_idx < end_frame:
            # 读取一批帧
            while len(frame_buffer) < self.FRAMES_PER_GLM_BATCH and current_frame_idx < end_frame:
                ret, frame = cap.read()
                if not ret or frame is None:
                    current_frame_idx += 1
                    continue

                # 跳帧逻辑
                if current_frame_idx % frame_skip != 0:
                    current_frame_idx += 1
                    continue

                frame_buffer.append((frame.copy(), current_frame_idx))
                current_frame_idx += 1

            if not frame_buffer:
                break

            # 2. 多线程并行处理这批帧的YOLO检测和抠图
            logger.debug(f"分段 {segment_idx}: 处理 {len(frame_buffer)} 帧的YOLO检测...")
            yolo_futures = []
            for frame, f_idx in frame_buffer:
                future = self.executor.submit(
                    self.engine.detect_text_regions,
                    frame=frame,
                    frame_idx=f_idx,
                    video_width=video_width,
                    video_height=video_height
                )
                yolo_futures.append(future)

            # 收集YOLO结果，提取所有待识别的文字区域
            pending_regions = []
            for future in as_completed(yolo_futures):
                try:
                    regions = future.result()
                    pending_regions.extend(regions)
                except Exception as e:
                    logger.error(f"YOLO任务失败: {str(e)[:80]}")

            frame_buffer = []  # 清空帧缓存，释放内存

            if not pending_regions:
                continue

            # 3. 【核心提速】第二阶段：单线程大Batch GLM推理
            logger.debug(f"分段 {segment_idx}: 批量识别 {len(pending_regions)} 个文字区域...")
            start_glm = time.time()
            recognized_texts = self.engine.batch_recognize_regions(pending_regions)
            elapsed_glm = time.time() - start_glm
            logger.debug(f"分段 {segment_idx}: GLM推理完成 | 耗时: {elapsed_glm:.2f}s | 区域数: {len(pending_regions)}")

            # 4. 合并YOLO位置信息和GLM识别结果
            for region, text in zip(pending_regions, recognized_texts):
                region["text"] = text
                all_raw_results.append(region)

            # 更新进度（简化版）
            progress = (current_frame_idx - start_frame) / (end_frame - start_frame)
            logger.info(
                f"分段 {segment_idx} 进度: {progress:.1%} | 已识别文本数: {len([r for r in all_raw_results if r.get('text')])}")

        # 5. 聚合结果
        logger.info(f"分段 {segment_idx} 预处理完成，正在聚合结果...")
        return self._aggregate_raw_results(all_raw_results, frame_skip, video_width)

    def process_video(self, video_path: str, frame_skip: int = 5, min_frames: int = 10,
                      pos_threshold: int = 100, segment_duration: int = 300) -> Tuple:
        """处理整个视频（主入口，新增完整断点续传能力）"""
        frame_skip = max(1, frame_skip)
        min_frames = max(1, min_frames)

        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"视频读取失败: {video_path}")
            return [], "", 0, 0, 0, 0, []

        # 获取视频元数据
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._init_video_dim_params(video_width, video_height, frame_skip)

        # 计算分段
        segment_frames = int(segment_duration * fps) if segment_duration > 0 else total_frames
        total_segments = math.ceil(total_frames / segment_frames)

        logger.info(
            f"视频信息: {fps:.1f}fps | {total_frames}帧 | {total_frames / fps / 60:.1f}分钟 | {video_width}x{video_height}")
        logger.info(
            f"分段配置: {total_segments}段 | 每段约{segment_frames}帧 | GLM每{self.FRAMES_PER_GLM_BATCH}帧推理一次")

        # -------------------------- 断点续传核心逻辑：加载断点 --------------------------
        # 生成视频+参数唯一标识
        breakpoint_key = generate_video_breakpoint_key(
            video_path, fps, total_frames, video_width, video_height, frame_skip, segment_duration
        )
        breakpoint_file = get_breakpoint_file_path(video_path, breakpoint_key)
        # 加载断点数据
        breakpoint_data = load_breakpoint(breakpoint_file)
        # 已处理的分段列表
        processed_segments = breakpoint_data.get("processed_segments", [])
        # 已处理分段的OCR结果
        global_stats = defaultdict(lambda: {"segments": []}, breakpoint_data.get("global_stats", {}))

        if processed_segments:
            logger.info(
                f"✅ 加载断点成功 | 已完成分段: {processed_segments} | 剩余分段: {total_segments - len(processed_segments)}")
        # --------------------------------------------------------------------------------

        try:
            for seg_idx in range(total_segments):
                if seg_idx in processed_segments:
                    logger.info(f"⏭️  分段 {seg_idx + 1}/{total_segments} 已完成，跳过")
                    continue

                start_frame = seg_idx * segment_frames
                end_frame = min((seg_idx + 1) * segment_frames, total_frames)

                seg_params = VideoSegmentParams(
                    cap=cap, start_frame=start_frame, end_frame=end_frame,
                    fps=fps, video_width=video_width, video_height=video_height,
                    frame_skip=frame_skip, video_path=video_path,
                    total_segments=total_segments, processed_segments=processed_segments,
                    segment_idx=seg_idx
                )

                # 处理分段
                seg_stats = self.process_video_segment(seg_params)

                # 合并结果到全局
                for text_key, data in seg_stats.items():
                    if text_key not in global_stats:
                        global_stats[text_key] = {"segments": []}
                    global_stats[text_key]["segments"].extend(data["segments"])

                # 标记分段为已完成
                processed_segments.append(seg_idx)
                # -------------------------- 断点续传核心逻辑：保存断点 --------------------------
                # 分段处理完成后，立刻持久化到本地
                safe_global_stats = convert_tuple_keys_to_str(dict(global_stats))

                save_breakpoint_atomically(
                    breakpoint_file,
                    {
                        "video_info": {
                            "path": video_path,
                            "fps": fps,
                            "total_frames": total_frames,
                            "width": video_width,
                            "height": video_height
                        },
                        "process_params": {
                            "frame_skip": frame_skip,
                            "segment_duration": segment_duration,
                            "min_frames": min_frames
                        },
                        "processed_segments": processed_segments,
                        "global_stats": safe_global_stats
                    }
                )
            subtitles = []
            for text_key, data in global_stats.items():
                for seg in data["segments"]:
                    if len(seg["frames"]) < min_frames:
                        continue
                    sub_dict = create_subtitle_structure(text_key, seg, fps)
                    subtitles.append(sub_dict)
            subtitles.sort(key=lambda x: ts_to_sec(x["time_stamp"].split(",")[0]))
            logger.info(f"全部处理完成 | 最终有效字幕数: {len(subtitles)}")
            if os.path.exists(breakpoint_file):
                os.remove(breakpoint_file)
            return subtitles, video_path, video_width, video_height, fps, total_segments, processed_segments
        finally:
            cap.release()
            self.executor.shutdown(wait=False)
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()