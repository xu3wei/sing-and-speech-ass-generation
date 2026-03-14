import logging
from typing import List, Optional
from dataclasses import dataclass
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForImageTextToText
logger = logging.getLogger(__name__)
@dataclass
class OCRConfig:
    """简化配置，专注核心参数"""
    yolo_conf: float = 0.75
    glm_max_batch_size: int = 32
    glm_max_new_tokens_per_region: int = 128
    text_region_min_area: int = 50
    use_mixed_precision: bool = True
    compile_model: bool = True
class OCRModelEngine:
    def __init__(self,yolo_model_path: str,glm_model_path: str = "GLM-OCR",device: str = 'cuda',config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()
        self.device = torch.device('cuda' if (torch.cuda.is_available() and device == 'cuda') else 'cpu')
        self.dtype = torch.bfloat16 if (self.config.use_mixed_precision and self.device.type == 'cuda') else torch.float32
        self.OCR_PROMPT = "Transcribe"
        self._load_yolo(yolo_model_path)
        self._load_glm(glm_model_path)
        logger.info(f"设备:{self.device}|BatchSize:{self.config.glm_max_batch_size}")

    def _load_yolo(self, model_path: str) -> None:
        try:
            self.yolo_detector = YOLO(model_path).to(self.device)
            try:
                self.yolo_detector.fuse()
            except Exception as e:
                logger.warning(f"YOLO融合失败: {str(e)[:50]}")
            self._warmup_yolo()
        except Exception as e:
            logger.error(f"YOLO加载失败: {str(e)}", exc_info=True)
            raise
    def _warmup_yolo(self):
        dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        for _ in range(2):
            _ = self.yolo_detector(dummy, device=self.device, conf=0.5, verbose=False)

    def _load_glm(self, model_path: str) -> None:
        try:
            self.glm_processor = AutoProcessor.from_pretrained(model_path)
            if self.glm_processor.tokenizer.pad_token is None:
                self.glm_processor.tokenizer.pad_token = self.glm_processor.tokenizer.eos_token
            self.glm_model = AutoModelForImageTextToText.from_pretrained(model_path,dtype=self.dtype,device_map="auto",low_cpu_mem_usage=True).eval()
            if self.config.compile_model and self.device.type == 'cuda':
                self.glm_model = torch.compile(self.glm_model, mode="reduce-overhead")
            self._warmup_glm()
        except Exception as e:
            logger.error(f"GLM加载失败: {str(e)}", exc_info=True)
            raise
    def _warmup_glm(self):
        dummy_img = Image.new("RGB", (224, 224), (255, 255, 255))
        dummy_msg = [[{"role": "user","content": [{"type": "image", "image": dummy_img}, {"type": "text", "text": self.OCR_PROMPT}]}]]
        with torch.no_grad():
            inputs = self.glm_processor.apply_chat_template(
                dummy_msg, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", padding=True,
            ).to(self.glm_model.device)
            inputs.pop("token_type_ids", None)
            for _ in range(2):
                _ = self.glm_model.generate(**inputs, max_new_tokens=10, num_beams=1, do_sample=False)

    def detect_text_regions(self,frame: np.ndarray,frame_idx: int,video_width: int,video_height: int) -> List[dict]:
        try:
            regions = []
            bboxes = self.yolo_detector(frame, device=self.device, conf=self.config.yolo_conf, verbose=False, imgsz=640)
            for result in bboxes:
                if result.boxes is None:
                    continue
                boxes_xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy().tolist()
                for box, conf in zip(boxes_xyxy, confs):
                    x1, y1, x2, y2 = box
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(video_width, x2), min(video_height, y2)
                    if (x2 - x1) * (y2 - y1) <= self.config.text_region_min_area:
                        continue
                    x_mid = (x1 + x2) / 2
                    y_top = y1
                    roi = frame[y1:y2, x1:x2]
                    pil_region = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)).convert('RGB')
                    regions.append({"frame_idx": frame_idx,"pil_image": pil_region,"x_mid": x_mid,"y_top": y_top,"pos_key": None})
            return regions
        except Exception as e:
            logger.warning(f"帧 {frame_idx} 检测失败: {str(e)[:80]}")
            return []
    def batch_recognize_regions(self, regions: List[dict]) -> List[str]:
        if not regions:
            return []
        total_size = len(regions)
        output_texts = [""] * total_size
        messages_list = []
        for r in regions:
            messages_list.append([{"role": "user", "content": [{"type": "image", "image": r["pil_image"]},{"type": "text", "text": self.OCR_PROMPT}]}])
        for batch_start in range(0, total_size, self.config.glm_max_batch_size):
            batch_end = min(batch_start + self.config.glm_max_batch_size, total_size)
            batch_messages = messages_list[batch_start:batch_end]
            try:
                inputs = self.glm_processor.apply_chat_template(
                    batch_messages, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt", padding=True,
                ).to(self.glm_model.device)
                inputs.pop("token_type_ids", None)
                with torch.no_grad():
                    generated_ids = self.glm_model.generate(**inputs,max_new_tokens=self.config.glm_max_new_tokens_per_region * len(batch_messages),
                        num_beams=1,do_sample=False,use_cache=True)
                batch_texts = self.glm_processor.batch_decode(generated_ids[:, inputs["input_ids"].shape[1]:],skip_special_tokens=True)
                for i, text in enumerate(batch_texts):
                    output_texts[batch_start + i] = text.strip()
                del generated_ids
                del inputs
                del batch_texts
                del batch_messages
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"GLM批量推理失败 [{batch_start}-{batch_end}): {str(e)[:100]}", exc_info=True)
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        del messages_list
        return output_texts