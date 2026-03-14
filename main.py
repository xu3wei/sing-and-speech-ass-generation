import time
import os
import torch
import argparse
import cv2
from ocr_models import OCRModelEngine
from video_processor import VideoProcessor,OCRConfig
from ass_generator import generate_ass_file
from asr_processor import extract_asr_subtitles
from speaker_diarization import diarize_speakers
import gc
def main():
    parser = argparse.ArgumentParser(description="视频字幕提取：OCR硬字幕 + ASR语音 + 说话人识别")
    parser.add_argument("--yolo-model", default="model.pt", help="YOLO模型路径")
    parser.add_argument("--glm-model", default="GLM-OCR", help="GLM-OCR模型路径")
    parser.add_argument("--video_path", default=r"2.mp4", help="视频路径")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="运行设备")
    parser.add_argument("--frame-skip", type=int, default=3, help="OCR跳帧间隔")
    parser.add_argument("--min-frames", type=int, default=3, help="OCR最小有效帧数")
    parser.add_argument("--pos-threshold", type=int, default=200, help="OCR位置聚合阈值")
    parser.add_argument("--segment-duration", type=int, default=120, help="OCR分段时长(秒)")
    parser.add_argument("--max-workers", type=int, default=2, help="OCR帧处理线程数")
    parser.add_argument("--yolo_conf", type=float, default=0.75, help="YOLO模型置信度")
    parser.add_argument("--glm_max_batch_size", type=int, default=32, help="GLM-OCR模型最大批量处理大小")
    parser.add_argument("--glm_max_new_tokens_per_region", type=int, default=128, help="GLM-OCR模型最大输出")
    parser.add_argument("--text_region_min_area", type=int, default=50, help="OCR文本区域最小面积")
    parser.add_argument("--use_mixed_precision", action="store_true", help="使用混合精度")
    parser.add_argument("--disable_ocr", action="store_true", help="禁用硬字幕提取（OCR）")
    parser.add_argument("--disable_asr", action="store_true", help="禁用语音识别（ASR）")
    parser.add_argument("--enable_speaker", action="store_true", help="禁用说话人识别")#不建议使用
    parser.add_argument("--disable_trans", action="store_true", help="启用翻译功能")
    parser.add_argument("--disable_aed", action="store_true", help="启用AED功能")
    parser.add_argument("--asr_model", default="large-v3",choices=["tiny", "base", "small", "medium", "large","large-v2","large-v3","turbo"],help="Whisper模型大小（默认large-v3）")
    parser.add_argument("--speaker-model-dir", default="speaker-diarization-community-1",help="本地说话人识别模型目录（默认speaker-diarization-community-1）")
    parser.add_argument("--speaker-token", default=None, help="pyannote HuggingFace认证token（本地模型可留空）")
    parser.add_argument("--min-speakers", type=int, default=None, help="说话人识别-最小说话人数（可选，默认自动推断）")
    parser.add_argument("--max-speakers", type=int, default=None, help="说话人识别-最大说话人数（可选，默认自动推断）")
    parser.add_argument("--asr_lg", default="auto", help="ASR模型语言")
    parser.add_argument("--asr_prompt", default="", help="ASR模型初始提示语")
    parser.add_argument("--tr_choice",default="transformers",choices=["transformers","ollama"],help="翻译平台")
    parser.add_argument("--tr_prompt", default="", help="翻译模型初始提示语")
    parser.add_argument("--tr_content", default='''''', help="翻译背景")
    parser.add_argument("--tr_model", default="", help="翻译模型")
    parser.add_argument("--tr_language", default="中文", help="翻译目标语言")
    parser.add_argument("--aed_model", default="FireRedVAD/AED", help="AED模型路径")
    parser.add_argument("--chunk_max_frame", default=30000, type=int, help="AED参数：音频块最大帧数")
    parser.add_argument("--smooth_window_size", default=5, type=int, help="AED参数：滑窗大小")
    parser.add_argument("--min_event_frame", default=20, type=int, help="AED参数：事件最小帧数")
    parser.add_argument("--max_event_frame", default=2000, type=int, help="AED参数：事件最大帧数")
    parser.add_argument("--min_silence_frame", default=20, type=int, help="AED参数：静音最小帧数")
    parser.add_argument("--merge_silence_frame", default=0, type=int, help="AED参数：合并静音帧数")
    parser.add_argument("--extend_speech_frame", default=0, type=int, help="AED参数：扩展语音帧数")
    parser.add_argument("--speech_threshold", default=0.4, type=float, help="AED参数：语音阈值")
    parser.add_argument("--singing_threshold", default=0.5, type=float, help="AED参数： singing阈值")
    parser.add_argument("--music_threshold", default=0.5, type=float, help="AED参数：音乐阈值")
    parser.add_argument("--single_tr", action="store_true", help="启用单句翻译功能")
    parser.add_argument("--font", default="方正准圆_GBK", help="字幕字体")
    parser.add_argument("--font_size", default=72, help="字幕字体大小")
    parser.add_argument("--PrimaryColour", default="&H00FFFFFF", help="字幕主色")
    parser.add_argument("--SecondaryColour", default="&H000000FF", help="字幕次色")
    parser.add_argument("--OutlineColour", default="&H00000000", help="字幕外框色")
    parser.add_argument("--Bold", default=0, help="字幕加粗")
    parser.add_argument("--Italic", default=0, help="字幕斜体")
    parser.add_argument("--Outline", default=2, help="字幕外框宽度")
    parser.add_argument("--BackColour", default="&H80000000", help="字幕背景色")
    parser.add_argument("--ScaleX", default=100, help="字幕X")
    parser.add_argument("--ScaleY", default=100, help="字幕Y")
    parser.add_argument("--Spacing", default=0, help="字幕间距")
    parser.add_argument("--Angle", default=0, help="字幕旋转角度")
    parser.add_argument("--BorderStyle", default=1, help="字幕边框样式")
    parser.add_argument("--Shadow", default=0, help="字幕阴影宽度")
    parser.add_argument("--Alignment", default=2, help="字幕对齐方式")
    parser.add_argument("--MarginL", default=10, help="字幕左外边距")
    parser.add_argument("--MarginR", default=10, help="字幕右外边距")
    parser.add_argument("--MarginV", default=10, help="字幕垂直外边距")
    parser.add_argument("--Encoding", default=0, help="字幕编码")
    parser.add_argument("--StrikeOut", default=0, help="字幕删除线")
    parser.add_argument("--Underline", default=0, help="字幕下划线")
    parser.add_argument("--subx", default=None, help="字幕pos(x,y)")
    parser.add_argument("--suby", default=None, help="字幕pos(x,y)")
    parser.add_argument("--singx", default=None, help="唱歌部分pos(x,y)")
    parser.add_argument("--singy", default=None, help="唱歌部分pos(x,y)")
    parser.add_argument("--tr_speech_save",default=True,help="保存说话原文")
    args = parser.parse_args()

    # 解析功能开关（默认启用所有）
    enable_ocr = not args.disable_ocr
    enable_asr = not args.disable_asr
    enable_speaker = args.enable_speaker
    enable_trans =not args.disable_trans
    enable_aed =not args.disable_aed
    enable_single_tr = args.single_tr
    # 参数校验
    if not enable_ocr and not enable_asr:
        parser.error("不能同时禁用OCR和ASR，至少保留一个识别功能")
    if enable_speaker and not (enable_ocr or enable_asr):
        parser.error("说话人识别依赖OCR/ASR结果，不能单独启用")
    if enable_aed and not enable_asr:
        parser.error("AED需要ASR")

    # 初始化配置
    ocr_config = OCRConfig()
    ocr_config.yolo_conf=args.yolo_conf
    ocr_config.glm_max_batch_size=args.glm_max_batch_size
    ocr_config.glm_max_new_tokens_per_region=args.glm_max_new_tokens_per_region
    ocr_config.text_region_min_area=args.text_region_min_area
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # 清空 CUDA 缓存
        torch.cuda.ipc_collect()  # 收集未使用的 GPU 内存
    gc.collect()

    final_subtitles = []
    video_width, video_height, fps = 0, 0, 0
    start_time = time.time()


    # 1. 硬字幕提取（OCR）：默认启用
    if enable_ocr:
        print("===== 加载OCR模型 =====")
        engine = OCRModelEngine(
            yolo_model_path=args.yolo_model,
            glm_model_path=args.glm_model,
            device=args.device,
            config=ocr_config
        )
        processor = VideoProcessor(engine,ocr_config)
        print("===== 开始OCR硬字幕提取 =====")
        ocr_subtitles, _, vw, vh, fps, total_seg, processed_seg = processor.process_video(
            args.video_path, args.frame_skip, args.min_frames,
            args.pos_threshold, args.segment_duration
        )
        video_width, video_height = vw, vh
        final_subtitles = ocr_subtitles
        print(f"OCR提取到字幕数：{len(ocr_subtitles)}")
        # 清理断点
        '''
        if len(processed_seg) == total_seg:
            delete_breakpoint(args.video_path)
        '''

        # 释放OCR缓存
        print("===== 释放GLM-OCR模型缓存 =====")
        del engine
        del processor
        if args.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

    # 2. 语音识别（ASR）：默认启用
    if enable_asr:
        # 仅ASR时获取视频宽高
        if not enable_ocr:
            cap = cv2.VideoCapture(args.video_path)
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

        # 提取ASR字幕
        if not enable_aed:
            final_subtitles = extract_asr_subtitles(video_path=args.video_path, device=args.device,model_size=args.asr_model, video_width=video_width,
            enable_aed=enable_aed, video_height=video_height,asr_prompt=args.asr_prompt, asr_lg=args.asr_lg,subx=args.subx,suby=args.suby,singx=args.singx,
            singy=args.singy,enable_ocr=enable_ocr,ocr_subtitles=final_subtitles)
        elif enable_aed:
            final_subtitles = extract_asr_subtitles(video_path=args.video_path, device=args.device,model_size=args.asr_model, video_width=video_width,
            enable_aed=enable_aed, video_height=video_height,chunk_max_frame=args.chunk_max_frame,smooth_window_size=args.smooth_window_size,
            min_event_frame=args.min_event_frame,max_event_frame=args.max_event_frame,min_silence_frame=args.min_silence_frame,merge_silence_frame=args.merge_silence_frame,
            extend_speech_frame=args.extend_speech_frame,speech_threshold=args.speech_threshold,singing_threshold=args.singing_threshold,music_threshold=args.music_threshold,
            asr_prompt=args.asr_prompt,asr_lg=args.asr_lg,aed_model=args.aed_model, subx=args.subx, suby=args.suby,singx=args.singx,singy=args.singy,enable_ocr=enable_ocr,
            ocr_subtitles=final_subtitles)
        print(f"ASR后提取到字幕数：{len(final_subtitles)}")
    if final_subtitles:
        ass_content=generate_ass_file(subtitles=final_subtitles,video_path=args.video_path,video_width=video_width,video_height=video_height,
        font=args.font, font_size=args.font_size, PrimaryColour=args.PrimaryColour,SecondaryColour=args.SecondaryColour, OutlineColour=args.OutlineColour,
        Bold=args.Bold,Italic=args.Italic,Outline=args.Outline,BackColour=args.BackColour,ScaleX=args.ScaleX,ScaleY=args.ScaleY,Spacing=args.Spacing,Angle=args.Angle,
        BorderStyle=args.BorderStyle, Shadow=args.Shadow, Alignment=args.Alignment,MarginL=args.MarginL, MarginR=args.MarginR, MarginV=args.MarginV,Encoding=args.Encoding,
        Underline=args.Underline, StrikeOut=args.StrikeOut)
        video_name = os.path.splitext(os.path.basename(args.video_path))[0]
        ass_path = f"{video_name}.ass"
        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(ass_content)

    if enable_speaker:
        print("===== 开始说话人识别 =====")
        final_subtitles = diarize_speakers(
            video_path=args.video_path,
            subtitles=final_subtitles,
            model_dir=args.speaker_model_dir,
            auth_token=args.speaker_token,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            device=args.device
        )
        # 统计说话人
        speakers = set(sub["speaker"] for sub in final_subtitles if "speaker" in sub)
        print(f"识别到说话人：{speakers}")
        # 打印多/无说话人统计
        multi_speaker_count = sum(
            1 for sub in final_subtitles if "all_speakers" in sub and len(sub["all_speakers"]) > 1)
        unknown_count = sum(1 for sub in final_subtitles if sub["speaker"] == "UNKNOWN")
        print(f"多说话人片段数：{multi_speaker_count}")
        print(f"无说话人片段数：{unknown_count}")

    # 4. 生成ASS文件
    if final_subtitles:
        ass_content = generate_ass_file(subtitles=final_subtitles, video_path=args.video_path, video_width=video_width,video_height=video_height,
        font=args.font,font_size=args.font_size,PrimaryColour=args.PrimaryColour,SecondaryColour=args.SecondaryColour,OutlineColour=args.OutlineColour,Bold=args.Bold,
        Italic=args.Italic,Outline=args.Outline, BackColour=args.BackColour,ScaleX=args.ScaleX, ScaleY=args.ScaleY, Spacing=args.Spacing, Angle=args.Angle,
        BorderStyle=args.BorderStyle,Shadow=args.Shadow, Alignment=args.Alignment,MarginL=args.MarginL,MarginR=args.MarginR, MarginV=args.MarginV,
        Encoding=args.Encoding, Underline=args.Underline, StrikeOut=args.StrikeOut,use_speaker_styles=enable_speaker,tr_choice=args.tr_choice,tr_content=args.tr_content,
        enable_trans=enable_trans, tr_model=args.tr_model, tr_prompt=args.tr_prompt, tr_language=args.tr_language,enable_single_tr=enable_single_tr,
        tr_speech_save=args.tr_speech_save)
        video_name = os.path.splitext(os.path.basename(args.video_path))[0]
        ass_path = f"{video_name}.ass"
        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(ass_content)

        # 输出统计信息
        end_time = time.time()
        print(f"\n===== 处理完成 =====")
        print(f"总耗时: {end_time - start_time:.2f}秒")
        print(f"ASS文件保存至: {ass_path}")
        print(f"最终字幕数: {len(final_subtitles)}")
        if enable_speaker:
            print(f"说话人数量: {len(speakers)}")
    else:
        print("\n未提取到有效字幕！")


if __name__ == "__main__":
    main()