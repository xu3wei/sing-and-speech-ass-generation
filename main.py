import time
import os
import argparse
import cv2
from ass_generator import generate_ass_file
from asr_processor import extract_asr_subtitles
import torch
import gc
def main():
    parser = argparse.ArgumentParser(description="ass字幕生成")
    parser.add_argument("--video_path", default="0.mp4", help="视频路径")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="运行设备")
    parser.add_argument("--disable_asr", action="store_true", help="禁用语音识别（ASR）")
    parser.add_argument("--disable_trans", action="store_true", help="禁用翻译功能")
    parser.add_argument("--disable_aed", action="store_true", help="禁用AED功能")
    parser.add_argument("--asr_model", default="large-v3",choices=["tiny", "base", "small", "medium", "large", "large-v3","turbo"],help="Whisper模型大小（默认large-v3）")
    parser.add_argument("--w_prompt", default="愛美", help="ASR模型初始提示语")
    parser.add_argument("--tr_prompt",default="あいみ翻译成爱美", help="翻译模型初始提示语")
    parser.add_argument("--tr_content",default='',help="翻译背景")
    parser.add_argument("--tr_model", default="HY-MT1.5-7B", help="翻译模型")
    parser.add_argument("--tr_language", default="中文", help="翻译目标语言")
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
    parser.add_argument("--font",default="方正准圆_GBK",help="字幕字体")
    parser.add_argument("--font_size",default=72,help="字幕字体大小")
    parser.add_argument("--PrimaryColour",default="&H00FFFFFF",help="字幕主色")
    parser.add_argument("--SecondaryColour",default="&H000000FF",help="字幕次色")
    parser.add_argument("--OutlineColour",default="&H00000000",help="字幕外框色")
    parser.add_argument("--Bold",default=0,help="字幕加粗")
    parser.add_argument("--Italic",default=0,help="字幕斜体")
    parser.add_argument("--Outline",default=2,help="字幕外框宽度")
    parser.add_argument("--BackColour",default="&H80000000",help="字幕背景色")
    parser.add_argument("--ScaleX",default=100,help="字幕X")
    parser.add_argument("--ScaleY",default=100,help="字幕Y")
    parser.add_argument("--Spacing",default=0,help="字幕间距")
    parser.add_argument("--Angle",default=0,help="字幕旋转角度")
    parser.add_argument("--BorderStyle",default=1,help="字幕边框样式")
    parser.add_argument("--Shadow",default=0,help="字幕阴影宽度")
    parser.add_argument("--Alignment",default=2,help="字幕对齐方式")
    parser.add_argument("--MarginL",default=10,help="字幕左外边距")
    parser.add_argument("--MarginR",default=10,help="字幕右外边距")
    parser.add_argument("--MarginV",default=10,help="字幕垂直外边距")
    parser.add_argument("--Encoding",default=0,help="字幕编码")
    parser.add_argument("--StrikeOut",default=0,help="字幕删除线")
    parser.add_argument("--Underline",default=0,help="字幕下划线")
    args = parser.parse_args()
    # 解析功能开关（默认启用所有）
    enable_asr=not args.disable_asr
    enable_trans=not args.disable_trans
    enable_aed=not args.disable_aed
    enable_single_tr=args.single_tr
    # 参数校验
    start_time = time.time()
    if enable_asr:
        cap = cv2.VideoCapture(args.video_path)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if not enable_aed:
            final_subtitles = extract_asr_subtitles(video_path=args.video_path,device=args.device,model_size=args.asr_model,video_width=video_width,
                                                    enable_aed=enable_aed,video_height=video_height)
        if enable_aed:
            final_subtitles = extract_asr_subtitles(video_path=args.video_path,device=args.device,model_size=args.asr_model,video_width=video_width,
                                                    enable_aed=enable_aed,video_height=video_height,chunk_max_frame=args.chunk_max_frame,
                                                    smooth_window_size=args.smooth_window_size,min_event_frame=args.min_event_frame,
                                                    max_event_frame=args.max_event_frame,min_silence_frame=args.min_silence_frame,
                                                    merge_silence_frame=args.merge_silence_frame,extend_speech_frame=args.extend_speech_frame,
                                                    speech_threshold=args.speech_threshold,singing_threshold=args.singing_threshold,
                                                    music_threshold=args.music_threshold)
        print(f"ASR后提取到字幕数：{len(final_subtitles)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
    if final_subtitles:
        ass_content = generate_ass_file(subtitles=final_subtitles,video_path=args.video_path,video_width=video_width,video_height=video_height,
                                        font=args.font,font_size=args.font_size,PrimaryColour=args.PrimaryColour,SecondaryColour=args.SecondaryColour,
                                        OutlineColour=args.OutlineColour,Bold=args.Bold,Italic=args.Italic,Outline=args.Outline,BackColour=args.BackColour,
                                        ScaleX=args.ScaleX,ScaleY=args.ScaleY,Spacing=args.Spacing,Angle=args.Angle,BorderStyle=args.BorderStyle,
                                        Shadow=args.Shadow,Alignment=args.Alignment,MarginL=args.MarginL,MarginR=args.MarginR,MarginV=args.MarginV,
                                        Encoding=args.Encoding,Underline=args.Underline,StrikeOut=args.StrikeOut,)
        video_name = os.path.splitext(os.path.basename(args.video_path))[0]
        ass_path = f"{video_name}.ass"
        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(ass_content)
    if enable_trans:
        ass_content = generate_ass_file(subtitles=final_subtitles,video_path=args.video_path,video_width=video_width,video_height=video_height,
                                        enable_trans=enable_trans,tr_prompt=args.tr_prompt,tr_content=args.tr_content,tr_model=args.tr_model,
                                        tr_language=args.tr_language,enable_single_tr=enable_single_tr,font=args.font,font_size=args.font_size,
                                        PrimaryColour=args.PrimaryColour,SecondaryColour=args.SecondaryColour,OutlineColour=args.OutlineColour,
                                        Bold=args.Bold,Italic=args.Italic,Outline=args.Outline,BackColour=args.BorderStyle,ScaleX=args.ScaleX,
                                        ScaleY=args.ScaleY,Spacing=args.Spacing,Angle=args.Angle,BorderStyle=args.BorderStyle,Shadow=args.Shadow,
                                        Alignment=args.Alignment,MarginL=args.MarginL,MarginR=args.MarginR,MarginV=args.MarginV,Encoding=args.Encoding,
                                        Underline=args.Underline,StrikeOut=args.StrikeOut)
        video_name = os.path.splitext(os.path.basename(args.video_path))[0]
        ass_path = f"{video_name}_tr.ass"
        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(ass_content)
        end_time = time.time()
        print(f"总耗时: {end_time - start_time:.2f}秒\nASS文件保存至: {ass_path}")
    else:
        print("\n未提取到有效字幕！")

if __name__ == "__main__":
    main()