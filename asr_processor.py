from typing import List, Dict
import whisper
from aed import aed2w

def extract_asr_subtitles(video_path,device="cuda",model_size="large-v3",video_width=1920,video_height=1080,enable_ocr=False,enable_aed=False,
    smooth_window_size=5, speech_threshold=0.4, singing_threshold=0.5, music_threshold=0.5,min_event_frame=20,max_event_frame=2000,
    min_silence_frame=20, merge_silence_frame=0, extend_speech_frame=0,chunk_max_frame=30000)-> List[Dict]:
    model = whisper.load_model(model_size, device=device)
    asr_x = round(video_width / 2)
    if enable_ocr:
        asr_y = int(video_height * 0.16)
    elif enable_aed:
        asr_y = int(video_height * 0.98)
        asy = int(video_height * 0.1)
        sppos=f"{{\\pos({asr_x},{asy})}}"
    else:
        asr_y = int(video_height * 0.95)  # 下顶点（视频高度）
    default_ass_pos = f"{{\\pos({asr_x},{asr_y})}}"
    # 音频识别（自动提取视频中的音频）
    subtitles = []
    if enable_aed:
        a=aed2w(video_path,smooth_window_size=smooth_window_size,speech_threshold=speech_threshold,singing_threshold=singing_threshold,music_threshold=music_threshold,min_event_frame=min_event_frame,
    max_event_frame=max_event_frame,min_silence_frame=min_silence_frame,merge_silence_frame=merge_silence_frame,extend_speech_frame=extend_speech_frame,chunk_max_frame=chunk_max_frame)
        for b in a:
            c,d,e=b
            if c=='speech':
                sp=model.transcribe(d,word_timestamps=True,verbose=False,language="Japanese",initial_prompt="愛美")
                for segment in sp["segments"]:
                    st=_sec_to_timestamp(segment["start"]+e)
                    ed=_sec_to_timestamp(segment["end"]+e)
                    subtitles.append({"text": segment["text"].strip(),"start": segment["start"],"end": segment["end"],"time_stamp": f"{st},{ed}","source": "asr","ass_pos": default_ass_pos})
            else:
                sg=model.transcribe(d,word_timestamps=True,verbose=False,language="Japanese",initial_prompt="愛美")
                for segment in sg["segments"]:
                    st=_sec_to_timestamp(segment["start"]+e)
                    ed=_sec_to_timestamp(segment["end"]+e)
                    wd=''
                    for word in segment["words"]:
                        wd+=f"{{\\K{int((word["end"]-word["start"])*100)}}}{word['word'].strip()}"
                    subtitles.append({"text": wd,"start": segment["start"],"end": segment["end"],"time_stamp": f"{st},{ed}","source": "singing","ass_pos":sppos})
    else:
        result = model.transcribe(video_path,word_timestamps=True,verbose=False,language="Japanese",initial_prompt="愛美")
        for segment in result["segments"]:
            start_ts = _sec_to_timestamp(segment["start"])
            end_ts = _sec_to_timestamp(segment["end"])
            subtitles.append({"text": segment["text"].strip(),"start":segment["start"],"end":segment["end"],"time_stamp":f"{start_ts},{end_ts}","source":"asr","ass_pos":default_ass_pos})
    return subtitles

# 工具函数：秒数转ASS时间戳(HH:MM:SS.xx)
def _sec_to_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:.2f}"
