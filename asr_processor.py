from typing import List, Dict
import whisper
from aed import aed2w
def ts_to_sec(timestamp: str) -> float:
    parts = timestamp.split(":")
    if len(parts) != 3:
        raise ValueError("Invalid timestamp format")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds
def _sec_to_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:.2f}"
def extract_asr_subtitles(video_path,device="cuda",model_size="large-v3",video_width=1920,video_height=1080,enable_ocr=False,enable_aed=False,
    smooth_window_size=5, speech_threshold=0.4, singing_threshold=0.5, music_threshold=0.5,min_event_frame=20,max_event_frame=2000,
    min_silence_frame=20, merge_silence_frame=0, extend_speech_frame=0,chunk_max_frame=30000,asr_prompt='',asr_lg='auto',aed_model="FireRedVAD/AED",
                          subx=None,suby=None,singx=None,singy=None,ocr_subtitles=None)-> List[Dict]:
    model = whisper.load_model(model_size, device=device)
    if subx is None:
        asr_x = round(video_width / 2)
    else:
        asr_x = subx
    if suby is None:
        if enable_ocr:
            asr_y = int(video_height)

        else:
            asr_y = int(video_height * 0.95)  # 下顶点（视频高度）
        default_ass_pos = f"{{\\pos({asr_x},{asr_y})}}"
    else:
        asr_y = suby
    if enable_aed:
        if singx is None and singy is None:
            asy = int(video_height * 0.1)
            sppos=f"{{\\pos({asr_x},{asy})}}"
        elif singx is not None and singy is not None:
            sppos=f"{{\\pos({singx},{singy})}}"
        elif singx is None and singy is not None:
            sppos=f"{{\\pos({asr_x},{singy})}}"
        elif singx is not None and singy is None:
            sppos=f"{{\\pos({singx},{asr_y})}}"
    # 音频识别（自动提取视频中的音频）
    subtitles = []
    if enable_aed:
        a = aed2w(video_path, frpt=aed_model, smooth_window_size=smooth_window_size, speech_threshold=speech_threshold,singing_threshold=singing_threshold,music_threshold=music_threshold,min_event_frame=min_event_frame,
    max_event_frame=max_event_frame,min_silence_frame=min_silence_frame,merge_silence_frame=merge_silence_frame,extend_speech_frame=extend_speech_frame,chunk_max_frame=chunk_max_frame)
        for b in a:
            c,d,e=b
            if c=='speech':
                sp=model.transcribe(d,word_timestamps=True,verbose=False,language="auto",initial_prompt=asr_prompt)
                for segment in sp["segments"]:
                    st=_sec_to_timestamp(segment["start"]+e)
                    ed=_sec_to_timestamp(segment["end"]+e)
                    subtitles.append({"text": segment["text"].strip(),"start": segment["start"],"end": segment["end"],"time_stamp": f"{st},{ed}","source": "asr","ass_pos": default_ass_pos})
            else:
                sg=model.transcribe(d,word_timestamps=True,verbose=False,language="auto",initial_prompt=asr_prompt)
                for segment in sg["segments"]:
                    st=_sec_to_timestamp(segment["start"]+e)
                    ed=_sec_to_timestamp(segment["end"]+e)
                    wd=''
                    for word in segment["words"]:
                        wd+=f"{{\\K{int((word["end"]-word["start"])*100)}}}{word['word'].strip()}"
                    subtitles.append({"text": wd,"start": segment["start"],"end": segment["end"],"time_stamp": f"{st},{ed}","source": "singing","ass_pos":sppos})
    else:
        result = model.transcribe(video_path,word_timestamps=True,verbose=False,language="auto",initial_prompt="")
        for segment in result["segments"]:
            start_ts = _sec_to_timestamp(segment["start"])
            end_ts = _sec_to_timestamp(segment["end"])
            subtitles.append({"text": segment["text"].strip(),"start":segment["start"],"end":segment["end"],"time_stamp":f"{start_ts},{end_ts}","source":"asr","ass_pos":default_ass_pos})
    processed_ocr = []
    if enable_ocr:
        for ocr in ocr_subtitles:
            if not isinstance(ocr, dict) or "text" not in ocr or "time_stamp" not in ocr:
                continue
            try:
                ts_parts = ocr["time_stamp"].split(",")
                if len(ts_parts) != 2:
                    continue
                start_sec = ts_to_sec(ts_parts[0])
                end_sec = ts_to_sec(ts_parts[1])
                ocr_processed = {
                    "text": ocr["text"].strip(),
                    "start": start_sec,
                    "end": end_sec,
                    "time_stamp": ocr["time_stamp"],
                    "source": ocr.get("source", "ocr"),
                    "ass_pos": ocr.get("ass_pos", "")
                }
                processed_ocr.append(ocr_processed)
            except Exception:
                continue
        subtitles = processed_ocr + subtitles
        subtitles.sort(key=lambda x: x["start"])
    return subtitles