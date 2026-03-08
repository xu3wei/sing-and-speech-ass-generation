import os
from typing import List, Dict
from tr import tr,trsg
def frame_to_timestamp(frame_idx: int, fps: float) -> str:
    total_seconds = frame_idx / fps
    hours = int(total_seconds // 3600)
    minutes = int(total_seconds % 3600 // 60)
    seconds = total_seconds % 60
    return f"{hours}:{minutes:02d}:{seconds:05.2f}"
def generate_ass_file(subtitles: List[Dict], video_path: str, video_width: int, video_height: int, use_speaker_styles: bool = False,
                      enable_trans:bool=False,tr_prompt:str='',tr_model:str='HY-MT1.5-7B',tr_content='',
                      tr_language:str='中文',enable_single_tr:bool=False,font="方正准圆_GBK",font_size=72,PrimaryColour="&H00FFFFFF",
                      SecondaryColour="&H000000FF",OutlineColour="&H00000000",Bold=0,Italic=0,Underline=0,StrikeOut=0,BackColour="&H80000000",ScaleX=100,ScaleY=100,Spacing=0,Angle=0,BorderStyle=1,Outline=2,Shadow=0,Alignment=2,MarginL=10,MarginR=10,MarginV=10,Encoding=0) -> str:
    ass_header = f"""[Script Info]
;本字幕由AI生成，该AI功能尚不完善，可能存在大量虚假错误信息，请仔细甄别内容
Title: {os.path.basename(video_path)}
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
PlayResX: {video_width}
PlayResY: {video_height}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
"""


    ass_styles = f"""Style: Default,{font},{font_size},{PrimaryColour},{SecondaryColour},{OutlineColour},{BackColour},{Bold},{Italic},{Underline},{StrikeOut},{ScaleX},{ScaleY},{Spacing},{Angle},{BorderStyle},{Outline},{Shadow},{Alignment},{MarginL},{MarginR},{MarginV},{Encoding}
"""

    # 事件部分
    ass_events = []
    # 格式化时间戳辅助函数
    def format_ts(ts):
        h, m, s = ts.split(":")
        return f"{h}:{m}:{s}"
    if enable_trans:
        a,b,c,d,e,f,g,h,j,k=[],[],[],[],0,0,0,[],0,[]  # a相关为翻译相关
    for sub in subtitles:
        if len(sub["text"])>0:
            start_ts, end_ts = sub["time_stamp"].split(",")
        # 处理文本换行符
            ass_text = f"{sub.get('ass_pos', '{\\pos(0,0)}')}{sub['text'].replace(chr(10), '\\N')}"
        # 选择样式（说话人专属/默认）
            style = sub.get("speaker","Default") if use_speaker_styles else "Default"
            if enable_trans and not enable_single_tr:
                if sub["source"]=="ocr":
                    a.append(f"{sub['text'].replace(chr(10),'\\N')}")
                    d.append("ocr")
                    h.append(f"Dialogue: 0,{format_ts(start_ts)},{format_ts(end_ts)},{style},,0,0,0,,{sub.get('ass_pos', '{\\pos(0,0)}')}")
                elif sub["source"]=="asr":
                    b.append(f"{sub['text'].replace(chr(10),'\\N')}")
                    d.append("asr")
                    h.append(f"Dialogue: 0,{format_ts(start_ts)},{format_ts(end_ts)},{style},,0,0,0,,{ass_text}")
                elif sub["source"]=="singing":
                    c.append(f"{sub['text'].replace(chr(10),'\\N')}")
                    d.append("singing")
                    h.append(f"Dialogue: 0,{format_ts(start_ts)},{format_ts(end_ts)},{style},,0,0,0,,{ass_text}")
                    k.append(f"Dialogue: 0,{format_ts(start_ts)},{format_ts(end_ts)},{style},,0,0,0,,{{\\pos({video_width//2},{int(video_height*0.98)})}}")
            elif enable_trans and enable_single_tr:
                if sub["source"]=="ocr":
                    a=trsg(a,ot2tt=tr_prompt,mp=tr_model,ct=tr_content,tl=tr_language)
                    ass_events.append(f"Dialogue: 0,{format_ts(start_ts)},{format_ts(end_ts)},{style},,0,0,0,,{sub.get('ass_pos', '{\\pos(960,1080)}')}{a}")
                    a=[]
                elif sub["source"]=="asr":
                    b=trsg(b,ot2tt=tr_prompt,mp=tr_model,ct=tr_content,tl=tr_language)
                    ass_events.append(f"Dialogue: 0,{format_ts(start_ts)},{format_ts(end_ts)},{style},,0,0,0,,{ass_text}\\N{b}")
                    b=[]
                elif sub["source"]=="singing":
                    c=trsg(c,ot2tt=tr_prompt,mp=tr_model,ct=tr_content,tl=tr_language)
                    ass_events.append(f"Dialogue: 0,{format_ts(start_ts)},{format_ts(end_ts)},{style},,0,0,0,,{{\\pos({video_width//2},{int(video_height*0.98)})}}{c}")
                    c=[]
            else:
                ass_events.append(f"Dialogue: 0,{format_ts(start_ts)},{format_ts(end_ts)},{style},,0,0,0,,{ass_text}")
    if enable_trans and not enable_single_tr:
        if a:
            a=tr(a,ot2tt=tr_prompt,mp=tr_model,ct=tr_content,tl=tr_language)
        if b:
            b=tr(b,ot2tt=tr_prompt,mp=tr_model,ct=tr_content,tl=tr_language)
        if c:
            c=tr(c,ot2tt=tr_prompt,mp=tr_model,ct=tr_content,tl=tr_language)
        if len(a)<d.count('ocr'):
            aa=d.count('ocr')-len(a)
            a+=['请检查内容，可能发生窜行']*aa
        if len(b)<d.count('asr'):
            bb=d.count('asr')-len(b)
            b+=['请检查内容，可能发生窜行']*bb
        if len(c)<d.count('singing'):
            cc=d.count('singing')-len(c)
            c+=['请检查内容，可能发生窜行']*cc
        for i in d:
            if i=="ocr":
                ass_events.append(h[j]+a[e])
                e+=1
                j+=1
            elif i=="asr":
                ass_events.append(h[j]+'\\N'+b[f])
                f+=1
                j+=1
            elif i=="singing":
                ass_events.append(h[j]+'\n'+k[g]+c[g])
                g+=1
                j+=1

    return ass_header + ass_styles + f"[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\nDialogue: 0,0:00:00.00,0:00:08.00,Default,,0,0,0,,{{\\move({video_width+1100},{int(video_height/9)},-1100,{int(video_height/9)},0,8000)\\b1}}本字幕由AI生成，该AI功能尚不完善，可能存在大量虚假错误信息，请仔细甄别内容\n" + "\n".join(ass_events)
