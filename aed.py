from FireRedASR2S.fireredasr2s.fireredvad import FireRedAed, FireRedAedConfig
from vad import frad,fradw

def merge_group_events(event_list, label, max_gap=15.0):
    merged_result = []
    for start_sec, end_sec in event_list:
        if not merged_result:
            merged_result.append({'lb': label, 'ss': start_sec, 'es': end_sec})
            continue
        last_item = merged_result[-1]
        if start_sec - last_item['es'] <= max_gap:
            last_item['es'] = end_sec
        else:
            merged_result.append({'lb': label, 'ss': start_sec, 'es': end_sec})
    return merged_result

def aed(pt,frpt="FireRedVAD/AED",use_gpu=True,smooth_window_size=5,speech_threshold=0.4,singing_threshold=0.6,music_threshold=0.5,min_event_frame=20,
    max_event_frame=2000,min_silence_frame=20,merge_silence_frame=0,extend_speech_frame=0,chunk_max_frame=30000,enable_music=False):
    aed_config=FireRedAedConfig(use_gpu=use_gpu,smooth_window_size=smooth_window_size,speech_threshold=speech_threshold,singing_threshold=singing_threshold,
    music_threshold=music_threshold,min_event_frame=min_event_frame,max_event_frame=max_event_frame,min_silence_frame=min_silence_frame,
    merge_silence_frame=merge_silence_frame,extend_speech_frame=extend_speech_frame,chunk_max_frame=chunk_max_frame)
    aed = FireRedAed.from_pretrained(frpt, aed_config)
    result, probs = aed.detect(frad(pt))
    lbgp = []
    speech_list = result['event2timestamps']['speech']
    if speech_list:
        lbgp += merge_group_events(speech_list, 'speech')
    singing_list = result['event2timestamps']['singing']
    if singing_list:
        a = merge_group_events(singing_list, 'singing', max_gap=3.)
        b = [item for item in a if item['es'] - item['ss']>=10.0]
        lbgp += b
    if enable_music:
        lbgp+=result['event2timestamps']['music']
    return lbgp
def aed2w(pt,frpt="FireRedVAD/AED",use_gpu=True,smooth_window_size=5,speech_threshold=0.4,singing_threshold=0.5,music_threshold=0.5,min_event_frame=20,
    max_event_frame=2000,min_silence_frame=20,merge_silence_frame=0,extend_speech_frame=0,chunk_max_frame=30000,enable_music=False):
    a=aed(pt,frpt,use_gpu=use_gpu,smooth_window_size=smooth_window_size,speech_threshold=speech_threshold,singing_threshold=singing_threshold,
    music_threshold=music_threshold,min_event_frame=min_event_frame,max_event_frame=max_event_frame,min_silence_frame=min_silence_frame,
    merge_silence_frame=merge_silence_frame,extend_speech_frame=extend_speech_frame,chunk_max_frame=chunk_max_frame,enable_music=enable_music)
    b=fradw(pt)
    d=[]
    for i in a:
        lb,ss,es=i['lb'],i['ss'],i['es']
        ssp,esp=int(ss*16000),int(es*16000)
        c=b[ssp:esp]
        d.append((lb,c,ss))
    return d