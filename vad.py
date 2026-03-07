from FireRedASR2S.fireredasr2s.fireredvad import FireRedVad, FireRedVadConfig
import ffmpeg
import numpy as np
def frad(input_path: str) -> np.ndarray:
    out, _ = ffmpeg.input(input_path).output('pipe:', format='s16le', acodec='pcm_s16le', ac=1, ar=16000).run(capture_stdout=True, capture_stderr=True)
    audio_array = np.frombuffer(out, dtype=np.int16)
    return audio_array
def fradw(input_path: str) -> np.ndarray:
    try:
        out,_=ffmpeg.input(input_path, threads=0).output('pipe:',format='f32le',acodec='pcm_f32le',ac=1,ar=16000).run(capture_stdout=True,capture_stderr=True,quiet=True)
    except ffmpeg.Error as e:
        print("FFmpeg 错误：", e.stderr.decode('utf-8'))
        raise
    audio_array = np.frombuffer(out, dtype=np.float32)
    return audio_array
def vad(pt,use_gpu=True,smooth_window_size=5,speech_threshold=0.4,min_speech_frame=20,max_speech_frame=2000,
    min_silence_frame=20,merge_silence_frame=0,extend_speech_frame=5,chunk_max_frame=30000):
    vad_config = FireRedVadConfig(use_gpu=use_gpu,smooth_window_size=smooth_window_size,speech_threshold=speech_threshold,min_speech_frame=min_speech_frame,
    max_speech_frame=max_speech_frame,min_silence_frame=min_silence_frame,merge_silence_frame=merge_silence_frame,extend_speech_frame=extend_speech_frame,chunk_max_frame=chunk_max_frame)
    vad = FireRedVad.from_pretrained("FireRedASR2S/FireRedVAD/VAD", vad_config)
    result, probs = vad.detect(frad(pt))
    return result['timestamps']

# {'dur': 2.32, 'timestamps': [(0.44, 1.82)], 'wav_path': 'assets/hello_zh.wav'}
