# ASS 字幕生成工具

## 项目简介
这是一个功能强大的 ASS 字幕生成工具，集成了 **自动语音识别（ASR）**、**音频事件检测（AED）** 和 **字幕翻译** 功能，支持高度自定义字幕样式，可快速为视频生成专业的 ASS 格式字幕。

## 功能特性
- 支持 Whisper 系列模型进行语音识别
- 集成音频事件检测（AED）优化字幕切分
- 支持自定义翻译模型进行字幕翻译
- 全方面的字幕样式自定义（字体、颜色、边框等）
- 支持 CPU/GPU 运行

## 安装依赖
确保已安装 Python 3.8+，然后安装所需库：

```bash
pip install opencv-python torch openai-whisper
# 如需使用翻译功能，还需安装相关依赖（如 transformers 等，根据翻译模型要求）
```

*注：请根据实际使用的翻译模型安装额外依赖。*

## 使用方法
基本命令格式：

```bash
python script.py --video_path <视频路径> [其他参数]
```

### 示例
1. **基础使用**（启用 ASR+AED，生成原文字幕）：
   ```bash
   python script.py --video_path input.mp4
   ```

2. **仅用 ASR（禁用 AED）**：
   ```bash
   python script.py --video_path input.mp4 --disable_aed
   ```

3. **启用翻译**（生成原文字幕+翻译字幕）：
   ```bash
   python script.py --video_path input.mp4
   ```
   *注：翻译功能默认启用，可通过 `--disable_trans` 禁用。*

4. **自定义字幕样式**：
   ```bash
   python script.py --video_path input.mp4 --font "微软雅黑" --font_size 60 --PrimaryColour "&H00FFFF00"
   ```

## 参数说明

### 通用参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--video_path` | `0.mp4` | 输入视频路径 |
| `--device` | `cuda` | 运行设备，可选 `cuda` 或 `cpu` |

### 功能开关
| 参数 | 说明 |
|------|------|
| `--disable_asr` | 禁用语音识别（ASR） |
| `--disable_trans` | 禁用翻译功能 |
| `--disable_aed` | 禁用音频事件检测（AED） |
| `--single_tr` | 启用单句翻译模式 |

### ASR 参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--asr_model` | `large-v3` | Whisper 模型大小，可选 `tiny`/`base`/`small`/`medium`/`large`/`large-v3`/`turbo` |
| `--asr_lg` | `auto` | ASR 模型语言（如 `zh`、`en`，默认自动检测） |
| `--asr_prompt` | 空 | ASR 模型初始提示语 |

### 翻译参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--tr_model` | `HY-MT1.5-7B` | 翻译模型名称 |
| `--tr_language` | `中文` | 翻译目标语言 |
| `--tr_prompt` | 空 | 翻译模型初始提示语 |
| `--tr_content` | 空 | 翻译背景信息 |

### AED 参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--chunk_max_frame` | `30000` | 音频块最大帧数 |
| `--smooth_window_size` | `5` | 滑窗大小 |
| `--min_event_frame` | `20` | 事件最小帧数 |
| `--max_event_frame` | `2000` | 事件最大帧数 |
| `--min_silence_frame` | `20` | 静音最小帧数 |
| `--merge_silence_frame` | `0` | 合并静音帧数 |
| `--extend_speech_frame` | `0` | 扩展语音帧数 |
| `--speech_threshold` | `0.4` | 语音检测阈值 |
| `--singing_threshold` | `0.5` | 歌声检测阈值 |
| `--music_threshold` | `0.5` | 音乐检测阈值 |

### 字幕样式参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--font` | `方正准圆_GBK` | 字幕字体 |
| `--font_size` | `72` | 字幕字体大小 |
| `--PrimaryColour` | `&H00FFFFFF` | 主色（ASS 格式：`&HBBGGRR`） |
| `--SecondaryColour` | `&H000000FF` | 次色 |
| `--OutlineColour` | `&H00000000` | 外框色 |
| `--BackColour` | `&H80000000` | 背景色 |
| `--Bold` | `0` | 加粗（0 否 / 1 是） |
| `--Italic` | `0` | 斜体（0 否 / 1 是） |
| `--Underline` | `0` | 下划线（0 否 / 1 是） |
| `--StrikeOut` | `0` | 删除线（0 否 / 1 是） |
| `--Outline` | `2` | 外框宽度 |
| `--Shadow` | `0` | 阴影宽度 |
| `--BorderStyle` | `1` | 边框样式 |
| `--Alignment` | `2` | 对齐方式（1 左下 / 2 中下 / 3 右下 / 5 左上 / 6 中上 / 7 右上） |
| `--MarginL` | `10` | 左外边距 |
| `--MarginR` | `10` | 右外边距 |
| `--MarginV` | `10` | 垂直外边距 |
| `--ScaleX` | `100` | X 轴缩放（%） |
| `--ScaleY` | `100` | Y 轴缩放（%） |
| `--Spacing` | `0` | 字符间距 |
| `--Angle` | `0` | 旋转角度 |
| `--Encoding` | `0` | 编码 |

## 注意事项
1. **模型下载**：首次使用 Whisper 模型时，会自动下载对应模型文件，需保持网络连接。
2. **GPU 加速**：建议使用 CUDA 设备运行以提升速度，需确保安装对应版本的 PyTorch。
3. **翻译模型**：翻译功能需根据所选模型配置环境，如使用 HuggingFace 模型需安装 `transformers` 等库。
4. **ASS 颜色格式**：颜色参数遵循 ASS 格式 `&HBBGGRR`（蓝-绿-红顺序，注意与常见 RGB 顺序相反）。

## 输出文件
- 原文字幕：`{视频名}.ass`
- 翻译字幕：`{视频名}_tr.ass`（若启用翻译）
