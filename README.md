# 视频字幕提取工具

一款视频字幕提取工具，集成 **OCR硬字幕识别**、**ASR语音识别**、**音频事件检测（AED）** 与 **双后端字幕翻译** 能力，最终生成全参数可定制的ASS格式字幕文件。

## 功能特性
- **OCR硬字幕提取**
  - 采用YOLO文本区域检测 + GLM-OCR文字识别的两级架构
  - 支持跳帧优化、同位置字幕聚合、分段并行处理，兼顾效率与准确率
  - 可自定义文本区域过滤、置信度阈值等核心参数

- **ASR语音字幕识别**
  - 基于OpenAI Whisper全系列模型，支持tiny到large-v3/turbo全尺寸选型
  - 支持指定识别语言、自定义初始提示词，优化特定场景识别准确率
  - 自动适配视频分辨率，匹配字幕显示位置

- **音频事件检测（AED）**
  - 精准区分语音、唱歌、背景音乐等音频场景
  - 全参数可自定义，支持阈值调整、滑窗平滑、事件长度过滤等优化
  - 可单独配置唱歌片段的字幕显示位置

- **双后端字幕翻译**
  - **Transformers后端**：提示词基于**通义千问Qwen3.5**构建
  - **Ollama后端**：提示词基于腾讯混元 hunyuan-mt-1.5构建
  - 支持自定义翻译提示词、背景知识设定、目标语言，支持单句/批量翻译模式

- **ASS字幕全自定义**
  - 字体、字号、颜色、边框、阴影、对齐方式等全样式参数可调
  - 支持字幕位置自定义，可分别设置普通字幕与唱歌字幕的显示坐标

## 环境依赖
```bash
# 核心依赖
pip install torch opencv-python

# ASR语音识别依赖
pip install openai-whisper ffmpeg-python

# Transformers后端翻译依赖（Qwen3.5）
pip install transformers accelerate sentencepiece

#如需使用ollama，请输入
ollama serve

# 其他工具依赖
pip install numpy tqdm
```
> 提示：请确保本地已安装FFmpeg并配置环境变量，Whisper与视频处理强依赖FFmpeg。

## 使用方法
### 基础命令（默认启用OCR+ASR，生成原生字幕）
```bash
python main.py --video_path "你的视频文件.mp4"
```

### 常用场景示例
1. **仅提取硬字幕（OCR模式）**
   ```bash
   python main.py --video_path "video.mp4" --disable_asr
   ```

2. **仅提取语音字幕（ASR模式）**
   ```bash
   python main.py --video_path "video.mp4" --disable_ocr
   ```

3. **启用AED音频事件检测，区分语音/唱歌场景**
   ```bash
   python main.py --video_path "video.mp4" --enable_aed
   ```

4. **使用Qwen3.5翻译（Transformers后端，中译日示例）**
   ```bash
   python main.py --video_path "video.mp4" --tr_choice "transformers" --tr_model "Qwen/Qwen3.5-4B" --tr_language "日语" --tr_prompt "请保持口语化，符合影视字幕表达习惯"
   ```

5. **使用hunyuan-mt-1.5翻译（Ollama后端，中译英示例）**
   ```bash
   python main.py --video_path "video.mp4" --tr_choice "ollama" --tr_model "hunyuan-mt-1.5" --tr_language "英语"
   ```

6. **自定义字幕样式（修改字体、字号、颜色）**
   ```bash
   python main.py --video_path "video.mp4" --font "微软雅黑" --font_size 64 --PrimaryColour "&H00FFFF00" --Outline 3
   ```

7. **单句翻译模式**
   ```bash
   python main.py --video_path "video.mp4" --single_tr --tr_choice "transformers" --tr_model "Qwen/Qwen3.5-4B-Instruct" --tr_language "中文"
   ```

## 完整参数说明
### 基础配置参数
| 参数名 | 默认值 | 功能说明 |
|--------|--------|----------|
| `--video_path` | `0.mp4` | 输入视频文件的路径，支持绝大多数主流视频格式 |
| `--device` | `cuda` | 推理运行设备，可选`cuda`（GPU）/`cpu`，自动检测CUDA可用性 |

### 模型路径配置
| 参数名 | 默认值 | 功能说明 |
|--------|--------|----------|
| `--yolo-model` | `model.pt` | YOLO文本区域检测模型的本地路径 |
| `--glm-model` | `GLM-OCR` | GLM-OCR文字识别模型的本地路径/仓库名 |
| `--asr_model` | `large-v3` | Whisper ASR模型尺寸，可选`tiny/base/small/medium/large/large-v2/large-v3/turbo` |
| `--aed_model` | `FireRedVAD/AED` | 音频事件检测模型的路径/仓库名 |
| `--tr_model` | `""` | 翻译模型路径/仓库名<br>- Transformers后端推荐：`Qwen/Qwen3.5-4B-Instruct`<br>- Ollama后端推荐：`hunyuan-mt-1.5` |

### OCR核心参数
| 参数名 | 默认值 | 功能说明 |
|--------|--------|----------|
| `--frame-skip` | `3` | OCR跳帧间隔，数值越大处理越快，过小会增加重复识别 |
| `--min-frames` | `3` | 字幕最小有效帧数，过滤一闪而过的无效文本 |
| `--pos-threshold` | `200` | 同位置字幕聚合阈值，控制相同位置字幕的合并逻辑 |
| `--segment-duration` | `120` | OCR分段处理时长（秒），大视频分段处理降低内存占用 |
| `--max-workers` | `2` | OCR帧处理线程数，根据CPU核心数调整 |
| `--yolo_conf` | `0.75` | YOLO检测置信度阈值，数值越高过滤越严格 |
| `--glm_max_batch_size` | `32` | GLM-OCR批量推理最大尺寸，显存不足请调小 |
| `--glm_max_new_tokens_per_region` | `128` | 单文本区域OCR最大输出token数 |
| `--text_region_min_area` | `50` | 文本区域最小面积，过滤过小的无效区域 |
| `--use_mixed_precision` | 关闭 | 启用混合精度推理，降低显存占用 |

### ASR与AED参数
| 参数名 | 默认值 | 功能说明 |
|--------|--------|----------|
| `--asr_lg` | `auto` | ASR识别语言，如`zh`/`ja`/`en`，auto为自动检测 |
| `--asr_prompt` | `""` | ASR初始提示词，用于优化特定场景/专有名词识别 |
| `--chunk_max_frame` | `30000` | AED音频块最大帧数 |
| `--smooth_window_size` | `5` | AED滑窗平滑窗口大小 |
| `--min_event_frame` | `20` | AED最小有效事件帧数 |
| `--max_event_frame` | `2000` | AED最大事件帧数 |
| `--min_silence_frame` | `20` | AED最小静音帧数 |
| `--merge_silence_frame` | `0` | AED相邻静音片段合并帧数 |
| `--extend_speech_frame` | `0` | AED语音片段前后扩展帧数 |
| `--speech_threshold` | `0.4` | 语音检测阈值 |
| `--singing_threshold` | `0.5` | 唱歌检测阈值 |
| `--music_threshold` | `0.5` | 背景音乐检测阈值 |
| `--subx` / `--suby` | `None` | 普通字幕固定显示坐标(x,y) |
| `--singx` / `--singy` | `None` | 唱歌片段字幕固定显示坐标(x,y) |

### 翻译功能参数
| 参数名 | 默认值 | 功能说明 |
|--------|--------|----------|
| `--tr_choice` | `transformers` | 翻译后端选择<br>- `transformers`：适配Qwen3.5系列模型<br>- `ollama`：适配hunyuan-mt-1.5本地模型 |
| `--tr_language` | `中文` | 翻译目标语言 |
| `--tr_prompt` | `""` | 翻译模型初始提示词，控制翻译风格与规则（Qwen3.5更适用） |
| `--tr_content` | `""` | 翻译背景知识，补充剧情/专有名词等上下文信息 |
| `--single_tr` | 关闭 | 启用单句翻译模式，逐句处理字幕避免上下文串扰 |

### 功能开关参数
| 参数名 | 功能说明 |
|--------|----------|
| `--disable_ocr` | 禁用硬字幕OCR提取功能 |
| `--disable_asr` | 禁用语音ASR识别功能 |
| `--disable_trans` | 禁用字幕翻译功能 |
| `--disable_aed` | 禁用音频事件检测功能 |

### ASS字幕样式参数
| 参数名 | 默认值 | 功能说明 |
|--------|--------|----------|
| `--font` | `方正准圆_GBK` | 字幕字体名称，需为系统已安装字体 |
| `--font_size` | `72` | 字幕字体大小 |
| `--PrimaryColour` | `&H00FFFFFF` | 字幕主色，格式为&H00BBGGRR |
| `--SecondaryColour` | `&H000000FF` | 字幕次色（卡拉OK变色用） |
| `--OutlineColour` | `&H00000000` | 字幕外框颜色 |
| `--BackColour` | `&H80000000` | 字幕背景阴影颜色 |
| `--Bold` | `0` | 字体加粗，0关闭/1开启 |
| `--Italic` | `0` | 字体斜体，0关闭/1开启 |
| `--Underline` | `0` | 字体下划线，0关闭/1开启 |
| `--StrikeOut` | `0` | 字体删除线，0关闭/1开启 |
| `--Outline` | `2` | 字幕外框宽度 |
| `--Shadow` | `0` | 字幕阴影深度 |
| `--BorderStyle` | `1` | 字幕边框样式，1为外边框+阴影，3为不透明背景 |
| `--ScaleX` / `--ScaleY` | `100` | 字体横向/纵向缩放百分比 |
| `--Spacing` | `0` | 字符间距 |
| `--Angle` | `0` | 字体旋转角度 |
| `--Alignment` | `2` | 字幕对齐方式，1=下左/2=下中/3=下右/5=中左/6=中中/7=中右/9=上左/10=上中/11=上右 |
| `--MarginL` / `--MarginR` | `10` | 字幕左/右外边距 |
| `--MarginV` | `10` | 字幕垂直外边距 |
| `--Encoding` | `0` | 字幕编码，0为ANSI，1为默认，134为GB2312，128为SHIFT-JIS |

## 输出说明
- 程序运行完成后，会在视频同级目录生成**与视频同名的.ass字幕文件**
- 控制台会实时输出处理进度、各环节提取字幕数量、总耗时等统计信息
- 若未提取到有效字幕，会给出对应提示，不会生成空字幕文件

## 注意事项
1. **参数校验规则**：不可同时禁用OCR和ASR功能，至少保留一项识别能力；AED音频事件检测依赖ASR功能，不可单独启用。
2. **显存优化**：若出现CUDA显存不足，可减小`--glm_max_batch_size`、增大`--frame-skip`，或切换至`--device cpu`运行。
3. **翻译模型使用**：
   请自行先下载好后再使用该功能
4. **字幕样式兼容性**：ASS字幕的字体需确保播放器所在系统已安装，否则会 fallback 到默认字体，导致显示效果不符预期。
