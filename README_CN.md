👉 **[English](https://github.com/freeyaers/ComfyUI-ClearerVoice-Studio/blob/main/README.md)**

# ComfyUI-ClearerVoice-Studio

ComfyUI 插件，使用 ClearVoice 模型进行音频处理，提供语音分离、降噪、增强和视频说话人提取功能。

## 安装

1. 将此仓库克隆到您的 ComfyUI `custom_nodes` 目录中。
2. **无需额外依赖项** - 插件包含 ClearVoice 库的本地副本。
3. 确保您已安装 FFmpeg 用于视频处理。
4. 下载所需的模型到 `models/ASR/ClearerVoice-Studio` 目录。

## 本地 ClearVoice 库

此插件包含 ClearVoice 库的本地副本，以确保兼容性并避免依赖冲突。本地库会自动使用，而不是任何系统安装的版本。

### 主要优势
- **自包含**：无需单独安装 ClearVoice
- **版本控制**：使用经过测试和兼容的版本
- **依赖隔离**：避免与其他包的冲突
- **更快加载**：预配置以获得最佳性能

### 库结构
```
ComfyUI-ClearerVoice-Studio/
├── clearvoice/           # 本地 ClearVoice 库
│   ├── models/           # 模型定义
│   ├── utils/            # 实用函数
│   └── networks.py       # 网络包装器
├── nodes.py              # ComfyUI 节点
└── __init__.py           # 插件初始化
```

## 模型

插件需要以下模型下载到 `models/ASR/ClearerVoice-Studio` 目录：

- FRCRN_SE_16K
- MossFormer2_SE_48K
- MossFormerGAN_SE_16K
- MossFormer2_SS_16K
- AV_MossFormer2_TSE_16K
- MossFormer2_SR_48K

### 模型下载

您可以从 ModelScope 下载所有必需的模型：

**官方模型仓库**：https://modelscope.cn/models/iic/ClearerVoice-Studio

#### 下载说明

1. 访问 ModelScope 仓库：https://modelscope.cn/models/iic/ClearerVoice-Studio
2. 或者直接 git clone https://www.modelscope.cn/iic/ClearerVoice-Studio.git 到指定目录
3. 模型放置目录：
   ```
   ComfyUI/models/ASR/ClearerVoice-Studio/
   ```

   ### 模型目录结构
   ```
   ClearerVoice-Studio/
   ├── AV_MossFormer2_TSE_16K/
   │   ├── last_best_checkpoint
   │   └── last_best_checkpoint.pt
   ├── FRCRN_SE_16K/
   │   ├── last_best_checkpoint
   │   └── last_best_checkpoint.pt
   ├── MossFormer2_SE_48K/
   │   ├── last_best_checkpoint
   │   └── last_best_checkpoint.pt
   ├── MossFormer2_SR_48K/
   │   ├── last_best_checkpoint
   │   ├── last_best_checkpoint_g.pt
   │   └── last_best_checkpoint_m.pt
   ├── MossFormer2_SS_16K/
   │   ├── last_best_checkpoint
   │   └── last_best_checkpoint.pt
   ├── MossFormerGAN_SE_16K/
   │   ├── last_best_checkpoint
   │   ├── last_best_checkpoint.disc.pt
   │   └── last_best_checkpoint.pt
   ├── .gitattributes
   ├── README.md
   ├── configuration.json
   └── sfd_face.pth
   ```

#### 模型说明

- **FRCRN_SE_16K**：用于 16KHz 音频的语音降噪模型
- **MossFormer2_SE_48K**：用于 48KHz 音频的语音降噪模型
- **MossFormerGAN_SE_16K**：使用 GAN 的语音降噪模型，用于 16KHz 音频
- **MossFormer2_SS_16K**：用于 16KHz 音频的语音分离模型（支持 2 个说话人）
- **AV_MossFormer2_TSE_16K**：用于 16KHz 视频的音频-视觉说话人提取模型
- **MossFormer2_SR_48K**：用于 48KHz 音频的语音超分辨率模型

#### 模型兼容性

每个模型都针对特定采样率进行了优化：
- 16KHz 模型：FRCRN_SE_16K, MossFormerGAN_SE_16K, MossFormer2_SS_16K, AV_MossFormer2_TSE_16K
- 48KHz 模型：MossFormer2_SE_48K, MossFormer2_SR_48K

插件会自动重采样音频以匹配模型的最佳采样率。

## 节点

### 1. ClearVoice 语音分离

**功能**：将混合语音分离为单独的说话人音频。

**输入**：
- `media`：音频或视频输入（*）
- `model_name`：用于分离的模型（MossFormer2_SS_16K）

**输出**：
- `audio_output_1`：第一个分离的说话人音频（AUDIO）
- `audio_output_2`：第二个分离的说话人音频（AUDIO，不可用则为空）
- `audio_output_3`：第三个分离的说话人音频（AUDIO，不可用则为空）
- `audio_output_4`：第四个分离的说话人音频（AUDIO，不可用则为空）
- `audio_output_5`：第五个分离的说话人音频（AUDIO，不可用则为空）

**使用方法**：将音频或视频输入连接到节点，选择模型，节点将把语音分离为单独的说话人。输出路径可以连接到其他音频处理节点或保存到磁盘。

### 2. ClearVoice 语音降噪

**功能**：从语音音频中移除噪声。

**输入**：
- `media`：音频或视频输入（*）
- `model_name`：用于降噪的模型（FRCRN_SE_16K, MossFormer2_SE_48K, MossFormerGAN_SE_16K）

**输出**：
- `audio_output`：降噪后的音频（AUDIO）
- `video_output`：替换音频的视频（VIDEO，如果输入是视频）

**使用方法**：将音频或视频输入连接到节点，选择模型，节点将对语音进行降噪。每个模型会自动使用其最佳采样率（FRCRN_SE_16K 和 MossFormerGAN_SE_16K 使用 16KHz，MossFormer2_SE_48K 使用 48KHz）。输出路径可以连接到其他音频处理节点或保存到磁盘。

### 3. ClearVoice 视频说话人提取

**功能**：从视频中提取单独的说话人并为每个说话人生成单独的视频。

**输入**：
- `video_input`：视频输入（*）
- `enable_crop`（可选）：是否输出裁剪的面部视频（224x224）。如果为 False，则输出带有面部边界框的原始视频。

**输出**：
- `video_output_1`：第一个说话人的视频（VIDEO）
- `video_output_2`：第二个说话人的视频（VIDEO，不可用则为空）
- `video_output_3`：第三个说话人的视频（VIDEO，不可用则为空）
- `video_output_4`：第四个说话人的视频（VIDEO，不可用则为空）
- `video_output_5`：第五个说话人的视频（VIDEO，不可用则为空）

**使用方法**：将视频输入连接到节点，节点将检测说话人，跟踪他们的面部，并为每个说话人生成带有相应音频的单独视频。输出路径可以连接到其他视频处理节点或保存到磁盘。

## 工作流示例

![Image text](https://github.com/freeyaers/ComfyUI-ClearerVoice-Studio/blob/main/workflows/img1.png)
![Image text](https://github.com/freeyaers/ComfyUI-ClearerVoice-Studio/blob/main/workflows/img2.png)
![Image text](https://github.com/freeyaers/ComfyUI-ClearerVoice-Studio/blob/main/workflows/img3.png)
![Image text](https://github.com/freeyaers/ComfyUI-ClearerVoice-Studio/blob/main/workflows/img4.png)


### 基本音频降噪
1. 使用 Load Audio 节点加载音频文件
2. 将音频路径连接到 ClearVoice Speech Denoise 节点
3. 选择适当的模型
4. 将输出连接到 Save Audio 节点

### 视频说话人提取
1. 使用 Load Video 节点加载视频文件
2. 将视频路径连接到 ClearVoice Video Speaker Extraction 节点
3. 设置 enable_crop 参数以选择输出类型
4. 将输出视频连接到 Save Video 节点

### 语音分离和增强
1. 使用 Load Audio 节点加载音频文件
2. 将音频路径连接到 ClearVoice Speech Separation 节点
3. 将分离的音频输出连接到 ClearVoice Speech Denoise 节点
4. 将降噪后的音频输出连接到 Save Audio 节点

## GPU 加速

插件会自动使用可用的 GPU 加速：
- **CUDA**：适用于 NVIDIA GPU
- **ROCm**：适用于 AMD GPU（如果 CUDA 不可用）

您可以在控制台输出中验证 GPU 使用情况：
```
ClearVoiceBaseNode: ✓ Model is on device: cuda:0
```

## 临时文件

临时文件存储在：
```
ComfyUI-ClearerVoice-Studio/temp/
```

这些文件会在每次启动Comfyui时自动清理（可在 nodes.py 中配置）。

## 故障排除

### 模型未找到
确保所有必需的模型都下载到 `models/ASR/ClearerVoice-Studio` 目录。

### FFmpeg 错误
确保 FFmpeg 已安装并在系统 PATH 中可访问。

### ClearVoice 导入错误
插件包含 ClearVoice 的本地副本，因此不需要单独安装。
如果遇到导入错误，请检查插件文件夹中是否存在 `clearvoice` 目录。

### CUDA 内存不足
降低输入视频分辨率或使用较小的模型。

### 面部检测器错误
面部检测器模型包含在本地库中。
检查 `clearvoice/models/av_mossformer2_tse/faceDetector/s3fd` 目录中是否存在 `sfd_face.pth`。

## 支持的格式

### 音频输入
- WAV, MP3, OGG, FLAC, AAC

### 视频输入
- MP4, AVI, MOV, MKV, WMV

## 许可证

此插件以 MIT 许可证发布。

## 致谢

- ClearVoice 团队提供音频处理模型
- ComfyUI 团队提供出色的 UI 框架
- PyTorch 团队提供深度学习框架
- librosa 团队提供音频分析库
