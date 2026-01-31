# ComfyUI-ClearerVoice-Studio

ComfyUI plugin for audio processing using ClearVoice models, providing voice separation, denoising, enhancement, and video speaker extraction functionality.

## Installation

1. Clone this repository into your ComfyUI `custom_nodes` directory.
2. **No additional dependencies** - The plugin includes a local copy of the ClearVoice library.
3. Ensure you have FFmpeg installed for video processing.
4. Download the required models to the `models/ASR/ClearerVoice-Studio` directory.

## Local ClearVoice Library

This plugin includes a local copy of the ClearVoice library to ensure compatibility and avoid dependency conflicts. The local library is automatically used instead of any system-installed version.

### Key Advantages
- **Self-contained**: No separate ClearVoice installation required
- **Version control**: Uses tested and compatible versions
- **Dependency isolation**: Avoids conflicts with other packages
- **Faster loading**: Pre-configured for optimal performance

### Library Structure
```
ComfyUI-ClearerVoice-Studio/
├── clearvoice/           # Local ClearVoice library
│   ├── models/           # Model definitions
│   ├── utils/            # Utility functions
│   └── networks.py       # Network wrappers
├── nodes.py              # ComfyUI nodes
└── __init__.py           # Plugin initialization
```

## Models

The plugin requires the following models downloaded to the `models/ASR/ClearerVoice-Studio` directory:

- FRCRN_SE_16K
- MossFormer2_SE_48K
- MossFormerGAN_SE_16K
- MossFormer2_SS_16K
- AV_MossFormer2_TSE_16K
- MossFormer2_SR_48K

### Model Download

You can download all required models from ModelScope:

**Official Model Repository**: https://modelscope.cn/models/iic/ClearerVoice-Studio

#### Download Instructions

1. Visit the ModelScope repository: https://modelscope.cn/models/iic/ClearerVoice-Studio
2. Or directly git clone https://www.modelscope.cn/iic/ClearerVoice-Studio.git to the specified directory
3. Model placement directory:
   ```
   ComfyUI/models/ASR/ClearerVoice-Studio/
   ```

   ### Model Directory Structure
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

#### Model Description

- **FRCRN_SE_16K**: Speech denoising model for 16KHz audio
- **MossFormer2_SE_48K**: Speech denoising model for 48KHz audio
- **MossFormerGAN_SE_16K**: GAN-based speech denoising model for 16KHz audio
- **MossFormer2_SS_16K**: Speech separation model for 16KHz audio (supports 2 speakers)
- **AV_MossFormer2_TSE_16K**: Audio-visual speaker extraction model for 16KHz video
- **MossFormer2_SR_48K**: Speech super-resolution model for 48KHz audio

#### Model Compatibility

Each model is optimized for specific sample rates:
- 16KHz models: FRCRN_SE_16K, MossFormerGAN_SE_16K, MossFormer2_SS_16K, AV_MossFormer2_TSE_16K
- 48KHz models: MossFormer2_SE_48K, MossFormer2_SR_48K

The plugin automatically resamples audio to match the model's optimal sample rate.

## Nodes

### 1. ClearVoice Speech Separation

**Function**: Separates mixed speech into individual speaker audios.

**Inputs**:
- `media`: Audio or video input (*)
- `model_name`: Model for separation (MossFormer2_SS_16K)

**Outputs**:
- `audio_output_1`: First separated speaker audio (AUDIO)
- `audio_output_2`: Second separated speaker audio (AUDIO, empty if not available)
- `audio_output_3`: Third separated speaker audio (AUDIO, empty if not available)
- `audio_output_4`: Fourth separated speaker audio (AUDIO, empty if not available)
- `audio_output_5`: Fifth separated speaker audio (AUDIO, empty if not available)

**Usage**: Connect audio or video input to the node, select model, the node will separate speech into individual speakers. Output paths can be connected to other audio processing nodes or saved to disk.

### 2. ClearVoice Speech Denoising

**Function**: Removes noise from speech audio.

**Inputs**:
- `media`: Audio or video input (*)
- `model_name`: Model for denoising (FRCRN_SE_16K, MossFormer2_SE_48K, MossFormerGAN_SE_16K)

**Outputs**:
- `audio_output`: Denoised audio (AUDIO)
- `video_output`: Video with replaced audio (VIDEO, if input is video)

**Usage**: Connect audio or video input to the node, select model, the node will denoise the speech. Each model automatically uses its optimal sample rate (FRCRN_SE_16K and MossFormerGAN_SE_16K use 16KHz, MossFormer2_SE_48K uses 48KHz). Output paths can be connected to other audio processing nodes or saved to disk.

### 3. ClearVoice Video Speaker Extraction

**Function**: Extracts individual speakers from video and generates separate videos for each speaker.

**Inputs**:
- `video_input`: Video input (*)
- `enable_crop` (optional): Whether to output cropped face videos (224x224). If False, outputs original video with face bounding boxes.

**Outputs**:
- `video_output_1`: First speaker's video (VIDEO)
- `video_output_2`: Second speaker's video (VIDEO, empty if not available)
- `video_output_3`: Third speaker's video (VIDEO, empty if not available)
- `video_output_4`: Fourth speaker's video (VIDEO, empty if not available)
- `video_output_5`: Fifth speaker's video (VIDEO, empty if not available)

**Usage**: Connect video input to the node, the node will detect speakers, track their faces, and generate separate videos with corresponding audio for each speaker. Output paths can be connected to other video processing nodes or saved to disk.

## Workflow Examples

### Basic Audio Denoising
1. Use Load Audio node to load audio file
2. Connect audio path to ClearVoice Speech Denoise node
3. Select appropriate model
4. Connect output to Save Audio node

### Video Speaker Extraction
1. Use Load Video node to load video file
2. Connect video path to ClearVoice Video Speaker Extraction node
3. Set enable_crop parameter to select output type
4. Connect output videos to Save Video node

### Speech Separation and Enhancement
1. Use Load Audio node to load audio file
2. Connect audio path to ClearVoice Speech Separation node
3. Connect separated audio outputs to ClearVoice Speech Denoise node
4. Connect denoised audio outputs to Save Audio node

## GPU Acceleration

The plugin automatically uses available GPU acceleration:
- **CUDA**: For NVIDIA GPUs
- **ROCm**: For AMD GPUs (if CUDA is not available)

You can verify GPU usage in the console output:
```
ClearVoiceBaseNode: ✓ Model is on device: cuda:0
```

## Temporary Files

Temporary files are stored in:
```
ComfyUI-ClearerVoice-Studio/temp/
```

These files are automatically cleaned up every time ComfyUI starts (configurable in nodes.py).

## Troubleshooting

### Model Not Found
Ensure all required models are downloaded to the `models/ASR/ClearerVoice-Studio` directory.

### FFmpeg Error
Ensure FFmpeg is installed and accessible in the system PATH.

### ClearVoice Import Error
The plugin includes a local copy of ClearVoice, so no separate installation is needed.
If you encounter import errors, check if the `clearvoice` directory exists in the plugin folder.

### CUDA Out of Memory
Reduce input video resolution or use smaller models.

### Face Detector Error
The face detector model is included in the local library.
Check if `sfd_face.pth` exists in the `clearvoice/models/av_mossformer2_tse/faceDetector/s3fd` directory.

## Supported Formats

### Audio Input
- WAV, MP3, OGG, FLAC, AAC

### Video Input
- MP4, AVI, MOV, MKV, WMV

## License

This plugin is released under the MIT License.

## Acknowledgements

- ClearVoice team for audio processing models
- ComfyUI team for the excellent UI framework
- PyTorch team for the deep learning framework
- librosa team for the audio analysis library