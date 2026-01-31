"""
ClearerVoice-Studio Node for ComfyUI
Author: ClearerVoice Team
Description: A node for audio processing using ClearVoice models
"""

import os
import sys
import folder_paths

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add local clearvoice directory to Python path to prioritize local import
clearvoice_path = os.path.join(current_dir, "clearvoice")
sys.path.insert(0, clearvoice_path)
print(f"ClearVoice-Studio: Added local clearvoice path: {clearvoice_path}")

# Add the custom nodes directory to the path for imports
sys.path.append(current_dir)

# Set up temp directory
temp_dir = os.path.join(current_dir, "temp")
print(f"ClearerVoice-Studio temp directory: {temp_dir}")

# Create temp directory if it doesn't exist
os.makedirs(temp_dir, exist_ok=True)

# Clear temp directory on startup
print("Clearing temp directory...")
try:
    for file in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                import shutil
                shutil.rmtree(file_path)
            print(f"Cleared: {file}")
        except Exception as e:
            print(f"Error clearing file {file}: {e}")
except Exception as e:
    print(f"Error clearing temp directory: {e}")

# Set up model paths
# ClearVoice default checkpoint directory (OUTDATED - now using ComfyUI's model path)
# checkpoints_dir = os.path.join(current_dir, "checkpoints")
# Custom model directory
model_dir = os.path.join(folder_paths.models_dir, "ASR", "ClearerVoice-Studio")

# Ensure model directory exists
try:
    os.makedirs(model_dir, exist_ok=True)
    print(f"Model directory: {model_dir}")
except Exception as e:
    print(f"Warning: Failed to create model directory {model_dir}: {e}")
    print("ClearerVoice-Studio will still load, but models may not be available if directory doesn't exist.")

# Create checkpoints directory if it doesn't exist (OUTDATED - now using ComfyUI's model path)
# if not os.path.exists(checkpoints_dir):
#     os.makedirs(checkpoints_dir, exist_ok=True)

# List of required models
required_models = [
    "FRCRN_SE_16K",
    "MossFormer2_SE_48K",
    "MossFormerGAN_SE_16K",
    "MossFormer2_SS_16K",
    "AV_MossFormer2_TSE_16K",
    "MossFormer2_SR_48K"
]

# Check if all models exist locally
print("Checking ClearerVoice models...")
missing_models = []
for model_name in required_models:
    model_path = os.path.join(model_dir, model_name)
    if os.path.exists(model_path) and os.listdir(model_path):
        print(f"✓ {model_name} exists locally")
    else:
        print(f"✗ {model_name} missing locally")
        missing_models.append(model_name)

if missing_models:
    print(f"\nMissing models: {missing_models}")
    print(f"Please download them to: {model_dir}")
else:
    print("\nAll models exist locally!")

# Create symlinks for each model (OUTDATED - now using ComfyUI's model path)
# print("\nSetting up model symlinks...")
# for model_name in required_models:
#     src = os.path.join(model_dir, model_name)
#     dst = os.path.join(checkpoints_dir, model_name)
#     
#     # If source exists and destination doesn't, create symlink
#     if os.path.exists(src) and not os.path.exists(dst):
#         if os.name == 'nt':  # Windows
#             try:
#                 import subprocess
#                 # Use mklink to create directory junction
#                 subprocess.run(['mklink', '/J', dst, src], shell=True, check=True)
#                 print(f"✓ Created symlink for {model_name}")
#             except Exception as e:
#                 print(f"✗ Failed to create symlink for {model_name}: {e}")
#                 # Fallback: create directory and copy files
#                 os.makedirs(dst, exist_ok=True)
#                 print(f"  Created directory fallback for {model_name}")
#         else:  # Unix-like
#             try:
#                 os.symlink(src, dst)
#                 print(f"✓ Created symlink for {model_name}")
#             except Exception as e:
#                 print(f"✗ Failed to create symlink for {model_name}: {e}")
#     else:
#         print(f"✓ {model_name} symlink already exists or source missing")

# Import nodes from nodes.py
from .nodes import (
    ClearVoiceSpeechSeparationNode,
    ClearVoiceSpeechDenoiseNode,
    ClearVoiceVideoSpeakerExtractionNode
)

# Register the nodes
NODE_CLASS_MAPPINGS = {
    "ClearVoiceSpeechSeparation": ClearVoiceSpeechSeparationNode,
    "ClearVoiceSpeechDenoise": ClearVoiceSpeechDenoiseNode,
    "ClearVoiceVideoSpeakerExtraction": ClearVoiceVideoSpeakerExtractionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ClearVoiceSpeechSeparation": "ClearVoice Speech Separation",
    "ClearVoiceSpeechDenoise": "ClearVoice Speech Denoise",
    "ClearVoiceVideoSpeakerExtraction": "ClearVoice Video Speaker Extraction"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']