"""
ClearerVoice-Studio Nodes for ComfyUI
Author: ClearerVoice Team
Description: Nodes for audio processing using ClearVoice models
"""

import os
import sys
import tempfile
import subprocess
import numpy as np
import torch
import torchaudio
import uuid
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import folder_paths

# Get current directory and temp directory
current_dir = os.path.dirname(os.path.abspath(__file__))
temp_dir = os.path.join(current_dir, "temp")
print("[1] ClearVoice nodes temp directory: {}".format(temp_dir))

# Create temp directory if it doesn't exist
os.makedirs(temp_dir, exist_ok=True)

# Add local clearvoice directory to Python path to prioritize local import
clearvoice_path = os.path.join(current_dir, "clearvoice")
sys.path.insert(0, clearvoice_path)
print("[2] Added local clearvoice path: {}".format(clearvoice_path))

# Import ClearVoice at module level - this is crucial for avoiding FakeModule errors
print("[3] Initializing ClearVoice import...")
try:
    from clearvoice import ClearVoice
    print("[4] ClearVoice imported successfully at module level")
    HAS_CLEARVOICE = True
except ImportError as e:
    print("[5] Error importing ClearVoice: {}".format(e))
    print("[6] Please install ClearVoice using: pip install clearvoice")
    HAS_CLEARVOICE = False

# Pre-import librosa and numba to avoid FakeModule errors
# This is crucial for models that use librosa.resample
print("[7] Pre-importing librosa and numba to avoid FakeModule errors...")
try:
    import librosa
    import numba
    # Force full import of librosa to avoid lazy loading issues
    import librosa.core
    import librosa.core.audio
    import librosa.core.convert
    import librosa.core.notation
    # Force numba jit to be loaded
    from numba import jit
    print("[8] librosa and numba imported successfully with forced loading")
except ImportError as e:
    print("[9] Warning: Failed to import librosa or numba: {}".format(e))
    print("[10] Some models may not work correctly")

class ClearVoiceBaseNode:
    """Base class for ClearVoice nodes"""
    
    # Control whether to clean up temporary resources after task completion
    # Default: False (disabled)
    CLEANUP_TEMP_RESOURCES = False
    
    def __init__(self):
        pass
    
    def process_media_input(self, media):
        """Process media input based on type (AUDIO or VIDEO)"""
        import os
        import tempfile
        
        input_path = None
        temp_audio_path = None
        
        # Process media input based on type
        if isinstance(media, dict) and 'waveform' in media and 'sample_rate' in media:
            # Process AUDIO type input
            print("[11] ClearVoiceBaseNode: Processing audio input")
            input_path = self.process_audio_input(media)
            print("[12] ClearVoiceBaseNode: input_path={}".format(input_path))
        else:
            # Process VIDEO type input
            print("[13] ClearVoiceBaseNode: Processing video input: {}".format(type(media)))
            
            # Extract video path from VIDEO object
            video_path = None
            
            # Print all attributes of the video object for debugging
            print("[14] ClearVoiceBaseNode: VIDEO object attributes: {}".format(dir(media)))
            
            # Try different ways to get video path
            if hasattr(media, 'video_path'):
                video_path = media.video_path
                print("[15] ClearVoiceBaseNode: Got video_path from media.video_path: {}".format(video_path))
            elif hasattr(media, 'path'):
                video_path = media.path
                print("[16] ClearVoiceBaseNode: Got video_path from media.path: {}".format(video_path))
            elif hasattr(media, 'file_path'):
                video_path = media.file_path
                print("[17] ClearVoiceBaseNode: Got video_path from media.file_path: {}".format(video_path))
            elif hasattr(media, 'filename'):
                video_path = media.filename
                print("[18] ClearVoiceBaseNode: Got video_path from media.filename: {}".format(video_path))
            elif hasattr(media, 'video_file_path'):
                video_path = media.video_file_path
                print("[19] ClearVoiceBaseNode: Got video_path from media.video_file_path: {}".format(video_path))
            elif hasattr(media, 'input_path'):
                video_path = media.input_path
                print("[20] ClearVoiceBaseNode: Got video_path from media.input_path: {}".format(video_path))
            elif hasattr(media, 'filepath'):
                video_path = media.filepath
                print("[21] ClearVoiceBaseNode: Got video_path from media.filepath: {}".format(video_path))
            elif hasattr(media, 'name'):
                video_path = media.name
                print("[22] ClearVoiceBaseNode: Got video_path from media.name: {}".format(video_path))
            elif isinstance(media, str):
                video_path = media
                print("[23] ClearVoiceBaseNode: Got video_path from string input: {}".format(video_path))
            
            # Try to get __dict__ to see all attributes
            if not video_path and hasattr(media, '__dict__'):
                print("[24] ClearVoiceBaseNode: VIDEO object __dict__: {}".format(media.__dict__))
                # Check if any attribute looks like a path
                for key, value in media.__dict__.items():
                    if isinstance(value, str) and ('.mp4' in value.lower() or '.avi' in value.lower() or '.mov' in value.lower()):
                        video_path = value
                        print("[25] ClearVoiceBaseNode: Got video_path from __dict__[{}]: {}".format(key, video_path))
                        break
            
            if video_path:
                print("[26] ClearVoiceBaseNode: Extracted video path: {}".format(video_path))
                temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir=temp_dir)
                temp_audio_path = temp_audio.name
                temp_audio.close()
                
                if self.extract_audio_from_video(video_path, temp_audio_path):
                    input_path = temp_audio_path
                    print("[27] ClearVoiceBaseNode: Extracted audio to {}".format(input_path))
            else:
                print("[28] ClearVoiceBaseNode: Failed to extract video path from VIDEO object")
        
        return input_path, temp_audio_path
    
    def initialize_model(self, task, model_name, model_dir):
        """Initialize ClearVoice model based on task and model name"""
        import os
        
        # Set the checkpoint directory for this specific model
        model_specific_dir = os.path.join(model_dir, model_name)
        print(f"ClearVoiceBaseNode: Model-specific directory: {model_specific_dir}")
        
        # Create model-specific directory if it doesn't exist
        if not os.path.exists(model_specific_dir):
            os.makedirs(model_specific_dir)
            print(f"ClearVoiceBaseNode: Created model-specific directory")
        
        # Use low-level API for better control
        from clearvoice.network_wrapper import network_wrapper
        
        # Create network wrapper
        wrapper = network_wrapper()
        wrapper.model_name = model_name
        
        # Load args based on task
        if task == 'speech_enhancement':
            wrapper.load_args_se()
        elif task == 'speech_separation':
            wrapper.load_args_ss()
        elif task == 'target_speaker_extraction':
            wrapper.load_args_tse()
        
        # Manually set the checkpoint directory
        wrapper.args.checkpoint_dir = model_specific_dir
        print(f"ClearVoiceBaseNode: Set checkpoint_dir to: {model_specific_dir}")
        
        # Set task and network
        wrapper.args.task = task
        wrapper.args.network = model_name
        
        # Enable GPU usage
        wrapper.args.use_cuda = 1
        print("[130] ClearVoiceBaseNode: Enabled GPU usage")
        
        # Initialize the network based on model name and task
        print("[131] ClearVoiceBaseNode: Initializing model...")
        
        # Import model classes based on task
        if task == 'speech_enhancement':
            from clearvoice.networks import CLS_FRCRN_SE_16K, CLS_MossFormer2_SE_48K, CLS_MossFormerGAN_SE_16K
            model_class_map = {
                'FRCRN_SE_16K': CLS_FRCRN_SE_16K,
                'MossFormer2_SE_48K': CLS_MossFormer2_SE_48K,
                'MossFormerGAN_SE_16K': CLS_MossFormerGAN_SE_16K
            }
        elif task == 'speech_separation':
            from clearvoice.networks import CLS_MossFormer2_SS_16K
            model_class_map = {
                'MossFormer2_SS_16K': CLS_MossFormer2_SS_16K
            }
        elif task == 'target_speaker_extraction':
            from clearvoice.networks import CLS_AV_MossFormer2_TSE_16K
            model_class_map = {
                'AV_MossFormer2_TSE_16K': CLS_AV_MossFormer2_TSE_16K
            }
        
        if model_name not in model_class_map:
            raise RuntimeError(f"Unsupported model: {model_name}")
        
        model_class = model_class_map[model_name]
        model = model_class(wrapper.args)
        
        # Check if model is on GPU
        try:
            param = next(model.model.parameters())
            print(f"ClearVoiceBaseNode: ✓ Model is on device: {param.device}")
        except Exception as e:
            print(f"ClearVoiceBaseNode: ⚠️  Error checking model device: {e}")
        
        print("[132] ClearVoiceBaseNode: ✓ ClearVoice initialized successfully with custom model directory")
        return model
    
    def cleanup_temp_resources(self, temp_resources):
        """Clean up temporary resources"""
        import os
        import shutil
        
        print("[133] ClearVoiceBaseNode: Cleaning up temporary resources...")
        
        # Clean up temp_audio_path
        if 'temp_audio_path' in temp_resources and temp_resources['temp_audio_path'] and os.path.exists(temp_resources['temp_audio_path']):
            try:
                os.unlink(temp_resources['temp_audio_path'])
                print(f"ClearVoiceBaseNode: Cleaned up temp_audio_path: {temp_resources['temp_audio_path']}")
            except Exception as e:
                print(f"ClearVoiceBaseNode: Error cleaning up temp_audio_path: {e}")
        
        # Clean up processed_input
        if 'processed_input' in temp_resources and temp_resources['processed_input'] and os.path.exists(temp_resources['processed_input']):
            try:
                os.unlink(temp_resources['processed_input'])
                print(f"ClearVoiceBaseNode: Cleaned up processed_input: {temp_resources['processed_input']}")
            except Exception as e:
                print(f"ClearVoiceBaseNode: Error cleaning up processed_input: {e}")
        
        # Clean up output_path
        if 'output_path' in temp_resources and temp_resources['output_path'] and os.path.exists(temp_resources['output_path']):
            try:
                os.unlink(temp_resources['output_path'])
                print(f"ClearVoiceBaseNode: Cleaned up output_path: {temp_resources['output_path']}")
            except Exception as e:
                print(f"ClearVoiceBaseNode: Error cleaning up output_path: {e}")
        
        # Clean up output_video_path (removed - video files are now saved to output directory)
        
        # Clean up output_files
        if 'output_files' in temp_resources:
            for file_path in temp_resources['output_files']:
                if os.path.exists(file_path):
                    try:
                        os.unlink(file_path)
                    except Exception as e:
                        print(f"ClearVoiceBaseNode: Error cleaning up output file: {e}")
        
        # Clean up temp_output_dir or output_dir
        if 'temp_output_dir' in temp_resources and temp_resources['temp_output_dir'] and os.path.exists(temp_resources['temp_output_dir']):
            try:
                shutil.rmtree(temp_resources['temp_output_dir'])
                print(f"ClearVoiceBaseNode: Cleaned up temp_output_dir: {temp_resources['temp_output_dir']}")
            except Exception as e:
                print(f"ClearVoiceBaseNode: Error cleaning up temp_output_dir: {e}")
        
        if 'output_dir' in temp_resources and temp_resources['output_dir'] and os.path.exists(temp_resources['output_dir']):
            try:
                shutil.rmtree(temp_resources['output_dir'])
                print(f"ClearVoiceBaseNode: Cleaned up output_dir: {temp_resources['output_dir']}")
            except Exception as e:
                print(f"ClearVoiceBaseNode: Error cleaning up output_dir: {e}")
        
        print("[134] ClearVoiceBaseNode: Temporary resources cleanup completed")
    
    def convert_to_audio_output(self, output_dir):
        """Convert processed audio to AUDIO type output"""
        import os
        
        # Find the actual output file created by ClearVoice
        output_path = None
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith('.wav'):
                    output_path = os.path.join(root, file)
                    print(f"ClearVoiceBaseNode: Found output file: {output_path}, size={os.path.getsize(output_path)} bytes")
                    break
            if output_path:
                break
        
        if not output_path:
            raise RuntimeError("No output file found in temporary directory")
        
        # Convert to AUDIO type
        print("[135] ClearVoiceBaseNode: Converting to AUDIO type")
        print(f"ClearVoiceBaseNode: Using output path: {output_path}")
        audio_output = self.audio_file_to_output(output_path)
        print(f"ClearVoiceBaseNode: audio_output={audio_output}")
        if audio_output:
            print(f"ClearVoiceBaseNode: audio_output waveform shape: {audio_output['waveform'].shape}")
            print(f"ClearVoiceBaseNode: audio_output sample rate: {audio_output['sample_rate']}")
        else:
            raise RuntimeError("Failed to create audio_output")
        
        return audio_output, output_path
    
    def extract_audio_from_video(self, video_path, output_audio_path):
        """Extract audio from video file using ffmpeg"""
        try:
            # Use ffmpeg to extract audio
            cmd = [
                "ffmpeg",
                "-i", str(video_path),
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # 16-bit PCM
                "-ar", "16000",  # 16kHz sample rate
                "-ac", "1",  # Mono
                "-y",  # Overwrite output
                str(output_audio_path)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except Exception as e:
            print(f"Error extracting audio from video: {e}")
            return False
    
    def ensure_wav_format(self, input_path):
        """Ensure input is in WAV format, convert if necessary"""
        input_path = Path(input_path)
        
        if input_path.suffix.lower() == '.wav':
            return str(input_path)
        
        # Create temporary WAV file
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir=temp_dir)
        temp_wav_path = temp_wav.name
        temp_wav.close()
        
        try:
            # Use ffmpeg to convert to WAV
            cmd = [
                "ffmpeg",
                "-i", str(input_path),
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                "-y",
                temp_wav_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            return temp_wav_path
        except Exception as e:
            print(f"Error converting to WAV: {e}")
            # Clean up temporary file
            if os.path.exists(temp_wav_path):
                os.unlink(temp_wav_path)
            return None
    
    def get_processed_audio_path(self, input_path, suffix):
        """Generate output audio path"""
        input_path = Path(input_path)
        output_dir = input_path.parent
        output_name = f"{input_path.stem}_{suffix}{input_path.suffix}"
        return str(output_dir / output_name)
    
    def process_audio_input(self, audio_input):
        """Process AUDIO type input and return a WAV file path"""
        if isinstance(audio_input, dict):
            # Handle dictionary format (waveform + sample_rate)
            audio_tensor = audio_input.get("waveform")
            sample_rate = audio_input.get("sample_rate")
            if audio_tensor is None or sample_rate is None:
                print("[136] Error: Invalid audio dictionary format, missing 'waveform' or 'sample_rate'")
                return None
        elif isinstance(audio_input, str):
            # Handle string path
            return audio_input
        elif isinstance(audio_input, tuple) or isinstance(audio_input, list):
            # Handle tuple/list format (tensor, sample_rate)
            if len(audio_input) >= 2:
                audio_tensor = audio_input[0]
                sample_rate = audio_input[1]
            else:
                print("[137] Error: Invalid audio format, expected (tensor, sample_rate)")
                return None
        else:
            print(f"[138] Error: Unsupported audio format: {type(audio_input)}")
            return None
        
        # Handle tensor input
        if isinstance(audio_input, dict) or (isinstance(audio_input, (tuple, list)) and len(audio_input) >= 2):
            # Create temporary WAV file
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir=temp_dir)
            temp_wav_path = temp_wav.name
            temp_wav.close()
            
            try:
                # Normalize audio tensor to 1D: [samples]
                print(f"ClearVoiceSpeechDenoise: Original audio tensor shape: {audio_tensor.shape}")
                
                # Handle different tensor dimensions
                if audio_tensor.dim() == 3:
                    # For shape [batch, channels, samples] like [1, 2, samples]
                    # Remove batch dimension: [1, 2, samples] -> [2, samples]
                    audio_tensor = audio_tensor.squeeze(0)
                    print(f"ClearVoiceSpeechDenoise: After removing batch dimension: {audio_tensor.shape}")
                
                if audio_tensor.dim() == 2:
                    # For shape [channels, samples] like [2, samples]
                    num_channels = audio_tensor.shape[0]
                    if num_channels == 1:
                        # Mono audio: [1, samples] -> [samples]
                        audio_tensor = audio_tensor.squeeze(0)
                    else:
                        # Multi-channel audio: [channels, samples] -> mix to mono [samples]
                        # Average all channels to create mono
                        audio_tensor = torch.mean(audio_tensor, dim=0)
                    print(f"ClearVoiceSpeechDenoise: After channel processing: {audio_tensor.shape}")
                
                if audio_tensor.dim() > 1:
                    # For any remaining multi-dimensional tensors, squeeze them
                    audio_tensor = audio_tensor.squeeze()
                    print(f"ClearVoiceSpeechDenoise: After squeezing: {audio_tensor.shape}")
                    
                    # If still not 1D, flatten it
                    if audio_tensor.dim() > 1:
                        audio_tensor = audio_tensor.flatten()
                        print(f"ClearVoiceSpeechDenoise: After flattening: {audio_tensor.shape}")
                
                if audio_tensor.dim() != 1:
                    print(f"Error: Invalid audio tensor dimensions after processing: {audio_tensor.shape}")
                    return None
                
                # Write to WAV file using wave module to avoid torchcodec dependency
                import wave
                import numpy as np
                
                # Convert tensor to numpy array
                audio_np = audio_tensor.numpy()
                
                # Normalize to int16 range
                audio_np = np.clip(audio_np * 32768, -32768, 32767).astype(np.int16)
                
                # Save to WAV file
                with wave.open(temp_wav_path, 'wb') as wf:
                    wf.setnchannels(1)  # Mono
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(int(sample_rate))
                    wf.writeframes(audio_np.tobytes())
                
                return temp_wav_path
            except Exception as e:
                print(f"Error processing audio tensor: {e}")
                import traceback
                print(f"Detailed error: {traceback.format_exc()}")
                # Clean up temporary file
                import os
                if temp_wav_path and os.path.exists(temp_wav_path):
                    try:
                        os.unlink(temp_wav_path)
                    except Exception as cleanup_error:
                        print(f"Error cleaning up temporary file: {cleanup_error}")
                return None
    
    def audio_file_to_output(self, audio_path):
        """Convert audio file to AUDIO type output"""
        if not audio_path:
            return None
        
        try:
            # Load audio file using wave module to avoid torchcodec dependency
            import wave
            import numpy as np
            
            with wave.open(audio_path, 'rb') as wf:
                sample_rate = wf.getframerate()
                num_channels = wf.getnchannels()
                num_frames = wf.getnframes()
                
                # Read audio data
                audio_data = wf.readframes(num_frames)
                
                # Convert to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # Normalize to float32
                audio_array = audio_array.astype(np.float32) / 32768.0
                
                # Convert to PyTorch tensor
                audio_tensor = torch.tensor(audio_array)
                
                # Handle multi-channel audio
                if num_channels > 1:
                    # Take first channel
                    audio_tensor = audio_tensor[::num_channels]
            
            # Create output dictionary
            # Ensure waveform is in ComfyUI standard format: [1, 1, samples]
            if audio_tensor.dim() == 1:
                # [samples] -> [1, 1, samples]
                audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
            elif audio_tensor.dim() == 2:
                # [channels, samples] -> [1, channels, samples]
                audio_tensor = audio_tensor.unsqueeze(0)
            
            audio_output = {
                "waveform": audio_tensor,
                "sample_rate": sample_rate
            }
            
            return audio_output
        except Exception as e:
            print(f"Error converting audio file to output: {e}")
            import traceback
            print(f"Detailed error: {traceback.format_exc()}")
            return None
    
    def replace_audio_in_video(self, input_video, input_audio, output_video):
        """Replace audio in video file using ffmpeg"""
        try:
            print(f"ClearVoiceBaseNode: Starting to replace audio in video: {input_video}")
            print(f"ClearVoiceBaseNode: Using audio: {input_audio}")
            
            # Use subprocess to call ffmpeg directly
            import subprocess
            import os
            
            # Construct ffmpeg command
            cmd = [
                "ffmpeg",
                "-i", input_video,
                "-i", input_audio,
                "-c:v", "copy",  # Copy video codec (no re-encoding)
                "-c:a", "aac",   # Encode audio as AAC
                "-map", "0:v:0",  # Use video from first input
                "-map", "1:a:0",  # Use audio from second input
                "-shortest",      # Make output duration match shortest input
                "-y",             # Overwrite output file
                output_video
            ]
            
            # Execute the command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"ClearVoiceBaseNode: Error running ffmpeg: {result.stderr}")
                return False
            
            # Check if output file was created
            if os.path.exists(output_video):
                print(f"ClearVoiceBaseNode: ✓ Successfully replaced audio in video: {output_video}")
                print(f"ClearVoiceBaseNode: Output file size: {os.path.getsize(output_video) / (1024*1024):.2f} MB")
                return True
            else:
                print(f"ClearVoiceBaseNode: ✗ Output video file was not created")
                return False
                
        except Exception as e:
            print(f"ClearVoiceBaseNode: Error replacing audio in video: {e}")
            import traceback
            print(f"ClearVoiceBaseNode: Detailed error: {traceback.format_exc()}")
            return False

class ClearVoiceSpeechSeparationNode(ClearVoiceBaseNode):
    """ComfyUI node for speech separation using ClearVoice
    
    Models:
    - MossFormer2_SS_16K: Best for 16KHz audio
    
    Note: Each model automatically uses its optimal sample rate.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        """Define the input types for the node"""
        return {
            "required": {
                "media": ("*",),
                "model_name": (["MossFormer2_SS_16K"], {"default": "MossFormer2_SS_16K"}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = ("audio_output_1", "audio_output_2", "audio_output_3", "audio_output_4", "audio_output_5")
    FUNCTION = "separate_speech"
    CATEGORY = "ClearVoice"
    TITLE = "ClearVoice Speech Separation"
    
    def separate_speech(self, media, model_name) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Perform speech separation"""
        import os
        import tempfile
        from comfy.utils import ProgressBar
        from comfy_api.latest._input_impl.video_types import VideoFromFile
        
        print(f"ClearVoiceSpeechSeparation: Starting separation with model={model_name}")
        print(f"ClearVoiceSpeechSeparation: media type={type(media)}")
        
        # ClearVoice is already imported at module level
        if not HAS_CLEARVOICE:
            raise RuntimeError("ClearVoice not installed")
        
        # Process media input
        input_path, temp_audio_path = self.process_media_input(media)
        
        if not input_path:
            raise RuntimeError("No valid input provided")
        
        pbar = ProgressBar(4)
        pbar.update_absolute(1, 4, None)
        
        # Ensure input is in WAV format
        print(f"ClearVoiceSpeechSeparation: Ensuring WAV format for {input_path}")
        processed_input = self.ensure_wav_format(input_path)
        print(f"ClearVoiceSpeechSeparation: processed_input={processed_input}")
        
        if not processed_input:
            raise RuntimeError("Failed to process input file")
        
        pbar.update_absolute(2, 4, None)
        
        # Initialize temporary resources
        temp_resources = {
            'temp_audio_path': temp_audio_path,
            'processed_input': processed_input,
            'output_dir': None,
            'output_files': []
        }
        
        try:
            # Set up model directory using ComfyUI's model path
            model_dir = os.path.join(folder_paths.models_dir, "ASR", "ClearerVoice-Studio")
            print(f"ClearVoiceSpeechSeparation: Using model directory: {model_dir}")
            
            # Create model directory if it doesn't exist
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
                print(f"ClearVoiceSpeechSeparation: Created model directory")
            
            # Set environment variable for ClearVoice to use our model directory
            os.environ['CLEARVOICE_CHECKPOINT_DIR'] = model_dir
            print(f"ClearVoiceSpeechSeparation: Set CLEARVOICE_CHECKPOINT_DIR to {model_dir}")
            
            # Initialize ClearVoice model
            model = self.initialize_model('speech_separation', model_name, model_dir)
            
            # Create output directory in plugin's temp folder using UUID to avoid Chinese path issues
            import tempfile
            import uuid
            unique_id = uuid.uuid4().hex[:8]
            output_dir = os.path.join(temp_dir, f"SS_{model_name}_{unique_id}")
            os.makedirs(output_dir, exist_ok=True)
            temp_resources['output_dir'] = output_dir
            print(f"ClearVoiceSpeechSeparation: Output directory: {output_dir}")
            
            # Process audio using the model directly
            print("[139] ClearVoiceSpeechSeparation: Processing audio with ClearVoice")
            print(f"ClearVoiceSpeechSeparation: Input file: {processed_input}")
            print(f"ClearVoiceSpeechSeparation: Output directory: {output_dir}")
            
            try:
                # Set input path and output directory in args
                model.args.input_path = processed_input
                model.args.output_dir = output_dir
                
                # Process the audio
                print("[140] ClearVoiceSpeechSeparation: Starting model.process()...")
                result = model.process(processed_input, online_write=True, output_path=output_dir)
                print("[141] ClearVoiceSpeechSeparation: ✓ ClearVoice processing completed")
                print(f"ClearVoiceSpeechSeparation: Processing result: {result}")
            except Exception as e:
                print(f"ClearVoiceSpeechSeparation: ✗ Error during model processing: {e}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Model processing failed: {e}")
            
            pbar.update_absolute(3, 4, None)
            
            # Find output files recursively
            output_files = []
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.endswith('.wav'):
                        file_path = os.path.join(root, file)
                        output_files.append(file_path)
                        temp_resources['output_files'].append(file_path)
            
            # Sort output files
            output_files.sort()
            print(f"ClearVoiceSpeechSeparation: Found {len(output_files)} output files")
            for file_path in output_files:
                print(f"ClearVoiceSpeechSeparation: Output file: {file_path}, size={os.path.getsize(file_path)} bytes")
            
            # Prepare output paths
            outputs = [None for _ in range(5)]
            
            if output_files:
                for i, file_path in enumerate(output_files[:5]):
                    print(f"ClearVoiceSpeechSeparation: Processing output file {i+1}: {file_path}")
                    if os.path.exists(file_path):
                        print(f"ClearVoiceSpeechSeparation: Output file exists, size={os.path.getsize(file_path)} bytes")
                        outputs[i] = self.audio_file_to_output(file_path)
                        print(f"ClearVoiceSpeechSeparation: Output {i+1} created successfully")
                    else:
                        print(f"ClearVoiceSpeechSeparation: Output file does not exist: {file_path}")
            else:
                print("[142] ClearVoiceSpeechSeparation: No output files found. This may indicate a model processing issue.")
                # Try to find any files in the output directory
                all_files = []
                for root, dirs, files in os.walk(output_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        all_files.append(file_path)
                if all_files:
                    print(f"ClearVoiceSpeechSeparation: Found {len(all_files)} files in output directory:")
                    for file_path in all_files:
                        print(f"  - {file_path}")
                else:
                    print("[143] ClearVoiceSpeechSeparation: No files found in output directory at all.")
            
            pbar.update_absolute(4, 4, None)
            
            print(f"Speech separation completed: {len(output_files)} speakers found")
            return tuple(outputs)
            
        except Exception as e:
            print(f"Error during speech separation: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Speech separation failed: {e}")
        finally:
            # Clean up temporary resources based on the CLEANUP_TEMP_RESOURCES flag
            if self.CLEANUP_TEMP_RESOURCES:
                self.cleanup_temp_resources(temp_resources)
            else:
                print("[144] ClearVoiceBaseNode: Temporary resource cleanup is disabled")

class ClearVoiceSpeechDenoiseNode(ClearVoiceBaseNode):
    """ComfyUI node for speech denoising using ClearVoice
    
    Models:
    - FRCRN_SE_16K: Best for 16KHz audio
    - MossFormer2_SE_48K: Best for 48KHz audio
    - MossFormerGAN_SE_16K: Best for 16KHz audio
    
    Note: Each model automatically uses its optimal sample rate.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        """Define the input types for the node"""
        return {
            "required": {
                "media": ("*",),
                "model_name": (["FRCRN_SE_16K", "MossFormer2_SE_48K", "MossFormerGAN_SE_16K"], {"default": "FRCRN_SE_16K"}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "VIDEO")
    RETURN_NAMES = ("audio_output", "video_output")
    FUNCTION = "denoise_speech"
    CATEGORY = "ClearVoice"
    TITLE = "ClearVoice Speech Denoise"
    
    def denoise_speech(self, media, model_name) -> Tuple[Dict[str, Any], object]:
        """Perform speech denoising"""
        import os
        import tempfile
        from comfy.utils import ProgressBar
        from comfy_api.latest._input_impl.video_types import VideoFromFile
        
        print(f"ClearVoiceSpeechDenoise: Starting denoising with model={model_name}")
        print(f"ClearVoiceSpeechDenoise: media type={type(media)}")
        
        # ClearVoice is already imported at module level
        if not HAS_CLEARVOICE:
            raise RuntimeError("ClearVoice not installed")
        
        # Extract original video path if input is video
        original_video_path = None
        if not isinstance(media, dict) or 'waveform' not in media:
            # This is likely a video input
            # Try to extract video path directly
            if hasattr(media, 'video_path'):
                original_video_path = media.video_path
                print(f"ClearVoiceSpeechDenoise: Got original video path from media.video_path: {original_video_path}")
            elif hasattr(media, 'path'):
                original_video_path = media.path
                print(f"ClearVoiceSpeechDenoise: Got original video path from media.path: {original_video_path}")
            elif hasattr(media, 'file_path'):
                original_video_path = media.file_path
                print(f"ClearVoiceSpeechDenoise: Got original video path from media.file_path: {original_video_path}")
            elif hasattr(media, 'filename'):
                original_video_path = media.filename
                print(f"ClearVoiceSpeechDenoise: Got original video path from media.filename: {original_video_path}")
            elif hasattr(media, 'video_file_path'):
                original_video_path = media.video_file_path
                print(f"ClearVoiceSpeechDenoise: Got original video path from media.video_file_path: {original_video_path}")
            elif hasattr(media, 'input_path'):
                original_video_path = media.input_path
                print(f"ClearVoiceSpeechDenoise: Got original video path from media.input_path: {original_video_path}")
            elif hasattr(media, 'filepath'):
                original_video_path = media.filepath
                print(f"ClearVoiceSpeechDenoise: Got original video path from media.filepath: {original_video_path}")
            elif hasattr(media, 'name'):
                original_video_path = media.name
                print(f"ClearVoiceSpeechDenoise: Got original video path from media.name: {original_video_path}")
            elif isinstance(media, str):
                original_video_path = media
                print(f"ClearVoiceSpeechDenoise: Got original video path from string input: {original_video_path}")
            
            # Try to get __dict__ to see all attributes
            if not original_video_path and hasattr(media, '__dict__'):
                print(f"ClearVoiceSpeechDenoise: Checking media.__dict__ for video path")
                # Check if any attribute looks like a path
                for key, value in media.__dict__.items():
                    if isinstance(value, str) and ('.mp4' in value.lower() or '.avi' in value.lower() or '.mov' in value.lower()):
                        original_video_path = value
                        print(f"ClearVoiceSpeechDenoise: Got original video path from __dict__[{key}]: {original_video_path}")
                        break
        
        # Process media input
        input_path, temp_audio_path = self.process_media_input(media)
        
        if not input_path:
            raise RuntimeError("No valid input provided")
        
        pbar = ProgressBar(4)
        pbar.update_absolute(1, 4, None)
        
        # Ensure input is in WAV format
        print(f"ClearVoiceSpeechDenoise: Ensuring WAV format for {input_path}")
        processed_input = self.ensure_wav_format(input_path)
        print(f"ClearVoiceSpeechDenoise: processed_input={processed_input}")
        
        if not processed_input:
            raise RuntimeError("Failed to process input file")
        
        pbar.update_absolute(2, 4, None)
        
        # Initialize temporary resources
        temp_resources = {
            'temp_audio_path': temp_audio_path,
            'processed_input': processed_input,
            'temp_output_dir': None,
            'output_path': None,
            'output_video_path': None
        }
        
        try:
            # Set up model directory using ComfyUI's model path
            model_dir = os.path.join(folder_paths.models_dir, "ASR", "ClearerVoice-Studio")
            print(f"ClearVoiceSpeechDenoise: Using model directory: {model_dir}")
            
            # Create model directory if it doesn't exist
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
                print(f"ClearVoiceSpeechDenoise: Created model directory")
            
            # Set environment variable for ClearVoice to use our model directory
            os.environ['CLEARVOICE_CHECKPOINT_DIR'] = model_dir
            print(f"ClearVoiceSpeechDenoise: Set CLEARVOICE_CHECKPOINT_DIR to {model_dir}")
            
            # Initialize ClearVoice model
            model = self.initialize_model('speech_enhancement', model_name, model_dir)
            
            # Generate temporary output directory in plugin's temp folder using UUID to avoid Chinese path issues
            import tempfile
            import uuid
            unique_id = uuid.uuid4().hex[:8]
            temp_output_dir = os.path.join(temp_dir, f"SE_{model_name}_{unique_id}")
            os.makedirs(temp_output_dir, exist_ok=True)
            temp_resources['temp_output_dir'] = temp_output_dir
            print(f"ClearVoiceSpeechDenoise: Temporary output directory: {temp_output_dir}")
            
            # Process audio using the model directly
            print("[145] ClearVoiceSpeechDenoise: Processing audio with ClearVoice...")
            print(f"ClearVoiceSpeechDenoise: Input file: {processed_input}")
            print(f"ClearVoiceSpeechDenoise: Output directory: {temp_output_dir}")
            
            try:
                # Set input path and output directory in args
                model.args.input_path = processed_input
                model.args.output_dir = temp_output_dir
                
                # Process the audio
                print("[146] ClearVoiceSpeechDenoise: Starting model.process()...")
                result = model.process(processed_input, online_write=True, output_path=temp_output_dir)
                print("[147] ClearVoiceSpeechDenoise: ✓ ClearVoice processing completed")
                print(f"ClearVoiceSpeechDenoise: Processing result: {result}")
            except Exception as e:
                print(f"ClearVoiceSpeechDenoise: ✗ Error during model processing: {e}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Model processing failed: {e}")
            
            pbar.update_absolute(3, 4, None)
            
            # Find the actual output file created by ClearVoice
            output_path = None
            for root, dirs, files in os.walk(temp_output_dir):
                for file in files:
                    if file.endswith('.wav'):
                        output_path = os.path.join(root, file)
                        temp_resources['output_path'] = output_path
                        print(f"ClearVoiceSpeechDenoise: Found output file: {output_path}, size={os.path.getsize(output_path)} bytes")
                        break
                if output_path:
                    break
            
            if not output_path:
                raise RuntimeError("No output file found in temporary directory")
            
            # Convert to AUDIO type
            print("[148] ClearVoiceSpeechDenoise: Converting to AUDIO type")
            print(f"ClearVoiceSpeechDenoise: Using output path: {output_path}")
            audio_output = self.audio_file_to_output(output_path)
            print(f"ClearVoiceSpeechDenoise: audio_output={audio_output}")
            if audio_output:
                print(f"ClearVoiceSpeechDenoise: audio_output waveform shape: {audio_output['waveform'].shape}")
                print(f"ClearVoiceSpeechDenoise: audio_output sample rate: {audio_output['sample_rate']}")
            else:
                raise RuntimeError("Failed to create audio_output")
            
            # Handle video output if input was a video
            output_video = None
            if original_video_path and os.path.exists(original_video_path):
                print(f"ClearVoiceSpeechDenoise: Original video path exists: {original_video_path}")
                # Create output video path in ComfyUI's output directory
                output_dir = folder_paths.get_output_directory()
                output_video_path = os.path.join(output_dir, "clearvoice_output.mp4")
                print(f"ClearVoiceSpeechDenoise: Output video path: {output_video_path}")
                # Don't add to temp_resources so it won't be cleaned up
                
                # Replace audio in video
                print(f"ClearVoiceSpeechDenoise: Replacing audio in video with processed audio")
                if self.replace_audio_in_video(original_video_path, output_path, output_video_path):
                    print(f"ClearVoiceSpeechDenoise: Successfully created output video: {output_video_path}")
                    # Create VideoFromFile object
                    output_video = VideoFromFile(output_video_path)
                    print(f"ClearVoiceSpeechDenoise: Created VideoFromFile object for output video")
                else:
                    print("[149] ClearVoiceSpeechDenoise: Failed to create output video")
            else:
                print("[150] ClearVoiceSpeechDenoise: No valid original video path provided, skipping video output")
            
            pbar.update_absolute(4, 4, None)
            
            print(f"Speech denoising completed: {output_path}")
            return (audio_output, output_video)
            
        except Exception as e:
            print(f"Error during speech denoising: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Speech denoising failed: {e}")
        finally:
            # Clean up temporary resources based on the CLEANUP_TEMP_RESOURCES flag
            if self.CLEANUP_TEMP_RESOURCES:
                self.cleanup_temp_resources(temp_resources)
            else:
                print("[151] ClearVoiceBaseNode: Temporary resource cleanup is disabled")


class ClearVoiceVideoSpeakerExtractionNode(ClearVoiceBaseNode):
    """ComfyUI node for video speaker extraction using ClearVoice"""
    
    @classmethod
    def INPUT_TYPES(cls):
        """Define the input types for the node"""
        return {
            "required": {
                "video_input": ("*",),
            },
            "optional": {
                "enable_crop": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("VIDEO", "VIDEO", "VIDEO", "VIDEO", "VIDEO")
    RETURN_NAMES = ("video_output_1", "video_output_2", "video_output_3", "video_output_4", "video_output_5")
    FUNCTION = "extract_speakers"
    CATEGORY = "ClearVoice"
    TITLE = "ClearVoice Video Speaker Extraction"
    
    def extract_speakers(self, video_input, enable_crop=False) -> Tuple[object, object, object, object, object]:
        """Extract speakers from video and generate individual videos
        
        Args:
            video_input: Input video file or VideoFromFile object
            enable_crop: If True, output cropped face videos (224x224). If False, output original videos with face bounding boxes.
        """
        import os
        import tempfile
        from comfy.utils import ProgressBar
        from comfy_api.latest._input_impl.video_types import VideoFromFile
        
        print("[100] ClearVoiceVideoSpeakerExtraction: Starting extraction for video={}".format(video_input))
        
        # ClearVoice is already imported at module level
        if not HAS_CLEARVOICE:
            raise RuntimeError("ClearVoice not installed")
        
        if not video_input:
            raise RuntimeError("No valid video input provided")
        
        pbar = ProgressBar(4)
        pbar.update_absolute(1, 4, None)
        
        # Extract video path from VIDEO object
        video_path = None
        
        # Print all attributes of the video object for debugging
        print("[101] ClearVoiceVideoSpeakerExtraction: VIDEO object attributes: {}".format(dir(video_input)))
        
        # Try different ways to get video path
        if hasattr(video_input, 'video_path'):
            video_path = video_input.video_path
            print("[102] ClearVoiceVideoSpeakerExtraction: Got video_path from video_input.video_path: {}".format(video_path))
        elif hasattr(video_input, 'path'):
            video_path = video_input.path
            print("[103] ClearVoiceVideoSpeakerExtraction: Got video_path from video_input.path: {}".format(video_path))
        elif hasattr(video_input, 'file_path'):
            video_path = video_input.file_path
            print("[104] ClearVoiceVideoSpeakerExtraction: Got video_path from video_input.file_path: {}".format(video_path))
        elif hasattr(video_input, 'filename'):
            video_path = video_input.filename
            print("[105] ClearVoiceVideoSpeakerExtraction: Got video_path from video_input.filename: {}".format(video_path))
        elif hasattr(video_input, 'video_file_path'):
            video_path = video_input.video_file_path
            print("[106] ClearVoiceVideoSpeakerExtraction: Got video_path from video_input.video_file_path: {}".format(video_path))
        elif hasattr(video_input, 'input_path'):
            video_path = video_input.input_path
            print("[107] ClearVoiceVideoSpeakerExtraction: Got video_path from video_input.input_path: {}".format(video_path))
        elif hasattr(video_input, 'filepath'):
            video_path = video_input.filepath
            print("[108] ClearVoiceVideoSpeakerExtraction: Got video_path from video_input.filepath: {}".format(video_path))
        elif hasattr(video_input, 'name'):
            video_path = video_input.name
            print("[109] ClearVoiceVideoSpeakerExtraction: Got video_path from video_input.name: {}".format(video_path))
        elif isinstance(video_input, str):
            video_path = video_input
            print("[110] ClearVoiceVideoSpeakerExtraction: Got video_path from string input: {}".format(video_path))
        
        # Try to get __dict__ to see all attributes
        if not video_path and hasattr(video_input, '__dict__'):
            print("[111] ClearVoiceVideoSpeakerExtraction: VIDEO object __dict__: {}".format(video_input.__dict__))
            # Check if any attribute looks like a path
            for key, value in video_input.__dict__.items():
                if isinstance(value, str) and ('.mp4' in value.lower() or '.avi' in value.lower() or '.mov' in value.lower()):
                    video_path = value
                    print("[112] ClearVoiceVideoSpeakerExtraction: Got video_path from __dict__[{}]: {}".format(key, video_path))
                    break
        
        if not video_path:
            raise RuntimeError("Failed to extract video path from VIDEO object")
        
        # Check if video file exists
        if not os.path.exists(video_path):
            raise RuntimeError("Video file does not exist: {}".format(video_path))
        
        print("[113] ClearVoiceVideoSpeakerExtraction: Video file exists, size={} bytes".format(os.path.getsize(video_path)))
        
        pbar.update_absolute(2, 4, None)
        
        # Initialize temporary resources
        temp_resources = {
            'output_dir': None,
            'output_files': []
        }
        
        try:
            # Set up model directory using ComfyUI's model path
            model_dir = os.path.join(folder_paths.models_dir, "ASR", "ClearerVoice-Studio")
            print("[114] ClearVoiceVideoSpeakerExtraction: Using model directory: {}".format(model_dir))
            
            # Create model directory if it doesn't exist
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
                print("[115] ClearVoiceVideoSpeakerExtraction: Created model directory")
            
            # For target speaker extraction, the task is target_speaker_extraction
            task = 'target_speaker_extraction'
            model_name = 'AV_MossFormer2_TSE_16K'
            
            # Initialize ClearVoice model
            model = self.initialize_model(task, model_name, model_dir)
            
            pbar.update_absolute(3, 4, None)
            
            # Create temporary output directory in plugin's temp folder using UUID to avoid Chinese path issues
            import tempfile
            import uuid
            unique_id = uuid.uuid4().hex[:8]
            output_dir = os.path.join(temp_dir, f"AV_MossFormer2_TSE_16K_{unique_id}")
            os.makedirs(output_dir, exist_ok=True)
            temp_resources['output_dir'] = output_dir
            print("[116] ClearVoiceVideoSpeakerExtraction: Output directory: {}".format(output_dir))
            
            # Process video using the model directly
            print("[117] ClearVoiceVideoSpeakerExtraction: Processing video with ClearVoice")
            print("[118] ClearVoiceVideoSpeakerExtraction: Input file: {}".format(video_path))
            print("[119] ClearVoiceVideoSpeakerExtraction: Output directory: {}".format(output_dir))
            
            try:
                # Set input path and output directory in args
                model.args.input_path = video_path
                model.args.output_dir = output_dir
                
                # Process the video
                print("[120] ClearVoiceVideoSpeakerExtraction: Starting model.process()...")
                result = model.process(video_path, online_write=True, output_path=output_dir)
                print("[121] ClearVoiceVideoSpeakerExtraction: ✓ ClearVoice processing completed")
                print("[122] ClearVoiceVideoSpeakerExtraction: Processing result: {}".format(result))
            except Exception as e:
                print("[123] ClearVoiceVideoSpeakerExtraction: ✗ Error during model processing: {}".format(e))
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Model processing failed: {e}")
            
            # Find output videos and face track videos
            import glob
            output_videos = glob.glob(os.path.join(output_dir, "**", "video_est_*.mp4"), recursive=True)
            output_videos.sort()
            print(f"ClearVoiceVideoSpeakerExtraction: Found {len(output_videos)} output videos (may be empty)")
            print(f"ClearVoiceVideoSpeakerExtraction: enable_crop={enable_crop}")
            
            # Find face track videos (cropped face videos from py_faceTracks)
            face_track_videos = glob.glob(os.path.join(output_dir, "**", "py_faceTracks", "*.avi"), recursive=True)
            face_track_videos.sort()
            print(f"ClearVoiceVideoSpeakerExtraction: Found {len(face_track_videos)} face track videos")
            
            # Find est_*.wav files (individual speaker audio from py_faceTracks)
            est_audio_files = glob.glob(os.path.join(output_dir, "**", "py_faceTracks", "est_*.wav"), recursive=True)
            est_audio_files.sort()
            print(f"ClearVoiceVideoSpeakerExtraction: Found {len(est_audio_files)} est audio files")
            
            # Find video_out_*.avi files (original videos with face boxes from py_video)
            video_out_files = glob.glob(os.path.join(output_dir, "**", "py_video", "video_out_*.avi"), recursive=True)
            video_out_files.sort()
            print(f"ClearVoiceVideoSpeakerExtraction: Found {len(video_out_files)} video_out files")
            
            # Prepare output paths
            from comfy_api.latest._input_impl.video_types import VideoFromFile
            outputs = [None for _ in range(5)]
            
            # Get ComfyUI's output directory
            comfy_output_dir = folder_paths.get_output_directory()
            
            # Determine number of speakers based on enable_crop
            if enable_crop:
                num_speakers = len(face_track_videos)
            else:
                num_speakers = len(video_out_files)
            print(f"ClearVoiceVideoSpeakerExtraction: Processing {num_speakers} speakers")
            
            for i in range(min(num_speakers, 5)):
                print(f"ClearVoiceVideoSpeakerExtraction: Processing speaker {i+1}")
                
                # Determine which video to use based on enable_crop
                if enable_crop:
                    # Use cropped face video (224x224) from py_faceTracks
                    # Files are named: 00000.avi, 00001.avi, etc.
                    source_video = face_track_videos[i]
                    output_video_path = os.path.join(comfy_output_dir, f"clearvoice_speaker_{i+1}_crop.mp4")
                    print(f"ClearVoiceVideoSpeakerExtraction: Using cropped face video: {source_video}")
                    
                    # For cropped videos, we need to merge with individual speaker audio (est_*.wav)
                    # The cropped videos currently have original audio, not the separated speaker audio
                    if i < len(est_audio_files):
                        est_audio = est_audio_files[i]
                        print(f"ClearVoiceVideoSpeakerExtraction: Merging with individual speaker audio: {est_audio}")
                        
                        # Create temporary output file with merged audio
                        temp_merged_path = os.path.join(temp_dir, f"temp_merged_{i}.avi")
                        try:
                            # Merge cropped video with individual speaker audio
                            cmd_merge = [
                                "ffmpeg",
                                "-i", source_video,
                                "-i", est_audio,
                                "-c:v", "copy",
                                "-c:a", "pcm_s16le",
                                "-map", "0:v:0",
                                "-map", "1:a:0",
                                "-y",
                                temp_merged_path
                            ]
                            result_merge = subprocess.run(cmd_merge, capture_output=True, text=True)
                            
                            if result_merge.returncode == 0 and os.path.exists(temp_merged_path):
                                # Use merged video as source
                                source_video = temp_merged_path
                                print(f"ClearVoiceVideoSpeakerExtraction: Successfully merged video with individual audio")
                            else:
                                print(f"ClearVoiceVideoSpeakerExtraction: Failed to merge audio, using original video: {result_merge.stderr}")
                        except Exception as e:
                            print(f"ClearVoiceVideoSpeakerExtraction: Error merging audio: {e}")
                else:
                    # Use original video with face bounding boxes from py_video
                    # Files are named: video_out_0.avi, video_out_1.avi, etc.
                    # These videos already have the correct individual speaker audio
                    source_video = video_out_files[i]
                    output_video_path = os.path.join(comfy_output_dir, f"clearvoice_speaker_{i+1}.mp4")
                    print(f"ClearVoiceVideoSpeakerExtraction: Using original video with face boxes: {source_video}")
                
                if os.path.exists(source_video):
                    file_size = os.path.getsize(source_video)
                    print(f"ClearVoiceVideoSpeakerExtraction: Source video exists, size={file_size} bytes")
                    
                    # Check if source video is valid (not empty)
                    if file_size == 0:
                        print(f"ClearVoiceVideoSpeakerExtraction: Source video is empty (0 bytes), skipping")
                        outputs[i] = None
                        continue
                    
                    print(f"ClearVoiceVideoSpeakerExtraction: Converting to output directory: {output_video_path}")
                    
                    # Convert video to MP4 format using ffmpeg
                    try:
                        cmd = [
                            "ffmpeg",
                            "-i", source_video,
                            "-c:v", "libx264",  # Use H.264 codec
                            "-c:a", "aac",      # Use AAC audio codec
                            "-y",               # Overwrite output
                            output_video_path
                        ]
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        
                        if result.returncode != 0:
                            print(f"ClearVoiceVideoSpeakerExtraction: Error converting video: {result.stderr}")
                            # Try alternative conversion (copy codec)
                            cmd_alt = [
                                "ffmpeg",
                                "-i", source_video,
                                "-c", "copy",
                                "-y",
                                output_video_path
                            ]
                            result_alt = subprocess.run(cmd_alt, capture_output=True, text=True)
                            if result_alt.returncode != 0:
                                print(f"ClearVoiceVideoSpeakerExtraction: Alternative conversion also failed: {result_alt.stderr}")
                                outputs[i] = None
                                continue
                        
                        # Check if output file was created and is valid
                        if os.path.exists(output_video_path):
                            output_size = os.path.getsize(output_video_path)
                            print(f"ClearVoiceVideoSpeakerExtraction: Successfully converted video, size={output_size} bytes")
                            
                            if output_size > 0:
                                # Create VideoFromFile object
                                video_object = VideoFromFile(output_video_path)
                                outputs[i] = video_object
                                print(f"ClearVoiceVideoSpeakerExtraction: Created VideoFromFile object for output video")
                            else:
                                print(f"ClearVoiceVideoSpeakerExtraction: Output video is empty (0 bytes)")
                                outputs[i] = None
                        else:
                            print(f"ClearVoiceVideoSpeakerExtraction: Output video was not created")
                            outputs[i] = None
                    except Exception as e:
                        print(f"ClearVoiceVideoSpeakerExtraction: Error converting video: {e}")
                        import traceback
                        traceback.print_exc()
                        outputs[i] = None
                else:
                    print(f"ClearVoiceVideoSpeakerExtraction: Source video does not exist: {source_video}")
                    outputs[i] = None
            
            pbar.update_absolute(4, 4, None)
            
            print(f"Video speaker extraction completed: {num_speakers} speakers found")
            return tuple(outputs)
            
        except Exception as e:
            print(f"Error during video speaker extraction: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Video speaker extraction failed: {e}")
        finally:
            # Clean up temporary resources based on the CLEANUP_TEMP_RESOURCES flag
            if self.CLEANUP_TEMP_RESOURCES:
                self.cleanup_temp_resources(temp_resources)
            else:
                print("[154] ClearVoiceBaseNode: Temporary resource cleanup is disabled")
