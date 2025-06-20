#!/usr/bin/env python3
# Disable screen access check
import os
import sys
import platform
# Load .env for local development
try:
    from dotenv import load_dotenv
    import os
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(dotenv_path, override=True)
    # Fallback: manually parse .env if variable is still missing
    if os.environ.get('AZURE_STORAGE_CONNECTION_STRING') is None:
        with open(dotenv_path, 'r') as f:
            for line in f:
                if line.strip().startswith('AZURE_STORAGE_CONNECTION_STRING='):
                    # Remove possible quotes and whitespace
                    value = line.strip().split('=', 1)[1].strip().strip('"').strip("'")
                    os.environ['AZURE_STORAGE_CONNECTION_STRING'] = value
                    print('DEBUG: Fallback loaded AZURE_STORAGE_CONNECTION_STRING from .env')
                    break
except ImportError:
    pass

# For macOS, implement comprehensive screen access check bypass
if platform.system() == 'darwin':
    # Set all required environment variables
    os.environ['PYTHONFRAMEWORK'] = '1'
    os.environ['DISPLAY'] = ':0'
    os.environ['WX_NO_DISPLAY_CHECK'] = '1'
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
    os.environ['WXMAC_NO_NATIVE_MENUBAR'] = '1'
    os.environ['PYOBJC_DISABLE_CONFIRMATION'] = '1'
    os.environ['QT_MAC_WANTS_LAYER'] = '1'
    os.environ['PYTHONHASHSEED'] = '1' 
    os.environ['WX_NO_NATIVE'] = '1'
    os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'
    
    # Function to handle uncaught exceptions in the app
    def handle_exception(exc_type, exc_value, exc_traceback):
        import traceback
        error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        
        # Try to show a dialog if possible, otherwise print to stderr
        try:
            # Check global WX_AVAILABLE flag which will be defined later
            # This will be caught in the except block if not yet defined
            if 'WX_AVAILABLE' in globals() and WX_AVAILABLE:
                import wx
                app = wx.App(False)
                wx.MessageBox(f"An error occurred:\n\n{error_msg}", "Application Error", wx.OK | wx.ICON_ERROR)
                app.MainLoop()
            else:
                # wx is not available or not imported yet, fall back to stderr
                sys.stderr.write(f"FATAL ERROR: {error_msg}\n")
        except:
            sys.stderr.write(f"FATAL ERROR: {error_msg}\n")
        
        # Exit with error code
        sys.exit(1)
    
    # Set the exception handler
    sys.excepthook = handle_exception

    # Note: wxPython patching will be done after the WX_AVAILABLE flag is defined

import json
import shutil
import tempfile
import threading
import time
from datetime import datetime
import requests
import base64
from io import BytesIO
import openai
from openai import AzureOpenAI
import wave
import uuid
import re
import io
import subprocess
import hashlib
import pickle
import types
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Make wx imports without errors
try:
    import wx
    import wx.adv
    WX_AVAILABLE = True
    
    # Now that wx is successfully imported, apply patches for macOS compatibility
    try:
        # Patch wx.App to avoid screen check
        if hasattr(wx, 'App'):
            original_init = wx.App.__init__
            
            def patched_init(self, *args, **kwargs):
                # Force redirect to False to avoid screen check issues
                kwargs['redirect'] = False
                return original_init(self, *args, **kwargs)
            
            wx.App.__init__ = patched_init
        
        # Try to patch _core directly
        if hasattr(wx, '_core') and hasattr(wx._core, '_macIsRunningOnMainDisplay'):
            # Replace with function that always returns True
            wx._core._macIsRunningOnMainDisplay = lambda: True
            
        print("Successfully applied wxPython patches for macOS compatibility")
    except Exception as e:
        # Not a fatal error, just log it
        print(f"Warning: Could not apply wxPython patches: {e}")
        
except ImportError:
    WX_AVAILABLE = False
    print("Could not import wxPython. GUI will not be available.")

# Try to import other dependencies with graceful fallback
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# Check if pydub is available for audio conversion
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# Check if pyannote is available for speaker diarization
try:
    import torch
    import pyannote.audio
    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
    from pyannote.audio import Audio
    from pyannote.core import Segment, Annotation
    
    # Fix for SpeechBrain when running as frozen application
    if getattr(sys, 'frozen', False):
        # Add a custom import hook to fix SpeechBrain paths
        import importlib.abc
        import importlib.machinery
        
        class SpeechBrainFixer(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname.startswith('speechbrain.utils.importutils'):
                    # Return empty list for find_imports to avoid file system access
                    return importlib.machinery.ModuleSpec(fullname, None)
                return None
        
        # Add the finder to sys.meta_path
        sys.meta_path.insert(0, SpeechBrainFixer())
    
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False

# Make PyAudio optional with silent failure
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except:
    PYAUDIO_AVAILABLE = False

# Ensure required directories exist
def ensure_directories():
    """Create necessary directories if they don't exist."""
    import os
    import platform
    import sys
    from pathlib import Path
    
    # First determine if app is running as a frozen executable
    is_frozen = getattr(sys, 'frozen', False)
    is_windows = platform.system() == 'windows' or platform.system() == 'Windows'
    is_macos = platform.system() == 'darwin'
    
    # Get application name
    app_name = "KeszAudio"
    
    # If frozen, determine the application directory
    if is_frozen:
        try:
            # For frozen applications, use platform-specific data locations
            if is_macos:
                # For macOS, use ~/Documents for user data
                home_dir = Path.home()
                
                # Create app directory in Documents
                app_dir = home_dir / "Documents" / app_name
                
            elif is_windows:
                # For Windows, use AppData/Local
                app_dir = Path(os.environ.get('LOCALAPPDATA', os.path.expanduser("~"))) / app_name
            else:
                # For Linux, use standard XDG data location
                app_dir = Path.home() / ".local" / "share" / app_name.lower()
            
            # Define required directories
            directories = [
                app_dir,
                app_dir / "Transcripts", 
                app_dir / "Summaries",
                app_dir / "diarization_cache"
            ]
            
            # Create the directories if they don't exist
            for directory in directories:
                if not directory.exists():
                    try:
                        directory.mkdir(parents=True, exist_ok=True)
                        print(f"Created directory: {directory}")
                    except Exception as e:
                        print(f"Error creating directory {directory}: {e}")
            
            # Return the app directory path for reference
            return str(app_dir)
        except Exception as e:
            # If we can't create directories in standard locations, use a fallback
            print(f"Error creating directories in standard location: {e}")
            try:
                # Use home directory as fallback
                fallback_dir = Path.home() / f".{app_name}"
                fallback_dir.mkdir(parents=True, exist_ok=True)
                
                # Create subdirectories
                (fallback_dir / "Transcripts").mkdir(exist_ok=True)
                (fallback_dir / "Summaries").mkdir(exist_ok=True)
                (fallback_dir / "diarization_cache").mkdir(exist_ok=True)
                
                print(f"Created fallback directories in {fallback_dir}")
                return str(fallback_dir)
            except Exception as e2:
                print(f"Failed to create directories in home directory: {e2}")
                
                # Last resort: use temp directory
                import tempfile
                temp_dir = Path(tempfile.gettempdir()) / app_name
                temp_dir.mkdir(exist_ok=True)
                print(f"Using temporary directory as last resort: {temp_dir}")
                return str(temp_dir)
    else:
        # For normal terminal execution, use relative paths but first check if they can be created
        try:
            directories = ["Transcripts", "Summaries", "diarization_cache"]
            for directory in directories:
                if not os.path.exists(directory):
                    os.makedirs(directory)
            
            # Return current directory for reference
            return os.path.abspath('.')
        except OSError as e:
            # If we can't create in current directory, try user's home directory
            print(f"Error creating directories in current path: {e}")
            try:
                home_dir = Path.home()
                app_dir = home_dir / app_name
                app_dir.mkdir(parents=True, exist_ok=True)
                
                # Create subdirectories
                (app_dir / "Transcripts").mkdir(exist_ok=True)
                (app_dir / "Summaries").mkdir(exist_ok=True)
                (app_dir / "diarization_cache").mkdir(exist_ok=True)
                
                print(f"Created directories in {app_dir}")
                return str(app_dir)
            except Exception as e2:
                print(f"Failed to create directories in home directory: {e2}")
                # Return current directory as a last resort, even if we can't write to it
                return os.path.abspath('.')

# Global variables
APP_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUPPORTED_LANGUAGES = ["en", "hu"]  # Add more as needed
LANGUAGE_DISPLAY_NAMES = {
    "en": "English",
    "hu": "Hungarian"
}
app_name = "KeszAudio"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"  # Azure OpenAI deployment name for chat
WHISPER_MODEL = "gpt-4o-transcribe"  # Azure OpenAI deployment name for whisper
client = None  # Azure OpenAI client instance

# Disable GUI requirement check for wxPython
os.environ['WXSUPPRESS_SIZER_FLAGS_CHECK'] = '1'
os.environ['WXSUPPRESS_APP_NAME_WARNING'] = '1'

# Add new imports for speaker diarization with graceful fallback
DIARIZATION_AVAILABLE = False
try:
    from pyannote.audio import Pipeline
    from pyannote.core import Segment, Timeline, Annotation
    import torch
    DIARIZATION_AVAILABLE = True
except ImportError:
    # Silently set flag without showing warning
    pass


# Simple CLI version for when GUI is not available
def run_cli():
    print("\n========= AI Assistant (CLI Mode) =========")
    print("1. Set Azure OpenAI API Key")
    print("2. Transcribe Audio")
    print("3. Chat with AI")
    print("4. Exit")
    
    api_key = os.environ.get("AZURE_OPENAI_API_KEY", "1Dhj1dCS5baiMwSLQ5IUp7jdMXCs9ja6jRbHDU2ThRLwg0N3rxr9JQQJ99BFACHYHv6XJ3w3AAAAACOGEpcL")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://slugb-mbkt1eqz-eastus2.cognitiveservices.azure.com")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")
    client = None
    if api_key and endpoint:
        try:
            client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint
            )
            print("Azure OpenAI configuration found in environment.")
        except Exception as e:
            print(f"Error initializing Azure OpenAI client: {e}")
    
    while True:
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == "1":
            api_key = input("Enter your Azure OpenAI API Key: ").strip()
            endpoint = input("Enter your Azure OpenAI Endpoint URL: ").strip()
            os.environ["AZURE_OPENAI_API_KEY"] = api_key
            os.environ["AZURE_OPENAI_ENDPOINT"] = endpoint
            try:
                client = AzureOpenAI(
                    api_key=api_key,
                    api_version="2025-03-01-preview",
                    azure_endpoint=endpoint
                )
                print("Azure OpenAI configuration set successfully.")
            except Exception as e:
                print(f"Error setting API key: {e}")
        
        elif choice == "2":
            if not client:
                print("Please set your Azure OpenAI API Key first (option 1).")
                continue
                
            audio_path = input("Enter the path to your audio file: ").strip()
            if not os.path.exists(audio_path):
                print(f"File not found: {audio_path}")
                continue
                
            print("Transcribing audio...")
            try:
                with open(audio_path, "rb") as audio_file:
                    response = client.audio.transcriptions.create(
                        file=audio_file,
                        model=whisper_deployment,
                        api_version="2025-03-01-preview"
                    )
                print("\n--- Transcription ---")
                print(response.text)
                print("---------------------")
            except Exception as e:
                print(f"Error transcribing audio: {e}")
        
        elif choice == "3":
            if not client:
                print("Please set your Azure OpenAI API Key first (option 1).")
                continue
                
            print("\nChat with AI (type 'exit' to end conversation)")
            chat_history = []
            
            while True:
                user_input = input("\nYou: ")
                if user_input.lower() == 'exit':
                    break
                    
                chat_history.append({"role": "user", "content": user_input})
                
                try:
                    response = client.chat.completions.create(
                        model=chat_deployment,
                        messages=chat_history,
                        temperature=0.7,
                        max_tokens=1000,
                        api_version="2025-03-01-preview"
                    )
                    
                    assistant_message = response.choices[0].message.content
                    chat_history.append({"role": "assistant", "content": assistant_message})
                    
                    print(f"\nAssistant: {assistant_message}")
                except Exception as e:
                    print(f"Error: {e}")
        
        elif choice == "4":
            print("Exiting AI Assistant. Goodbye!")
            break
            
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")


def main():
    # Check if we're running in CLI mode explicitly
    if "--cli" in sys.argv:
        run_cli()
        return 0
    
    # Check if wxPython is available
    if not WX_AVAILABLE:
        print("wxPython is not available. Running in CLI mode.")
        run_cli()
        return 0
    
    # Try to run in GUI mode
    try:
        app = MainApp()
        app.MainLoop()
        return 0
    except Exception as e:
        print(f"Error starting application: {e}")
        print("Falling back to CLI mode.")
        run_cli()
        return 1

class AudioProcessor:
    """Audio processing functionality for transcription and diarization."""
    def __init__(self, client, update_callback=None, config_manager=None):
        self.client = client
        self.update_callback = update_callback
        self.config_manager = config_manager
        self.transcript = None  # Initialize transcript attribute
        self.speech_client = None
        self.initialize_speech_client()
    
    def initialize_speech_client(self):
        """Initialize Azure Speech client."""
        try:
            import azure.cognitiveservices.speech as speechsdk
            
            # Get API key and region from config
            speech_config = self.config_manager.get_azure_speech_config()
            if speech_config is None:
                print("Failed to get Azure Speech configuration")
                if isinstance(self.update_callback, MainFrame):
                    self.update_callback.show_azure_speech_config_dialog()
                return False
                
            api_key = speech_config.get("api_key", "")
            region = speech_config.get("region", "")
            
            # Print diagnostic information
            print("Initializing Speech client with configuration:")
            print(f"Region: {region}")
            print(f"API Key: {'[SET]' if api_key else '[NOT SET]'}")
            
            # Validate basic requirements
            if not api_key:
                print("No API key found in configuration")
                if isinstance(self.update_callback, MainFrame):
                    self.update_callback.show_azure_speech_config_dialog()
                return False
                
            if not region:
                print("No region found in configuration")
                if isinstance(self.update_callback, MainFrame):
                    self.update_callback.show_azure_speech_config_dialog()
                return False
            
            # Create speech config using subscription key and region
            try:
                print(f"Creating SpeechConfig with region: {region}")
                self.speech_client = speechsdk.SpeechConfig(subscription=api_key, region=region)
                print("Successfully created SpeechConfig")
                
                # Test the configuration by creating a recognizer
                print("Testing speech configuration...")
                audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
                test_recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_client, audio_config=audio_config)
                print("Successfully created test recognizer")
                
                # Set language if specified
                language = self.config_manager.get_language()
                if language:
                    print(f"Setting recognition language to: {language}")
                    self.speech_client.speech_recognition_language = language
                
                print("Speech client initialized successfully")
                return True
                
            except Exception as e:
                print(f"Error creating SpeechConfig: {str(e)}")
                if isinstance(self.update_callback, MainFrame):
                    self.update_callback.show_azure_speech_config_dialog()
                return False
            
        except Exception as e:
            error_msg = f"Error initializing Azure Speech client: {str(e)}"
            print(error_msg)
            if hasattr(e, '__traceback__'):
                import traceback
                traceback.print_exc()
            self.speech_client = None
            
            # Show configuration dialog if there's an error and we have a UI
            if isinstance(self.update_callback, MainFrame):
                self.update_callback.show_azure_speech_config_dialog()
            return False
    
    def update_status(self, message, percent=None):
        """Update status with message and optional progress percentage."""
        if self.update_callback:
            self.update_callback(message, percent)
    
    def convert_audio_file(self, file_path, target_format=".wav"):
        """Convert audio file to a different format using FFmpeg."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
            
        # Get ffmpeg command - check for bundled version first
        ffmpeg_cmd = "ffmpeg"  # Default command
        
        # Check for bundled FFmpeg
        if hasattr(sys, '_MEIPASS'):
            # Running as a PyInstaller bundle
            base_dir = sys._MEIPASS
            bundled_ffmpeg_dir = os.path.join(base_dir, 'ffmpeg')
            
            if os.path.exists(bundled_ffmpeg_dir):
                # Choose executable based on platform
                if platform.system() == 'Windows':
                    bundled_ffmpeg = os.path.join(bundled_ffmpeg_dir, 'ffmpeg.exe')
                else:
                    bundled_ffmpeg = os.path.join(bundled_ffmpeg_dir, 'ffmpeg')
                
                if os.path.exists(bundled_ffmpeg):
                    ffmpeg_cmd = bundled_ffmpeg
                    # Update PATH environment variable to include bundled ffmpeg
                    os.environ["PATH"] = bundled_ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        
        # Check if FFmpeg is available
        try:
            subprocess.run(
                [ffmpeg_cmd, "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            raise RuntimeError("FFmpeg is required for audio conversion but was not found.")
        
        # Create output file path with new extension
        output_path = os.path.splitext(file_path)[0] + target_format
        
        # Run conversion
        self.update_status(f"Converting audio to {target_format} format...", percent=10)
        try:
            subprocess.run(
                [ffmpeg_cmd, "-i", file_path, "-y", output_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            self.update_status(f"Conversion complete: {os.path.basename(output_path)}", percent=20)
            return output_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error during audio conversion: {e.stderr.decode('utf-8', errors='replace')}")
    
    def transcribe_audio(self, audio_path, language=None):
        """Transcribe audio file using the optimal Azure service based on file duration and size."""
        temp_wav_path = None
        try:
            self.update_status("Preparing audio for transcription...", percent=5)

            # Always convert to .wav if not already
            file_ext = os.path.splitext(audio_path)[1].lower()
            if file_ext != '.wav':
                self.update_status(f"Converting {file_ext} to .wav for compatibility...", percent=5)
                try:
                    temp_wav_path = self.convert_audio_file(audio_path, ".wav")
                    wav_path = temp_wav_path
                except Exception as e:
                    raise ValueError(f"Unable to convert to .wav: {str(e)}")
            else:
                wav_path = audio_path

            # Get file size in MB (on .wav file)
            file_size = os.path.getsize(wav_path) / (1024 * 1024)

            # Check audio duration (on .wav file)
            duration_sec = None
            try:
                if LIBROSA_AVAILABLE:
                    duration_sec = librosa.get_duration(path=wav_path)
                else:
                    with wave.open(wav_path, 'rb') as wf:
                        frames = wf.getnframes()
                        rate = wf.getframerate()
                        duration_sec = frames / float(rate)
            except Exception as e:
                duration_sec = None  # If duration can't be determined, fallback to file size only

            # Use Azure OpenAI Whisper for files <= 1500 seconds (25 minutes) and <= 25MB
            if duration_sec is not None:
                if duration_sec <= 1500 and file_size <= 25:
                    result = self._transcribe_with_azure_openai(wav_path, language)
                else:
                    result = self._transcribe_with_azure_speech_batch(wav_path, language)
            else:
                # Fallback: if duration can't be determined, use file size as before
                if file_size <= 25:
                    result = self._transcribe_with_azure_openai(wav_path, language)
                else:
                    result = self._transcribe_with_azure_speech_batch(wav_path, language)

            return result
        except Exception as e:
            error_msg = f"Error in transcription: {str(e)}"
            self.update_status(error_msg, percent=0)
            self.transcript = error_msg
            return error_msg
        finally:
            # Clean up temp wav file if created
            if temp_wav_path and os.path.exists(temp_wav_path):
                try:
                    os.remove(temp_wav_path)
                except Exception:
                    pass

    def _transcribe_with_azure_openai(self, audio_path, language=None):
        """Use Azure OpenAI Whisper for faster transcription of smaller files."""
        try:
            # Get Azure OpenAI configuration
            deployment_name = self.config_manager.get_azure_deployment("whisper")
            if not deployment_name:
                raise ValueError("Whisper deployment name not configured in Azure OpenAI settings.")

            api_key = self.config_manager.get_azure_api_key()
            if not api_key:
                raise ValueError("Azure OpenAI API key not configured.")

            endpoint = self.config_manager.get_azure_endpoint()
            if not endpoint:
                raise ValueError("Azure OpenAI endpoint not configured.")

            # Prepare headers for Azure OpenAI API
            headers = {
                "api-key": api_key
                # Do NOT set Content-Type here; requests will handle it for multipart
            }

            # Read audio file and prepare for upload
            with open(audio_path, 'rb') as audio_file:
                files = {
                    'file': (os.path.basename(audio_path), audio_file, 'application/octet-stream'),
                    'model': (None, deployment_name),
                    'language': (None, language if language else 'en')
                }

                # Submit to Azure OpenAI Whisper endpoint
                self.update_status("Transcribing with Azure OpenAI Whisper...", percent=20)
                response = requests.post(
                    f"{endpoint}/openai/deployments/{deployment_name}/audio/transcriptions?api-version={self.config_manager.get_azure_api_version()}",
                    headers=headers,
                    files=files
                )
                
                if response.status_code != 200:
                    raise Exception(f"Transcription failed: {response.text}")

                result = response.json()
                self.transcript = result.get('text', '')
                
                self.update_status("Transcription complete", percent=100)
                return self.transcript

        except Exception as e:
            error_msg = f"Error in Azure OpenAI transcription: {str(e)}"
            self.update_status(error_msg, percent=0)
            self.transcript = error_msg  # Set transcript to error message
            return error_msg

    def _transcribe_with_azure_speech_batch(self, audio_path, language=None):
        """Use Azure Speech Batch with Whisper model for larger files. Uploads local files to Azure Blob Storage and uses SAS URL. Implements parallel chunking for long files."""
        try:
            import math
            from azure.storage.blob import BlobServiceClient, ContentSettings, generate_blob_sas, BlobSasPermissions
            import uuid
            from datetime import datetime, timedelta
            import os
            import concurrent.futures
            import wave

            speech_config = self.config_manager.get_azure_speech_config()
            subscription_key = speech_config["api_key"]
            region = speech_config["region"]

            storage_connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
            container_name = os.environ.get("AZURE_STORAGE_CONTAINER", "audio-batch")
            if not storage_connection_string:
                raise Exception("Azure Storage connection string not set in environment variable AZURE_STORAGE_CONNECTION_STRING")

            # Helper to split audio into chunks if needed
            def split_audio_to_chunks(audio_path, chunk_length_sec=600):
                """Split audio into chunks of chunk_length_sec (default 10 min). Returns list of chunk file paths."""
                with wave.open(audio_path, 'rb') as wf:
                    framerate = wf.getframerate()
                    nframes = wf.getnframes()
                    nchannels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    duration = nframes / float(framerate)
                    chunk_frames = int(chunk_length_sec * framerate)
                    num_chunks = math.ceil(duration / chunk_length_sec)
                    chunk_paths = []
                    for i in range(num_chunks):
                        wf.setpos(i * chunk_frames)
                        frames = wf.readframes(chunk_frames)
                        chunk_path = f"{audio_path}_chunk_{i+1}.wav"
                        with wave.open(chunk_path, 'wb') as out_wf:
                            out_wf.setnchannels(nchannels)
                            out_wf.setsampwidth(sampwidth)
                            out_wf.setframerate(framerate)
                            out_wf.writeframes(frames)
                        chunk_paths.append(chunk_path)
                    return chunk_paths

            # Check audio duration
            with wave.open(audio_path, 'rb') as wf:
                framerate = wf.getframerate()
                nframes = wf.getnframes()
                duration_sec = nframes / float(framerate)

            # If audio is longer than 10 minutes, split into 10-min chunks
            if duration_sec > 600:
                self.update_status("Splitting audio into 10-min chunks for parallel batch processing...", percent=10)
                chunk_paths = split_audio_to_chunks(audio_path, chunk_length_sec=600)
            else:
                chunk_paths = [audio_path]

            # Upload all chunks in parallel to blob storage and collect SAS URLs
            blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)
            container_client = blob_service_client.get_container_client(container_name)
            try:
                container_client.create_container()
            except Exception:
                pass  # Already exists

            def upload_chunk(chunk_path):
                blob_name = f"{uuid.uuid4()}_{os.path.basename(chunk_path)}"
                blob_client = container_client.get_blob_client(blob_name)
                with open(chunk_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True, content_settings=ContentSettings(content_type="audio/wav"))
                sas_token = generate_blob_sas(
                    account_name=blob_client.account_name,
                    container_name=container_name,
                    blob_name=blob_name,
                    account_key=blob_service_client.credential.account_key,
                    permission=BlobSasPermissions(read=True),
                    expiry=datetime.utcnow() + timedelta(hours=2)
                )
                return f"{blob_client.url}?{sas_token}"

            self.update_status("Uploading audio chunks to Azure Blob Storage in parallel...", percent=20)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                sas_urls = list(executor.map(upload_chunk, chunk_paths))

            # Prepare the API request (use latest API version and correct endpoint)
            api_version = "2024-11-15"
            endpoint = f"https://{region}.api.cognitive.microsoft.com/speechtotext/transcriptions:submit?api-version={api_version}"
            headers = {
                "Ocp-Apim-Subscription-Key": subscription_key,
                "Content-Type": "application/json"
            }
            body = {
                "contentUrls": sas_urls,
                "locale": language or "en-US",
                "displayName": os.path.basename(audio_path),
                "properties": {
                    "displayFormWordLevelTimestampsEnabled": True,
                    "timeToLiveHours": 48
                }
            }

            self.update_status("Submitting batch transcription job...", percent=40)
            response = requests.post(endpoint, headers=headers, json=body)
            response.raise_for_status()
            transcription = response.json()
            transcription_url = transcription["self"]

            # Poll for completion
            self.update_status("Processing transcription... (polling)", percent=60)
            while True:
                response = requests.get(transcription_url, headers=headers)
                response.raise_for_status()
                status = response.json()
                if status["status"] == "Succeeded":
                    break
                elif status["status"] in ["Failed", "Deleted"]:
                    raise Exception(f"Transcription failed with status: {status['status']}")
                time.sleep(5)

            # Get results
            self.update_status("Retrieving transcription results...", percent=90)
            files_url = status["links"]["files"]
            response = requests.get(files_url, headers=headers)
            response.raise_for_status()
            files = response.json()
            combined_transcript = []
            for file in files["values"]:
                if file["kind"] == "Transcription":
                    response = requests.get(file["links"]["contentUrl"])
                    response.raise_for_status()
                    transcription_result = response.json()
                    for segment in transcription_result["combinedRecognizedPhrases"]:
                        combined_transcript.append(segment["display"])
            self.transcript = " ".join(combined_transcript)
            self.update_status("Transcription complete", percent=100)
            # Clean up chunk files if created
            for chunk_path in chunk_paths:
                if chunk_path != audio_path and os.path.exists(chunk_path):
                    try:
                        os.remove(chunk_path)
                    except Exception:
                        pass
            return self.transcript
        except Exception as e:
            error_msg = f"Error in Azure Speech Batch transcription: {str(e)}"
            self.update_status(error_msg, percent=0)
            self.transcript = error_msg
            return error_msg
    
    def _get_ffmpeg_install_instructions(self):
        """Return platform-specific FFmpeg installation instructions."""
        import platform
        system = platform.system().lower()
        
        if system == 'darwin':  # macOS
            return "On macOS:\n1. Install Homebrew from https://brew.sh if you don't have it\n2. Run: brew install ffmpeg"
        elif system == 'windows':
            return "On Windows:\n1. Download from https://ffmpeg.org/download.html\n2. Add to PATH or use a package manager like Chocolatey (choco install ffmpeg)"
        elif system == 'linux':
            return "On Linux:\n- Ubuntu/Debian: sudo apt install ffmpeg\n- Fedora: sudo dnf install ffmpeg\n- Arch: sudo pacman -S ffmpeg"
        else:
            return "Please download FFmpeg from https://ffmpeg.org/download.html"

class MainApp(wx.App):
    def OnInit(self):
        try:
            # Force GUI to work on macOS without Framework build
            self.SetExitOnFrameDelete(True)
            self.frame = MainFrame(None, title="AI Assistant", base_dir=APP_BASE_DIR)
            self.frame.Show()
            # Set top window explicitly for macOS
            self.SetTopWindow(self.frame)
            return True
        except AttributeError as e:
            # Add missing methods to MainFrame that might be referenced but don't exist
            if "'MainFrame' object has no attribute" in str(e):
                attr_name = str(e).split("'")[-2]
                print(f"Adding missing attribute: {attr_name}")
                setattr(MainFrame, attr_name, lambda self, *args, **kwargs: None)
                # Try again
                return self.OnInit()
            else:
                print(f"Error initializing main frame: {e}")
                return False
        except Exception as e:
            print(f"Error initializing main frame: {e}")
            return False

class MainFrame(wx.Frame):
    def __init__(self, parent, title, base_dir):
        super(MainFrame, self).__init__(parent, title=title, size=(1200, 800))
        
        # Initialize config manager
        self.config_manager = ConfigManager(base_dir)
        
        # Initialize attributes
        self.client = None
        self.api_key = self.config_manager.get_azure_api_key()
        self.language = self.config_manager.get_language()  # This will always return a valid language code
        self.hf_token = self.config_manager.get_pyannote_token()
        
        # Initialize other attributes that might be referenced
        self.identify_speakers_btn = None
        self.speaker_id_help_text = None
        self.transcript = None
        self.last_audio_path = None
        
        # Check for API key and initialize client
        self.initialize_openai_client()
        
        # Initialize processors
        self.audio_processor = AudioProcessor(client, self.update_status, self.config_manager)
        self.llm_processor = LLMProcessor(client, self.config_manager, self.update_status)
        
        # Set up the UI - use either create_ui or init_ui, not both
        # Initialize menus and status bar using create_ui
        self.create_ui() # Create notebook and panels
        
        # Event bindings
        self.bind_events()
        
        # Center the window
        self.Centre()
        
        # Status update
        self.update_status("Application ready.", percent=0)
        
        # Display info about supported audio formats
        wx.CallLater(1000, self.show_format_info)
        
        # Check for PyAnnote and display installation message if needed
        wx.CallLater(1500, self.check_pyannote)
    
    def initialize_openai_client(self):
        """Initialize Azure OpenAI client."""
        global client
        api_key = self.config_manager.get_azure_api_key()
        endpoint = self.config_manager.get_azure_endpoint()
        
        if not api_key or not endpoint or endpoint == "https://your-resource-name.openai.azure.com":
            dlg = wx.Dialog(self, title="Azure OpenAI Configuration Required", size=(400, 200))
            panel = wx.Panel(dlg)
            sizer = wx.BoxSizer(wx.VERTICAL)
            
            # API Key input
            key_label = wx.StaticText(panel, label="Azure OpenAI API Key:")
            key_input = wx.TextCtrl(panel, style=wx.TE_PASSWORD)
            key_input.SetValue(api_key)
            
            # Endpoint input
            endpoint_label = wx.StaticText(panel, label="Azure OpenAI Endpoint:")
            endpoint_input = wx.TextCtrl(panel)
            endpoint_input.SetValue(endpoint)
            
            # Add to sizer
            sizer.Add(key_label, 0, wx.ALL, 5)
            sizer.Add(key_input, 0, wx.EXPAND | wx.ALL, 5)
            sizer.Add(endpoint_label, 0, wx.ALL, 5)
            sizer.Add(endpoint_input, 0, wx.EXPAND | wx.ALL, 5)
            
            # Buttons
            btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
            ok_btn = wx.Button(panel, wx.ID_OK, "OK")
            cancel_btn = wx.Button(panel, wx.ID_CANCEL, "Cancel")
            btn_sizer.Add(ok_btn, 0, wx.ALL, 5)
            btn_sizer.Add(cancel_btn, 0, wx.ALL, 5)
            sizer.Add(btn_sizer, 0, wx.ALIGN_CENTER | wx.ALL, 5)
            
            panel.SetSizer(sizer)
            
            if dlg.ShowModal() == wx.ID_OK:
                api_key = key_input.GetValue().strip()
                endpoint = endpoint_input.GetValue().strip()
                self.config_manager.set_azure_api_key(api_key)
                self.config_manager.set_azure_endpoint(endpoint)
            dlg.Destroy()
        
        try:
            client = AzureOpenAI(
                api_key=api_key,
                api_version=self.config_manager.get_azure_api_version(),
                azure_endpoint=endpoint
            )
        except Exception as e:
            wx.MessageBox(f"Error initializing Azure OpenAI client: {str(e)}", "Error", wx.OK | wx.ICON_ERROR)
            
    def create_ui(self):
        """Create the user interface."""
        # Create status bar
        self.status_bar = self.CreateStatusBar()
        self.status_bar.SetStatusText("Ready")
        
        # Create menu bar
        menu_bar = wx.MenuBar()
        
        # File menu
        file_menu = wx.Menu()
        
        # Audio submenu
        audio_menu = wx.Menu()
        upload_audio_item = audio_menu.Append(wx.ID_ANY, "&Upload Audio File", "Upload audio file for transcription")
        self.Bind(wx.EVT_MENU, self.on_upload_audio, upload_audio_item)
        
        file_menu.AppendSubMenu(audio_menu, "&Audio")
        
        # Document submenu
        doc_menu = wx.Menu()
        upload_doc_item = doc_menu.Append(wx.ID_ANY, "&Upload Document", "Upload document for LLM context")
        select_docs_item = doc_menu.Append(wx.ID_ANY, "&Select Documents", "Select documents to load into context")
        
        self.Bind(wx.EVT_MENU, self.on_upload_document, upload_doc_item)
        self.Bind(wx.EVT_MENU, self.on_select_documents, select_docs_item)
        
        file_menu.AppendSubMenu(doc_menu, "&Documents")
        
        # Settings menu item
        settings_item = file_menu.Append(wx.ID_ANY, "&Settings", "Application settings")
        self.Bind(wx.EVT_MENU, self.on_settings, settings_item)
        
        # Exit menu item
        exit_item = file_menu.Append(wx.ID_EXIT, "E&xit", "Exit application")
        self.Bind(wx.EVT_MENU, self.on_exit, exit_item)
        
        menu_bar.Append(file_menu, "&File")
        
        # Help menu
        help_menu = wx.Menu()
        about_item = help_menu.Append(wx.ID_ABOUT, "&About", "About this application")
        self.Bind(wx.EVT_MENU, self.on_about, about_item)
        
        menu_bar.Append(help_menu, "&Help")
        
        self.SetMenuBar(menu_bar)
        
        # Create notebook for tabbed interface
        self.notebook = wx.Notebook(self)
        
        # Create panels for each tab
        self.audio_panel = wx.Panel(self.notebook)
        self.chat_panel = wx.Panel(self.notebook)
        self.settings_panel = wx.Panel(self.notebook)
        
        # Add panels to notebook
        self.notebook.AddPage(self.audio_panel, "Audio Processing")
        self.notebook.AddPage(self.chat_panel, "Chat")
        self.notebook.AddPage(self.settings_panel, "Settings")
        
        # Bind the notebook page change event
        self.notebook.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.on_notebook_page_changed)
        
        # Create UI for each panel
        if hasattr(self, 'create_audio_panel'):
            self.create_audio_panel()
        
        # Add placeholder method if not exists
        if not hasattr(self, 'create_chat_panel'):
            def create_chat_panel(self):
                chat_sizer = wx.BoxSizer(wx.VERTICAL)
                placeholder = wx.StaticText(self.chat_panel, label="Chat panel")
                chat_sizer.Add(placeholder, 1, wx.EXPAND | wx.ALL, 5)
                self.chat_panel.SetSizer(chat_sizer)
            self.create_chat_panel = types.MethodType(create_chat_panel, self)
        self.create_chat_panel()
        
        # Add placeholder method if not exists
        if not hasattr(self, 'create_settings_panel'):
            def create_settings_panel(self):
                settings_sizer = wx.BoxSizer(wx.VERTICAL)
                
                # Create actual content instead of just a placeholder
                # Azure OpenAI API Key
                azure_key_label = wx.StaticText(self.settings_panel, label="Azure OpenAI API Key:")
                self.azure_api_key_input = wx.TextCtrl(self.settings_panel, style=wx.TE_PASSWORD)
                self.azure_api_key_input.SetValue(self.config_manager.get_azure_api_key())
                
                # Azure OpenAI Endpoint
                azure_endpoint_label = wx.StaticText(self.settings_panel, label="Azure OpenAI Endpoint:")
                self.azure_endpoint_input = wx.TextCtrl(self.settings_panel)
                self.azure_endpoint_input.SetValue(self.config_manager.get_azure_endpoint())
                
                # HuggingFace API Key
                hf_label = wx.StaticText(self.settings_panel, label="HuggingFace Token:")
                self.hf_input = wx.TextCtrl(self.settings_panel, style=wx.TE_PASSWORD)
                self.hf_input.SetValue(self.config_manager.get_pyannote_token())
                
                # Add elements to sizer
                settings_sizer.Add(azure_key_label, 0, wx.ALL, 5)
                settings_sizer.Add(self.azure_api_key_input, 0, wx.EXPAND | wx.ALL, 5)
                settings_sizer.Add(azure_endpoint_label, 0, wx.ALL, 5)
                settings_sizer.Add(self.azure_endpoint_input, 0, wx.EXPAND | wx.ALL, 5)
                settings_sizer.Add(hf_label, 0, wx.ALL, 5)
                settings_sizer.Add(self.hf_input, 0, wx.EXPAND | wx.ALL, 5)
                
                # Add the prominent save button
                settings_sizer.Add(add_save_all_settings_button(self.settings_panel, self), 0, wx.EXPAND)
                
                self.settings_panel.SetSizer(settings_sizer)
            self.create_settings_panel = types.MethodType(create_settings_panel, self)
        self.create_settings_panel()
        
        # Main sizer
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(self.notebook, 1, wx.EXPAND | wx.ALL, 5)
        
        self.SetSizer(main_sizer)
        
    def on_notebook_page_changed(self, event):
        """Handle notebook page change event."""
        old_page = event.GetOldSelection()
        new_page = event.GetSelection()
        
        # If user switched from settings to audio tab, update the speaker ID button styling
        if old_page == 2 and new_page == 0:  # 2 = settings, 0 = audio
            self.identify_speakers_btn.SetLabel(self.get_speaker_id_button_label())
            self.speaker_id_help_text.SetLabel(self.get_speaker_id_help_text())
            self.update_speaker_id_button_style()
            self.audio_panel.Layout()
        
        # If user switched to settings tab, refresh the settings values
        if new_page == 2:  # 2 = settings
            # Update HuggingFace token in settings tab from config
            if hasattr(self, 'hf_input'):
                self.hf_input.SetValue(self.config_manager.get_pyannote_token())
            
            # Update Azure OpenAI API key in settings tab from config
            if hasattr(self, 'azure_api_key_input'):
                self.azure_api_key_input.SetValue(self.config_manager.get_azure_api_key())
        
        event.Skip()  # Allow default event processing
    
    def get_speaker_id_button_label(self):
        """Get label for speaker identification button based on token availability."""
        has_token = bool(self.config_manager.get_pyannote_token())
        return "Identify Speakers (Advanced)" if has_token else "Identify Speakers (Basic)"
    
    def get_speaker_id_help_text(self):
        """Get help text for speaker identification based on token availability."""
        has_token = bool(self.config_manager.get_pyannote_token())
        if has_token:
            return "Using PyAnnote for advanced speaker identification"
        else:
            return "Using basic speaker identification (Add PyAnnote token in Settings for better results)"
            
    def update_speaker_id_button_style(self):
        """Update the style of the speaker identification button based on token availability."""
        if hasattr(self, 'identify_speakers_btn'):
            has_token = bool(self.config_manager.get_pyannote_token())
            if has_token:
                self.identify_speakers_btn.SetBackgroundColour(wx.Colour(50, 200, 50))
            else:
                self.identify_speakers_btn.SetBackgroundColour(wx.NullColour)
    
    def check_api_key(self):
        """Check if Azure OpenAI API key and endpoint are available and initialize the client."""
        api_key = self.config_manager.get_azure_api_key()
        endpoint = self.config_manager.get_azure_endpoint()
        
        if api_key and endpoint and endpoint != "https://your-resource-name.openai.azure.com":
            try:
                self.client = AzureOpenAI(
                    api_key=api_key,
                    api_version=self.config_manager.get_azure_api_version(),
                    azure_endpoint=endpoint
                )
                self.status_bar.SetStatusText("Azure OpenAI configuration loaded")
                return
            except Exception as e:
                print(f"Error loading Azure OpenAI configuration: {e}")
        
        # If API key or endpoint is not in config or invalid, show the configuration dialog
        self.show_azure_config_dialog()

    def show_azure_config_dialog(self):
        """Show dialog to enter Azure OpenAI configuration."""
        dlg = wx.Dialog(self, title="Azure OpenAI Configuration Required", size=(400, 200))
        panel = wx.Panel(dlg)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # API Key input
        key_label = wx.StaticText(panel, label="Azure OpenAI API Key:")
        key_input = wx.TextCtrl(panel, style=wx.TE_PASSWORD)
        key_input.SetValue(self.config_manager.get_azure_api_key())
        
        # Endpoint input
        endpoint_label = wx.StaticText(panel, label="Azure OpenAI Endpoint:")
        endpoint_input = wx.TextCtrl(panel)
        endpoint_input.SetValue(self.config_manager.get_azure_endpoint())
        
        # Add to sizer
        sizer.Add(key_label, 0, wx.ALL, 5)
        sizer.Add(key_input, 0, wx.EXPAND | wx.ALL, 5)
        sizer.Add(endpoint_label, 0, wx.ALL, 5)
        sizer.Add(endpoint_input, 0, wx.EXPAND | wx.ALL, 5)
        
        # Buttons
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        ok_btn = wx.Button(panel, wx.ID_OK, "OK")
        cancel_btn = wx.Button(panel, wx.ID_CANCEL, "Cancel")
        btn_sizer.Add(ok_btn, 0, wx.ALL, 5)
        btn_sizer.Add(cancel_btn, 0, wx.ALL, 5)
        sizer.Add(btn_sizer, 0, wx.ALIGN_CENTER | wx.ALL, 5)
        
        panel.SetSizer(sizer)
        
        if dlg.ShowModal() == wx.ID_OK:
            api_key = key_input.GetValue().strip()
            endpoint = endpoint_input.GetValue().strip()
            if api_key and endpoint:
                # Save the configuration
                self.config_manager.set_azure_api_key(api_key)
                self.config_manager.set_azure_endpoint(endpoint)
                
                try:
                    self.client = AzureOpenAI(
                        api_key=api_key,
                        api_version=self.config_manager.get_azure_api_version(),
                        azure_endpoint=endpoint
                    )
                    self.status_bar.SetStatusText("Azure OpenAI configuration saved")
                except Exception as e:
                    wx.MessageBox(f"Error initializing Azure OpenAI client: {e}", "Error", wx.OK | wx.ICON_ERROR)
                    self.show_azure_config_dialog()
            else:
                wx.MessageBox("Both API Key and Endpoint are required to use this application.", "Error", wx.OK | wx.ICON_ERROR)
                self.show_azure_config_dialog()
        else:
            wx.MessageBox("Azure OpenAI configuration is required to use this application.", "Error", wx.OK | wx.ICON_ERROR)
            self.show_azure_config_dialog()
        dlg.Destroy()
    
    def init_ui(self):
        # Create status bar
        self.status_bar = self.CreateStatusBar()
        self.status_bar.SetStatusText("Ready")
        
        # Create menu bar
        menu_bar = wx.MenuBar()
        
        # File menu
        file_menu = wx.Menu()
        
        # Audio submenu
        audio_menu = wx.Menu()
        upload_audio_item = audio_menu.Append(wx.ID_ANY, "&Upload Audio File", "Upload audio file for transcription")
        self.Bind(wx.EVT_MENU, self.on_upload_audio, upload_audio_item)
        
        file_menu.AppendSubMenu(audio_menu, "&Audio")
        
        # Document submenu
        doc_menu = wx.Menu()
        upload_doc_item = doc_menu.Append(wx.ID_ANY, "&Upload Document", "Upload document for LLM context")
        select_docs_item = doc_menu.Append(wx.ID_ANY, "&Select Documents", "Select documents to load into context")
        
        self.Bind(wx.EVT_MENU, self.on_upload_document, upload_doc_item)
        self.Bind(wx.EVT_MENU, self.on_select_documents, select_docs_item)
        
        file_menu.AppendSubMenu(doc_menu, "&Documents")
        
        # Settings menu item
        settings_item = file_menu.Append(wx.ID_ANY, "&Settings", "Application settings")
        self.Bind(wx.EVT_MENU, self.on_settings, settings_item)
        
        # Exit menu item
        exit_item = file_menu.Append(wx.ID_EXIT, "E&xit", "Exit application")
        self.Bind(wx.EVT_MENU, self.on_exit, exit_item)
        
        menu_bar.Append(file_menu, "&File")
        
        # Help menu
        help_menu = wx.Menu()
        about_item = help_menu.Append(wx.ID_ABOUT, "&About", "About this application")
        self.Bind(wx.EVT_MENU, self.on_about, about_item)
        
        menu_bar.Append(help_menu, "&Help")
        
        self.SetMenuBar(menu_bar)
        
        # Main panel with notebook
        self.panel = wx.Panel(self)
        self.notebook = wx.Notebook(self.panel)
        
        # Chat tab
        self.chat_tab = wx.Panel(self.notebook)
        chat_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Chat history
        self.chat_display = wx.TextCtrl(self.chat_tab, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2)
        
        # Input area
        input_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.chat_input = wx.TextCtrl(self.chat_tab, style=wx.TE_MULTILINE)
        send_button = wx.Button(self.chat_tab, label="Send")
        
        input_sizer.Add(self.chat_input, proportion=1, flag=wx.EXPAND | wx.RIGHT, border=5)
        input_sizer.Add(send_button, proportion=0, flag=wx.EXPAND)
        
        chat_sizer.Add(self.chat_display, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
        chat_sizer.Add(input_sizer, proportion=0, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, border=5)
        
        self.chat_tab.SetSizer(chat_sizer)
        
        # Transcription tab
        self.transcription_tab = wx.Panel(self.notebook)
        transcription_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Transcription display
        self.transcription_display = wx.TextCtrl(self.transcription_tab, style=wx.TE_MULTILINE | wx.TE_RICH2)
        
        # Speaker panel
        speaker_panel = wx.Panel(self.transcription_tab)
        speaker_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.speaker_list = wx.ListCtrl(speaker_panel, style=wx.LC_REPORT)
        self.speaker_list.InsertColumn(0, "Speaker")
        self.speaker_list.InsertColumn(1, "Name")
        
        speaker_button_sizer = wx.BoxSizer(wx.VERTICAL)
        rename_speaker_button = wx.Button(speaker_panel, label="Rename Speaker")
        regenerate_button = wx.Button(speaker_panel, label="Regenerate Transcript")
        
        speaker_button_sizer.Add(rename_speaker_button, flag=wx.EXPAND | wx.BOTTOM, border=5)
        speaker_button_sizer.Add(regenerate_button, flag=wx.EXPAND)
        
        speaker_sizer.Add(self.speaker_list, proportion=1, flag=wx.EXPAND | wx.RIGHT, border=5)
        speaker_sizer.Add(speaker_button_sizer, proportion=0, flag=wx.EXPAND)
        
        speaker_panel.SetSizer(speaker_sizer)
        
        # Summarization panel
        summary_panel = wx.Panel(self.transcription_tab)
        summary_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        templates_label = wx.StaticText(summary_panel, label="Template:")
        self.templates_combo = wx.ComboBox(summary_panel, choices=["Meeting Notes", "Interview Summary", "Lecture Notes"])
        summarize_button = wx.Button(summary_panel, label="Summarize")
        
        summary_sizer.Add(templates_label, flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=5)
        summary_sizer.Add(self.templates_combo, proportion=1, flag=wx.EXPAND | wx.RIGHT, border=5)
        summary_sizer.Add(summarize_button, proportion=0, flag=wx.EXPAND)
        
        summary_panel.SetSizer(summary_sizer)
        
        transcription_sizer.Add(self.transcription_display, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
        transcription_sizer.Add(speaker_panel, proportion=0, flag=wx.EXPAND | wx.LEFT | wx.RIGHT, border=5)
        transcription_sizer.Add(summary_panel, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)
        
        self.transcription_tab.SetSizer(transcription_sizer)
        
        # Settings tab (NEW)
        self.settings_tab = wx.Panel(self.notebook)
        settings_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # API Keys section
        api_box = wx.StaticBox(self.settings_tab, label="API Keys")
        api_box_sizer = wx.StaticBoxSizer(api_box, wx.VERTICAL)
        
        # Azure OpenAI API Key
        azure_sizer = wx.BoxSizer(wx.HORIZONTAL)
        azure_label = wx.StaticText(self.settings_tab, label="Azure OpenAI API Key:")
        self.azure_api_key_input = wx.TextCtrl(self.settings_tab, value=self.api_key, style=wx.TE_PASSWORD)
        
        azure_sizer.Add(azure_label, flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=5)
        azure_sizer.Add(self.azure_api_key_input, proportion=1)
        
        # HuggingFace API Key
        hf_sizer = wx.BoxSizer(wx.HORIZONTAL)
        hf_label = wx.StaticText(self.settings_tab, label="HuggingFace Token:")
        self.hf_input = wx.TextCtrl(self.settings_tab, value=self.hf_token, style=wx.TE_PASSWORD)
        
        hf_sizer.Add(hf_label, flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=5)
        hf_sizer.Add(self.hf_input, proportion=1)
        
        api_box_sizer.Add(azure_sizer, flag=wx.EXPAND | wx.ALL, border=5)
        api_box_sizer.Add(hf_sizer, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, border=5)
        
        # Language settings section
        lang_box = wx.StaticBox(self.settings_tab, label="Language Settings")
        lang_box_sizer = wx.StaticBoxSizer(lang_box, wx.VERTICAL)
        
        lang_sizer = wx.BoxSizer(wx.HORIZONTAL)
        lang_label = wx.StaticText(self.settings_tab, label="Transcription Language:")
        self.lang_combo = wx.ComboBox(self.settings_tab, 
                                     choices=[LANGUAGE_DISPLAY_NAMES[lang] for lang in SUPPORTED_LANGUAGES],
                                     style=wx.CB_READONLY)
        
        # Set initial selection based on saved language
        current_lang = self.config_manager.get_language()
        self.lang_combo.SetSelection(SUPPORTED_LANGUAGES.index(current_lang))
        
        lang_sizer.Add(lang_label, flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=5)
        lang_sizer.Add(self.lang_combo, proportion=1)
        
        lang_box_sizer.Add(lang_sizer, flag=wx.EXPAND | wx.ALL, border=5)
        
        # Save button for settings
        save_button = wx.Button(self.settings_tab, label="Save Settings")
        save_button.Bind(wx.EVT_BUTTON, self.on_save_settings)
        
        # Add all sections
        settings_sizer.Add(api_box_sizer, flag=wx.EXPAND | wx.ALL, border=10)
        settings_sizer.Add(lang_box_sizer, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, border=10)
        settings_sizer.Add(save_button, flag=wx.ALIGN_RIGHT | wx.LEFT | wx.RIGHT | wx.BOTTOM, border=10)
        
        self.settings_tab.SetSizer(settings_sizer)
        
        # Add tabs to notebook
        self.notebook.AddPage(self.chat_tab, "Chat")
        self.notebook.AddPage(self.transcription_tab, "Transcription")
        self.notebook.AddPage(self.settings_tab, "Settings")
        
        # Main sizer
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(self.notebook, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
        
        self.panel.SetSizer(main_sizer)
        
        # Bind events
        send_button.Bind(wx.EVT_BUTTON, self.on_send_message)
        self.chat_input.Bind(wx.EVT_KEY_DOWN, self.on_key_down)
        rename_speaker_button.Bind(wx.EVT_BUTTON, self.on_rename_speaker)
        regenerate_button.Bind(wx.EVT_BUTTON, self.on_regenerate_transcript)
        summarize_button.Bind(wx.EVT_BUTTON, self.on_summarize)
    
    def on_key_down(self, event):
        key_code = event.GetKeyCode()
        if key_code == wx.WXK_RETURN and event.ShiftDown():
            # Allow Shift+Enter to insert a newline
            event.Skip()
        elif key_code == wx.WXK_RETURN:
            # Enter key sends the message
            self.on_send_message(event)
        else:
            event.Skip()
    
    def on_send_message(self, event):
        """Handle sending a message in the chat."""
        user_input = self.user_input.GetValue()
        if not user_input:
            return
            
        # Generate response
        response = self.llm_processor.generate_response(user_input)
        
        # Update chat history
        self.chat_history_text.AppendText(f"You: {user_input}\n")
        self.chat_history_text.AppendText(f"Assistant: {response}\n\n")
        
        # Clear user input
        self.user_input.SetValue("")
        
    def on_clear_chat_history(self, event):
        """Clear the chat history."""
        self.llm_processor.clear_chat_history()
        self.chat_history_text.SetValue("")
        
    def on_save_api_key(self, event):
        """Save the Azure OpenAI API key."""
        api_key = self.azure_api_key_input.GetValue()
        endpoint = self.azure_endpoint_input.GetValue()
        
        self.config_manager.set_azure_api_key(api_key)
        self.config_manager.set_azure_endpoint(endpoint)
        
        # Reinitialize the client with new settings
        self.initialize_openai_client()
        
        wx.MessageBox("Azure OpenAI settings saved successfully.", "Success", wx.OK | wx.ICON_INFORMATION)
    
    def on_save_pyannote_token(self, event):
        """Save the PyAnnote token."""
        token = self.pyannote_token_input.GetValue()
        self.config_manager.set_pyannote_token(token)
        
        # Update the speaker identification button style
        self.identify_speakers_btn.SetLabel(self.get_speaker_id_button_label())
        self.speaker_id_help_text.SetLabel(self.get_speaker_id_help_text())
        self.update_speaker_id_button_style()
        self.audio_panel.Layout()
        
        wx.MessageBox("PyAnnote token saved successfully.", "Success", wx.OK | wx.ICON_INFORMATION)
        
    def on_save_model(self, event):
        """Save the selected Azure OpenAI deployment."""
        chat_deployment = self.chat_deployment_input.GetValue()
        whisper_deployment = self.whisper_deployment_input.GetValue()
        
        self.config_manager.set_azure_deployment("chat", chat_deployment)
        self.config_manager.set_azure_deployment("whisper", whisper_deployment)
        
        wx.MessageBox("Azure OpenAI deployments saved successfully.", "Success", wx.OK | wx.ICON_INFORMATION)
        
    def on_save_temperature(self, event):
        """Save the temperature value."""
        temperature = self.temperature_slider.GetValue() / 10.0
        self.config_manager.set_temperature(temperature)
        wx.MessageBox("Temperature saved successfully.", "Success", wx.OK | wx.ICON_INFORMATION)
        
    def on_save_language(self, event):
        """Save the selected language."""
        language = self.language_settings_choice.GetString(self.language_settings_choice.GetSelection()).lower()
        self.config_manager.set_language(language)
        wx.MessageBox("Language saved successfully.", "Success", wx.OK | wx.ICON_INFORMATION)
        
    def populate_template_list(self):
        """Populate the template list with available templates."""
        if not hasattr(self, 'template_list'):
            return
            
        self.template_list.Clear()
        templates = self.config_manager.get_templates()
        for name in templates.keys():
            self.template_list.Append(name)
            
    def on_add_template(self, event):
        """Add a new template."""
        name = self.template_name_input.GetValue()
        content = self.template_content_input.GetValue()
        
        if not name or not content:
            wx.MessageBox("Please enter both name and content for the template.", "Error", wx.OK | wx.ICON_ERROR)
            return
            
        self.config_manager.add_template(name, content)
        self.populate_template_list()
        self.template_name_input.SetValue("")
        self.template_content_input.SetValue("")
        
    def on_remove_template(self, event):
        """Remove the selected template."""
        if not hasattr(self, 'template_list'):
            return
            
        selected = self.template_list.GetSelection()
        if selected == wx.NOT_FOUND:
            wx.MessageBox("Please select a template to remove.", "Error", wx.OK | wx.ICON_ERROR)
            return
            
        template_name = self.template_list.GetString(selected)
        
        # Confirm deletion
        dlg = wx.MessageDialog(self, f"Are you sure you want to delete the template '{template_name}'?",
                              "Confirm Deletion", wx.YES_NO | wx.ICON_QUESTION)
        if dlg.ShowModal() == wx.ID_YES:
            # Delete template
            self.config_manager.remove_template(template_name)
            
            # Update lists
            self.populate_template_list()
            
            # Update template choice in audio panel
            templates = list(self.config_manager.get_templates().keys())
            self.template_choice.SetItems(["None"] + templates)
            self.template_choice.SetSelection(0)
        
        dlg.Destroy()
    
    def on_upload_audio(self, event):
        # Check if PyAudio is available
        if not PYAUDIO_AVAILABLE:
            wx.MessageBox("PyAudio is not available. Recording functionality will be limited.", 
                         "PyAudio Missing", wx.OK | wx.ICON_WARNING)
        
        # File dialog to select audio file - fix wildcard and dialog settings
        wildcard = "Audio files (*.mp3;*.wav;*.m4a)|*.mp3;*.wav;*.m4a|All files (*.*)|*.*"
        with wx.FileDialog(
            self, 
            message="Choose an audio file",
            defaultDir=os.path.expanduser("~"),  # Start in user's home directory
            defaultFile="",
            wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
        ) as file_dialog:
            
            # Show the dialog and check if user clicked OK
            if file_dialog.ShowModal() == wx.ID_CANCEL:
                return  # User cancelled the dialog
            
            # Get the selected file path
            audio_path = file_dialog.GetPath()
            self.last_audio_path = audio_path  # Store for potential retry
            
            # Show a message that we're processing
            wx.MessageBox(f"Selected file: {audio_path}\n\nStarting transcription...", 
                         "Transcription Started", wx.OK | wx.ICON_INFORMATION)
            
            # Disable UI during processing
            self.notebook.Disable()
            self.status_bar.SetStatusText(f"Transcribing audio...")
            
            # Start transcription in a thread
            threading.Thread(target=self.transcribe_audio, args=(audio_path,), daemon=True).start()
    
    
    def _fallback_speaker_detection(self):
        """Use a basic approach to detect speakers when diarization is not available"""
        paragraphs = self.transcript.split("\n\n")
        speaker_count = min(len(paragraphs), 3)
        
        self.speakers = []
        self.speaker_names = {}
        
        # Use language-appropriate speaker names
        current_lang = self.config_manager.get_language()
        if current_lang == "hu":
            speaker_prefix = "Beszélő"  # Hungarian for "Speaker"
        else:
            speaker_prefix = "Speaker"
            
        for i in range(speaker_count):
            speaker_id = f"{speaker_prefix} {i+1}"
            self.speakers.append(speaker_id)
            self.speaker_names[speaker_id] = speaker_id
    
    def combine_transcript_with_speakers(self, whisper_response, speaker_segments):
        """
        Combine the word-level transcription from Whisper with speaker information from diarization.
        
        Args:
            whisper_response: The response from Whisper API with timestamps
            speaker_segments: Dictionary of speaker segments {speaker_id: [(start_time, end_time), ...]}
            
        Returns:
            Formatted transcript with speaker labels
        """
        try:
            # Get words with timestamps from Whisper response
            segments = whisper_response.segments
            
            # Build a new transcript with speaker information
            formatted_lines = []
            current_speaker = None
            current_line = []
            
            for segment in segments:
                for word_info in segment.words:
                    word = word_info.word
                    start_time = word_info.start
                    
                    # Find which speaker was talking at this time
                    speaker_at_time = None
                    for speaker, time_segments in speaker_segments.items():
                        for start, end in time_segments:
                            if start <= start_time <= end:
                                speaker_at_time = speaker
                                break
                        if speaker_at_time:
                            break
                    
                    # If no speaker found or couldn't determine, use the first speaker
                    if not speaker_at_time and self.speakers:
                        speaker_at_time = self.speakers[0]
                    
                    # Start a new line if the speaker changes
                    if speaker_at_time != current_speaker:
                        if current_line:
                            formatted_lines.append(f"{self.speaker_names.get(current_speaker, 'Unknown')}: {' '.join(current_line)}")
                            current_line = []
                        current_speaker = speaker_at_time
                    
                    # Add the word to the current line
                    current_line.append(word)
            
            # Add the last line
            if current_line:
                formatted_lines.append(f"{self.speaker_names.get(current_speaker, 'Unknown')}: {' '.join(current_line)}")
            
            return "\n\n".join(formatted_lines)
            
        except Exception as e:
            print(f"Error combining transcript with speakers: {e}")
            return whisper_response.text  # Fall back to the original transcript
    
    def show_hf_token_dialog(self):
        # Localize dialog text based on language
        current_lang = self.config_manager.get_language()
        if current_lang == "hu":
            dialog_title = "HuggingFace Token Szükséges"
            dialog_message = "Kérjük, add meg a HuggingFace hozzáférési tokened a beszélők azonosításához:\n" \
                            "(Szerezz egyet innen: https://huggingface.co/settings/tokens)"
            error_message = "A HuggingFace token szükséges a beszélők azonosításához."
        else:
            dialog_title = "HuggingFace Token Required"
            dialog_message = "Please enter your HuggingFace Access Token for speaker identification:\n" \
                            "(You can get one from https://huggingface.co/settings/tokens)"
            error_message = "HuggingFace token is required for speaker identification."
        
        dialog = wx.TextEntryDialog(
            self, 
            dialog_message,
            dialog_title
        )
        
        if dialog.ShowModal() == wx.ID_OK:
            self.hf_token = dialog.GetValue().strip()
            if self.hf_token:
                # Save the token to environment
                os.environ["HF_TOKEN"] = self.hf_token
                
                # Save to config manager
                self.config_manager.set_pyannote_token(self.hf_token)
                
                # Update the speaker identification button style if it exists
                if hasattr(self, 'identify_speakers_btn') and self.identify_speakers_btn:
                    self.identify_speakers_btn.SetLabel(self.get_speaker_id_button_label())
                    self.speaker_id_help_text.SetLabel(self.get_speaker_id_help_text())
                    self.update_speaker_id_button_style()
                    self.audio_panel.Layout()
                
                # Retry transcription
                self.notebook.Disable()
                self.status_bar.SetStatusText("Retrying transcription...")
                threading.Thread(target=self.transcribe_audio, args=(self.last_audio_path,), daemon=True).start()
            else:
                self.show_error(error_message)
                self.notebook.Enable()
        else:
            self.show_error(error_message)
            self.notebook.Enable()
        
        dialog.Destroy()
    
    def update_transcript_display(self):
        if hasattr(self, 'transcript_text'):
            self.transcript_text.Clear()
            
            # Set the transcript to the display control
            self.transcript_text.SetValue(self.transcript)
    
    def update_speaker_list(self):
        self.speaker_list.DeleteAllItems()
        
        if hasattr(self, 'speakers') and self.speakers:
            # Initialize speaker_names dictionary if it doesn't exist
            if not hasattr(self, 'speaker_names'):
                self.speaker_names = {}
            
            # Track unique speaker IDs to avoid duplicates
            unique_speakers = set()
            
            for i, speaker_data in enumerate(self.speakers):
                # Extract speaker ID from the dictionary format
                if isinstance(speaker_data, dict) and "speaker" in speaker_data:
                    speaker_id = speaker_data["speaker"]
                else:
                    speaker_id = speaker_data
                
                # Skip duplicate speaker IDs
                if speaker_id in unique_speakers:
                    continue
                unique_speakers.add(speaker_id)
                
                # Ensure this speaker is in the speaker_names dictionary
                if speaker_id not in self.speaker_names:
                    self.speaker_names[speaker_id] = speaker_id
                
                # Add to list control
                index = self.speaker_list.InsertItem(i, speaker_id)
                self.speaker_list.SetItem(index, 1, self.speaker_names.get(speaker_id, speaker_id))
    
    def on_rename_speaker(self, event):
        # Get selected speaker
        selected = self.speaker_list.GetFirstSelected()
        if selected == -1:
            self.show_error("Please select a speaker to rename")
            return
        
        speaker_id = self.speaker_list.GetItemText(selected, 0)
        current_name = self.speaker_list.GetItemText(selected, 1)
        
        # Show dialog to get new name
        dialog = wx.TextEntryDialog(self, f"Enter new name for {speaker_id}:", "Rename Speaker", value=current_name)
        if dialog.ShowModal() == wx.ID_OK:
            new_name = dialog.GetValue().strip()
            if new_name:
                # Update speaker name
                self.speaker_names[speaker_id] = new_name
                self.speaker_list.SetItem(selected, 1, new_name)
                
                # Update any other instances of this speaker in the speaker list
                for i in range(self.speaker_list.GetItemCount()):
                    if i != selected and self.speaker_list.GetItemText(i, 0) == speaker_id:
                        self.speaker_list.SetItem(i, 1, new_name)
                
                # Automatically update the transcript with the new speaker name
                if hasattr(self, 'transcript') and self.transcript:
                    new_transcript = self.transcript
                    new_transcript = new_transcript.replace(f"{current_name}:", f"{new_name}:")
                    self.transcript = new_transcript
                    self.update_transcript_display()
                    self.status_bar.SetStatusText(f"Speaker '{speaker_id}' renamed to '{new_name}'")
        dialog.Destroy()
    
    def on_regenerate_transcript(self, event):
        if not self.transcript or not self.speakers:
            self.show_error("No transcript available to regenerate")
            return
        
        # In a real implementation, you would regenerate the transcript with proper speaker names
        # For now, let's just simulate it by replacing "Speaker X" with the assigned names
        new_transcript = self.transcript
        for speaker_id, name in self.speaker_names.items():
            if speaker_id != name:  # Only replace if name has been changed
                new_transcript = new_transcript.replace(speaker_id, name)
        
        self.transcript = new_transcript
        self.update_transcript_display()
        self.status_bar.SetStatusText("Transcript regenerated with speaker names")
    
    def on_summarize(self, event):
        """Generate a summary of the transcript."""
        if not self.audio_processor.transcript:
            wx.MessageBox("Please transcribe an audio file first.", "No Transcript", wx.OK | wx.ICON_INFORMATION)
            return
            
        # Get selected template
        template_idx = self.template_choice.GetSelection()
        template_name = None
        if template_idx > 0:  # 0 is "None"
            template_name = self.template_choice.GetString(template_idx)
            
        # Disable button during processing
        self.summarize_btn.Disable()
        
        # Start summarization in a separate thread
        transcript = self.transcript_text.GetValue()
        threading.Thread(target=self.summarize_thread, args=(transcript, template_name)).start()
        
    def summarize_thread(self, transcript, template_name):
        """Thread function for transcript summarization."""
        try:
            summary = self.llm_processor.summarize_transcript(transcript, template_name)
            
            # Show summary in a dialog
            wx.CallAfter(self.show_summary_dialog, summary)
        except Exception as e:
            wx.CallAfter(wx.MessageBox, f"Summarization error: {str(e)}", "Error", wx.OK | wx.ICON_ERROR)
        finally:
            wx.CallAfter(self.summarize_btn.Enable)
            
    def show_summary_dialog(self, summary):
        """Show summary in a dialog."""
        dlg = wx.Dialog(self, title="Summary", size=(600, 400))
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        text_ctrl = wx.TextCtrl(dlg, style=wx.TE_MULTILINE | wx.TE_READONLY)
        text_ctrl.SetValue(summary)
        
        sizer.Add(text_ctrl, 1, wx.EXPAND | wx.ALL, 10)
        
        # Add Close button
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        close_btn = wx.Button(dlg, wx.ID_CLOSE)
        btn_sizer.Add(close_btn, 0, wx.ALL, 5)
        
        sizer.Add(btn_sizer, 0, wx.ALIGN_CENTER | wx.ALL, 5)
        
        dlg.SetSizer(sizer)
        
        close_btn.Bind(wx.EVT_BUTTON, lambda event: dlg.EndModal(wx.ID_CLOSE))
        
        dlg.ShowModal()
        dlg.Destroy()
        
    def update_button_states(self):
        """Update the enabled/disabled states of buttons based on current state."""
        has_audio_file = bool(self.audio_file_path.GetValue())
        has_transcript = hasattr(self.audio_processor, 'transcript') and bool(self.audio_processor.transcript)
        has_speakers = hasattr(self.audio_processor, 'speakers') and bool(self.audio_processor.speakers)
        
        if hasattr(self, 'transcribe_btn'):
            self.transcribe_btn.Enable(has_audio_file)
            
        if hasattr(self, 'identify_speakers_btn'):
            self.identify_speakers_btn.Enable(has_transcript)
            
        if hasattr(self, 'apply_speaker_names_btn'):
            self.apply_speaker_names_btn.Enable(has_speakers)
            
        if hasattr(self, 'summarize_btn'):
            self.summarize_btn.Enable(has_transcript)
    
    def on_upload_document(self, event):
        # Improved file dialog to select document
        wildcard = "Text files (*.txt)|*.txt|PDF files (*.pdf)|*.pdf|All files (*.*)|*.*"
        with wx.FileDialog(
            self, 
            message="Choose a document to add",
            defaultDir=os.path.expanduser("~"),  # Start in user's home directory
            defaultFile="",
            wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
        ) as file_dialog:
            
            # Show the dialog and check if user clicked OK
            if file_dialog.ShowModal() == wx.ID_CANCEL:
                return  # User cancelled the dialog
            
            # Get the selected file path
            doc_path = file_dialog.GetPath()
            filename = os.path.basename(doc_path)
            dest_path = os.path.join(self.documents_folder, filename)
            
            # Check if file already exists
            if os.path.exists(dest_path):
                dialog = wx.MessageDialog(self, f"File {filename} already exists. Replace it?",
                                         "File exists", wx.YES_NO | wx.ICON_QUESTION)
                if dialog.ShowModal() == wx.ID_NO:
                    dialog.Destroy()
                    return
                dialog.Destroy()
            
            # Copy file to documents folder
            try:
                shutil.copy2(doc_path, dest_path)
                self.status_bar.SetStatusText(f"Document {filename} uploaded")
                
                # Show success message
                wx.MessageBox(f"Document '{filename}' has been successfully added.", 
                             "Document Added", wx.OK | wx.ICON_INFORMATION)
            except Exception as e:
                self.show_error(f"Error uploading document: {str(e)}")
    
    def on_select_documents(self, event):
        # Get list of documents
        try:
            files = os.listdir(self.documents_folder)
            files = [f for f in files if os.path.isfile(os.path.join(self.documents_folder, f))]
        except Exception as e:
            self.show_error(f"Error listing documents: {str(e)}")
            return
        
        if not files:
            self.show_error("No documents found. Please upload documents first.")
            return
        
        # Create a dialog with checkboxes for each document
        dialog = wx.Dialog(self, title="Select Documents", size=(400, 300))
        panel = wx.Panel(dialog)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Instruction text
        instructions = wx.StaticText(panel, label="Select documents to load into context:")
        sizer.Add(instructions, flag=wx.ALL, border=10)
        
        # Checkboxes for each document
        checkboxes = {}
        for filename in files:
            checkbox = wx.CheckBox(panel, label=filename)
            checkbox.SetValue(filename in self.loaded_documents)
            checkboxes[filename] = checkbox
            sizer.Add(checkbox, flag=wx.ALL, border=5)
        
        # Buttons
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        ok_button = wx.Button(panel, wx.ID_OK, "OK")
        cancel_button = wx.Button(panel, wx.ID_CANCEL, "Cancel")
        
        button_sizer.Add(ok_button, flag=wx.RIGHT, border=5)
        button_sizer.Add(cancel_button)
        
        sizer.Add(button_sizer, flag=wx.ALIGN_RIGHT | wx.ALL, border=10)
        
        panel.SetSizer(sizer)
        
        # Show dialog
        if dialog.ShowModal() == wx.ID_OK:
            # Load selected documents
            selected = [filename for filename, checkbox in checkboxes.items() if checkbox.GetValue()]
            
            # Clear previously loaded documents
            self.loaded_documents = {}
            
            # Load new selections
            for filename in selected:
                try:
                    with open(os.path.join(self.documents_folder, filename), 'r', encoding='utf-8') as f:
                        self.loaded_documents[filename] = f.read()
                except Exception as e:
                    self.show_error(f"Error loading {filename}: {str(e)}")
            
            self.status_bar.SetStatusText(f"Loaded {len(self.loaded_documents)} documents")
        
        dialog.Destroy()
    
    def on_settings(self, event):
        # Switch to the Settings tab
        self.notebook.SetSelection(2)  # Index 2 is the Settings tab
    
    def on_exit(self, event):
        self.Close()
    
    def on_about(self, event):
        info = wx.adv.AboutDialogInfo()
        info.SetName("AI Assistant")
        info.SetVersion("1.0")
        info.SetDescription("An AI assistant application for transcription, summarization, and document processing.")
        info.SetCopyright("(C) 2023")
        
        wx.adv.AboutBox(info)
    
    def show_error(self, message):
        wx.MessageBox(message, "Error", wx.OK | wx.ICON_ERROR)

    def on_save_settings(self, event):
        """Save settings from the Settings tab"""
        # Save Azure OpenAI settings
        new_api_key = self.azure_api_key_input.GetValue().strip()
        new_endpoint = self.azure_endpoint_input.GetValue().strip()
        new_hf_token = self.hf_input.GetValue().strip()
        
        # Get language selection
        lang_selection = self.lang_combo.GetSelection()
        if 0 <= lang_selection < len(SUPPORTED_LANGUAGES):
            new_language = SUPPORTED_LANGUAGES[lang_selection]  # Get language code directly from selection
        else:
            new_language = "en"  # Default to English if selection is invalid
        
        # Update Azure OpenAI settings if changed
        if new_api_key != self.config_manager.get_azure_api_key() or new_endpoint != self.config_manager.get_azure_endpoint():
            os.environ["AZURE_OPENAI_API_KEY"] = new_api_key
            os.environ["AZURE_OPENAI_ENDPOINT"] = new_endpoint
            self.config_manager.set_azure_api_key(new_api_key)
            self.config_manager.set_azure_endpoint(new_endpoint)
            
            # Update client
            if new_api_key and new_endpoint:
                try:
                    self.client = AzureOpenAI(
                        api_key=new_api_key,
                        api_version=self.config_manager.get_azure_api_version(),
                        azure_endpoint=new_endpoint
                    )
                except Exception as e:
                    self.show_error(f"Error setting Azure OpenAI client: {e}")
        
        # Update HuggingFace token if changed
        if new_hf_token != self.hf_token:
            self.hf_token = new_hf_token
            os.environ["HF_TOKEN"] = self.hf_token
            self.config_manager.set_pyannote_token(new_hf_token)
            
            # Update the speaker identification button style
            if hasattr(self, 'identify_speakers_btn') and self.identify_speakers_btn:
                self.identify_speakers_btn.SetLabel(self.get_speaker_id_button_label())
                self.speaker_id_help_text.SetLabel(self.get_speaker_id_help_text())
                self.update_speaker_id_button_style()
                if hasattr(self, 'audio_panel'):
                    self.audio_panel.Layout()
        
        # Update language if changed
        if new_language != self.language:
            self.language = new_language
            os.environ["TRANSCRIPTION_LANGUAGE"] = self.language
            self.config_manager.set_language(new_language)
        
        wx.MessageBox("Settings saved successfully!", "Success", wx.OK | wx.ICON_INFORMATION)
        self.status_bar.SetStatusText("Settings saved successfully")

    def _identify_speakers_chunked(self, paragraphs, chunk_size):
        """Process long transcripts in chunks for speaker identification."""
        self.update_status("Processing transcript in chunks...", percent=0.1)
        
        # Group paragraphs into chunks
        chunks = []
        current_chunk = []
        current_length = 0
        
        for p in paragraphs:
            if current_length + len(p) > chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = [p]
                current_length = len(p)
            else:
                current_chunk.append(p)
                current_length += len(p)
                
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk)
            
        self.update_status(f"Processing transcript in {len(chunks)} chunks...", percent=0.15)
        
        # Process first chunk to establish speaker patterns
        model_to_use = self.config_manager.get_azure_deployment("chat")
        
        # Initialize result container
        all_results = []
        speaker_characteristics = {}
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            # Calculate progress percentage (0-1)
            progress = (i / len(chunks)) * 0.7 + 0.2  # 20% to 90% of total progress
            
            self.update_status(f"Processing chunk {i+1}/{len(chunks)}...", percent=progress)
            
            # For first chunk, get detailed analysis
            if i == 0:
                prompt = f"""
                Analyze this transcript segment and identify exactly two speakers (A and B).
                
                TASK:
                1. Determine which paragraphs belong to which speaker
                2. Identify each speaker's characteristics and speaking style
                3. Ensure logical conversation flow
                
                Return JSON in this exact format:
                {{
                    "analysis": {{
                        "speaker_a_characteristics": ["characteristic 1", "characteristic 2"],
                        "speaker_b_characteristics": ["characteristic 1", "characteristic 2"]
                    }},
                    "paragraphs": [
                        {{
                            "id": {len(all_results)},
                            "speaker": "A",
                            "text": "paragraph text"
                        }},
                        ...
                    ]
                }}
                
                Transcript paragraphs:
                {json.dumps([{"id": len(all_results) + j, "text": p} for j, p in enumerate(chunk)])}
                """
            else:
                # For subsequent chunks, use characteristics from first analysis
                prompt = f"""
                Continue assigning speakers to this transcript segment.
                
                Speaker A characteristics: {json.dumps(speaker_characteristics.get("speaker_a_characteristics", []))}
                Speaker B characteristics: {json.dumps(speaker_characteristics.get("speaker_b_characteristics", []))}
                
                Return JSON with speaker assignments:
                {{
                    "paragraphs": [
                        {{
                            "id": {len(all_results)},
                            "speaker": "A or B",
                            "text": "paragraph text"
                        }},
                        ...
                    ]
                }}
                
                Transcript paragraphs:
                {json.dumps([{"id": len(all_results) + j, "text": p} for j, p in enumerate(chunk)])}
                """
            
            # Make API call for this chunk
            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": "You are an expert conversation analyst who identifies speaker turns in transcripts with high accuracy."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Save speaker characteristics from first chunk
            if i == 0 and "analysis" in result:
                speaker_characteristics = result["analysis"]
            
            # Add results from this chunk
            if "paragraphs" in result:
                all_results.extend(result["paragraphs"])
            
            # Update progress
            after_progress = (i + 0.5) / len(chunks) * 0.7 + 0.2
            self.update_status(f"Processed chunk {i+1}/{len(chunks)}...", percent=after_progress)
        
        # Map Speaker A/B to Speaker 1/2
        speaker_map = {
            "A": "Speaker 1", 
            "B": "Speaker 2",
            "Speaker A": "Speaker 1", 
            "Speaker B": "Speaker 2"
        }
        
        self.update_status("Finalizing speaker assignments...", percent=0.95)
        
        # Create final speakers list
        self.speakers = []
        for item in sorted(all_results, key=lambda x: x.get("id", 0)):
            speaker_label = item.get("speaker", "Unknown")
            mapped_speaker = speaker_map.get(speaker_label, speaker_label)
            
            self.speakers.append({
                "speaker": mapped_speaker,
                "text": item.get("text", "")
            })
        
        # Ensure we have the right number of paragraphs
        if len(self.speakers) != len(paragraphs):
            self.update_status(f"Warning: Received {len(self.speakers)} segments but expected {len(paragraphs)}. Fixing...", percent=0.98)
            self.speakers = [
                {"speaker": self.speakers[min(i, len(self.speakers)-1)]["speaker"] if self.speakers else f"Speaker {i % 2 + 1}", 
                 "text": p}
                for i, p in enumerate(paragraphs)
            ]
        
        self.update_status(f"Speaker identification complete. Found 2 speakers across {len(chunks)} chunks.", percent=1.0)
        return self.speakers

    def identify_speakers_simple(self, transcript):
        """Identify speakers using a simplified and optimized approach."""
        self.update_status("Analyzing transcript for speaker identification...", percent=0.1)
        
        # First, split transcript into paragraphs
        paragraphs = self._create_improved_paragraphs(transcript)
        self.speaker_segments = paragraphs
        
        # Setup model
        model_to_use = self.config_manager.get_azure_deployment("chat")
        
        # For very long transcripts, we'll analyze in chunks
        MAX_CHUNK_SIZE = 8000  # characters per chunk
        
        if len(transcript) > MAX_CHUNK_SIZE:
            self.update_status("Long transcript detected. Processing in chunks...", percent=0.15)
            return self._identify_speakers_chunked(paragraphs, MAX_CHUNK_SIZE)
        
        # Enhanced single-pass approach for shorter transcripts
        prompt = f"""
        Analyze this transcript and identify exactly two speakers (A and B).
        
        TASK:
        1. Determine which paragraphs belong to which speaker
        2. Focus on conversation pattern and speaking style
        3. Ensure logical conversation flow (e.g., questions are followed by answers)
        4. Maintain consistency in first-person statements
        
        Return JSON in this exact format:
        {{
            "analysis": {{
                "speaker_a_characteristics": ["characteristic 1", "characteristic 2"],
                "speaker_b_characteristics": ["characteristic 1", "characteristic 2"],
                "speaker_count": 2,
                "conversation_type": "interview/discussion/etc"
            }},
            "paragraphs": [
                {{
                    "id": 0,
                    "speaker": "A",
                    "text": "paragraph text"
                }},
                ...
            ]
        }}
        
        Transcript paragraphs:
        {json.dumps([{"id": i, "text": p} for i, p in enumerate(paragraphs)])}
        """
        
        try:
            # Single API call to assign speakers
            self.update_status("Sending transcript for speaker analysis...", percent=0.3)
            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": "You are an expert conversation analyst who identifies speaker turns in transcripts with high accuracy."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            self.update_status("Processing speaker identification results...", percent=0.7)
            result = json.loads(response.choices[0].message.content)
            
            # Get paragraph assignments
            assignments = result.get("paragraphs", [])
            
            # Map Speaker A/B to Speaker 1/2 for compatibility with existing system
            speaker_map = {
                "A": "Speaker 1", 
                "B": "Speaker 2",
                "Speaker A": "Speaker 1", 
                "Speaker B": "Speaker 2"
            }
            
            # Create speakers list with proper mapping
            self.speakers = []
            for item in sorted(assignments, key=lambda x: x.get("id", 0)):
                speaker_label = item.get("speaker", "Unknown")
                mapped_speaker = speaker_map.get(speaker_label, speaker_label)
                
                self.speakers.append({
                    "speaker": mapped_speaker,
                    "text": item.get("text", "")
                })
            
            # Ensure we have the right number of paragraphs
            if len(self.speakers) != len(paragraphs):
                self.update_status(f"Warning: Received {len(self.speakers)} segments but expected {len(paragraphs)}. Fixing...", percent=0.9)
                self.speakers = [
                    {"speaker": self.speakers[min(i, len(self.speakers)-1)]["speaker"] if self.speakers else f"Speaker {i % 2 + 1}", 
                     "text": p}
                    for i, p in enumerate(paragraphs)
                ]
            
            self.update_status(f"Speaker identification complete. Found {2} speakers.", percent=1.0)
            return self.speakers
            
        except Exception as e:
            self.update_status(f"Error in speaker identification: {str(e)}", percent=0)
            # Fallback to basic alternating speaker assignment
            self.speakers = [
                {"speaker": f"Speaker {i % 2 + 1}", "text": p}
                for i, p in enumerate(paragraphs)
            ]
            return self.speakers
            
    def _create_improved_paragraphs(self, transcript):
        """Create more intelligent paragraph breaks based on semantic analysis."""
        import re
        # Split transcript into sentences
        sentences = re.split(r'(?<=[.!?])\s+', transcript.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Group sentences into paragraphs
        paragraphs = []
        current_para = []
        
        # These phrases often signal the start of a new speaker's turn
        new_speaker_indicators = [
            "yes", "no", "I think", "I believe", "so,", "well,", "actually", 
            "to be honest", "in my opinion", "I agree", "I disagree",
            "let me", "I'd like to", "I would", "you know", "um", "uh", 
            "hmm", "but", "however", "from my perspective", "wait", "okay",
            "right", "sure", "exactly", "absolutely", "definitely", "perhaps",
            "look", "listen", "basically", "frankly", "honestly", "now", "so",
            "thank you", "thanks", "good point", "interesting", "true", "correct",
            "first of all", "firstly", "secondly", "finally", "in conclusion"
        ]
        
        # Words/phrases that indicate continuation by the same speaker
        continuation_indicators = [
            "and", "also", "additionally", "moreover", "furthermore", "plus",
            "then", "after that", "next", "finally", "lastly", "in addition",
            "consequently", "as a result", "therefore", "thus", "besides",
            "for example", "specifically", "in particular", "especially",
            "because", "since", "due to", "as such", "which means"
        ]
        
        for i, sentence in enumerate(sentences):
            # Start a new paragraph if:
            start_new_para = False
            
            # 1. This is the first sentence
            if i == 0:
                start_new_para = True
                
            # 2. Previous sentence ended with a question mark
            elif i > 0 and sentences[i-1].endswith('?'):
                start_new_para = True
                
            # 3. Current sentence begins with a common new speaker phrase
            elif any(sentence.lower().startswith(indicator.lower()) for indicator in new_speaker_indicators):
                start_new_para = True
                
            # 4. Not a continuation and not a pronoun reference
            elif (i > 0 and 
                  not any(sentence.lower().startswith(indicator.lower()) for indicator in continuation_indicators) and
                  not re.match(r'^(It|This|That|These|Those|They|He|She|We|I)\b', sentence, re.IGNORECASE) and
                  len(current_para) >= 2):
                start_new_para = True
                
            # 5. Natural length limit to avoid overly long paragraphs
            elif len(current_para) >= 4:
                start_new_para = True
            
            # Start a new paragraph if needed
            if start_new_para and current_para:
                paragraphs.append(' '.join(current_para))
                current_para = []
            
            current_para.append(sentence)
        
        # Add the last paragraph
        if current_para:
            paragraphs.append(' '.join(current_para))
        
        return paragraphs

    def assign_speaker_names(self, speaker_map):
        """Apply custom speaker names to the transcript."""
        if not hasattr(self, 'speakers') or not self.speakers:
            return self.transcript
            
        # Create a formatted transcript with the new speaker names
        formatted_text = []
        
        for segment in self.speakers:
            original_speaker = segment.get("speaker", "Unknown")
            new_speaker = speaker_map.get(original_speaker, original_speaker)
            text = segment.get("text", "")
            
            formatted_text.append(f"{new_speaker}: {text}")
            
        return "\n\n".join(formatted_text)

    def create_audio_panel(self):
        """Create the audio processing panel."""
        panel = self.audio_panel
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # File upload section
        file_box = wx.StaticBox(panel, label="Audio File")
        file_sizer = wx.StaticBoxSizer(file_box, wx.VERTICAL)
        
        file_select_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.audio_file_path = wx.TextCtrl(panel, style=wx.TE_READONLY)
        browse_btn = wx.Button(panel, label="Browse")
        browse_btn.Bind(wx.EVT_BUTTON, self.on_browse_audio)
        
        file_select_sizer.Add(self.audio_file_path, proportion=1, flag=wx.EXPAND | wx.RIGHT, border=5)
        file_select_sizer.Add(browse_btn, proportion=0, flag=wx.EXPAND)
        
        file_sizer.Add(file_select_sizer, 1, wx.EXPAND | wx.ALL, 5)
        
        sizer.Add(file_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Create transcription controls
        transcribe_box = wx.StaticBox(panel, label="Transcription")
        transcribe_sizer = wx.StaticBoxSizer(transcribe_box, wx.VERTICAL)
        
        # Language selector
        lang_sizer = wx.BoxSizer(wx.HORIZONTAL)
        lang_label = wx.StaticText(panel, label="Language:")
        self.language_choice = wx.Choice(panel, choices=[LANGUAGE_DISPLAY_NAMES[lang] for lang in SUPPORTED_LANGUAGES])
        current_lang = self.config_manager.get_language()
        self.language_choice.SetSelection(SUPPORTED_LANGUAGES.index(current_lang))
        
        lang_sizer.Add(lang_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        lang_sizer.Add(self.language_choice, 1, wx.EXPAND)
        
        transcribe_sizer.Add(lang_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Transcribe button
        self.transcribe_btn = wx.Button(panel, label="Transcribe Audio")
        self.transcribe_btn.Bind(wx.EVT_BUTTON, self.on_transcribe)
        self.transcribe_btn.Disable()  # Start disabled
        transcribe_sizer.Add(self.transcribe_btn, 0, wx.EXPAND | wx.ALL, 5)
        
        sizer.Add(transcribe_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Speaker identification
        speaker_box = wx.StaticBox(panel, label="Speaker Identification")
        speaker_sizer = wx.StaticBoxSizer(speaker_box, wx.VERTICAL)
        
        self.identify_speakers_btn = wx.Button(panel, label=self.get_speaker_id_button_label())
        self.identify_speakers_btn.Bind(wx.EVT_BUTTON, self.on_identify_speakers)
        self.identify_speakers_btn.Disable()  # Start disabled
        
        # Set button styling based on PyAnnote availability
        self.update_speaker_id_button_style()
        
        speaker_sizer.Add(self.identify_speakers_btn, 0, wx.EXPAND | wx.ALL, 5)
        
        # Add help text
        self.speaker_id_help_text = wx.StaticText(panel, label=self.get_speaker_id_help_text())
        self.speaker_id_help_text.SetForegroundColour(wx.Colour(100, 100, 100))  # Gray text
        speaker_sizer.Add(self.speaker_id_help_text, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)
        
        sizer.Add(speaker_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Transcript display
        transcript_box = wx.StaticBox(panel, label="Transcript")
        transcript_sizer = wx.StaticBoxSizer(transcript_box, wx.VERTICAL)
        
        self.transcript_text = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2)
        self.transcript_text.SetMinSize((400, 200))
        transcript_sizer.Add(self.transcript_text, 1, wx.EXPAND | wx.ALL, 5)
        
        # Speaker list
        speaker_list_box = wx.StaticBox(panel, label="Speakers")
        speaker_list_sizer = wx.StaticBoxSizer(speaker_list_box, wx.VERTICAL)
        
        self.speaker_list = wx.ListCtrl(panel, style=wx.LC_REPORT)
        self.speaker_list.InsertColumn(0, "ID", width=50)
        self.speaker_list.InsertColumn(1, "Name", width=150)
        speaker_list_sizer.Add(self.speaker_list, 1, wx.EXPAND | wx.ALL, 5)
        
        # Edit speakers button
        rename_speaker_btn = wx.Button(panel, label="Rename Speaker")
        rename_speaker_btn.Bind(wx.EVT_BUTTON, self.on_rename_speaker)
        speaker_list_sizer.Add(rename_speaker_btn, 0, wx.EXPAND | wx.ALL, 5)
        
        # Create a horizontal sizer for transcript and speaker list
        transcript_speaker_sizer = wx.BoxSizer(wx.HORIZONTAL)
        transcript_speaker_sizer.Add(transcript_sizer, 2, wx.EXPAND | wx.RIGHT, 5)
        transcript_speaker_sizer.Add(speaker_list_sizer, 1, wx.EXPAND)
        
        sizer.Add(transcript_speaker_sizer, 1, wx.EXPAND | wx.ALL, 5)
        
        # Summarization section
        summary_box = wx.StaticBox(panel, label="Summarization")
        summary_sizer = wx.StaticBoxSizer(summary_box, wx.VERTICAL)
        
        # Template selection
        template_sizer = wx.BoxSizer(wx.HORIZONTAL)
        template_label = wx.StaticText(panel, label="Template:")
        
        # Get templates from config
        template_names = list(self.config_manager.get_templates().keys())
        self.template_choice = wx.Choice(panel, choices=["None"] + template_names)
        self.template_choice.SetSelection(0)
        self.template_choice.Bind(wx.EVT_CHOICE, self.on_template_selected)
        
        template_sizer.Add(template_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        template_sizer.Add(self.template_choice, 1, wx.EXPAND)
        
        summary_sizer.Add(template_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Summarize button
        self.summarize_btn = wx.Button(panel, label="Summarize Transcript")
        self.summarize_btn.Bind(wx.EVT_BUTTON, self.on_summarize)
        self.summarize_btn.Disable()  # Start disabled
        summary_sizer.Add(self.summarize_btn, 0, wx.EXPAND | wx.ALL, 5)
        
        sizer.Add(summary_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        panel.SetSizer(sizer)
        
        return panel

    def bind_events(self):
        """Bind events to handlers."""
        # Enter key in prompt input
        if hasattr(self, 'prompt_input'):
            self.prompt_input.Bind(wx.EVT_TEXT_ENTER, self.on_send_prompt)
        
    def on_close(self, event):
        """Handle application close event."""
        self.Destroy()
        
    def update_status(self, message, percent=None):
        """Update status bar with message and optional progress percentage."""
        if percent is not None:
            self.status_bar.SetStatusText(f"{message} ({percent:.0f}%)")
        else:
            self.status_bar.SetStatusText(message)
            
    def on_identify_speakers(self, event):
        """Handle speaker identification button click."""
        if hasattr(self, 'transcript') and self.transcript:
            # Make sure we have a client available
            if self.client is None:
                global client
                self.client = client

            has_token = bool(self.config_manager.get_pyannote_token())
            if has_token and hasattr(self, 'last_audio_path') and self.last_audio_path:
                # Use advanced speaker identification with diarization
                threading.Thread(
                    target=self.identify_speakers_with_diarization,
                    args=(self.last_audio_path, self.transcript),
                    daemon=True
                ).start()
                
                # Add a timer to check for completion and update UI
                self.speaker_id_timer = wx.Timer(self)
                self.Bind(wx.EVT_TIMER, self.check_speaker_id_complete, self.speaker_id_timer)
                self.speaker_id_timer.Start(1000)  # Check every second
            else:
                # Use basic speaker identification
                speakers = self.identify_speakers_simple(self.transcript)
                
                # Update UI with results
                if hasattr(self, 'speakers') and self.speakers:
                    # Format transcript with speaker names
                    speaker_transcript = self.assign_speaker_names({s["speaker"]: s["speaker"] for s in self.speakers})
                    self.transcript = speaker_transcript
                    
                    # Update the UI
                    self.update_transcript_display()
                    self.update_speaker_list()
        else:
            wx.MessageBox("No transcript available. Please transcribe audio first.", 
                         "No Transcript", wx.OK | wx.ICON_INFORMATION)
    
    def check_speaker_id_complete(self, event):
        """Check if speaker identification is complete and update UI if it is."""
        if hasattr(self, 'speakers') and self.speakers:
            # Stop the timer
            self.speaker_id_timer.Stop()
            
            # Format transcript with speaker names
            speaker_transcript = self.assign_speaker_names({s["speaker"]: s["speaker"] for s in self.speakers})
            self.transcript = speaker_transcript
            
            # Update the UI
            self.update_transcript_display()
            self.update_speaker_list()
    
    def on_browse_audio(self, event):
        """Handle audio file browse button."""
        wildcard = (
            "Audio files|*.flac;*.m4a;*.mp3;*.mp4;*.mpeg;*.mpga;*.oga;*.ogg;*.wav;*.webm|"
            "FLAC files (*.flac)|*.flac|"
            "M4A files (*.m4a)|*.m4a|"
            "MP3 files (*.mp3)|*.mp3|"
            "MP4 files (*.mp4)|*.mp4|"
            "OGG files (*.ogg;*.oga)|*.ogg;*.oga|"
            "WAV files (*.wav)|*.wav|"
            "All files (*.*)|*.*"
        )
        
        with wx.FileDialog(self, "Choose an audio file", wildcard=wildcard,
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as file_dialog:
            if file_dialog.ShowModal() == wx.ID_CANCEL:
                return
                
            path = file_dialog.GetPath()
            
            # Validate file extension
            file_ext = os.path.splitext(path)[1].lower()
            supported_formats = ['.flac', '.m4a', '.mp3', '.mp4', '.mpeg', '.mpga', '.oga', '.ogg', '.wav', '.webm']
            
            if file_ext not in supported_formats:
                # If user selected "All files" and chose an unsupported format
                wx.MessageBox(
                    f"The selected file has an unsupported format: {file_ext}\n"
                    f"Supported formats are: {', '.join(supported_formats)}", 
                    "Unsupported Format", 
                    wx.OK | wx.ICON_WARNING
                )
                return
                
            # Check file size
            file_size_mb = os.path.getsize(path) / (1024 * 1024)
            if file_size_mb > 25:
                wx.MessageBox(
                    f"The selected file is {file_size_mb:.1f}MB, which exceeds the 25MB limit for OpenAI's Whisper API.\n"
                    f"Please choose a smaller file or compress this one.",
                    "File Too Large",
                    wx.OK | wx.ICON_WARNING
                )
                return
                
            self.audio_file_path.SetValue(path)
            self.update_status(f"Selected audio file: {os.path.basename(path)} ({file_size_mb:.1f}MB)", percent=0)
            self.update_button_states()
            
    def on_transcribe(self, event):
        """Handle audio transcription."""
        if not self.audio_file_path.GetValue():
            wx.MessageBox("Please select an audio file first.", "No File Selected", wx.OK | wx.ICON_INFORMATION)
            return
            
        # Check if Azure Speech configuration is set
        if not self.config_manager.get_azure_speech_api_key():
            self.show_azure_speech_config_dialog()
            return
            
        # Check if Azure OpenAI configuration is set
        if not self.config_manager.get_azure_api_key() or not self.config_manager.get_azure_endpoint():
            wx.MessageBox("Please set your Azure OpenAI configuration in the Settings tab.", "Configuration Required", wx.OK | wx.ICON_INFORMATION)
            return
            
        # Get the whisper deployment name
        whisper_deployment = self.config_manager.get_azure_deployment("whisper")
        if not whisper_deployment:
            wx.MessageBox("Whisper deployment name not configured in Azure OpenAI settings.", "Configuration Error", wx.OK | wx.ICON_ERROR)
            return
            
        # Get language selection and ensure it's valid
        selection = self.language_choice.GetSelection()
        if 0 <= selection < len(SUPPORTED_LANGUAGES):
            language = SUPPORTED_LANGUAGES[selection]
        else:
            language = "en"  # Default to English if selection is invalid
        
        # Save language choice to config
        self.config_manager.set_language(language)
        
        # Store the audio file path in the AudioProcessor
        # This ensures it's available for diarization later
        self.audio_processor.audio_file_path = self.audio_file_path.GetValue()
        
        # Update status message with display name
        self.update_status(f"Transcribing in {LANGUAGE_DISPLAY_NAMES[language]}...", percent=0)
        
        # Disable buttons during processing
        self.transcribe_btn.Disable()
        self.identify_speakers_btn.Disable()
        self.summarize_btn.Disable()
        
        # Start transcription in a separate thread
        threading.Thread(target=self.transcribe_thread, args=(self.audio_file_path.GetValue(), language)).start()
        
    def transcribe_thread(self, file_path, language):
        """Thread function for audio transcription."""
        try:
            # Check if Azure Speech configuration is set
            if not self.config_manager.get_azure_speech_api_key():
                wx.CallAfter(self.show_azure_speech_config_dialog)
                return
                
            # Check if Azure OpenAI configuration is set
            if not self.config_manager.get_azure_api_key() or not self.config_manager.get_azure_endpoint():
                wx.CallAfter(wx.MessageBox, "Please set your Azure OpenAI configuration in the Settings tab.", "Configuration Required", wx.OK | wx.ICON_INFORMATION)
                return
                
            # Get the whisper deployment name
            whisper_deployment = self.config_manager.get_azure_deployment("whisper")
            if not whisper_deployment:
                wx.CallAfter(wx.MessageBox, "Whisper deployment name not configured in Azure OpenAI settings.", "Configuration Error", wx.OK | wx.ICON_ERROR)
                return
            
            # Get file extension for better error reporting
            file_ext = os.path.splitext(file_path)[1].lower()
            supported_formats = ['.flac', '.m4a', '.mp3', '.mp4', '.mpeg', '.mpga', '.oga', '.ogg', '.wav', '.webm']
            
            # Check if file format is supported
            if file_ext not in supported_formats:
                # Attempt to convert to WAV if FFmpeg is available
                if self._is_ffmpeg_available():
                    wx.CallAfter(self.update_status, f"Converting {file_ext} file to WAV format...", percent=10)
                    try:
                        output_path = os.path.splitext(file_path)[0] + ".wav"
                        subprocess.run(
                            ["ffmpeg", "-i", file_path, "-y", output_path],
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE,
                            check=True
                        )
                        wx.CallAfter(self.update_status, f"Conversion complete. Starting transcription...", percent=20)
                        file_path = output_path
                    except Exception as e:
                        error_msg = f"Error converting file: {str(e)}"
                        wx.CallAfter(wx.MessageBox, error_msg, "Conversion Error", wx.OK | wx.ICON_ERROR)
                        wx.CallAfter(self.update_status, "Ready", percent=0)
                        wx.CallAfter(self.transcribe_btn.Enable)
                        return
                else:
                    error_msg = f"The file format {file_ext} is not supported by Azure OpenAI's Whisper API.\n\nSupported formats: {', '.join(supported_formats)}"
                    wx.CallAfter(wx.MessageBox, error_msg, "Unsupported Format", wx.OK | wx.ICON_ERROR)
                    wx.CallAfter(self.update_status, "Ready", percent=0)
                    wx.CallAfter(self.transcribe_btn.Enable)
                    return
            
            # For M4A files that often have issues, try to convert to WAV if FFmpeg is available
            if file_ext == '.m4a' and self._is_ffmpeg_available():
                wx.CallAfter(self.update_status, "Converting M4A to WAV format for better compatibility...", percent=10)
                try:
                    output_path = os.path.splitext(file_path)[0] + ".wav"
                    subprocess.run(
                        ["ffmpeg", "-i", file_path, "-y", output_path],
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE,
                        check=True
                    )
                    wx.CallAfter(self.update_status, "Conversion complete. Starting transcription...", percent=20)
                    file_path = output_path
                except Exception as e:
                    # Continue with original file, just log the warning
                    wx.CallAfter(self.update_status, f"Warning: Could not convert M4A file. Attempting to use original file.", percent=20)
            
            response = self.audio_processor.transcribe_audio(file_path, language)
            
            # Add a note about speaker identification at the top of the transcript
            transcription_notice = "--- TRANSCRIPTION COMPLETE ---\n" + \
                                  "To identify speakers in this transcript, click the 'Identify Speakers' button below.\n\n"
            
            # Ensure transcript is not None before concatenation
            transcript_text = self.audio_processor.transcript if self.audio_processor.transcript is not None else ""
            wx.CallAfter(self.transcript_text.SetValue, transcription_notice + transcript_text)
            # Set the transcript attribute in the MainFrame instance
            wx.CallAfter(self.set_transcript, transcript_text)
            wx.CallAfter(self.update_button_states)
            wx.CallAfter(self.update_status, f"Transcription complete: {len(transcript_text)} characters", percent=100)
            
            # Show a dialog informing the user to use speaker identification
            wx.CallAfter(self.show_speaker_id_hint)
            
        except FileNotFoundError as e:
            wx.CallAfter(wx.MessageBox, f"File not found: {str(e)}", "Error", wx.OK | wx.ICON_ERROR)
        except ValueError as e:
            error_msg = str(e)
            title = "Format Error"
            
            # Special handling for common error cases
            if 'ffprobe' in error_msg or 'ffmpeg' in error_msg:
                title = "FFmpeg Missing"
                error_msg = error_msg.replace('[Errno 2] No such file or directory:', 'Missing required component:')
                # Installation instructions are already in the error message from _get_ffmpeg_install_instructions
            elif file_ext == '.m4a' and 'Invalid file format' in error_msg:
                error_msg = (
                    "There was an issue with your M4A file. Some M4A files have compatibility issues with the Azure OpenAI API.\n\n"
                    "Possible solutions:\n"
                    "1. Install FFmpeg on your system (required for m4a processing)\n"
                    "2. Convert the file to WAV or MP3 format manually\n"
                    "3. Try a different M4A file (some are more compatible than others)"
                )
                title = "M4A Compatibility Issue"
                wx.CallAfter(wx.MessageBox, error_msg, title, wx.OK | wx.ICON_ERROR)
            else:
                wx.CallAfter(wx.MessageBox, error_msg, title, wx.OK | wx.ICON_ERROR)
        except openai.RateLimitError:
            wx.CallAfter(wx.MessageBox, "Azure OpenAI rate limit exceeded. Please try again later.", "Rate Limit Error", wx.OK | wx.ICON_ERROR)
        except openai.AuthenticationError:
            wx.CallAfter(wx.MessageBox, "Authentication error. Please check your Azure OpenAI API key in the Settings tab.", "Authentication Error", wx.OK | wx.ICON_ERROR)
        except openai.BadRequestError as e:
            error_msg = str(e)
            title = "API Error"
            
            if "Invalid file format" in error_msg:
                # Try to extract the current format from the error message
                format_match = re.search(r"supported formats: \[(.*?)\]", error_msg.lower())
                if format_match:
                    supported = format_match.group(1)
                else:
                    supported = "flac, m4a, mp3, mp4, mpeg, mpga, oga, ogg, wav, webm"
                
                if file_ext == '.m4a':
                    error_msg = (
                        "Your M4A file format is not compatible with the Azure OpenAI API.\n\n"
                        "Possible solutions:\n"
                        "1. Install FFmpeg on your system (required for m4a processing)\n"
                        "2. Convert the file to WAV or MP3 format manually\n"
                        "3. Try a different M4A file (some are more compatible than others)"
                    )
                    title = "M4A Format Error"
                else:
                    error_msg = (
                        f"The file format {file_ext} is not supported or has compatibility issues.\n\n"
                        f"Supported formats: {supported}\n\n"
                        "Recommended: Convert your file to WAV format using FFmpeg."
                    )
                    title = "Format Error"
            wx.CallAfter(wx.MessageBox, error_msg, title, wx.OK | wx.ICON_ERROR)
        except Exception as e:
            error_msg = str(e)
            if 'ffprobe' in error_msg or 'ffmpeg' in error_msg:
                # Handle FFmpeg-related errors not caught by previous handlers
                install_instructions = self.audio_processor._get_ffmpeg_install_instructions()
                error_msg = f"FFmpeg/FFprobe is required but not found. Please install it to process audio files.\n\n{install_instructions}"
                wx.CallAfter(wx.MessageBox, error_msg, "FFmpeg Required", wx.OK | wx.ICON_ERROR)
            else:
                wx.CallAfter(wx.MessageBox, f"Transcription error: {error_msg}", "Error", wx.OK | wx.ICON_ERROR)
        finally:
            wx.CallAfter(self.transcribe_btn.Enable)
            wx.CallAfter(self.update_status, "Ready", percent=0)
            
    def show_speaker_id_hint(self):
        """Show a hint dialog about using speaker identification."""
        # Check if PyAnnote is available
        if PYANNOTE_AVAILABLE:
            message = (
                "Transcription is complete!\n\n"
                "To identify different speakers in this transcript, click the 'Identify Speakers' button.\n\n"
                "This system will use advanced audio-based speaker diarization to detect different "
                "speakers by analyzing voice characteristics (pitch, tone, speaking style) from the "
                "original audio file.\n\n"
                "This approach is significantly more accurate than text-based analysis since it "
                "uses the actual voice patterns to distinguish between speakers."
            )
        else:
            message = (
                "Transcription is complete!\n\n"
                "To identify different speakers in this transcript, click the 'Identify Speakers' button.\n\n"
                "Currently, the system will analyze the text patterns to detect different speakers.\n\n"
                "For more accurate speaker identification, consider installing PyAnnote which uses "
                "audio analysis to distinguish speakers based on their voice characteristics. "
                "Click 'Yes' for installation instructions."
            )
            
        dlg = wx.MessageDialog(
            self,
            message,
            "Speaker Identification",
            wx.OK | (wx.CANCEL | wx.YES_NO if not PYANNOTE_AVAILABLE else wx.OK) | wx.ICON_INFORMATION
        )
        
        result = dlg.ShowModal()
        dlg.Destroy()
        
        # If user wants to install PyAnnote
        if result == wx.ID_YES:
            self.show_pyannote_setup_guide()
        
        # Highlight the identify speakers button
        self.identify_speakers_btn.SetFocus()

    def show_format_info(self):
        """Show information about supported audio formats and requirements."""
        # Check if we already showed this info
        if self.config_manager.config.get("shown_format_info", False):
            return
            
        # Check if FFmpeg is available
        ffmpeg_available = self._is_ffmpeg_available()
        ffmpeg_missing = not ffmpeg_available
        
        # List required tools
        needed_tools = []
        if ffmpeg_missing:
            needed_tools.append("FFmpeg")
            
        # No tools needed, we're all set
        if not needed_tools:
            if hasattr(sys, '_MEIPASS'):
                # Running as a PyInstaller bundle - FFmpeg is bundled
                self.update_status("FFmpeg is bundled with this application", percent=0)
                return
            else:
                # Running from source with FFmpeg installed
                self.update_status("All audio tools available", percent=0)
                return
        
        # Generate installation instructions
        ffmpeg_install = self._get_ffmpeg_install_instructions() if hasattr(self, '_get_ffmpeg_install_instructions') else self.audio_processor._get_ffmpeg_install_instructions()
        
        # Prepare message
        msg = (
            "Audio Format Compatibility Information:\n\n"
            "• Directly supported formats: WAV, MP3, FLAC, OGG\n"
            "• M4A files require FFmpeg for conversion\n\n"
            "For better audio file compatibility, especially with M4A files, "
            f"you need to install the following tools:\n\n{', '.join(needed_tools)}\n\n"
        )
        
        if ffmpeg_missing:
            if hasattr(sys, '_MEIPASS'):
                # Running as a PyInstaller bundle, but FFmpeg directory not found
                msg = (
                    "There was an issue with the bundled FFmpeg. This is unusual and indicates "
                    "a problem with the application package. Please contact support.\n\n"
                    "As a workaround, you can install FFmpeg manually:\n\n"
                )
                msg += ffmpeg_install
            else:
                # Running from source without FFmpeg
                msg += f"FFmpeg installation instructions:\n{ffmpeg_install}\n\n"
                msg += "FFmpeg is required for processing M4A files. Without it, M4A transcription will likely fail."
        
        self.update_status("FFmpeg required for M4A support - please install it", percent=0)
        
        # Always show FFmpeg warning because it's critical
        if ffmpeg_missing:
            wx.MessageBox(msg, "FFmpeg Required for M4A Files", wx.OK | wx.ICON_WARNING)
            self.config_manager.config["shown_format_info"] = True
            self.config_manager.save_config()
        # Only show other warnings if not shown before
        elif not self.config_manager.config.get("shown_format_info", False):
            wx.MessageBox(msg, "Audio Format Information", wx.OK | wx.ICON_INFORMATION)
            self.config_manager.config["shown_format_info"] = True
            self.config_manager.save_config()

    def _is_ffmpeg_available(self):
        """Check if ffmpeg is available on the system."""
        # First try the bundled FFmpeg
        bundled_ffmpeg = None
        
        # Get the application directory
        if hasattr(sys, '_MEIPASS'):
            # Running as a PyInstaller bundle
            base_dir = sys._MEIPASS
            bundled_ffmpeg_dir = os.path.join(base_dir, 'ffmpeg')
            
            if os.path.exists(bundled_ffmpeg_dir):
                # Check for ffmpeg executable based on platform
                if platform.system() == 'Windows':
                    bundled_ffmpeg = os.path.join(bundled_ffmpeg_dir, 'ffmpeg.exe')
                else:
                    bundled_ffmpeg = os.path.join(bundled_ffmpeg_dir, 'ffmpeg')
                
                if bundled_ffmpeg and os.path.exists(bundled_ffmpeg):
                    # Update PATH environment variable to include bundled ffmpeg
                    os.environ["PATH"] = bundled_ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
                    return True
        
        # Then try the standard way (using PATH)
        try:
            subprocess.run(
                ["ffmpeg", "-version"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                check=True
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            # On macOS, try common Homebrew path
            if platform.system() == 'darwin':
                common_mac_paths = [
                    "/opt/homebrew/bin/ffmpeg",
                    "/usr/local/bin/ffmpeg",
                    "/opt/local/bin/ffmpeg"  # MacPorts
                ]
                
                for path in common_mac_paths:
                    try:
                        subprocess.run(
                            [path, "-version"],
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE,
                            check=True
                        )
                        # If found, update the PATH environment variable
                        os.environ["PATH"] = os.path.dirname(path) + ":" + os.environ.get("PATH", "")
                        return True
                    except (subprocess.SubprocessError, FileNotFoundError):
                        continue
            
            return False

    def check_pyannote(self):
        """Check if PyAnnote is available and show installation instructions if not."""
        if not PYANNOTE_AVAILABLE:
            dlg = wx.MessageDialog(
                self,
                "PyAnnote is not installed. PyAnnote provides more accurate speaker diarization "
                "by analyzing audio directly, rather than just text.\n\n"
                "To install PyAnnote and set it up, click 'Yes' for detailed instructions.",
                "Speaker Diarization Enhancement",
                wx.YES_NO | wx.ICON_INFORMATION
            )
            if dlg.ShowModal() == wx.ID_YES:
                self.show_pyannote_setup_guide()
            dlg.Destroy()
    
    def show_pyannote_setup_guide(self):
        """Show detailed setup instructions for PyAnnote."""
        dlg = wx.Dialog(self, title="PyAnnote Setup Guide", size=(650, 550))
        
        panel = wx.Panel(dlg)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Create a styled text control for better formatting
        text = wx.TextCtrl(
            panel, 
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2,
            size=(-1, 400)
        )
        
        # Set up the instructions
        guide = """PYANNOTE SETUP GUIDE

Step 1: Install Required Dependencies
--------------------------------------
Run the following commands in your terminal:

pip install torch torchaudio
pip install pyannote.audio

Step 2: Get HuggingFace Access Token
------------------------------------
1. Create a HuggingFace account at https://huggingface.co/join
2. Go to https://huggingface.co/pyannote/speaker-diarization
3. Accept the user agreement
4. Go to https://huggingface.co/settings/tokens
5. Create a new token with READ access
6. Copy the token

Step 3: Configure the Application
--------------------------------
1. After installing, restart this application
2. Go to the Settings tab
3. Paste your token in the "PyAnnote Speaker Diarization" section
4. Click "Save Token"
5. Return to the Audio Processing tab
6. Click "Identify Speakers" to use audio-based speaker identification

Important Notes:
---------------
• PyAnnote requires at least 4GB of RAM
• GPU acceleration (if available) will make processing much faster
• For best results, use high-quality audio with minimal background noise
• The first run may take longer as models are downloaded

Troubleshooting:
---------------
• If you get CUDA errors, try installing a compatible PyTorch version for your GPU
• If you get "Access Denied" errors, check that your token is valid and you've accepted the license agreement
• For long audio files (>10 min), processing may take several minutes
"""
        
        # Add the text with some styling
        text.SetValue(guide)
        
        # Style the headers
        text.SetStyle(0, 19, wx.TextAttr(wx.BLUE, wx.NullColour, wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)))
        
        # Find all the section headers and style them
        for section in ["Step 1:", "Step 2:", "Step 3:", "Important Notes:", "Troubleshooting:"]:
            start = guide.find(section)
            if start != -1:
                end = start + len(section)
                text.SetStyle(start, end, wx.TextAttr(wx.Colour(128, 0, 128), wx.NullColour, wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)))
        
        # Add to sizer
        sizer.Add(text, 1, wx.EXPAND | wx.ALL, 10)
        
        # Add buttons
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Add a button to copy installation commands
        copy_btn = wx.Button(panel, label="Copy Installation Commands")
        copy_btn.Bind(wx.EVT_BUTTON, lambda e: self.copy_to_clipboard("pip install torch torchaudio\npip install pyannote.audio"))
        btn_sizer.Add(copy_btn, 0, wx.RIGHT, 10)
        
        # Add a button to open HuggingFace token page
        hf_btn = wx.Button(panel, label="Open HuggingFace Token Page")
        hf_btn.Bind(wx.EVT_BUTTON, lambda e: wx.LaunchDefaultBrowser("https://huggingface.co/settings/tokens"))
        btn_sizer.Add(hf_btn, 0, wx.RIGHT, 10)
        
        # Add button to go to settings tab
        settings_btn = wx.Button(panel, label="Go to Settings Tab")
        settings_btn.Bind(wx.EVT_BUTTON, lambda e: (self.notebook.SetSelection(2), dlg.EndModal(wx.ID_CLOSE)))
        btn_sizer.Add(settings_btn, 0, wx.RIGHT, 10)
        
        # Add close button
        close_btn = wx.Button(panel, wx.ID_CLOSE)
        close_btn.Bind(wx.EVT_BUTTON, lambda e: dlg.EndModal(wx.ID_CLOSE))
        btn_sizer.Add(close_btn, 0)
        
        sizer.Add(btn_sizer, 0, wx.ALIGN_CENTER | wx.ALL, 10)
        
        panel.SetSizer(sizer)
        dlg.ShowModal()
        dlg.Destroy()
    
    def copy_to_clipboard(self, text):
        """Copy text to clipboard."""
        if wx.TheClipboard.Open():
            wx.TheClipboard.SetData(wx.TextDataObject(text))
            wx.TheClipboard.Close()
            wx.MessageBox("Commands copied to clipboard", "Copied", wx.OK | wx.ICON_INFORMATION)

    def on_template_selected(self, event):
        """Handle selection of a template."""
        if not hasattr(self, 'template_list'):
            return
            
        selected = self.template_list.GetSelection()
        if selected == wx.NOT_FOUND:
            return
            
        template_name = self.template_list.GetString(selected)
        templates = self.config_manager.get_templates()
        
        if template_name in templates:
            self.template_content_input.SetValue(templates[template_name])
        else:
            self.template_content_input.Clear()

    def _quick_consistency_check(self):
        """Ultra-quick consistency check for short files"""
        if len(self.speakers) < 3:
            return
            
        # Look for isolated speaker segments
        for i in range(1, len(self.speakers) - 1):
            prev_speaker = self.speakers[i-1]["speaker"]
            curr_speaker = self.speakers[i]["speaker"]
            next_speaker = self.speakers[i+1]["speaker"]
            
            # If current speaker is sandwiched between different speakers
            if prev_speaker == next_speaker and curr_speaker != prev_speaker:
                # Fix the segment only if very short (likely error)
                if len(self.speakers[i]["text"].split()) < 15:
                    self.speakers[i]["speaker"] = prev_speaker

    def _process_audio_in_chunks(self, pipeline, audio_file, total_duration, chunk_size):
        """Process long audio files in chunks to optimize memory usage and speed."""
        from pyannote.core import Segment, Annotation
        import concurrent.futures
        from threading import Lock
        
        # Initialize a combined annotation object
        combined_diarization = Annotation()
        
        # Calculate number of chunks
        num_chunks = int(np.ceil(total_duration / chunk_size))
        self.update_status(f"Processing audio in {num_chunks} chunks...", percent=0.1)
        
        # Optimize number of workers based on file length and available memory
        # More chunks = more workers (up to cpu_count), but limit for very long files
        # to avoid excessive memory usage
        cpu_count = os.cpu_count() or 4
        
        # Determine optimal number of workers based on audio duration
        if total_duration > 7200:  # > 2 hours
            max_workers = max(2, min(cpu_count - 1, 4))  # For very long files, use fewer workers to avoid memory issues
        else:
            max_workers = max(2, min(cpu_count, 6))  # Use more workers for shorter files
        
        # Function to process a single chunk
        def process_chunk(chunk_idx, start_time):
            end_time = min(start_time + chunk_size, total_duration)
            chunk_segment = Segment(start=start_time, end=end_time)
            
            # Create optimized pipeline parameters for this chunk
            pipeline_params = {
                "segmentation": {
                    "min_duration_on": 0.25,
                    "min_duration_off": 0.25,
                },
                "clustering": {
                    "min_cluster_size": 6,
                    "method": "centroid"
                },
                "segmentation_batch_size": 64,  # Larger batch size for speed
                "embedding_batch_size": 64      # Larger batch size for speed
            }
            
            # Apply optimized parameters
            pipeline.instantiate(pipeline_params)
            
            # Apply diarization to this chunk
            chunk_result = pipeline(audio_file, segmentation=chunk_segment)
            
            return chunk_idx, chunk_result
        
        # Create thread lock for combining results
        lock = Lock()
        chunk_results = [None] * num_chunks
        progress_counter = [0]
        
        # Function to update progress
        def update_progress():
            with lock:
                progress_counter[0] += 1
                progress = 0.15 + (progress_counter[0] / num_chunks) * 0.7
                self.update_status(f"Processed {progress_counter[0]}/{num_chunks} chunks...", percent=progress)
        
        try:
            # Process chunks in parallel using ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all chunks for processing
                futures = {}
                for i in range(num_chunks):
                    start_time = i * chunk_size
                    future = executor.submit(process_chunk, i, start_time)
                    futures[future] = i
                
                # Handle results as they complete
                for future in concurrent.futures.as_completed(futures):
                    chunk_idx, chunk_result = future.result()
                    chunk_results[chunk_idx] = chunk_result
                    update_progress()
            
            # Combine results
            self.update_status(f"Combining results from {num_chunks} chunks...", percent=0.85)
            
            # Create a new combined annotation
            combined_diarization = Annotation()
            
            # Merge annotations while preserving speaker identities across chunks
            speaker_mapping = {}  # Map to track speakers across chunks
            global_speaker_count = 0
            
            # First pass: collect all speaker embeddings
            all_speakers = {}
            for chunk_idx, chunk_result in enumerate(chunk_results):
                if chunk_result is None:
                    continue
                    
                # Get speakers from this chunk
                for segment, track, speaker in chunk_result.itertracks(yield_label=True):
                    if speaker not in all_speakers:
                        all_speakers[speaker] = []
                        
                    # Store segment information
                    all_speakers[speaker].append((chunk_idx, segment, track))
            
            # Second pass: establish global speaker mapping based on temporal proximity
            for local_speaker, segments in all_speakers.items():
                # Sort segments by time
                segments.sort(key=lambda x: x[1].start)
                
                # Check if this speaker should be mapped to an existing global speaker
                mapped = False
                for global_speaker, global_segments in speaker_mapping.items():
                    # Check if any segment overlaps or is very close to an existing global speaker
                    for chunk_idx, segment, track in segments:
                        for g_chunk_idx, g_segment, g_track in global_segments:
                            # Consider speakers the same if segments are close in time
                            if (abs(segment.start - g_segment.end) < 2.0 or 
                                abs(g_segment.start - segment.end) < 2.0):
                                mapped = True
                                break
                                
                        if mapped:
                            break
                        
                    if mapped:
                        # Add segments to existing global speaker
                        speaker_mapping[global_speaker].extend(segments)
                        break
                        
                if not mapped:
                    # Create new global speaker
                    global_speaker = f"SPEAKER_{global_speaker_count}"
                    global_speaker_count += 1
                    speaker_mapping[global_speaker] = segments
            
            # Third pass: add all segments to the combined diarization
            for global_speaker, segments in speaker_mapping.items():
                for chunk_idx, segment, track in segments:
                    combined_diarization[segment, track] = global_speaker
            
            # Return the combined diarization
            return combined_diarization
            
        except Exception as e:
            self.update_status(f"Error in parallel processing: {str(e)}", percent=0.5)
            
            # Fall back to simpler sequential processing
            self.update_status("Falling back to sequential processing...", percent=0.5)
            
            # Process chunks sequentially as fallback
            combined_diarization = Annotation()
            for i in range(num_chunks):
                start_time = i * chunk_size
                end_time = min(start_time + chunk_size, total_duration)
                segment = Segment(start=start_time, end=end_time)
                
                # Update progress
                progress = 0.5 + (i / num_chunks) * 0.4
                self.update_status(f"Processing chunk {i+1}/{num_chunks} (sequential mode)...", percent=progress)
                
                # Process this chunk
                chunk_result = pipeline(audio_file, segmentation=segment)
                
                # Add results to combined annotation
                for s, t, spk in chunk_result.itertracks(yield_label=True):
                    # Create a global speaker ID
                    global_spk = f"SPEAKER_{spk.split('_')[-1]}"
                    combined_diarization[s, t] = global_spk
            
            return combined_diarization

    def identify_speakers_with_diarization(self, audio_file_path, transcript):
        """Identify speakers using audio diarization with PyAnnote."""
        self.update_status("Performing audio diarization analysis...", percent=0.05)
        
        # Check if PyAnnote is available
        if not PYANNOTE_AVAILABLE:
            self.update_status("PyAnnote not available. Install with: pip install pyannote.audio", percent=0)
            return self.identify_speakers_simple(transcript)
        
        # Check if we have cached results - if so, skip to mapping
        if self._check_diarization_cache(audio_file_path):
            self.update_status("Using cached diarization results...", percent=0.4)
            
            # Get audio information for status reporting
            audio_duration = librosa.get_duration(path=audio_file_path)
            is_short_file = audio_duration < 300
            
            # Skip to mapping step
            if is_short_file:
                self.update_status("Fast mapping diarization to transcript...", percent=0.8)
                return self._fast_map_diarization(transcript)
            else:
                return self._map_diarization_to_transcript(transcript)
        
        # No cache, proceed with normal processing
        # Step 1: Initialize PyAnnote pipeline
        try:
            # Get token from config_manager if available
            token = None
            if self.config_manager:
                token = self.config_manager.get_pyannote_token()
            
            # If not found, check for a token file as a fallback
            if not token:
                # Use APP_BASE_DIR if available
                if APP_BASE_DIR:
                    token_file = os.path.join(APP_BASE_DIR, "pyannote_token.txt")
                else:
                    token_file = "pyannote_token.txt"
                
                if os.path.exists(token_file):
                    with open(token_file, "r") as f:
                        file_token = f.read().strip()
                        if not file_token.startswith("#") and len(file_token) >= 10:
                            token = file_token
            
            # If still no token, show message and fall back to text-based identification
            if not token:
                self.update_status("PyAnnote token not found in settings. Please add your token in the Settings tab.", percent=0)
                return self.identify_speakers_simple(transcript)
            
            self.update_status("Initializing diarization pipeline...", percent=0.1)
            
            # Initialize the PyAnnote pipeline
            pipeline = pyannote.audio.Pipeline.from_pretrained(
                "pyannote/speaker-diarization@2.1",
                use_auth_token=token
            )
            
            # Set device (GPU if available)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipeline = pipeline.to(torch.device(device))
            
            # Convert the file to WAV format if needed
            if not audio_file_path.lower().endswith('.wav'):
                self.update_status("Converting audio to WAV format for diarization...", percent=0.15)
                converted_file = self.convert_to_wav(audio_file_path)
                diarization_file = converted_file
            else:
                diarization_file = audio_file_path
            
            # Get audio file information
            audio_duration = librosa.get_duration(path=diarization_file)
            self.update_status(f"Audio duration: {audio_duration:.1f} seconds", percent=0.2)
            
            # Very short files need very different processing approach
            is_short_file = audio_duration < 300  # Less than 5 minutes
            
            if is_short_file:
                # Ultra fast mode for short files (5 min or less) - direct processing with optimized parameters
                self.update_status("Short audio detected, using ultra-fast mode...", percent=0.25)
                
                # Use ultra-optimized parameters for short files
                pipeline.instantiate({
                    # More aggressive voice activity detection for speed
                    "segmentation": {
                        "min_duration_on": 0.25,      # Shorter minimum speech (default 0.1s)
                        "min_duration_off": 0.25,     # Shorter minimum silence (default 0.1s)
                    },
                    # Faster clustering with fewer speakers expected in short clips
                    "clustering": {
                        "min_cluster_size": 6,        # Require fewer samples (default 15)
                        "method": "centroid"          # Faster than "average" linkage
                    },
                    # Skip post-processing for speed
                    "segmentation_batch_size": 32,    # Larger batch for speed
                    "embedding_batch_size": 32,       # Larger batch for speed
                })
                
                # Apply diarization directly for short files
                self.update_status("Processing audio (fast mode)...", percent=0.3)
                self.diarization = pipeline(diarization_file)
                
                # For very short files, optimize the diarization results
                if audio_duration < 60:  # Less than 1 minute
                    # Further optimize by limiting max speakers for very short clips
                    num_speakers = len(set(s for _, _, s in self.diarization.itertracks(yield_label=True)))
                    if num_speakers > 3:
                        self.update_status("Optimizing speaker count for short clip...", percent=0.7)
                        # Re-run with max_speakers=3 for very short clips
                        self.diarization = pipeline(diarization_file, num_speakers=3)
            else:
                # Determine chunk size based on audio duration - longer files use chunking
                if audio_duration > 10800:  # > 3 hours
                    # For extremely long recordings, use very small 3-minute chunks
                    MAX_CHUNK_DURATION = 180  # 3 minutes per chunk
                    self.update_status("Extremely long audio detected (>3 hours). Using highly optimized micro-chunks.", percent=0.22)
                elif audio_duration > 5400:  # > 1.5 hours
                    # For very long recordings, use 4-minute chunks
                    MAX_CHUNK_DURATION = 240  # 4 minutes per chunk
                    self.update_status("Very long audio detected (>1.5 hours). Using micro-chunks for improved performance.", percent=0.22)
                elif audio_duration > 3600:  # > 1 hour
                    # For long recordings, use 5-minute chunks
                    MAX_CHUNK_DURATION = 300  # 5 minutes per chunk
                    self.update_status("Long audio detected (>1 hour). Using optimized chunk size.", percent=0.22)
                elif audio_duration > 1800:  # > 30 minutes
                    # For medium recordings, use 7.5-minute chunks
                    MAX_CHUNK_DURATION = 450  # 7.5 minutes per chunk
                    self.update_status("Medium-length audio detected (>30 minutes). Using optimized chunk size.", percent=0.22)
                else:
                    # Default 10-minute chunks for shorter files
                    MAX_CHUNK_DURATION = 600  # 10 minutes per chunk
                
                # Process in chunks for longer files
                self.update_status("Processing in chunks for optimized performance...", percent=0.25)
                self.diarization = self._process_audio_in_chunks(pipeline, diarization_file, audio_duration, MAX_CHUNK_DURATION)
            
            # Clean up converted file if needed
            if diarization_file != audio_file_path and os.path.exists(diarization_file):
                os.unlink(diarization_file)
            
            # Save diarization results to cache for future use
            self._save_diarization_cache(audio_file_path)
            
            # Now we have diarization data, map it to the transcript using word timestamps
            # Use optimized mapping for short files
            if is_short_file:
                self.update_status("Fast mapping diarization to transcript...", percent=0.8)
                return self._fast_map_diarization(transcript)
            else:
                return self._map_diarization_to_transcript(transcript)
            
        except Exception as e:
            self.update_status(f"Error in diarization: {str(e)}", percent=0)
            # Fall back to text-based approach
            return self.identify_speakers_simple(transcript)

    def _fast_map_diarization(self, transcript):
        """Simplified and faster mapping for short files."""
        self.update_status("Fast mapping diarization results to transcript...", percent=0.85)
        
        if not hasattr(self, 'word_by_word') or not self.word_by_word or not self.diarization:
            return self.identify_speakers_simple(transcript)
        
        try:
            # Create speaker timeline map at higher granularity (every 0.2s)
            timeline_map = {}
            speaker_set = set()
            
            # Extract all speakers and their time ranges
            for segment, _, speaker in self.diarization.itertracks(yield_label=True):
                start_time = segment.start
                end_time = segment.end
                speaker_set.add(speaker)
                
                # For short files, we can afford fine-grained sampling
                step = 0.1  # 100ms steps
                for t in np.arange(start_time, end_time, step):
                    timeline_map[round(t, 1)] = speaker
            
            # Create paragraphs if they don't exist
            if hasattr(self, 'speaker_segments') and self.speaker_segments:
                paragraphs = self.speaker_segments
            else:
                paragraphs = self._create_improved_paragraphs(transcript)
                self.speaker_segments = paragraphs
            
            # Calculate overall speakers - short clips typically have 1-3 speakers
            num_speakers = len(speaker_set)
            self.update_status(f"Detected {num_speakers} speakers in audio", percent=0.9)
            
            # Map each word to a speaker
            word_speakers = {}
            for word_info in self.word_by_word:
                if not hasattr(word_info, "start") or not hasattr(word_info, "end"):
                    continue
                
                # Take the middle point of each word
                word_time = round((word_info.start + word_info.end) / 2, 1)
                
                # Find closest time in our map
                closest_time = min(timeline_map.keys(), key=lambda x: abs(x - word_time), default=None)
                if closest_time is not None and abs(closest_time - word_time) < 1.0:
                    word_speakers[word_info.word] = timeline_map[closest_time]
            
            # Now assign speakers to paragraphs based on word majority
            self.speakers = []
            for paragraph in paragraphs:
                para_speakers = []
                
                # Count speakers in this paragraph
                words = re.findall(r'\b\w+\b', paragraph.lower())
                for word in words:
                    if word in word_speakers:
                        para_speakers.append(word_speakers[word])
                
                # Find most common speaker
                if para_speakers:
                    from collections import Counter
                    speaker_counts = Counter(para_speakers)
                    most_common_speaker = speaker_counts.most_common(1)[0][0]
                    speaker_id = f"Speaker {most_common_speaker.split('_')[-1]}"
                else:
                    # Fallback for paragraphs with no identified speaker
                    speaker_id = f"Speaker 1"
                
                self.speakers.append({
                    "speaker": speaker_id,
                    "text": paragraph
                })
            
            # Final quick consistency check for short files
            if len(self.speakers) > 1:
                self._quick_consistency_check()
            
            self.update_status(f"Diarization complete. Found {num_speakers} speakers.", percent=1.0)
            return self.speakers
            
        except Exception as e:
            self.update_status(f"Error in fast mapping: {str(e)}", percent=0)
            return self.identify_speakers_simple(transcript)
    
    def _map_diarization_to_transcript(self, transcript):
        """Memory-efficient mapping for long files by using sparse sampling and batch processing."""
        self.update_status("Mapping diarization results to transcript (optimized for long files)...", percent=0.8)
        
        if not hasattr(self, 'word_by_word') or not self.word_by_word or not self.diarization:
            return self.identify_speakers_simple(transcript)
            
        try:
            # Get initial speaker count for progress reporting
            speaker_set = set()
            segment_count = 0
            
            # Quick scan to count speakers and segments - don't store details yet
            for segment, _, speaker in self.diarization.itertracks(yield_label=True):
                speaker_set.add(speaker)
                segment_count += 1
                
            num_speakers = len(speaker_set)
            self.update_status(f"Detected {num_speakers} speakers across {segment_count} segments", percent=0.82)
            
            # Create paragraphs if they don't exist
            if hasattr(self, 'speaker_segments') and self.speaker_segments:
                paragraphs = self.speaker_segments
            else:
                paragraphs = self._create_improved_paragraphs(transcript)
                self.speaker_segments = paragraphs
                
            # OPTIMIZATION 1: For long files, use sparse sampling of the timeline
            # Instead of creating a dense timeline map which is memory-intensive,
            # we'll create a sparse map with only the segment boundaries
            timeline_segments = []
            
            # Use diarization_cache directory for temporary storage if needed
            if APP_BASE_DIR:
                cache_dir = os.path.join(APP_BASE_DIR, "diarization_cache")
            else:
                cache_dir = "diarization_cache"
                
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
                
            # OPTIMIZATION 2: For very long files, process diarization in chunks to avoid memory issues
            chunk_size = 1000  # Process 1000 segments at a time
            use_temp_storage = segment_count > 5000  # Only use temp storage for very large files
            
            # If using temp storage, save intermediate results to avoid memory buildup
            if use_temp_storage:
                self.update_status("Using temporary storage for large diarization data...", percent=0.83)
                temp_file = os.path.join(cache_dir, f"diarization_map_{int(time.time())}.json")
                
                # Process in chunks to avoid memory buildup
                processed = 0
                segment_chunk = []
                
                for segment, _, speaker in self.diarization.itertracks(yield_label=True):
                    # Skip very short segments
                    if segment.duration < 0.5:
                        continue
                        
                    segment_chunk.append({
                        "start": segment.start,
                        "end": segment.end,
                        "speaker": speaker
                    })
                    
                    processed += 1
                    
                    # When chunk is full, process it
                    if len(segment_chunk) >= chunk_size:
                        timeline_segments.extend(segment_chunk)
                        # Save intermediate results
                        with open(temp_file, 'w') as f:
                            json.dump(timeline_segments, f)
                        # Clear memory
                        timeline_segments = []
                        segment_chunk = []
                        # Update progress
                        progress = 0.83 + (processed / segment_count) * 0.05
                        self.update_status(f"Processed {processed}/{segment_count} diarization segments...", percent=progress)
                
                # Process remaining segments
                if segment_chunk:
                    timeline_segments.extend(segment_chunk)
                    with open(temp_file, 'w') as f:
                        json.dump(timeline_segments, f)
                
                # Load from file to continue processing
                with open(temp_file, 'r') as f:
                    timeline_segments = json.load(f)
            else:
                # For smaller files, process all at once
                for segment, _, speaker in self.diarization.itertracks(yield_label=True):
                    # Skip very short segments
                    if segment.duration < 0.5:
                        continue
                        
                    timeline_segments.append({
                        "start": segment.start,
                        "end": segment.end,
                        "speaker": speaker
                    })
            
            self.update_status("Matching words to speaker segments...", percent=0.89)
            
            # OPTIMIZATION 3: Optimize word-to-speaker mapping for long files
            # Sort segments by start time for faster searching
            timeline_segments.sort(key=lambda x: x["start"])
            
            # Initialize paragraph mapping structures
            paragraph_speaker_counts = [{} for _ in paragraphs]
            
            # Batch process words to reduce computation
            batch_size = 500
            num_words = len(self.word_by_word)
            
            # Calculate which paragraph each word belongs to
            word_paragraphs = {}
            
            para_start_idx = 0
            for i, word_info in enumerate(self.word_by_word):
                if not hasattr(word_info, "start") or not hasattr(word_info, "end"):
                    continue
                    
                # Binary search to find the paragraph for this word
                # This is much faster than iterating through all paragraphs for each word
                word = word_info.word.lower()
                
                # Find paragraph for this word only once
                if i % 100 == 0:  # Only update progress occasionally
                    progress = 0.89 + (i / num_words) * 0.05
                    self.update_status(f"Matching words to paragraphs ({i}/{num_words})...", percent=progress)
                
                # Find which paragraph this word belongs to
                found_para = False
                for p_idx in range(para_start_idx, len(paragraphs)):
                    if word in paragraphs[p_idx].lower():
                        word_paragraphs[word] = p_idx
                        para_start_idx = p_idx  # Optimization: start next search from here
                        found_para = True
                        break
                
                if not found_para:
                    # If we didn't find it moving forward, try searching all paragraphs
                    for p_idx in range(len(paragraphs)):
                        if word in paragraphs[p_idx].lower():
                            word_paragraphs[word] = p_idx
                            para_start_idx = p_idx
                            found_para = True
                            break
            
            # Process words in batches to assign speakers efficiently
            for batch_start in range(0, num_words, batch_size):
                batch_end = min(batch_start + batch_size, num_words)
                
                for i in range(batch_start, batch_end):
                    if i >= len(self.word_by_word):
                        break
                        
                    word_info = self.word_by_word[i]
                    if not hasattr(word_info, "start") or not hasattr(word_info, "end"):
                        continue
                    
                    word = word_info.word.lower()
                    word_time = (word_info.start + word_info.end) / 2
                    
                    # Find segment for this word using binary search for speed
                    left, right = 0, len(timeline_segments) - 1
                    segment_idx = -1
                    
                    while left <= right:
                        mid = (left + right) // 2
                        if timeline_segments[mid]["start"] <= word_time <= timeline_segments[mid]["end"]:
                            segment_idx = mid
                            break
                        elif word_time < timeline_segments[mid]["start"]:
                            right = mid - 1
                        else:
                            left = mid + 1
                    
                    # If we found a segment, update the paragraph speaker counts
                    if segment_idx != -1:
                        speaker = timeline_segments[segment_idx]["speaker"]
                        
                        # If we know which paragraph this word belongs to, update its speaker count
                        if word in word_paragraphs:
                            para_idx = word_paragraphs[word]
                            paragraph_speaker_counts[para_idx][speaker] = paragraph_speaker_counts[para_idx].get(speaker, 0) + 1
                
                # Update progress
                progress = 0.94 + (batch_end / num_words) * 0.05
                self.update_status(f"Processed {batch_end}/{num_words} words...", percent=progress)
            
            # Assign speakers to paragraphs based on majority vote
            self.speakers = []
            for i, paragraph in enumerate(paragraphs):
                # Get speaker counts for this paragraph
                speaker_counts = paragraph_speaker_counts[i]
                
                # Assign the most common speaker, or default if none
                if speaker_counts:
                    # Find speaker with highest count
                    most_common_speaker = max(speaker_counts.items(), key=lambda x: x[1])[0]
                    speaker_id = f"Speaker {most_common_speaker.split('_')[-1]}"
                else:
                    # Default speaker if no match found
                    speaker_id = f"Speaker 1"
                
                self.speakers.append({
                    "speaker": speaker_id,
                    "text": paragraph
                })
            
            # Quick consistency check
            if len(self.speakers) > 2:
                self._quick_consistency_check()
            
            # Clean up temp file if used
            if use_temp_storage and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass
            
            self.update_status(f"Diarization mapping complete. Found {num_speakers} speakers.", percent=1.0)
            return self.speakers
            
        except Exception as e:
            self.update_status(f"Error in diarization mapping: {str(e)}", percent=0)
            # Fall back to text-based approach
            return self.identify_speakers_simple(transcript)

    
    def on_send_prompt(self, event):
        """Handle sending a prompt from the chat input."""
        prompt = self.chat_input.GetValue()
        if prompt.strip():
            self.llm_processor.generate_response(prompt)
            self.chat_input.SetValue("")

    def set_transcript(self, transcript_text):
        """Set the transcript attribute directly."""
        self.transcript = transcript_text
        self.last_audio_path = getattr(self.audio_processor, 'audio_file_path', None)

    def _check_diarization_cache(self, audio_file_path):
        """Check if diarization results exist in cache for the given audio file.
        
        If found, it loads the results into self.diarization.
        Returns True if cache was found and loaded, False otherwise.
        """
        try:
            # Create hash of file path and modification time to use as cache key
            file_stats = os.stat(audio_file_path)
            file_hash = hashlib.md5(f"{audio_file_path}_{file_stats.st_mtime}".encode()).hexdigest()
            
            # Use APP_BASE_DIR if available
            if 'APP_BASE_DIR' in globals() and APP_BASE_DIR:
                cache_dir = os.path.join(APP_BASE_DIR, "diarization_cache")
            else:
                cache_dir = "diarization_cache"
                
            if not os.path.exists(cache_dir):
                return False
                
            # Cache file path
            cache_file = os.path.join(cache_dir, f"{file_hash}.diar")
            
            # Check if cache file exists
            if not os.path.exists(cache_file):
                return False
                
            # Load cached results
            self.update_status("Loading cached diarization results...", percent=0.3)
            with open(cache_file, 'rb') as f:
                self.diarization = pickle.load(f)
                
            self.update_status("Successfully loaded cached results", percent=0.35)
            return True
        except Exception as e:
            self.update_status(f"Error loading from cache: {str(e)}", percent=0.05)
            return False
            
    def _save_diarization_cache(self, audio_file_path):
        """Save diarization results to cache for future use."""
        try:
            # Create hash of file path and modification time to use as cache key
            file_stats = os.stat(audio_file_path)
            file_hash = hashlib.md5(f"{audio_file_path}_{file_stats.st_mtime}".encode()).hexdigest()
            
            # Use APP_BASE_DIR if available
            if 'APP_BASE_DIR' in globals() and APP_BASE_DIR:
                cache_dir = os.path.join(APP_BASE_DIR, "diarization_cache")
            else:
                cache_dir = "diarization_cache"
                
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
                
            # Cache file path
            cache_file = os.path.join(cache_dir, f"{file_hash}.diar")
            
            # Save results
            self.update_status("Saving diarization results to cache for future use...", percent=0.95)
            with open(cache_file, 'wb') as f:
                pickle.dump(self.diarization, f)
            
            # Clean up old cache files if there are more than 20
            cache_files = [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith('.diar')]
            if len(cache_files) > 20:
                # Sort by modification time and remove oldest
                cache_files.sort(key=os.path.getmtime)
                for old_file in cache_files[:-20]:  # Keep the 20 most recent
                    os.unlink(old_file)
                    
            self.update_status("Successfully cached results for future use", percent=0.98)
        except Exception as e:
            self.update_status(f"Error saving to cache: {str(e)}", percent=0.95)
            # Continue without caching - non-critical error

    def show_azure_speech_config_dialog(self):
        """Show dialog to configure Azure Speech settings."""
        dialog = wx.Dialog(self, title="Configure Azure Speech", size=(400, 300))
        
        # Create main panel with padding
        panel = wx.Panel(dialog)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Get current configuration
        speech_config = self.config_manager.get_azure_speech_config()
        current_key = speech_config.get("api_key", "") if speech_config else ""
        current_region = speech_config.get("region", "eastus") if speech_config else "eastus"
        
        # Add description
        description = wx.StaticText(panel, label=(
            "Configure your Azure Speech service settings.\n"
            "These are required for audio transcription.\n\n"
            "1. The API key should be from your Azure Speech service resource\n"
            "2. The region should match your Azure Speech service region (e.g., eastus)\n\n"
            "You can find these in your Azure portal under your Speech service resource."
        ))
        description.Wrap(380)
        sizer.Add(description, 0, wx.ALL | wx.EXPAND, 10)
        
        # Create input fields
        key_label = wx.StaticText(panel, label="API Key:")
        key_input = wx.TextCtrl(panel, value=current_key, size=(300, -1))
        key_input.SetHint("Enter your Azure Speech API key")
        
        region_label = wx.StaticText(panel, label="Region:")
        region_input = wx.TextCtrl(panel, value=current_region, size=(300, -1))
        region_input.SetHint("e.g., eastus, westus, etc.")
        
        # Add fields to sizer
        field_sizer = wx.FlexGridSizer(2, 2, 5, 5)
        field_sizer.Add(key_label, 0, wx.ALIGN_CENTER_VERTICAL)
        field_sizer.Add(key_input, 1, wx.EXPAND)
        field_sizer.Add(region_label, 0, wx.ALIGN_CENTER_VERTICAL)
        field_sizer.Add(region_input, 1, wx.EXPAND)
        
        sizer.Add(field_sizer, 0, wx.ALL | wx.EXPAND, 10)
        
        # Add buttons
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        save_button = wx.Button(panel, label="Save")
        cancel_button = wx.Button(panel, label="Cancel")
        
        button_sizer.Add(save_button, 0, wx.ALL, 5)
        button_sizer.Add(cancel_button, 0, wx.ALL, 5)
        
        sizer.Add(button_sizer, 0, wx.ALIGN_CENTER | wx.ALL, 10)
        
        # Set up event handlers
        def on_save(event):
            api_key = key_input.GetValue().strip()
            region = region_input.GetValue().strip()
            
            # Validate inputs
            if not api_key:
                wx.MessageBox("Please enter an API key", "Validation Error", wx.OK | wx.ICON_ERROR)
                return
                
            if not region:
                wx.MessageBox("Please enter a region", "Validation Error", wx.OK | wx.ICON_ERROR)
                return
            
            # Save configuration
            if self.config_manager.set_azure_speech_config(api_key, None, region):
                wx.MessageBox("Configuration saved successfully!", "Success", wx.OK | wx.ICON_INFORMATION)
                dialog.EndModal(wx.ID_OK)
            else:
                wx.MessageBox("Failed to save configuration", "Error", wx.OK | wx.ICON_ERROR)
        
        def on_cancel(event):
            dialog.EndModal(wx.ID_CANCEL)
        
        save_button.Bind(wx.EVT_BUTTON, on_save)
        cancel_button.Bind(wx.EVT_BUTTON, on_cancel)
        
        # Set up the dialog
        panel.SetSizer(sizer)
        dialog.Centre()
        
        # Show the dialog
        if dialog.ShowModal() == wx.ID_OK:
            # Reinitialize the speech client with new configuration
            if hasattr(self, 'audio_processor'):
                self.audio_processor.initialize_speech_client()
        
        dialog.Destroy()

class LLMProcessor:
    """LLM processing functionality for chat and summarization."""
    def __init__(self, client, config_manager, update_callback=None):
        self.client = client
        self.config_manager = config_manager
        self.update_callback = update_callback
        self.chat_history = []
        
    def update_status(self, message, percent=None):
        if self.update_callback:
            wx.CallAfter(self.update_callback, message, percent)
            
    def generate_response(self, prompt, temperature=None):
        """Generate a response from the LLM."""
        if temperature is None:
            temperature = self.config_manager.get_temperature()
        
        messages = self.prepare_messages(prompt)
        
        try:
            self.update_status("Generating response...", percent=0)
            chat_deployment = self.config_manager.get_azure_deployment("chat")
            response = self.client.chat.completions.create(
                model=chat_deployment,
                messages=messages,
                temperature=temperature,
                api_version=self.config_manager.get_azure_api_version()
            )
            
            response_text = response.choices[0].message.content
            
            # Add to chat history
            self.chat_history.append({"role": "user", "content": prompt})
            self.chat_history.append({"role": "assistant", "content": response_text})
            
            self.update_status("Response generated.", percent=100)
            return response_text
            
        except Exception as e:
            self.update_status(f"Error generating response: {str(e)}", percent=50)
            return f"Error: {str(e)}"
            
    def prepare_messages(self, prompt):
        """Prepare messages for the LLM, including chat history."""
        messages = []
        
        # Add system message
        system_content = "You are a helpful assistant that can analyze transcripts."
        messages.append({"role": "system", "content": system_content})
        
        # Add chat history (limit to last 10 messages to avoid token limits)
        if self.chat_history:
            messages.extend(self.chat_history[-10:])
            
        # Add the current prompt
        if prompt not in [msg["content"] for msg in messages if msg["role"] == "user"]:
            messages.append({"role": "user", "content": prompt})
            
        return messages
        
    def clear_chat_history(self):
        """Clear the chat history."""
        self.chat_history = []
        self.update_status("Chat history cleared.", percent=0)
        
    def summarize_transcript(self, transcript, template_name=None):
        """Summarize a transcript, optionally using a template."""
        if not transcript:
            return "No transcript to summarize."
            
        self.update_status("Generating summary...", percent=0)
        
        prompt = f"Summarize the following transcript:"
        template = None
        
        if template_name:
            templates = self.config_manager.get_templates()
            if template_name in templates:
                template = templates[template_name]
                prompt += f" Follow this template format:\n\n{template}"
                
        prompt += f"\n\nTranscript:\n{transcript}"
        
        try:
            chat_deployment = self.config_manager.get_azure_deployment("chat")
            response = self.client.chat.completions.create(
                model=chat_deployment,
                messages=[
                    {"role": "system", "content": "You are an assistant that specializes in summarizing transcripts."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                api_version=self.config_manager.get_azure_api_version()
            )
            
            summary = response.choices[0].message.content
            
            # Save summary to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Use APP_BASE_DIR if available
            if APP_BASE_DIR:
                summary_dir = os.path.join(APP_BASE_DIR, "Summaries")
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                summary_filename = os.path.join(summary_dir, f"summary_{timestamp}.txt")
            else:
                summary_filename = f"Summaries/summary_{timestamp}.txt"
                
            with open(summary_filename, 'w', encoding='utf-8') as f:
                f.write(summary)
                
            self.update_status(f"Summary generated and saved to {summary_filename}.", percent=100)
            return summary
            
        except Exception as e:
            self.update_status(f"Error generating summary: {str(e)}", percent=50)
            return f"Error: {str(e)}"

class ConfigManager:
    def __init__(self, base_dir):
        """Initialize the config manager with the base directory."""
        print(f"Initializing ConfigManager with base_dir: {base_dir}")
        self.base_dir = base_dir
        self.config_file = os.path.join(base_dir, "config.json")
        print(f"Config file will be at: {self.config_file}")
        self.config = self.load_config()
        # Ensure the config has the correct API keys
        self.ensure_correct_config()
        
    def load_config(self):
        """Load configuration from file."""
        try:
            # Get absolute path to config file
            config_path = os.path.abspath(os.path.join(self.base_dir, 'config.json'))
            print(f"Loading config from: {config_path}")
            
            if not os.path.exists(config_path):
                print(f"Config file not found at {config_path}")
                return self.config if hasattr(self, 'config') else {}
                
            # Read the raw file content first
            with open(config_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
                print("Raw file content:", raw_content)
                config = json.loads(raw_content)
                
            # Remove any OpenAI-specific keys that might have been added
            if 'api_key' in config:
                del config['api_key']
            if 'model' in config:
                del config['model']
                
            # Ensure required sections exist
            if 'azure_speech' not in config:
                config['azure_speech'] = {}
                
            # Get the azure_speech section
            speech_config = config.get('azure_speech', {})
            
            # Print raw config for debugging
            print("Raw config from file:", config)
            print("Raw azure_speech section:", speech_config)
            
            # Store the original API key
            original_api_key = speech_config.get('api_key', '')
            print(f"Original API key from file: {original_api_key}")
            
            # Ensure the API key is properly loaded and preserved
            if not original_api_key:
                print("API key is missing or empty in azure_speech config")
            else:
                # Make sure we keep the original API key
                speech_config['api_key'] = original_api_key
                print(f"Preserved API key: {original_api_key}")
                
            if not speech_config.get('region'):
                print("Region is missing or empty in azure_speech config")
                speech_config['region'] = 'eastus'
                
            # Update the config with the validated speech_config
            config['azure_speech'] = speech_config
            
            # Save the config back to ensure it's properly formatted
            self.save_config(config)
                
            print("Successfully loaded configuration")
            print(f"Azure Speech config: {config.get('azure_speech', {})}")
            return config
            
        except json.JSONDecodeError as e:
            print(f"Error decoding config file: {str(e)}")
            if hasattr(self, 'config'):
                print("Returning existing configuration")
                return self.config
            print("No existing configuration found, returning empty dict")
            return {}
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            if hasattr(self, 'config'):
                print("Returning existing configuration")
                return self.config
            print("No existing configuration found, returning empty dict")
            return {}

    def save_config(self, config=None):
        """Save configuration to file."""
        try:
            if config is not None:
                self.config = config
                
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            # Ensure azure_speech section exists
            if 'azure_speech' not in self.config:
                self.config['azure_speech'] = {}
                
            # Save with proper formatting
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
                
            # Verify the save was successful
            with open(self.config_file, 'r', encoding='utf-8') as f:
                saved_config = json.load(f)
                if saved_config.get('azure_speech', {}).get('api_key') != self.config.get('azure_speech', {}).get('api_key'):
                    print("Warning: API key may not have been saved correctly")
                    
            return True
        except Exception as e:
            print(f"Error saving config: {str(e)}")
            return False
            
    def ensure_correct_config(self):
        """Ensure the config file has the correct API keys."""
        try:
            # Load the current config
            current_config = self.load_config()
            
            # Check if we need to update the speech API key
            speech_config = current_config.get('azure_speech', {})
            if speech_config.get('api_key') == current_config.get('azure_api_key'):
                print("Fixing incorrect speech API key...")
                # Update with the correct speech API key
                speech_config['api_key'] = "EqEmEYR5wbZF5EVkIB612IigqBsZW5sh3qhU6o97k2CeZc0ITP9NJQQJ99BFACYeBjFXJ3w3AAAYACOGXEbk"
                current_config['azure_speech'] = speech_config
                self.save_config(current_config)
                print("Updated speech API key in config")
                
            return True
        except Exception as e:
            print(f"Error ensuring correct config: {str(e)}")
            return False

    def get_azure_api_key(self):
        """Get Azure API key."""
        return self.config.get("azure_api_key", "")

    def set_azure_api_key(self, api_key):
        """Set Azure API key."""
        self.config["azure_api_key"] = api_key
        self.save_config()

    def get_azure_endpoint(self):
        """Get Azure endpoint."""
        return self.config.get("azure_endpoint", "")

    def set_azure_endpoint(self, endpoint):
        """Set Azure endpoint."""
        self.config["azure_endpoint"] = endpoint
        self.save_config()

    def get_azure_api_version(self):
        """Get Azure API version."""
        return self.config.get("azure_api_version", "2023-05-15")

    def get_azure_deployment(self, deployment_type):
        """Get Azure deployment name for the specified type."""
        deployments = self.config.get("azure_deployments", {})
        return deployments.get(deployment_type, "")

    def set_azure_deployment(self, deployment_type, deployment_name):
        """Set Azure deployment name for the specified type."""
        if "azure_deployments" not in self.config:
            self.config["azure_deployments"] = {}
        self.config["azure_deployments"][deployment_type] = deployment_name
        self.save_config()

    def get_pyannote_token(self):
        """Get Pyannote token."""
        return self.config.get("pyannote_token", "")

    def set_pyannote_token(self, token):
        """Set Pyannote token."""
        self.config["pyannote_token"] = token
        self.save_config()

    def get_temperature(self):
        """Get temperature setting."""
        return float(self.config.get("temperature", 0.7))

    def set_temperature(self, temperature):
        """Set temperature setting."""
        try:
            self.config["temperature"] = float(temperature)
            self.save_config()
            return True
        except ValueError:
            return False

    def get_language(self):
        """Get language setting."""
        return self.config.get("language", "en")

    def set_language(self, language):
        """Set language setting."""
        self.config["language"] = language
        self.save_config()

    def get_templates(self):
        """Get all templates."""
        return self.config.get("templates", {})

    def get_template(self, name):
        """Get specific template by name."""
        templates = self.get_templates()
        return templates.get(name, "")

    def add_template(self, name, content):
        """Add a new template."""
        if "templates" not in self.config:
            self.config["templates"] = {}
        self.config["templates"][name] = content
        self.save_config()

    def remove_template(self, name):
        """Remove a template."""
        if "templates" in self.config and name in self.config["templates"]:
            del self.config["templates"][name]
            self.save_config()

    def get_azure_speech_config(self):
        """Get Azure Speech configuration."""
        try:
            # Get the azure_speech section from the current config
            speech_config = self.config.get('azure_speech', {})
            print("Loading Azure Speech configuration...")
            print(f"Config content: {speech_config}")
            print(f"Available keys in azure_speech: {list(speech_config.keys())}")
            
            # Check if we have the required fields
            if not speech_config:
                print("No Azure Speech configuration found")
                return None
                
            # Get and validate required fields
            api_key = speech_config.get('api_key', '')
            region = speech_config.get('region', '')
            
            # Print diagnostic information
            print(f"API Key length: {len(api_key)}")
            print(f"Region: {region}")
            
            # Validate the API key
            if not api_key or not isinstance(api_key, str):
                print("API key is empty or invalid in azure_speech config")
                # Try to reload from file
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    file_speech_config = file_config.get('azure_speech', {})
                    api_key = file_speech_config.get('api_key', '')
                    if api_key:
                        print("Successfully reloaded API key from file")
                        speech_config['api_key'] = api_key
                        self.config['azure_speech'] = speech_config
                        self.save_config()
                    else:
                        return None
                
            # Validate the region
            if not region or not isinstance(region, str):
                print("Region is empty or invalid in azure_speech config")
                return None
                
            # Return a copy of the configuration to prevent modification
            return {
                'api_key': api_key,
                'region': region,
                'endpoint': speech_config.get('endpoint', '')
            }
        except Exception as e:
            print(f"Error getting Azure Speech configuration: {str(e)}")
            return None

    def set_azure_speech_config(self, api_key, region, endpoint=None):
        """Set Azure Speech configuration."""
        try:
            # Validate inputs
            if not api_key or not isinstance(api_key, str):
                raise ValueError("API key must be a non-empty string")
            if not region or not isinstance(region, str):
                raise ValueError("Region must be a non-empty string")
                
            # Strip whitespace
            api_key = api_key.strip()
            region = region.strip()
            
            # Ensure azure_speech section exists
            if 'azure_speech' not in self.config:
                self.config['azure_speech'] = {}
                
            # Set the configuration
            self.config['azure_speech']['api_key'] = api_key
            self.config['azure_speech']['region'] = region
            if endpoint:
                self.config['azure_speech']['endpoint'] = endpoint
                
            # Save the configuration
            if not self.save_config():
                raise Exception("Failed to save Azure Speech configuration")
                
            # Verify the configuration was saved correctly
            saved_config = self.get_azure_speech_config()
            if not saved_config or saved_config.get('api_key') != api_key:
                raise Exception("Failed to save Azure Speech configuration")
                
            return True
        except Exception as e:
            print(f"Error setting Azure Speech configuration: {str(e)}")
            return False

    def get_azure_speech_api_key(self):
        """Get Azure Speech API key."""
        # Print diagnostic information
        print("Attempting to load Azure Speech API key from config...")
        
        # Get the azure_speech section
        speech_config = self.config.get("azure_speech", {})
        
        # Print diagnostic information about the config
        print(f"Config content: {speech_config}")
        print(f"Available keys in azure_speech: {list(speech_config.keys())}")
        
        # Get the API key
        api_key = speech_config.get("api_key", "")
        
        # Print diagnostic information about the API key
        print(f"Raw API key type: {type(api_key)}")
        print(f"Raw API key length: {len(api_key)}")
        
        # Validate the API key
        if not api_key or not isinstance(api_key, str):
            print("API key is empty or invalid in azure_speech config")
            return ""
            
        # Ensure the API key is properly formatted
        api_key = api_key.strip()
        
        # Additional validation
        if len(api_key) < 10:  # Basic length check
            print("API key appears to be too short")
            return ""
            
        print("Successfully loaded API key from config")
        return api_key

    def get_azure_speech_region(self):
        """Get Azure Speech region."""
        speech_config = self.config.get("azure_speech", {})
        return speech_config.get("region", "eastus")

def add_save_all_settings_button(panel, parent_frame):
    """Create a prominent save all settings button and add it to the panel."""
    
    # Create a function to safely save settings and verify the result
    def on_save_button_click(event):
        try:
            # Get Azure API key and endpoint from appropriate input fields
            if hasattr(parent_frame, 'azure_api_key_input'):
                api_key = parent_frame.azure_api_key_input.GetValue().strip()
            else:
                api_key = parent_frame.config_manager.get_azure_api_key()

            if hasattr(parent_frame, 'azure_endpoint_input'):
                endpoint = parent_frame.azure_endpoint_input.GetValue().strip()
            else:
                endpoint = parent_frame.config_manager.get_azure_endpoint()

            # Get Azure OpenAI deployments
            if hasattr(parent_frame, 'chat_deployment_input'):
                chat_deployment = parent_frame.chat_deployment_input.GetValue().strip()
            else:
                chat_deployment = parent_frame.config_manager.get_azure_deployment("chat")

            if hasattr(parent_frame, 'whisper_deployment_input'):
                whisper_deployment = parent_frame.whisper_deployment_input.GetValue().strip()
            else:
                whisper_deployment = parent_frame.config_manager.get_azure_deployment("whisper")

            # Get HF token from appropriate input field
            if hasattr(parent_frame, 'hf_input'):
                hf_token = parent_frame.hf_input.GetValue().strip()
            elif hasattr(parent_frame, 'pyannote_token_input'):
                hf_token = parent_frame.pyannote_token_input.GetValue().strip()
            else:
                hf_token = parent_frame.hf_token if hasattr(parent_frame, 'hf_token') else ""
                
            # Update parent_frame attributes
            parent_frame.hf_token = hf_token
            
            # Save to config manager
            parent_frame.config_manager.set_azure_api_key(api_key)
            parent_frame.config_manager.set_azure_endpoint(endpoint)
            parent_frame.config_manager.set_azure_deployment("chat", chat_deployment)
            parent_frame.config_manager.set_azure_deployment("whisper", whisper_deployment)
            parent_frame.config_manager.set_pyannote_token(hf_token)
            
            # Set environment variables
            os.environ["AZURE_OPENAI_API_KEY"] = api_key
            os.environ["AZURE_OPENAI_ENDPOINT"] = endpoint
            os.environ["HF_TOKEN"] = hf_token
            
            # Update client if needed
            if hasattr(parent_frame, 'client') and api_key and endpoint:
                try:
                    parent_frame.client = AzureOpenAI(
                        api_key=api_key,
                        api_version=parent_frame.config_manager.get_azure_api_version(),
                        azure_endpoint=endpoint
                    )
                    wx.MessageBox("Settings saved successfully!", "Success", wx.OK | wx.ICON_INFORMATION)
                except Exception as e:
                    wx.MessageBox(f"Error setting Azure OpenAI client: {e}", "Error", wx.OK | wx.ICON_ERROR)
            else:
                wx.MessageBox("Settings saved successfully!", "Success", wx.OK | wx.ICON_INFORMATION)
                
        except Exception as e:
            wx.MessageBox(f"Error saving settings: {e}", "Error", wx.OK | wx.ICON_ERROR)
            
    # Create button sizer
    button_sizer = wx.BoxSizer(wx.VERTICAL)
    
    # Create save button with custom style
    save_button = wx.Button(panel, label="Save All Settings")
    save_button.SetBackgroundColour(wx.Colour(50, 200, 50))  # Green background
    save_button.SetForegroundColour(wx.Colour(255, 255, 255))  # White text
    save_button.Bind(wx.EVT_BUTTON, on_save_button_click)
    
    button_sizer.Add(save_button, 0, wx.EXPAND | wx.ALL, 10)
    
    # Add a verify button to check settings at any time
    verify_button = wx.Button(panel, label="Verify Saved Settings")
    verify_button.Bind(wx.EVT_BUTTON, lambda e: wx.MessageBox(
        verify_saved_settings(parent_frame.config_manager.config_file),
        "Current Settings", wx.OK | wx.ICON_INFORMATION))
    
    button_sizer.Add(verify_button, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
    
    return button_sizer

def verify_saved_settings(config_file_path):
    """Verify that settings are properly saved and return a status message."""
    try:
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        status_lines = []
        
        # Check Azure OpenAI settings
        azure_api_key = config.get('azure_api_key', '')
        azure_endpoint = config.get('azure_endpoint', '')
        azure_api_version = config.get('azure_api_version', '')
        azure_deployments = config.get('azure_deployments', {})
        
        if azure_api_key:
            status_lines.append("✓ Azure OpenAI API Key is set")
        else:
            status_lines.append("✗ Azure OpenAI API Key is not set")
            
        if azure_endpoint and azure_endpoint != "https://your-resource-name.openai.azure.com":
            status_lines.append("✓ Azure OpenAI Endpoint is set")
        else:
            status_lines.append("✗ Azure OpenAI Endpoint is not set")
            
        if azure_api_version:
            status_lines.append(f"✓ Azure OpenAI API Version: {azure_api_version}")
        else:
            status_lines.append("✗ Azure OpenAI API Version is not set")
            
        if azure_deployments:
            status_lines.append("\nAzure OpenAI Deployments:")
            for deployment_type, deployment_name in azure_deployments.items():
                status_lines.append(f"✓ {deployment_type}: {deployment_name}")
        else:
            status_lines.append("✗ No Azure OpenAI deployments configured")
        
        # Check PyAnnote token
        pyannote_token = config.get('pyannote_token', '')
        if pyannote_token:
            status_lines.append("\n✓ PyAnnote Token is set")
        else:
            status_lines.append("\n✗ PyAnnote Token is not set")
        
        # Check language setting
        language = config.get('language', '')
        if language:
            status_lines.append(f"✓ Language is set to: {language}")
        else:
            status_lines.append("✗ Language is not set")
        
        # Check templates
        templates = config.get('templates', {})
        if templates:
            status_lines.append(f"\nFound {len(templates)} templates:")
            for name in templates.keys():
                status_lines.append(f"✓ {name}")
        else:
            status_lines.append("\n✗ No templates found")
        
        return "\n".join(status_lines)
        
    except FileNotFoundError:
        return "✗ Config file not found"
    except json.JSONDecodeError:
        return "✗ Config file is not valid JSON"
    except Exception as e:
        return f"✗ Error verifying settings: {str(e)}"

# Fallback: Load AZURE_STORAGE_CONNECTION_STRING and AZURE_STORAGE_CONTAINER from config.json if not set in environment
try:
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        if os.environ.get('AZURE_STORAGE_CONNECTION_STRING') is None:
            storage_conn = config_data.get('AZURE_STORAGE_CONNECTION_STRING')
            if not storage_conn:
                storage_conn = config_data.get('azure_storage_connection_string')
            if not storage_conn:
                storage_conn = config_data.get('azure_speech', {}).get('AZURE_STORAGE_CONNECTION_STRING')
            if not storage_conn:
                storage_conn = config_data.get('azure_speech', {}).get('azure_storage_connection_string')
            if not storage_conn:
                storage_conn = config_data.get('storage_connection_string')
            if storage_conn:
                os.environ['AZURE_STORAGE_CONNECTION_STRING'] = storage_conn
        if os.environ.get('AZURE_STORAGE_CONTAINER') is None:
            storage_container = config_data.get('AZURE_STORAGE_CONTAINER')
            if not storage_container:
                storage_container = config_data.get('azure_storage_container')
            if not storage_container:
                storage_container = config_data.get('azure_speech', {}).get('AZURE_STORAGE_CONTAINER')
            if not storage_container:
                storage_container = config_data.get('azure_speech', {}).get('azure_storage_container')
            if not storage_container:
                storage_container = config_data.get('storage_container')
            if storage_container:
                os.environ['AZURE_STORAGE_CONTAINER'] = storage_container
except Exception as e:
    print(f"Warning: Could not load storage connection info from config.json: {e}")

if __name__ == "__main__":
    try:
        print("Starting KeszAudio...")
        
        # Check if application is frozen (running as executable)
        is_frozen = getattr(sys, 'frozen', False)
        is_windows = platform.system() == 'windows' or platform.system() == 'Windows'
        is_macos = platform.system() == 'darwin'
        
        # For frozen applications, set specific environment variables
        if is_frozen:
            # Set application name for display
            app_name = "KeszAudio"
            
            # For macOS, make sure GUI works correctly in app bundle
            if is_macos:
                # Set additional environment variables for macOS App Bundle
                os.environ['PYTHONFRAMEWORK'] = '1'
                os.environ['DISPLAY'] = ':0'
                os.environ['WX_NO_DISPLAY_CHECK'] = '1'
                os.environ['WXMAC_NO_NATIVE_MENUBAR'] = '1'
            
            # For Windows, set working directory correctly
            if is_windows:
                # Make sure working directory is set to the executable location
                if hasattr(sys, '_MEIPASS'):
                    # PyInstaller sets _MEIPASS
                    os.chdir(sys._MEIPASS)
                else:
                    # Otherwise use executable's directory
                    os.chdir(os.path.dirname(sys.executable))
        
        # Handle CLI mode explicitly
        if "--cli" in sys.argv:
            # CLI mode explicitly requested
            main()
        else:
            # Ensure required directories exist and get base directory
            # Critical step: this must succeed before proceeding
            try:
                APP_BASE_DIR = ensure_directories()
                print(f"Using application directory: {APP_BASE_DIR}")
                
                # Verify directories are created and writable
                for subdir in ["Transcripts", "Summaries", "diarization_cache"]:
                    test_dir = os.path.join(APP_BASE_DIR, subdir)
                    if not os.path.exists(test_dir):
                        os.makedirs(test_dir, exist_ok=True)
                    
                    # Verify we can write to the directory
                    test_file = os.path.join(test_dir, ".write_test")
                    try:
                        with open(test_file, 'w') as f:
                            f.write("test")
                        if os.path.exists(test_file):
                            os.remove(test_file)
                        print(f"Directory {test_dir} is writable")
                    except Exception as e:
                        print(f"WARNING: Directory {test_dir} is not writable: {e}")
                        # Try to find an alternative location
                        APP_BASE_DIR = os.path.join(os.path.expanduser("~"), "KeszAudio")
                        os.makedirs(APP_BASE_DIR, exist_ok=True)
                        print(f"Using alternative directory: {APP_BASE_DIR}")
                        break
            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                print(f"FATAL ERROR setting up directories: {error_msg}")
                
                # Use a simple directory in the user's home folder as a last resort
                APP_BASE_DIR = os.path.join(os.path.expanduser("~"), "KeszAudio")
                os.makedirs(APP_BASE_DIR, exist_ok=True)
                print(f"Using fallback directory: {APP_BASE_DIR}")
            
            # Short delay to ensure filesystem operations complete
            import time
            time.sleep(0.5)
            
            # Use main() function to start the appropriate mode
            sys.exit(main())
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"FATAL ERROR: {error_msg}")
        
        # Try to show error dialog if possible
        try:
            # Check if we can import wx and if it's available
            if WX_AVAILABLE:
                if not hasattr(wx, 'App') or not wx.GetApp():
                    error_app = wx.App(False)
                wx.MessageBox(f"Fatal error starting application:\n\n{error_msg}", 
                             "Application Error", wx.OK | wx.ICON_ERROR)
            else:
                # wx is not available, just print the error
                print("Could not show error dialog because wxPython is not available.")
        except Exception as dialog_error:
            # If we can't even show a dialog, just report the error
            print(f"Could not show error dialog: {dialog_error}")
        
        sys.exit(1)