#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil
import tempfile
import site
import importlib
from pathlib import Path
import glob
import platform
import datetime
import urllib.request
import zipfile
import tarfile
from PyInstaller.utils.win32.versioninfo import VSVersionInfo, FixedFileInfo, StringFileInfo, StringTable, StringStruct, VarFileInfo, VarStruct

def download_ffmpeg():
    """Download FFmpeg for the current platform and return the path to the executable."""
    print("Downloading FFmpeg for bundling with the application...")
    temp_dir = tempfile.mkdtemp(prefix="ffmpeg_temp_")
    ffmpeg_dir = os.path.join(temp_dir, "ffmpeg")
    os.makedirs(ffmpeg_dir, exist_ok=True)
    
    system = platform.system().lower()
    is_64bit = sys.maxsize > 2**32
    
    if system == 'windows':
        # Download Windows static build
        url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
        zip_path = os.path.join(temp_dir, "ffmpeg.zip")
        
        print(f"Downloading FFmpeg from {url}")
        urllib.request.urlretrieve(url, zip_path)
        
        # Extract
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find the extracted folder (it might have a version name)
        extracted_dirs = [d for d in os.listdir(temp_dir) if d.startswith('ffmpeg-') and os.path.isdir(os.path.join(temp_dir, d))]
        if extracted_dirs:
            extracted_dir = os.path.join(temp_dir, extracted_dirs[0])
            
            # Copy the bin directory to our ffmpeg directory
            bin_dir = os.path.join(extracted_dir, 'bin')
            if os.path.exists(bin_dir):
                for file in os.listdir(bin_dir):
                    file_path = os.path.join(bin_dir, file)
                    if file.endswith('.exe') and os.path.isfile(file_path):
                        shutil.copy2(file_path, ffmpeg_dir)
                
                print(f"FFmpeg executables copied to {ffmpeg_dir}")
                return ffmpeg_dir
    
    elif system == 'darwin':  # macOS
        if platform.machine() == 'arm64':  # Apple Silicon
            url = "https://evermeet.cx/ffmpeg/getrelease/zip"
        else:  # Intel
            url = "https://evermeet.cx/ffmpeg/getrelease/zip"
        
        zip_path = os.path.join(temp_dir, "ffmpeg.zip")
        
        print(f"Downloading FFmpeg from {url}")
        urllib.request.urlretrieve(url, zip_path)
        
        # Extract
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(ffmpeg_dir)
        
        print(f"FFmpeg executables copied to {ffmpeg_dir}")
        return ffmpeg_dir
    
    elif system == 'linux':
        # For Linux, we'll use static builds
        if is_64bit:
            url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
        else:
            url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-i686-static.tar.xz"
        
        tar_path = os.path.join(temp_dir, "ffmpeg.tar.xz")
        
        print(f"Downloading FFmpeg from {url}")
        urllib.request.urlretrieve(url, tar_path)
        
        # Extract
        with tarfile.open(tar_path, 'r:xz') as tar_ref:
            tar_ref.extractall(temp_dir)
        
        # Find the extracted folder (it might have a version name)
        extracted_dirs = [d for d in os.listdir(temp_dir) if d.startswith('ffmpeg-') and os.path.isdir(os.path.join(temp_dir, d))]
        if extracted_dirs:
            extracted_dir = os.path.join(temp_dir, extracted_dirs[0])
            
            # Copy the executables to our ffmpeg directory
            for file in ['ffmpeg', 'ffprobe']:
                file_path = os.path.join(extracted_dir, file)
                if os.path.isfile(file_path):
                    shutil.copy2(file_path, ffmpeg_dir)
                    # Make sure they're executable
                    os.chmod(os.path.join(ffmpeg_dir, file), 0o755)
            
            print(f"FFmpeg executables copied to {ffmpeg_dir}")
            return ffmpeg_dir
    
    print("Warning: Could not download FFmpeg for your platform. Audio conversion may not work.")
    return None

def find_package_paths():
    """Find paths to required packages."""
    package_paths = {}
    
    # Get site-packages directory
    site_packages = site.getsitepackages()[0]
    
    # Find speechbrain path
    try:
        import speechbrain
        package_paths['speechbrain'] = os.path.dirname(speechbrain.__file__)
    except ImportError:
        print("Warning: speechbrain not found")
    
    # Find pyannote path
    try:
        import pyannote.audio
        package_paths['pyannote'] = os.path.dirname(pyannote.audio.__file__)
    except ImportError:
        print("Warning: pyannote.audio not found")
        
    # Find Azure Speech SDK path
    try:
        import azure.cognitiveservices.speech
        package_paths['azure.cognitiveservices.speech'] = os.path.dirname(azure.cognitiveservices.speech.__file__)
    except ImportError:
        print("Warning: azure.cognitiveservices.speech not found")
    
    # Find lightning_fabric path
    try:
        import lightning_fabric
        package_paths['lightning_fabric'] = os.path.dirname(lightning_fabric.__file__)
    except ImportError:
        print("Warning: lightning_fabric not found")
    
    return package_paths

def create_version_info():
    """Create needed version files for libraries that might require them."""
    temp_dir = tempfile.mkdtemp(prefix="build_temp_")
    os.makedirs(os.path.join(temp_dir, "lightning_fabric"), exist_ok=True)
    
    # Create version_info.py for lightning_fabric
    with open(os.path.join(temp_dir, "lightning_fabric", "version_info.py"), 'w') as f:
        f.write('version_info = "2.2.0"\n__version__ = version_info\n')
    
    return temp_dir

def create_speechbrain_utils_fix(temp_dir, package_paths):
    """Create a fixed version of speechbrain's importutils.py to avoid file system access."""
    if 'speechbrain' in package_paths:
        print("Creating SpeechBrain utility fix...")
        utils_dir = os.path.join(temp_dir, "speechbrain", "utils")
        os.makedirs(utils_dir, exist_ok=True)
        
        # Copy __init__.py
        src_init = os.path.join(package_paths['speechbrain'], "utils", "__init__.py")
        if os.path.exists(src_init):
            with open(src_init, 'r') as src:
                with open(os.path.join(utils_dir, "__init__.py"), 'w') as dst:
                    dst.write(src.read())
        
        # Create a patched version of importutils.py
        importutils_path = os.path.join(package_paths['speechbrain'], "utils", "importutils.py")
        if os.path.exists(importutils_path):
            with open(importutils_path, 'r') as f:
                content = f.read()
            
            # Replace problematic function that tries to read directories
            patched_content = content.replace(
                "def find_imports(path):",
                """def find_imports(path):
    # PyInstaller-friendly version that doesn't rely on filesystem
    return []"""
            )
            
            with open(os.path.join(utils_dir, "importutils.py"), 'w') as f:
                f.write(patched_content)
            
            print("Created patched SpeechBrain importutils.py")
    else:
        print("Warning: SpeechBrain package not found, skipping patch")

def generate_spec_file(temp_dir, package_paths, icon_path=None, ffmpeg_dir=None):
    """Generate a PyInstaller spec file with all required configurations."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    main_script = os.path.join(current_dir, 'main.py')
    
    # Convert paths to use forward slashes for spec file
    main_script = main_script.replace('\\', '/')
    current_dir = current_dir.replace('\\', '/')
    if temp_dir:
        temp_dir = temp_dir.replace('\\', '/')
    if ffmpeg_dir:
        ffmpeg_dir = ffmpeg_dir.replace('\\', '/')
    
    if icon_path is None:
        # Try to detect icon file
        for ext in ['.ico', '.icns']:
            potential_icon = os.path.join(current_dir, f'icon{ext}')
            if os.path.exists(potential_icon):
                icon_path = potential_icon.replace('\\', '/')
                break
    
    # Prepare data files list with forward slashes
    datas_list = []
    binaries_list = []
    datas_list.append(f"('{current_dir}/*.json', '.')")
    datas_list.append(f"('{temp_dir}/lightning_fabric/version_info.py', 'lightning_fabric')")
    
    # Add FFmpeg binaries if available
    if ffmpeg_dir and os.path.exists(ffmpeg_dir):
        datas_list.append(f"('{ffmpeg_dir}', 'ffmpeg')")
    
    # Add SpeechBrain patch
    if os.path.exists(os.path.join(temp_dir, "speechbrain")):
        datas_list.append(f"('{temp_dir}/speechbrain', 'speechbrain')")

    # Add model files and data from packages
    for package, path in package_paths.items():
        # Convert path to use forward slashes
        path = path.replace('\\', '/')
        
        if package == 'speechbrain':
            # Add all speechbrain data files
            pretrained_path = os.path.join(path, 'pretrained').replace('\\', '/')
            if os.path.exists(pretrained_path):
                datas_list.append(f"('{pretrained_path}', 'speechbrain/pretrained')")
        
        if package == 'pyannote':
            # Add pyannote data directories
            for subpkg in ['audio', 'core', 'database', 'metrics', 'pipeline']:
                pkg_path = os.path.join(path, subpkg).replace('\\', '/')
                if os.path.exists(pkg_path):
                    datas_list.append(f"('{pkg_path}', 'pyannote/{subpkg}')")
                    
        if package == 'azure.cognitiveservices.speech':
            # Add Azure Speech SDK DLLs to binaries list to ensure they're in root
            sdk_path = os.path.dirname(path)
            if os.path.exists(sdk_path):
                datas_list.append(f"('{sdk_path}', 'azure/cognitiveservices/speech')")
                # Add DLLs to binaries list for root placement
                for dll in glob.glob(os.path.join(sdk_path, '*.dll').replace('\\', '/')):
                    binaries_list.append(f"('{dll}', '.')")
    
    # Determine platform-specific settings
    is_windows = sys.platform.startswith('win')
    is_macos = sys.platform == 'darwin'
    
    # Set console mode based on platform (False for GUI apps)
    console_mode = True
    
    # App metadata
    app_name = "KeszAudio"
    app_version = "1.0.0"
    app_company = "KeszAudio"
    app_description = "Audio transcription and processing tool"
    
    # Windows-specific version information
    version_info = ''
    if is_windows:
        version_info = f'''
# Windows version info
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=({app_version.replace(".", ", ")}, 0),
    prodvers=({app_version.replace(".", ", ")}, 0),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
        StringTable(
          u'040904B0',
          [StringStruct(u'CompanyName', u'{app_company}'),
           StringStruct(u'FileDescription', u'{app_description}'),
           StringStruct(u'FileVersion', u'{app_version}'),
           StringStruct(u'InternalName', u'{app_name}'),
           StringStruct(u'LegalCopyright', u'(C) {datetime.datetime.now().year} {app_company}'),
           StringStruct(u'OriginalFilename', u'{app_name}.exe'),
           StringStruct(u'ProductName', u'{app_name}'),
           StringStruct(u'ProductVersion', u'{app_version}')])
      ]), 
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
'''
    
    # Create the spec file content
    spec_content = f'''# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata
import datetime
from PyInstaller.utils.win32.versioninfo import VSVersionInfo, FixedFileInfo, StringFileInfo, StringTable, StringStruct, VarFileInfo, VarStruct

block_cipher = None

{version_info}

# Collect all modules needed for complex dependencies
pyannote_datas = []
pyannote_binaries = []
pyannote_hiddenimports = []

pytorch_datas = []
pytorch_binaries = []
pytorch_hiddenimports = []

lightning_datas = []
lightning_binaries = []
lightning_hiddenimports = []

speechbrain_datas = []
speechbrain_binaries = []
speechbrain_hiddenimports = []

azure_speech_datas = []
azure_speech_binaries = []
azure_speech_hiddenimports = []

# Collect all for problematic modules
for module_name in ['pyannote', 'pyannote.audio', 'pytorch_lightning', 'lightning_fabric', 'torchaudio', 'speechbrain', 'azure.cognitiveservices.speech']:
    try:
        datas, binaries, hiddenimports = collect_all(module_name)
        if module_name.startswith('pyannote'):
            pyannote_datas.extend(datas)
            pyannote_binaries.extend(binaries)
            pyannote_hiddenimports.extend(hiddenimports)
        elif module_name.startswith('torch'):
            pytorch_datas.extend(datas)
            pytorch_binaries.extend(binaries)
            pytorch_hiddenimports.extend(hiddenimports)
        elif module_name.startswith('speech'):
            speechbrain_datas.extend(datas)
            speechbrain_binaries.extend(binaries)
            speechbrain_hiddenimports.extend(hiddenimports)
        elif module_name.startswith('azure'):
            azure_speech_datas.extend(datas)
            azure_speech_binaries.extend(binaries)
            azure_speech_hiddenimports.extend(hiddenimports)
        else:
            lightning_datas.extend(datas)
            lightning_binaries.extend(binaries)
            lightning_hiddenimports.extend(hiddenimports)
    except Exception as e:
        print(f"Warning: Could not collect all for {{module_name}}: {{e}}")

# Additional imports for torch modules
pytorch_hiddenimports.extend(['torch', 'torch.nn', 'torch.optim', 'torch.utils', 'torch.distributions'])
lightning_hiddenimports.extend(['lightning_fabric', 'lightning_fabric.__version__'])
speechbrain_hiddenimports.extend(['speechbrain.utils.importutils', 'speechbrain.utils.data_utils'])
azure_speech_hiddenimports.extend(['azure.cognitiveservices.speech'])

# Add metadata for huggingface libraries
hf_metadata = []
for pkg in ['huggingface_hub', 'transformers', 'tokenizers']:
    try:
        hf_metadata.extend(copy_metadata(pkg))
    except Exception:
        pass

# Add Azure Speech SDK DLLs to binaries
azure_speech_binaries.extend([{", ".join(binaries_list)}])

a = Analysis(
    ['{main_script}'],
    pathex=['{current_dir}'],
    binaries=[*pyannote_binaries, *pytorch_binaries, *lightning_binaries, *speechbrain_binaries, *azure_speech_binaries],
    datas=[
        # Include all data files
        *pyannote_datas,
        *pytorch_datas, 
        *lightning_datas,
        *speechbrain_datas,
        *azure_speech_datas,
        *hf_metadata,
        {", ".join(datas_list)}
    ],
    hiddenimports=[
        *pyannote_hiddenimports,
        *pytorch_hiddenimports,
        *lightning_hiddenimports,
        *speechbrain_hiddenimports,
        *azure_speech_hiddenimports,
        'numpy', 'librosa', 'soundfile', 'pyaudio', 'pydub', 'torch', 
        'pyannote.audio', 'pyannote.core', 'pyannote.pipeline',
        'wx', 'wx.adv', 'concurrent.futures', 'openai', 'requests', 
        'pyannote.audio.pipelines.speaker_verification',
        'pyannote.audio.pipelines.speaker_diarization',
        'speechbrain', 'speechbrain.inference.interfaces', 'speechbrain.inference.classifiers',
        'speechbrain.inference.encoders', 'speechbrain.pretrained.interfaces',
        'lightning_fabric.__version__',
        'importlib_metadata',
        'packaging',
        'typing_extensions',
        'subprocess',
        'ffmpeg',
        'wave', 
        're',
        'hashlib',
        'uuid',
        'json',
        'threading',
        'io'
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

'''

    # Add platform-specific sections to the spec file
    if is_macos:
        # For macOS, create a .app bundle
        spec_content += f'''
# Create macOS .app bundle
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='KeszAudio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console={console_mode},
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    {f"icon='{icon_path}'" if icon_path else ""}
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='KeszAudio'
)

# Create .app bundle structure
app = BUNDLE(
    coll,
    name='KeszAudio.app',
    icon=None,
    bundle_identifier='com.keszaudio.app',
    info_plist={{
        'NSHighResolutionCapable': 'True',
        'NSRequiresAquaSystemAppearance': 'False',
        'NSMicrophoneUsageDescription': 'KeszAudio needs access to the microphone for audio recording and processing.',
        'CFBundleShortVersionString': '{app_version}',
        'CFBundleInfoDictionaryVersion': '6.0',
        'CFBundleExecutable': 'KeszAudio',
        'CFBundleName': 'KeszAudio',
        'CFBundleDisplayName': 'KeszAudio',
        'CFBundlePackageType': 'APPL',
        'LSApplicationCategoryType': 'public.app-category.utilities',
        'LSMinimumSystemVersion': '10.14',
    }},
)
'''
    elif is_windows:
        # For Windows, create a proper GUI .exe
        spec_content += f'''
# Create Windows .exe
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='KeszAudio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console={console_mode},
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# Create the directory structure with all dependencies
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='KeszAudio',
)
'''
    else:
        # For Linux and other platforms, use standard output format
        spec_content += f'''
# Create the executable
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='KeszAudio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console={console_mode},
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    {f"icon='{icon_path}'" if icon_path else ""}
)

# Create the collection
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='KeszAudio',
)
'''
    
    # Write the spec file
    spec_file = os.path.join(temp_dir, 'app.spec')
    with open(spec_file, 'w') as f:
        f.write(spec_content)
    
    return spec_file

def check_pyinstaller():
    """Check if PyInstaller is installed, install if needed."""
    try:
        subprocess.run([sys.executable, '-m', 'PyInstaller', '--version'], 
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print("PyInstaller is already installed.")
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        print("PyInstaller not found. Installing...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'pyinstaller'], check=True)
            print("PyInstaller installed successfully.")
            return True
        except subprocess.SubprocessError:
            print("Failed to install PyInstaller. Please install manually: pip install pyinstaller")
            return False

def build_app(one_file=False):
    """Build the application using PyInstaller."""
    if not check_pyinstaller():
        return False
    
    print("Setting up build environment...")
    temp_dir = create_version_info()
    print(f"Created temporary directory at: {temp_dir}")
    
    # Find package paths
    package_paths = find_package_paths()
    
    # Create SpeechBrain utility fix
    create_speechbrain_utils_fix(temp_dir, package_paths)
    
    # Download FFmpeg
    ffmpeg_dir = download_ffmpeg()
    
    # Find icon file - use platform-specific icons if available
    icon_path = None
    is_windows = sys.platform.startswith('win')
    is_macos = sys.platform == 'darwin'
    
    # Look for platform-specific icons first
    if is_windows:
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icon.ico')
    elif is_macos:
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icon.icns')
    
    # Fall back to any available icon
    if not icon_path or not os.path.exists(icon_path):
        for ext in ['.ico', '.icns']:
            potential_icon = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'icon{ext}')
            if os.path.exists(potential_icon):
                icon_path = potential_icon
                print(f"Found icon file: {icon_path}")
                break
    
    if not icon_path:
        print("No icon file found. The application will use the default PyInstaller icon.")
    
    # Generate spec file
    spec_file = generate_spec_file(temp_dir, package_paths, icon_path, ffmpeg_dir)
    print(f"Generated spec file at: {spec_file}")
    
    # Run PyInstaller
    print("\nBuilding application with PyInstaller (this may take a while)...")
    try:
        build_cmd = [
            sys.executable, 
            '-m', 'PyInstaller',
            '--clean',
            '--distpath', './dist',
            '--workpath', './build',
        ]
        
        # Add platform-specific options
        if is_macos:
            # For macOS, add --target-architecture if needed
            if platform.machine() == 'arm64':
                build_cmd.append('--target-architecture=universal2')
            
        build_cmd.append(spec_file)
        
        print(f"Running build command: {' '.join(build_cmd)}")
        subprocess.run(build_cmd, check=True)
        
        print("\nBuild completed successfully!")
        
        if is_macos:
            print(f"Application bundle can be found at: {os.path.abspath('./dist/KeszAudio.app')}")
            
            # Set executable permissions for the app binary
            app_binary = os.path.abspath('./dist/KeszAudio.app/Contents/MacOS/KeszAudio')
            if os.path.exists(app_binary):
                os.chmod(app_binary, 0o755)
                print(f"Set executable permissions for {app_binary}")
                
        elif is_windows:
            print(f"Application files can be found in: {os.path.abspath('./dist/KeszAudio')}")
            print("The executable is KeszAudio.exe in this directory.")
            
            # Create a shortcut on Windows if possible
            try:
                exe_path = os.path.abspath('./dist/KeszAudio/KeszAudio.exe')
                shortcut_path = os.path.abspath('./dist/KeszAudio.lnk')
                
                # Try to create a Windows shortcut using PowerShell
                if os.path.exists(exe_path):
                    ps_command = f'''
                    $WshShell = New-Object -comObject WScript.Shell
                    $Shortcut = $WshShell.CreateShortcut("{shortcut_path}")
                    $Shortcut.TargetPath = "{exe_path}"
                    $Shortcut.WorkingDirectory = "{os.path.dirname(exe_path)}"
                    $Shortcut.Save()
                    '''
                    
                    subprocess.run(['powershell', '-Command', ps_command], shell=True, check=False)
                    if os.path.exists(shortcut_path):
                        print(f"Created shortcut at: {shortcut_path}")
            except Exception as e:
                print(f"Note: Could not create shortcut: {e}")
                
        else:
            print(f"Application files can be found in: {os.path.abspath('./dist/KeszAudio')}")
        
        # Create a one-file version if requested
        if one_file and (is_windows or not is_macos):  # One-file not ideal for macOS
            print("\nCreating one-file executable version...")
            onefile_spec = os.path.join(temp_dir, 'app_onefile.spec')
            
            # Read the spec file
            with open(spec_file, 'r') as f:
                spec_content = f.read()
            
            # Modify for one-file build
            onefile_content = spec_content.replace(
                "exclude_binaries=True",
                "exclude_binaries=False"
            )
            
            # Replace COLLECT section and BUNDLE section if it exists
            import re
            onefile_content = re.sub(
                r'# Create the collection\ncoll = COLLECT\([^)]+\)',
                '',
                onefile_content
            )
            
            # Remove BUNDLE section for macOS
            onefile_content = re.sub(
                r'# Create \.app bundle structure\napp = BUNDLE\([^)]+\)',
                '',
                onefile_content
            )
            
            # Add runtime_tmpdir to EXE
            onefile_content = onefile_content.replace(
                f"console={str(not is_windows).lower()},",
                f"console={str(not is_windows).lower()},\n    runtime_tmpdir=None,"
            )
            
            # Write the onefile spec file
            with open(onefile_spec, 'w') as f:
                f.write(onefile_content)
            
            print(f"Generated one-file spec at: {onefile_spec}")
            
            # Build one-file version
            onefile_cmd = [
                sys.executable, 
                '-m', 'PyInstaller',
                '--clean',
                '--distpath', './dist',
                '--workpath', './build',
                onefile_spec
            ]
            
            print(f"Running one-file build command: {' '.join(onefile_cmd)}")
            subprocess.run(onefile_cmd, check=True)
            
            print("\nOne-file build completed successfully!")
            if is_windows:
                print(f"One-file executable can be found at: {os.path.abspath('./dist/KeszAudio.exe')}")
            else:
                print(f"One-file executable can be found at: {os.path.abspath('./dist/KeszAudio')}")
        
        return True
    except subprocess.SubprocessError as e:
        print(f"Build failed: {e}")
        return False
    finally:
        print(f"Cleaning up temporary directory: {temp_dir}")
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Could not clean up temporary directory: {e}")

def main():
    # Set environment variables to bypass macOS GUI restrictions
    os.environ['WXMAC_NO_NATIVE_MENUBAR'] = '1'
    os.environ['PYOBJC_DISABLE_CONFIRMATION'] = '1'
    os.environ['QT_MAC_WANTS_LAYER'] = '1'
    os.environ['PYTHONHASHSEED'] = '1'
    os.environ['WX_NO_NATIVE'] = '1'
    os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'
    
    # Check if the user wants to build the app
    # Handle both `--build` and combined `--build--onefile` cases
    should_build = False
    one_file = False
    
    for arg in sys.argv[1:]:
        if arg == '--build' or arg.startswith('--build'):
            should_build = True
        if '--onefile' in arg:
            one_file = True
    
    if should_build:
        build_app(one_file)
        return 0
    
    # Otherwise, run the app normally
    main_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main.py')
    
    # Ask if user wants GUI or CLI mode
    try:
        mode = input("Run in GUI mode (y/n)? ").strip().lower()
        if mode == 'y' or mode == 'yes':
            # Try running with pythonw for macOS GUI
            if sys.platform == 'darwin':
                try:
                    # Check if pythonw exists
                    pythonw_path = os.path.join(os.path.dirname(sys.executable), 'pythonw')
                    if os.path.exists(pythonw_path):
                        print("Starting GUI with pythonw...")
                        subprocess.run([pythonw_path, main_script])
                        return 0
                except Exception:
                    pass
            
            # Fall back to standard python with GUI
            print("Starting GUI mode...")
            subprocess.run([sys.executable, main_script])
        else:
            # Run in CLI mode
            print("Starting CLI mode...")
            subprocess.run([sys.executable, main_script, "--cli"])
    except KeyboardInterrupt:
        print("\nApplication startup canceled by user.")
    except Exception as e:
        print(f"Error running application: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 