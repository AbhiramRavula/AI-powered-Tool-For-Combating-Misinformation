#!/usr/bin/env python3
"""
TruthGuard Setup Script
Automatically detects system and installs compatible dependencies
"""

import subprocess
import sys
import platform
import os

def run_command(cmd):
    """Run a command and return success status"""
    try:
        subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def detect_system():
    """Detect system specifications"""
    system_info = {
        'os': platform.system(),
        'architecture': platform.machine(),
        'python_version': sys.version_info
    }
    
    print(f"🔍 Detected system:")
    print(f"   OS: {system_info['os']}")
    print(f"   Architecture: {system_info['architecture']}")
    print(f"   Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    return system_info

def install_pytorch():
    """Install PyTorch with system-appropriate version"""
    print("\n🔧 Installing PyTorch...")
    
    # Try different PyTorch installation methods
    pytorch_commands = [
        # Latest stable version (CPU only - most compatible)
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
        
        # Alternative: Let pip decide the best version
        "pip install torch torchvision torchaudio",
        
        # Fallback: Specific version ranges
        "pip install 'torch>=2.0.0' 'torchvision>=0.15.0' 'torchaudio>=2.0.0'",
    ]
    
    for i, cmd in enumerate(pytorch_commands, 1):
        print(f"   Attempt {i}: {cmd}")
        if run_command(cmd):
            print("   ✅ PyTorch installed successfully!")
            return True
        else:
            print(f"   ❌ Attempt {i} failed, trying next method...")
    
    print("   ⚠️  All PyTorch installation attempts failed!")
    return False

def install_other_dependencies():
    """Install other required packages"""
    print("\n📦 Installing other dependencies...")
    
    # Core dependencies that usually work
    core_deps = [
        "flask>=3.0.0",
        "flask-cors>=4.0.0", 
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "numpy>=1.24.0,<2.0.0",
        "transformers>=4.35.0"
    ]
    
    # Optional dependencies
    optional_deps = [
        "scipy>=1.10.0",
        "lxml>=4.9.0",
        "accelerate>=0.20.0",
        "safetensors>=0.4.0"
    ]
    
    # Install core dependencies
    print("   Installing core dependencies...")
    for dep in core_deps:
        print(f"   Installing {dep}...")
        if not run_command(f"pip install '{dep}'"):
            print(f"   ⚠️  Failed to install {dep}")
            return False
    
    print("   ✅ Core dependencies installed!")
    
    # Install optional dependencies (don't fail if these don't work)
    print("   Installing optional dependencies...")
    for dep in optional_deps:
        print(f"   Installing {dep}...")
        if run_command(f"pip install '{dep}'"):
            print(f"   ✅ {dep} installed")
        else:
            print(f"   ⚠️  {dep} failed (optional)")
    
    return True

def verify_installation():
    """Verify that key packages can be imported"""
    print("\n🧪 Verifying installation...")
    
    test_imports = [
        ("flask", "Flask web framework"),
        ("torch", "PyTorch machine learning"),
        ("transformers", "Hugging Face transformers"),
        ("requests", "HTTP requests"),
        ("bs4", "Beautiful Soup HTML parsing"),
        ("numpy", "NumPy numerical computing")
    ]
    
    all_good = True
    for module, description in test_imports:
        try:
            __import__(module)
            print(f"   ✅ {description}")
        except ImportError:
            print(f"   ❌ {description} - FAILED")
            all_good = False
    
    return all_good

def main():
    """Main setup function"""
    print("🚀 TruthGuard Setup Script")
    print("=" * 50)
    
    # Detect system
    system_info = detect_system()
    
    # Upgrade pip first
    print("\n⬆️  Upgrading pip...")
    run_command("python -m pip install --upgrade pip")
    
    # Install PyTorch
    if not install_pytorch():
        print("\n❌ PyTorch installation failed!")
        print("Please try manual installation:")
        print("   For CPU: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
        print("   For CUDA: Visit https://pytorch.org/get-started/locally/")
        return False
    
    # Install other dependencies
    if not install_other_dependencies():
        print("\n❌ Dependency installation failed!")
        return False
    
    # Verify installation
    if not verify_installation():
        print("\n⚠️  Some packages failed to install properly")
        print("The app might still work, but with limited functionality")
    else:
        print("\n🎉 All dependencies installed successfully!")
    
    print("\n🏁 Setup complete!")
    print("To run TruthGuard:")
    print("   1. cd backend")
    print("   2. python app.py")
    print("   3. Open frontend/index.html in browser")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)