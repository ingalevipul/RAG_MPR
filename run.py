#!/usr/bin/env python3

import os
import sys
import subprocess

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT_DIR)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def check_api_key():
    """Check if GOOGLE_API_KEY exists in .env"""
    env_path = os.path.join(ROOT_DIR, ".env")
    
    if not os.path.exists(env_path):
        print("Error: .env file not found")
        print("Create .env and add: GOOGLE_API_KEY=your_key_here")
        print("Get key at: https://aistudio.google.com/apikey")
        sys.exit(1)

    with open(env_path) as f:
        for line in f:
            if line.strip().startswith("GOOGLE_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key and key != "your_gemini_api_key_here":
                    return
    
    print("Error: GOOGLE_API_KEY not set in .env")
    print("Get key at: https://aistudio.google.com/apikey")
    sys.exit(1)


def install_requirements():
    """Install dependencies from requirements.txt"""
    req = os.path.join(ROOT_DIR, "requirements.txt")
    if not os.path.exists(req):
        print("Warning: requirements.txt not found, skipping...")
        return
    
    print("Installing dependencies...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", req, "-q"],
        cwd=ROOT_DIR
    )


if __name__ == "__main__":
    check_api_key()
    install_requirements()
    
    print("\nStarting server at http://localhost:8000")
    print("Press Ctrl+C to stop\n")
    
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "backend.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ], cwd=ROOT_DIR)