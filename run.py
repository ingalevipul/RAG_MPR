#!/usr/bin/env python3
"""
run.py  -  One-command launcher for Supply Chain Risk Assessment System

USAGE (run from INSIDE the supply_chain_risk/ folder):
    cd supply_chain_risk
    python run.py
"""

import os
import sys
import subprocess

# Always run from the directory this file lives in (supply_chain_risk/)
# This is the critical fix for the ModuleNotFoundError on Windows.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT_DIR)

# Also add root to sys.path so imports work when called from elsewhere
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def check_env():
    env_path = os.path.join(ROOT_DIR, ".env")
    example_path = os.path.join(ROOT_DIR, ".env.example")

    if not os.path.exists(env_path):
        print("  No .env file found. Copying from .env.example...")
        import shutil
        shutil.copy(example_path, env_path)
        print("  Please edit .env and add your GOOGLE_API_KEY")
        print("  Get a FREE key at: https://aistudio.google.com/apikey")
        sys.exit(1)

    # Read .env manually to avoid dotenv import issues before deps are installed
    api_key = ""
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("GOOGLE_API_KEY="):
                api_key = line.split("=", 1)[1].strip().strip('"').strip("'")

    if not api_key or api_key == "your_gemini_api_key_here":
        print("  GOOGLE_API_KEY not set in .env")
        print("  Get a FREE key at: https://aistudio.google.com/apikey")
        sys.exit(1)

    print("  API key found OK")


def install_deps():
    req = os.path.join(ROOT_DIR, "requirements.txt")
    print("  Installing dependencies (this may take a minute the first time)...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", req, "-q"],
        cwd=ROOT_DIR,
    )
    print("  Dependencies installed")


def run_server():
    print("\n  Starting Supply Chain Risk Assessment Server...")
    print("  Dashboard: http://localhost:8000")
    print("  API Docs:  http://localhost:8000/docs")
    print("  Press Ctrl+C to stop\n")

    # Launch uvicorn from ROOT_DIR so Python can find the `backend` package
    subprocess.run(
        [
            sys.executable, "-m", "uvicorn",
            "backend.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
        ],
        cwd=ROOT_DIR,   # <-- this is what fixes ModuleNotFoundError
    )


if __name__ == "__main__":
    print("=" * 55)
    print("  SUPPLYGUARD - AI Supply Chain Risk Assessment")
    print("=" * 55)
    check_env()
    install_deps()
    run_server()
