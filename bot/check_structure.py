# bot/check_structure.py
import os
import sys

def check_structure():
    print("🔍 Checking project structure...")
    
    # Current working directory (where you run the script from)
    print(f"📁 Current working directory: {os.getcwd()}")
    
    # Where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"📁 Bot directory: {script_dir}")
    
    # Project root (one level up from bot)
    project_root = os.path.dirname(script_dir)
    print(f"📁 Project root: {project_root}")
    
    # List files in project root
    print(f"📂 Files in project root:")
    for file in os.listdir(project_root):
        print(f"   - {file}")
    
    # Check if main.py exists in project root
    main_path = os.path.join(project_root, "main.py")
    print(f"🔎 Looking for main.py at: {main_path}")
    print(f"✅ File exists: {os.path.exists(main_path)}")
    
    # List files in bot directory
    print(f"📂 Files in bot directory:")
    for file in os.listdir(script_dir):
        print(f"   - {file}")

if __name__ == "__main__":
    check_structure()