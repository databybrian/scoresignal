# check_dependencies.py
import importlib
import sys

def check_dependencies():
    """Check if all required dependencies are installed"""
    dependencies = [
        'lightgbm',
        'xgboost', 
        'catboost',
        'sklearn',
        'pandas',
        'numpy',
        'joblib',
        'pytz'
    ]
    
    missing = []
    for dep in dependencies:
        try:
            importlib.import_module(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep}")
            missing.append(dep)
    
    if missing:
        print(f"\n🚨 Missing dependencies: {missing}")
        print("Install with: pip install " + " ".join(missing))
        return False
    else:
        print("\n🎉 All dependencies installed!")
        return True

if __name__ == "__main__":
    check_dependencies()