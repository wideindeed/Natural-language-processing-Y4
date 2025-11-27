import sys
import os
import subprocess

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    src_folder = os.path.join(project_root, 'src')
    
    main_script = os.path.join(src_folder, 'main.py')

    print("="*50)
    print("🚀 NLP Pipeline Launcher")
    print(f"📂 Project Root: {project_root}")
    print(f"🐍 Python Executable: {sys.executable}")
    print("="*50)

    if not os.path.exists(main_script):
        print(f"❌ Error: Could not find 'main.py' at:\n{main_script}")
        input("\nPress Enter to exit...")
        return

    env = os.environ.copy()
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = src_folder + os.pathsep + env['PYTHONPATH']
    else:
        env['PYTHONPATH'] = src_folder

    try:
        print("\nStarting pipeline...\n")
        subprocess.call([sys.executable, main_script], env=env)
        
        print("\n✅ Pipeline finished successfully.")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user.")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

    print("\n" + "="*50)
    input("Press Enter to close this window...")

if __name__ == "__main__":
    main()