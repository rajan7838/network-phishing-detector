import subprocess
import sys
import os

def run_command(cmd, desc):
    print(f"\n{'='*60}")
    print(f"Running: {desc}")
    print(f"Command: {cmd}")
    print('='*60)
    result = subprocess.run(cmd, shell=True, text=True)
    if result.returncode != 0:
        print(f"Command failed with code {result.returncode}")
        sys.exit(result.returncode)
    else:
        print(f"Finished: {desc}")

def train():
    run_command("python src/train.py", "Training CNN Model")

def streamlit():
    run_command("streamlit run app/streamlit_app.py --server.port 8501", "Launching Streamlit App")

def docker_build():
    run_command("docker build -t phishing-detector .", "Building Docker Image")

def docker_run():
    run_command("docker run -p 8501:8501 phishing-detector", "Running Docker Container")

def main():
    print("\n" + "="*60)
    print("NETWORK PHISHING DETECTION")
    print("="*60)
    print("1. Train Model")
    print("2. Run Streamlit App")
    print("3. Docker Build")
    print("4. Docker Run")
    print("5. Run All")
    print("="*60)
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == '1':
        train()
    elif choice == '2':
        streamlit()
    elif choice == '3':
        docker_build()
    elif choice == '4':
        docker_run()
    elif choice == '5':
        train()
        docker_build()
        print("\nProcess finished. Run: python run.py 4")
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()
