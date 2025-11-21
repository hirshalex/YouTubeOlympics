"""
Download Kaggle dataset into data/raw using the Kaggle API.
Run this file inside VS Code (Run File / Run Python File in Terminal or Run Cell in Interactive window).
No manual terminal steps required.

Notes:
- Make sure ~/.kaggle/kaggle.json exists on the remote server (you already uploaded it).
- Select the correct Python interpreter in the bottom-right of VS Code (choose your remote venv).
"""

import os, sys, traceback

def ensure_package(pkg):
    try:
        __import__(pkg)
        return True
    except ImportError:
        print(f"Package '{pkg}' not found. Installing into the current Python environment...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            print(f"Installed {pkg}.")
            return True
        except Exception as e:
            print("Failed to install", pkg, ":", e)
            return False

# Ensure Kaggle client
if not ensure_package("kaggle"):
    print("Unable to install 'kaggle'. Please open the Python interpreter selection in VS Code and try again.")
    raise SystemExit(1)

# Now run download
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    home = os.path.expanduser("~")
    kaggle_key = os.path.join(home, ".kaggle", "kaggle.json")
    if not os.path.exists(kaggle_key):
        raise FileNotFoundError(f"No kaggle.json found at {kaggle_key}. Place your kaggle.json there (VS Code: upload to ~/.kaggle/kaggle.json).")

    dest = os.path.join(os.getcwd(), "data", "raw")
    os.makedirs(dest, exist_ok=True)

    print("Authenticating to Kaggle API...")
    api = KaggleApi()
    api.authenticate()
    print("Authenticated. Downloading dataset...")

    # dataset slug
    slug = "rsrishav/youtube-trending-video-dataset"
    api.dataset_download_files(slug, path=dest, unzip=True, quiet=False)
    print("Download + unzip complete. Files written to:", dest)

    print("\nListing files downloaded:")
    for root, dirs, files in os.walk(dest):
        for f in files[:50]:
            print(" -", os.path.join(root, f))
        break

except Exception as exc:
    print("Error during download:")
    traceback.print_exc()
    raise
