Python312

# for windows git bash terminal
cd PeopleTrackAI
/c/Users/admin/AppData/Local/Programs/Python/Python312/python.exe -m venv .venv
source .venv/Scripts/activate

pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio
pip install ultralytics supervision opencv-python deep-sort-realtime
pip install pyqt5
pip install filterpy scikit-learn lap pandas seaborn


# for mac terminal
/opt/homebrew/bin/python312 -m venv .venv
source .venv/bin/activate

<!-- pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio -->
pip install ultralytics supervision opencv-python deep-sort-realtime
pip install pyqt5
pip install filterpy scikit-learn lap pandas seaborn
