Background Remover AK
A high-performance, local background removal tool designed for students, researchers, and creators. Quickly and easily remove or replace image backgrounds for posters, presentations, and illustrations. The application runs entirely on your machine, ensuring your data remains private.
![alt text](https://img.shields.io/badge/platform-Windows%20%7C%20macOS-blue)

![alt text](https://img.shields.io/badge/license-MIT-green)
<img width="1400" alt="Main Interface" src="https://github.com/user-attachments/assets/6a7b9a09-bfc6-4b21-8f15-90edcea6db02" />
Key Features
Fully Local & Private: All processing happens on your computer. No data is ever uploaded to the cloud.
AI-Powered Removal: Uses state-of-the-art models via rembg for one-click background removal.
Precise Edge Refinement: Fine-tune results with intuitive "Keep" and "Remove" brushes.
Magic Wand Tool: Quickly select and modify large, contiguous color areas with adjustable tolerance.
Non-Destructive Background Fill: Add a solid color background that can be changed or removed at any time.
Intuitive Workflow: Features undo/redo, zoom/pan, and full keyboard shortcut support for efficient editing.
Drag & Drop: Simply drag an image file onto the window to start editing.
Getting Started (Running from Source)
Follow these steps to run the application directly using Python.
1. Prerequisites
Python 3.8+: Ensure you have Python installed on your system. You can download it from python.org.
2. Installation
Clone or Download the Repository
Download the BACKGROUND_REMOVER_AK.py file and, if you wish to compile, the associated .spec and hook files.
Set Up a Virtual Environment (Recommended)
Generated bash
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
Use code with caution.
Bash
Install Requirements
You can install the required packages directly using this command:
Generated bash
pip install PySide6 Pillow numpy scipy rembg requests onnxruntime
Use code with caution.
Bash
3. Run the Application
With your virtual environment active, run the script:
Generated bash
python BACKGROUND_REMOVER_AK.py
Use code with caution.
Bash
First-Time Run Note: The first time you use the background removal feature, the program will download the required AI models (several hundred MB). This may take a few minutes depending on your internet connection. Subsequent runs will be instant.
Creating a Standalone Application (Optional)
If you want to compile the script into a standalone .exe (Windows) or .app (macOS) file that doesn't require a Python installation, follow these instructions.
1. Install PyInstaller
Generated bash
pip install pyinstaller
Use code with caution.
Bash
2. Download the Models (Crucial Step)
Before compiling, you must run the script at least once to ensure the rembg models are downloaded to your user directory (C:\Users\YourUser\.u2net on Windows or /Users/YourUser/.u2net on macOS). PyInstaller needs these files to bundle them into your application.
3. Compile for Windows
Place the build_windows.spec file in the same directory as your Python script.
(Optional) Create an icon file named icon.ico and place it in an assets sub-folder.
Run the PyInstaller command from your terminal:
Generated bash
pyinstaller build_windows.spec
Use code with caution.
Bash
The final application will be in the dist folder as Background Remover AK.exe.
4. Compile for macOS
Place build_macos.spec and runtime_hook.py in the same directory as your Python script.
(Optional) Create an icon file named icon.icns and place it in an assets sub-folder.
Run the PyInstaller command from your terminal:
Generated bash
pyinstaller build_macos.spec
Use code with caution.
Bash
The final application will be in the dist folder as Background Remover AK.app.
Fix macOS Gatekeeper Issue: To run the app, you must first run this command in the terminal to allow it:
Generated bash
xattr -cr "dist/Background Remover AK.app"
Use code with caution.
Bash
Screenshots
<table>
<tr>
<td align="center"><b>Crop Tool</b></td>
<td align="center"><b>Magic Wand Selection</b></td>
</tr>
<tr>
<td><img width="700" alt="image" src="https://github.com/user-attachments/assets/48bbed82-b8ef-4415-b197-33ce1656e36e" /></td>
<td><img width="700" alt="image" src="https://github.com/user-attachments/assets/158e8777-22fa-4639-aac7-3bcb187ff0c0" /></td>
</tr>
<tr>
<td align="center"><b>Refinement Brushes</b></td>
<td align="center"><b>Color Fill Background</b></td>
</tr>
<tr>
<td><img width="700" alt="image" src="https://github.com/user-attachments/assets/5695e44a-a930-4552-9c06-d9258428474d" /></td>
<td><img width="700" alt="image" src="https://github.com/user-attachments/assets/d5be3e37-0951-46d1-87ec-0badf60db0b" /></td>
</tr>
</table>
Acknowledgements
This tool is powered by the incredible rembg library by Daniel Gatis.
The sample cell image is provided by the NIH NIAID Bioart collection (Credit: NIAID).
License
This project is licensed under the MIT License. See the LICENSE file for details.
