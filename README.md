# Background Remover AK

A high-performance, local background removal tool designed for students, researchers, and creators. Quickly and easily remove or replace image backgrounds for posters, presentations, and illustrations. The application runs entirely on your machine, ensuring your data remains private.

![Platform: Windows, macOS](https://img.shields.io/badge/platform-Windows%20%7C%20macOS-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

<img width="999" alt="image" src="https://github.com/user-attachments/assets/b4ed3bf4-967f-4d3e-a288-158961f377f2" />


## Table of Contents
- [Key Features](#key-features)
- [Getting Started (Running from Source)](#getting-started-running-from-source)
- [Creating a Standalone Application (Optional)](#creating-a-standalone-application-optional)
- [Screenshots](#screenshots)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Key Features

*   **Fully Local & Private:** All processing happens on your computer. No data is ever uploaded to the cloud.
*   **AI-Powered Removal:** Uses state-of-the-art models via `rembg` for one-click background removal.
*   **Precise Edge Refinement:** Fine-tune results with intuitive "Keep" and "Remove" brushes.
*   **Magic Wand Tool:** Quickly select and modify large, contiguous color areas with adjustable tolerance.
*   **Non-Destructive Background Fill:** Add a solid color background that can be changed or removed at any time.
*   **Intuitive Workflow:** Features undo/redo, zoom/pan, and full keyboard shortcut support for efficient editing.
*   **Drag & Drop:** Simply drag an image file onto the window to start editing.

---

## Getting Started (Running from Source)

Follow these steps to run the application directly using Python.

### 1. Prerequisites

- **Python 3.8+**: Ensure you have Python installed on your system. You can download it from [python.org](https://python.org).

### 2. Get the Code and Requirements File

First, download the project files, especially `BACKGROUND_REMOVER_AK.py` and the new `requirements.txt` file.

### 3. Set Up a Virtual Environment (Recommended)

```bash
# Create a virtual environment in your project folder
python -m venv venv

# Activate the environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 4. Install Requirements

With your virtual environment active, install all the necessary packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
python BACKGROUND_REMOVER_AK.py
```

> **First-Time Run Note:** The first time you use the background removal feature, the program will download the required AI models (several hundred MB). This may take a few minutes depending on your internet connection. Subsequent runs will be instant.

---

## Creating a Standalone Application (Optional)

If you want to compile the script into a single `.exe` (Windows) or `.app` (macOS) file that doesn't require a Python installation, follow these instructions.

### 1. Install PyInstaller

```bash
pip install pyinstaller
```

### 2. Download the Models (Crucial Step)

Before compiling, you **must** run the script at least once (`python BACKGROUND_REMOVER_AK.py`) to ensure the `rembg` models are downloaded to your user directory. PyInstaller needs these files to bundle them into your application.

### 3. Compile the Application

#### For Windows

1.  Place the `build_windows.spec` file in the same directory as your Python script.
2.  Download the `assets` sub-folder and place it in the same directory as your Python script.
3.  Run the PyInstaller command from your terminal:
    ```bash
    pyinstaller build_windows.spec
    ```
4.  The final application will be in the `dist` folder as `Background Remover.exe`.

#### For macOS

1.  Place `build_macos.spec` and `runtime_hook.py` in the same directory as your Python script.
2.  Download the `assets` sub-folder and place it in the same directory as your Python script.
3.  Run the PyInstaller command from your terminal:
    ```bash
    pyinstaller build_macos.spec
    ```
4.  The final application will be in the `dist` folder as `Background Remover.app`.
5.  **Fix macOS Gatekeeper Issue**: To run the app, you may need to run this command first in the terminal (Only try if it does not work) :
    ```bash
    xattr -cr "dist/Background Remover AK.app"
    ```

---

## Screenshots

<table>
  <tr>
    <td align="center"><b>Crop Tool</b></td>
    <td align="center"><b>Magic Wand Selection</b></td>
  </tr>
  <tr>
    <td><img width="700" alt="Crop Tool in Action" src="https://github.com/user-attachments/assets/dd04c796-559a-484f-869c-78cd6dbccf0d" /></td>
    <td><img width="700" alt="Magic Wand Selecting an Area" src="https://github.com/user-attachments/assets/edbc8dfe-d334-450d-b5ee-2a60c9a1cfeb" /></td>
  </tr>
  <tr>
    <td align="center"><b>Refinement Brushes</b></td>
    <td align="center"><b>Color Fill Background</b></td>
  </tr>
  <tr>
    <td><img width="700" alt="Using Brushes to Refine Edges" src="https://github.com/user-attachments/assets/6290f6ac-a16e-42bf-919c-62b37206bb13" /></td>
    <td><img width="700" alt="Filling the Background with a Solid Color" src="https://github.com/user-attachments/assets/3adaa353-4ea8-4508-9550-d8a21620b47d" /></td>
  </tr>
</table>

---

## Acknowledgements

*   This tool is powered by the incredible [**rembg** library](https://github.com/danielgatis/rembg) by Daniel Gatis.
*   The sample cell image is provided by the [NIH NIAID Bioart collection](https://bioart.niaid.nih.gov/bioart/231) (Credit: NIAID).

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
