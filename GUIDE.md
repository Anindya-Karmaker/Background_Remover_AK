# Quick Start Guide: Background Remover

Welcome to Background Remover! This guide will walk you through the main features and settings to help you get professional-looking results quickly and easily.

## Table of Contents
1. [The Basics: Loading & Saving](#the-basics-loading--saving)
2. [A Recommended Workflow](#a-recommended-workflow)
3. [Understanding the Tools & Settings](#understanding-the-tools--settings)
   - [AI Tools Tab](#tab-1-ai-tools)
   - [Manual Edit Tab](#tab-2-manual-edit)
4. [Viewing & Navigation](#viewing--navigation)

---

## The Basics: Loading & Saving

- **Open an Image**: Click the **Open...** button or press `Ctrl+O`.
- **Paste from Clipboard**: Copy an image from anywhere, then click the **Paste Image** button or press `Ctrl+V`.
- **Drag & Drop**: Simply drag an image file from your computer directly onto the application window.
- **Save Your Work**: Click **Save As...** or press `Ctrl+Shift+S`.

> **Pro Tip:** Saving as a **PNG** file will preserve the transparent background. Saving as a **JPG** will add a white background.

---

## A Recommended Workflow

For the most efficient editing, follow these steps:

1.  **Start with AI**: Go to the **AI Tools** tab and click **Remove Background**. This will do 90% of the work for you instantly.
2.  **Crop the Image**: If needed, go to the **Manual Edit** tab, click **Select Crop Area**, draw a box around your subject, and click **Apply Crop**.
3.  **Refine the Edges**: Use the **Drawing Tools** (Keep/Remove brushes) or the **Magic Wand** to clean up any mistakes the AI made.
4.  **Add Your Background**: Once you are happy with the cutout, use the **Fill Background** button to add a solid color.

---

## Understanding the Tools & Settings

Here’s a breakdown of what each setting does.

### Tab 1: AI Tools

This tab is for the main automatic background removal.

-   **Model**: This dropdown lets you choose different AI models.
    -   `u2net`: The best all-around model. **Always start with this one.**
    -   `u2net_human_seg`: Specialized for cutting out people.
    -   `isnet-anime`: Use this for anime, cartoons, or graphic illustrations.
    -   Other models: Good alternatives if the main ones don't give a perfect result.

-   **Enable Alpha Matting**: An advanced feature for fine details.
    -   **What it does**: Creates smoother, more detailed edges. It's fantastic for **hair, fur, and fuzzy objects**.
    -   **When to use it**: Check this box if the standard removal leaves blocky or rough edges on complex objects.
    -   **FG / BG Threshold**: Controls how aggressively the matting works. You can usually leave these at their defaults.
    -   **Erode Size**: Slightly shrinks the final cutout. This is great for removing a thin "glow" or halo of the old background that might be left around the edges. A small value like 5-10 is often helpful.

### Tab 2: Manual Edit

This is where you can manually perfect your image.

#### Drawing & Crop Tools
- **Crop**:
    1. Click **Select Crop Area** (your cursor becomes a cross).
    2. Click and drag a box on your image.
    3. Click **Apply Crop** to make the change permanent.
- **Mark Keep (Green Brush)**: Use this to paint back parts of the **original image** that were accidentally removed.
- **Mark Remove (Red Brush)**: Use this to erase parts of the current image, making them transparent.
- **Brush Size**: Adjust the slider to make your brush bigger or smaller for detail work.
- **Apply Keep/Remove Marks**: **(Important!)** After using the brushes, you **must** click this button to apply your changes.

#### Magic Wand Tool
- **Magic Wand Select**: Click on any color in your image to select all connected pixels of a similar color. A blue overlay will show your selection.
- **Tolerance**: This is the most important setting for the wand.
    - **Low Tolerance (e.g., 10)**: Selects only colors that are *very* similar to where you clicked.
    - **High Tolerance (e.g., 50)**: Selects a much wider range of similar colors.
- **Remove Selected Area**: Makes the blue selected area transparent.
- **Keep Selected Area**: Makes **everything else** transparent, keeping only the blue selected area.

#### Other Manual Tools
- **Fill Background**: Adds a solid color background to your image. You can change or remove it at any time.
- **Remove Fill**: Removes the color background and shows the transparent checkerboard again.
- **Show Original (Hold)**: Click and hold this button to quickly see what your original image looked like for comparison.

---

## Viewing & Navigation

- **Zoom**: Use `Ctrl` + `Mouse Wheel` or the shortcuts:
    - Zoom In: `Ctrl` + `+`
    - Zoom Out: `Ctrl` + `-`
- **Fit to View**: Press `Ctrl+0` to instantly fit the entire image perfectly in the window.
- **Pan**: Use the scrollbars on the sides of the image preview to move around when zoomed in.
