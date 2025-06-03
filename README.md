# Fresh Frame 
Menu Image Processing & Optimization

## Purpose of the Tool

This tool is designed to process and optimize images automatically
## Main Functionalities

  - **Automatic Orientation Correction**: Detects and corrects image orientation based on EXIF metadata, preventing incorrectly rotated images.
  - **Background Removal**: Uses advanced algorithms to remove the background and create images with transparency (PNG format), ideal for professional presentations.
  - **Main Object Detection**: Employs YOLOv3 (an artificial intelligence model) to identify the main object in the image and crop the image around it.
  - **Size Optimization**: Reduces file size without compromising quality, ensuring that images are smaller than 10 MB and in JPEG or PNG format.
  - **Quality Control**: Checks the sharpness and lighting of the image, applying automatic corrections to improve visual quality.
  - **Format Conversion**: Supports conversion from less common formats (e.g., HEIC) to standard formats such as JPEG or PNG.
  - **Batch Processing**: Allows for the automatic processing of entire folders of images, saving time.

## Installation & Usage

1. **Download `yolov3.weights`**  
   Download the file from the [official YOLOv3 weights](https://github.com/patrick013/Object-Detection---Yolov3/blob/master/model/yolov3.weights) and place it in your project's **root directory**.

2. **Install Tesseract OCR**
   ```bash
   sudo apt install tesseract-ocr
   ```

3. **Install dependencies**  
   Install dependencies using Poetry:
   ```bash
   poetry install
   ```

4. **Process images**  
   Run the following command to process all images in the `input_images` folder:
   ```bash
   poetry run image_process
   ```
   Run the process with `Text Removal` feature:
   ```bash
   poetry run image_process --enable-text-removal
   ```

5. **View results**  
   Processed images will be saved to the `processed_images` folder.

---

### Notes:
- Ensure the `input_images` folder exists in your project root and contains images to process.
- If the folders (`input_images`/`processed_images`) don’t exist, create them first:
  ```bash
  mkdir -p input_images processed_images
  ```

## Technologies Used

  - **OpenCV**: For image processing (cropping, resizing, color correction).
  - **YOLOv3**: For object detection and intelligent cropping.
  - **Pillow**: For managing image formats and EXIF metadata.
  - **Rembg**: For background removal with high-quality results.
  - **Pillow-HEIC**: For supporting HEIC files (common on Apple devices).
  - **Tesseract OCR**: For removing Text

