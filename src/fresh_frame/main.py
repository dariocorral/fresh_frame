from fresh_frame.image_processing import BatchImageProcessor
import os
import pytesseract 
import sys 
import argparse # Added import

def main():
    parser = argparse.ArgumentParser(description="Batch process images with Fresh Frame.")
    parser.add_argument(
        '--enable-text-removal',
        default=False, # Changed default to False
        action=argparse.BooleanOptionalAction, # Creates --enable-text-removal and --no-enable-text-removal
        help="Enable or disable the text removal feature (default: disabled)."
    )
    parser.add_argument(
        '--input-folder',
        default='input_images',
        type=str,
        help="Folder with input images (default: input_images)."
    )
    parser.add_argument(
        '--output-folder',
        default='processed_images',
        type=str,
        help="Folder for processed output images (default: processed_images)."
    )
    parser.add_argument(
        '--yolo-config',
        default='yolov3.cfg',
        type=str,
        help="Path to YOLO config file (default: yolov3.cfg)."
    )
    parser.add_argument(
        '--yolo-weights',
        default='yolov3.weights',
        type=str,
        help="Path to YOLO weights file (default: yolov3.weights)."
    )

    args = parser.parse_args()

    # --- Configuration from parsed arguments ---
    input_folder = args.input_folder
    output_folder = args.output_folder
    yolo_config_path = args.yolo_config
    yolo_weights_path = args.yolo_weights
    enable_text_removal = args.enable_text_removal # Use parsed argument
    # --- End Configuration ---

    # Check if Tesseract is available
    try:
        pytesseract.get_tesseract_version()
        print(f"Tesseract OCR version {pytesseract.get_tesseract_version()} detected.")
    except pytesseract.TesseractNotFoundError:
        print("Error: Tesseract OCR is not installed or not found in your system's PATH.")
        print("Text removal feature will not work if enabled. Please install Tesseract OCR.")
        print("Installation guide: https://tesseract-ocr.github.io/tessdoc/Installation.html")
        if enable_text_removal:
            print("Exiting because text removal is enabled but Tesseract is not found.")
            sys.exit(1) 
    except Exception as e:
        print(f"Could not verify Tesseract version: {e}")

    # Check if YOLO files exist
    if not os.path.exists(yolo_config_path) or not os.path.exists(yolo_weights_path):
         print(f"Error: YOLOv3 config ('{yolo_config_path}') or weights ('{yolo_weights_path}') not found.")
         print("Please ensure these files are present. Download from appropriate sources if necessary.")
         print("Object detection feature will not work. Exiting.")
         sys.exit(1) 

    # Check if input directory exists
    if not os.path.exists(input_folder):
         print(f"Error: Input directory '{input_folder}' not found. Please create it and add images.")
         sys.exit(1) 

    # Ensure output directory exists (BatchImageProcessor also does this, but good practice here too)
    os.makedirs(output_folder, exist_ok=True)

    batch_processor = BatchImageProcessor(input_folder, output_folder)
    
    # Set configuration on the batch_processor instance
    batch_processor.yolo_config_path = yolo_config_path
    batch_processor.yolo_weights_path = yolo_weights_path
    batch_processor.perform_text_removal = enable_text_removal
    
    batch_processor.process_folder()
    print(f"\nProcessing complete. Processed images are in '{output_folder}' folder.")
    print("Look for files ending with '_bg_original.*' and '_bg_removed.*'")

if __name__ == '__main__':
    main()
