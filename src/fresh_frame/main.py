from fresh_frame.image_processing import BatchImageProcessor

def main():
    input_folder = 'input_images'  
    output_folder = 'processed_images'  

    batch_processor = BatchImageProcessor(input_folder, output_folder)
    batch_processor.process_folder()
    print(f"\nProcessing complete. Processed images are in '{output_folder}' folder.")

if __name__ == '__main__':
    main()
