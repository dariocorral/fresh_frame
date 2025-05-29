import cv2
import numpy as np
import os
from PIL import Image, ExifTags
from rembg import remove
from pillow_heif import register_heif_opener
import traceback
import pytesseract # Added import

register_heif_opener()  # Enable HEIC support in Pillow

class ImageProcessor:
    def __init__(self, image_path, output_base_path): 
        self.image_path = image_path
        self.output_base_path = output_base_path
        self.img_pil = None
        self.img_cv2 = None
        self.original_img_cv2_for_color_sampling = None
        # Add yolo_config and yolo_weights paths to the constructor
        self.yolo_config_path = "yolov3.cfg" # Default, can be overridden
        self.yolo_weights_path = "yolov3.weights" # Default, can be overridden
        try:
            # Step 1: Open image with Pillow
            img_pil_opened = Image.open(image_path)

            # Step 2: Correct orientation
            self.img_pil = self.correct_orientation(img_pil_opened)

            # Step 3: Convert to OpenCV format
            cv_img = self.pil_to_cv2(self.img_pil)

            # Step 4: Store copies if successful
            if cv_img is not None:
                self.img_cv2 = cv_img
                self.original_img_cv2_for_color_sampling = cv_img.copy() # Keep original for color sampling
                print(f"Image loaded successfully: {image_path}")
            else:
                print(f"Error: Failed to convert PIL image to OpenCV format for {image_path}")
                self._reset_image_attributes()

        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            self._reset_image_attributes()
        except Exception as e:
            print(f"Error during image loading/initial processing of {image_path}: {e}")
            traceback.print_exc() # More details on the error
            self._reset_image_attributes()

    def _reset_image_attributes(self):
        """Helper to reset image attributes on error."""
        self.img_pil = None
        self.img_cv2 = None
        self.original_img_cv2_for_color_sampling = None

    def correct_orientation(self, img_pil_input):
        """Applies EXIF orientation correction to a PIL image and returns the corrected PIL image."""
        if img_pil_input is None:
            return None
        img_pil_corrected = img_pil_input
        try:
            exif = img_pil_corrected.getexif()
            if exif:
                orientation_tag = next((k for k, v in ExifTags.TAGS.items() if v == 'Orientation'), None)
                if orientation_tag in exif:
                    orientation = exif[orientation_tag]
                    rotations = {3: 180, 6: 270, 8: 90}
                    if orientation in rotations:
                        print(f"Applying EXIF orientation correction (Rotation: {rotations[orientation]} degrees)")
                        img_pil_corrected = img_pil_corrected.rotate(rotations[orientation], expand=True)
        except Exception as e:
            print(f"Warning: Could not parse/apply EXIF orientation for {self.image_path}: {str(e)}")
        return img_pil_corrected

    def pil_to_cv2(self, img_pil):
        """Converts a PIL image to an OpenCV image (BGR or BGRA), handling various modes."""
        if img_pil is None: return None
        try:
            mode = img_pil.mode
            print(f"Converting PIL image mode: {mode}")
            # Convert complex modes first
            if mode == 'P': img_pil = img_pil.convert('RGBA'); mode = 'RGBA'
            elif mode == 'LA': img_pil = img_pil.convert('RGBA'); mode = 'RGBA'
            elif mode == 'L': img_pil = img_pil.convert('RGB'); mode = 'RGB'
            elif mode not in ['RGB', 'RGBA']:
                print(f"Attempting conversion of mode '{mode}' to RGBA...")
                try: img_pil = img_pil.convert('RGBA'); mode = 'RGBA'
                except Exception as conv_err:
                    print(f"Could not convert mode '{mode}' to RGBA: {conv_err}")
                    try:
                        print(f"Attempting conversion of mode '{mode}' to RGB...")
                        img_pil = img_pil.convert('RGB'); mode = 'RGB'
                    except Exception as conv_err2:
                        print(f"Could not convert mode '{mode}' to RGB either: {conv_err2}")
                        return None

            img_array = np.array(img_pil)
            if mode == 'RGB': return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            elif mode == 'RGBA': return cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGRA)
            else: # Should not be reached if conversions worked
                print(f"Warning: Unexpected array shape or mode after conversion: {img_array.shape}")
                return None
        except Exception as e:
            print(f"Error during PIL to CV2 conversion: {e}")
            traceback.print_exc()
            return None

    def remove_text_objects(self, confidence_threshold=60, inpaint_radius=5, padding=2, tesseract_config='--psm 11'):
        """Attempts to detect and remove text from self.img_cv2 using OCR and inpainting."""
        if self.img_cv2 is None:
            print("Skipping text removal: Image not loaded.")
            return

        print("Attempting to detect and remove text...")
        img_for_ocr_processing = self.img_cv2.copy()
        
        # Ensure image is BGR for Tesseract and inpainting
        original_alpha = None
        if img_for_ocr_processing.shape[2] == 4: # BGRA
            original_alpha = img_for_ocr_processing[:, :, 3].copy()
            img_bgr_for_ocr = cv2.cvtColor(img_for_ocr_processing, cv2.COLOR_BGRA2BGR)
        elif img_for_ocr_processing.shape[2] == 3: # BGR
            img_bgr_for_ocr = img_for_ocr_processing
        else:
            print("Warning: Text removal requires 3 or 4 channel image.")
            return

        # Convert BGR to RGB for pytesseract
        img_rgb_for_tesseract = cv2.cvtColor(img_bgr_for_ocr, cv2.COLOR_BGR2RGB)

        # --- Preprocessing for OCR ---
        # Convert to grayscale for thresholding
        gray_for_ocr = cv2.cvtColor(img_rgb_for_tesseract, cv2.COLOR_RGB2GRAY)
        
        # Apply adaptive thresholding
        # Block size and C value might need tuning depending on image characteristics and text size
        # A larger block size can help with uneven illumination.
        # C is a constant subtracted from the mean or weighted sum.
        thresh_for_ocr = cv2.adaptiveThreshold(gray_for_ocr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY_INV, 11, 2) # block_size=11, C=2
        # You might want to experiment with THRESH_BINARY instead of THRESH_BINARY_INV
        # depending on whether text is darker or lighter than background.
        # For Tesseract, typically white text on black background (THRESH_BINARY_INV if text is dark on light)
        # or black text on white background (THRESH_BINARY if text is dark on light, then invert if needed by Tesseract)
        # Tesseract generally prefers black text on a white background.
        # If your original text is dark on a light background, THRESH_BINARY will make text black, background white.
        # If original text is light on a dark background, THRESH_BINARY_INV will make text black, background white.
        # Let's assume text is darker than its immediate background, so THRESH_BINARY is a good start.
        # If Tesseract expects black text on white, and adaptiveThreshold gives white text on black, invert it.
        # if np.mean(thresh_for_ocr) < 128: # Heuristic: if mostly black, text was likely light
        #    thresh_for_ocr = cv2.bitwise_not(thresh_for_ocr)
        # For simplicity, let's try one version first. Gaussian with INV is common.

        print(f"Using Tesseract config: '{tesseract_config}'")
        try:
            data = pytesseract.image_to_data(thresh_for_ocr, output_type=pytesseract.Output.DICT, config=tesseract_config)
        except pytesseract.TesseractNotFoundError:
            print("Error: Tesseract OCR is not installed or not found in your system's PATH.")
            print("Please install Tesseract OCR: https://github.com/tesseract-ocr/tesseract")
            print("Skipping text removal for this image.")
            return
        except Exception as e:
            print(f"Error during Tesseract OCR processing: {e}")
            traceback.print_exc()
            return

        mask = np.zeros(img_bgr_for_ocr.shape[:2], dtype=np.uint8)
        num_boxes = len(data['level'])
        text_regions_found = False

        print(f"Tesseract processing {num_boxes} potential text boxes. Min confidence for removal: {confidence_threshold}")
        for i in range(num_boxes):
            conf = int(data['conf'][i])
            text = data['text'][i].strip()
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])

            # Log every detected box's details
            print(f"  Box {i+1}/{num_boxes}: Text='{text}', Confidence={conf}, Coords=({x},{y},{w},{h})")

            if not text: # Skip if text is empty
                # print(f"    - Skipping Box {i+1}: Empty text.") # Optional: for very verbose logging
                continue

            is_alnum = any(c.isalnum() for c in text)
            if not is_alnum:
                print(f"    - Skipping Box {i+1} ('{text}'): Not alphanumeric.")
                continue
            
            if conf <= confidence_threshold:
                print(f"    - Skipping Box {i+1} ('{text}'): Confidence {conf} <= threshold {confidence_threshold}.")
                continue

            # If we reach here, the text is valid and confident enough to be removed
            print(f"    - Adding Box {i+1} ('{text}') to mask for removal.")
            cv2.rectangle(mask, 
                          (max(0, x - padding), max(0, y - padding)), 
                          (min(img_bgr_for_ocr.shape[1], x + w + padding), min(img_bgr_for_ocr.shape[0], y + h + padding)), 
                          255, 
                          -1)
            text_regions_found = True


        if not text_regions_found:
            print("No significant text regions found for removal.")
            return

        print("Applying inpainting to remove detected text regions...")
        inpainted_img_bgr = cv2.inpaint(img_bgr_for_ocr, mask, inpaint_radius, cv2.INPAINT_TELEA)

        if original_alpha is not None: # Original image was BGRA
            # Create the new alpha channel: original alpha, but opaque (255) where inpainting occurred
            final_alpha = np.where(mask == 255, 255, original_alpha)
            self.img_cv2 = cv2.cvtColor(inpainted_img_bgr, cv2.COLOR_BGR2BGRA)
            self.img_cv2[:, :, 3] = final_alpha
        else: # Original image was BGR
            self.img_cv2 = inpainted_img_bgr
        
        print("Text removal process finished.")


    def is_image_in_focus(self, threshold=100):
        if self.img_cv2 is None: return False, None
        gray = cv2.cvtColor(self.img_cv2, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        print(f"Focus measure: {variance:.2f}")
        return variance > threshold, variance

    def is_well_illuminated(self, brightness_threshold=30):
        if self.img_cv2 is None: return False, None
        gray = cv2.cvtColor(self.img_cv2, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        print(f"Average brightness: {avg_brightness:.2f}")
        return avg_brightness > brightness_threshold, avg_brightness

    def auto_correct_lighting(self):
        if self.img_cv2 is not None:
            lab = cv2.cvtColor(self.img_cv2, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            self.img_cv2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def remove_background(self, alpha_matting=True, model = "u2net", fg_threshold=230, bg_threshold=20, erode_size=3):
        """Removes the background using rembg, optionally with alpha matting. Modifies self.img_cv2."""
        if self.img_cv2 is None:
            print("Skipping background removal: Image not loaded.")
            return

        print(f"Applying background removal (Alpha Matting: {alpha_matting})...")
        try:
            # Determine input format for rembg (needs PIL RGB/RGBA)
            if self.img_cv2.shape[2] == 4: # BGRA
                img_rgba = cv2.cvtColor(self.img_cv2, cv2.COLOR_BGRA2RGBA)
                img_pil_input = Image.fromarray(img_rgba)
            elif self.img_cv2.shape[2] == 3: # BGR
                img_rgb = cv2.cvtColor(self.img_cv2, cv2.COLOR_BGR2RGB)
                img_pil_input = Image.fromarray(img_rgb)
            else:
                print("Warning: Background removal requires 3 or 4 channel image.")
                return

            # Use rembg.remove
            img_removed_pil = remove(
                img_pil_input,
                alpha_matting=alpha_matting,
                model=model,
                alpha_matting_foreground_threshold=fg_threshold,
                alpha_matting_background_threshold=bg_threshold,
                alpha_matting_erode_size=erode_size
            )

            # Convert the resulting PIL RGBA image back to OpenCV BGRA format
            self.img_cv2 = cv2.cvtColor(np.array(img_removed_pil), cv2.COLOR_RGBA2BGRA)
            print("Background removal finished.")

        except Exception as e:
            print(f"Error during background removal: {e}")
            traceback.print_exc()
            # Optionally revert self.img_cv2 or leave it potentially modified

    # --- NEW Helper function for saving ---
    def _save_optimized(self, img_data_to_save, target_output_path, target_size=10*1024*1024, allow_jpg_conversion=True):
        """Optimizes and saves the provided image data to the target path."""
        if img_data_to_save is None:
            print(f"Skipping save for {target_output_path}: Image data is None.")
            return False

        img_has_alpha = img_data_to_save.shape[2] == 4
        # Determine extension based ONLY on alpha channel presence for this save attempt
        preferred_ext = '.png' if img_has_alpha else '.jpg'
        # Ensure the target_output_path has the correct extension initially
        base_output, _ = os.path.splitext(target_output_path)
        current_output_path = f"{base_output}{preferred_ext}"
        current_ext = preferred_ext

        print(f"Attempting to save optimized image to {os.path.basename(current_output_path)}")

        saved_successfully = False

        # --- PNG Optimization Loop (if applicable) ---
        if current_ext == '.png':
            for compression in range(9, -1, -1):
                params = [cv2.IMWRITE_PNG_COMPRESSION, compression]
                try:
                    cv2.imwrite(current_output_path, img_data_to_save, params) # Use passed image data
                    file_size = os.path.getsize(current_output_path)
                    print(f"Saved PNG {os.path.basename(current_output_path)} ({file_size // 1024}KB, compression: {compression})")
                    if file_size <= target_size:
                        saved_successfully = True
                        break # Success
                except Exception as e:
                    print(f"Error writing PNG file {current_output_path}: {e}")
                    break # Stop trying PNG if write fails

            if saved_successfully:
                print(f"Final image saved: {current_output_path}")
                return True # PNG saved within size limit

            # If PNG still too large after trying all compressions
            print(f"Warning: PNG size ({file_size // 1024}KB) exceeds target size ({target_size // 1024}KB) even at max compression.")
            if allow_jpg_conversion:
                print("Attempting JPG conversion as fallback...")
                current_ext = '.jpg'
                current_output_path = f"{base_output}.jpg" # Change path for JPG fallback
            else:
                print("Saving oversized PNG as transparency is prioritized.")
                print(f"Final image saved (oversized): {current_output_path}")
                return True # Save the large PNG anyway

        # --- JPG Optimization Loop (if applicable, either preferred or fallback) ---
        if current_ext == '.jpg' or current_ext == '.jpeg':
            img_to_save_jpg = img_data_to_save # Use passed image data
            if img_has_alpha: # Convert to BGR for JPEG saving if needed
                img_to_save_jpg = cv2.cvtColor(img_data_to_save, cv2.COLOR_BGRA2BGR)

            quality = 95
            last_successful_size = -1
            while quality > 10:
                params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                try:
                    cv2.imwrite(current_output_path, img_to_save_jpg, params)
                    file_size = os.path.getsize(current_output_path)
                    print(f"Saved JPG {os.path.basename(current_output_path)} ({file_size // 1024}KB, quality: {quality})")
                    last_successful_size = file_size
                    if file_size <= target_size:
                        saved_successfully = True
                        break # Success
                except Exception as e:
                    print(f"Error writing JPG file {current_output_path}: {e}")
                    saved_successfully = False # Ensure flag is false on error
                    break # Stop trying JPG if write fails

                quality -= 5

            if saved_successfully:
                 print(f"Final image saved: {current_output_path}")
                 return True

            # If loop finishes and JPG is still too large (or failed)
            if last_successful_size > target_size:
                 print(f"Warning: JPG size ({last_successful_size // 1024}KB) exceeds target ({target_size // 1024}KB) even at low quality.")
                 print(f"Final image saved (oversized): {current_output_path}")
                 return True # Save the large JPG anyway
            elif last_successful_size == -1: # Means saving failed even once
                 print(f"Failed to save JPG: {current_output_path}")
                 return False

        # Fallthrough case if saving didn't happen or logic failed
        print(f"Could not save image {target_output_path} successfully according to criteria.")
        return False

    def detect_object_yolo(self): # Removed config_path and weights_path from parameters
        """Detects the largest object by area using YOLOv3"""
        if self.img_cv2 is None: return None, None
        try:
            image_for_yolo = self.img_cv2
            if image_for_yolo.shape[2] == 4:
                image_for_yolo = cv2.cvtColor(image_for_yolo, cv2.COLOR_BGRA2BGR)

            # Use instance attributes for YOLO paths
            if not os.path.exists(self.yolo_config_path) or not os.path.exists(self.yolo_weights_path):
                print(f"Error: YOLO config ('{self.yolo_config_path}') or weights ('{self.yolo_weights_path}') not found.")
                print("Skipping YOLO detection.")
                return None, None
                
            net = cv2.dnn.readNetFromDarknet(self.yolo_config_path, self.yolo_weights_path)
            ln = net.getLayerNames()
            try: ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
            except TypeError: ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

            height, width = image_for_yolo.shape[:2]
            input_size = (608, 608) if width > 2000 else (416, 416)
            blob = cv2.dnn.blobFromImage(image_for_yolo, 1/255.0, input_size, swapRB=True, crop=False)
            net.setInput(blob)
            outputs = net.forward(ln)

            boxes, confidences = [], []
            H, W = self.img_cv2.shape[:2] # Use original dimensions for coordinate mapping

            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    confidence = scores[np.argmax(scores)]
                    if confidence > 0.5:
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width_box, height_box) = box.astype("int")
                        x = int(centerX - (width_box / 2))
                        y = int(centerY - (height_box / 2))
                        boxes.append([x, y, int(width_box), int(height_box)])
                        confidences.append(float(confidence))

            largest_area, best_box, best_confidence = 0, None, 0
            for i, (x, y, w, h) in enumerate(boxes):
                area = w * h
                if area > largest_area:
                    largest_area, best_box, best_confidence = area, boxes[i], confidences[i]

            if best_box is not None:
                x, y, w, h = best_box
                print(f"Largest object detected (Area: {largest_area}px², Confidence: {best_confidence:.2f})")
                return (x, y, x + w, y + h), best_confidence
            else:
                print("No objects detected with sufficient confidence.")
                return None, None
        except Exception as e:
            print(f"YOLO detection error: {str(e)}")
            traceback.print_exc()
            return None, None

    def check_angle(self):
        """Checks image skew using moments. Handles BGR/BGRA."""
        if self.img_cv2 is None: return True
        if self.img_cv2.shape[2] == 4: gray = cv2.cvtColor(self.img_cv2, cv2.COLOR_BGRA2GRAY)
        else: gray = cv2.cvtColor(self.img_cv2, cv2.COLOR_BGR2GRAY)
        moments = cv2.moments(gray)
        mu02, mu11 = moments.get('mu02', 0), moments.get('mu11', 0)
        if mu02 == 0: return True # Perfectly horizontal/vertical is okay
        skew = abs(mu11 / (mu02 + 1e-6))
        return skew < 0.2

    def check_tidiness(self):
        """Checks composition tidiness using Canny edge density. Handles BGR/BGRA."""
        if self.img_cv2 is None: return False
        img_for_canny = self.img_cv2
        if img_for_canny.shape[2] == 4: img_for_canny = cv2.cvtColor(img_for_canny, cv2.COLOR_BGRA2BGR)
        elif img_for_canny.shape[2] != 3: print("Warning: Tidiness check needs 3/4 channels."); return False
        edges = cv2.Canny(img_for_canny, 100, 200)
        edge_density = np.count_nonzero(edges) / edges.size
        return edge_density < 0.15

    def detect_multiple_objects(self):
        """Detect if there are multiple large objects using contours. Handles BGR/BGRA."""
        if self.img_cv2 is None: return False
        if self.img_cv2.shape[2] == 4: gray = cv2.cvtColor(self.img_cv2, cv2.COLOR_BGRA2GRAY)
        elif self.img_cv2.shape[2] == 3: gray = cv2.cvtColor(self.img_cv2, cv2.COLOR_BGR2GRAY)
        else: print("Warning: Multiple object detection needs 3/4 channels."); return False
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area_threshold = max(1000, gray.size * 0.005)
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area_threshold]
        print(f"Found {len(large_contours)} large distinct contours.")
        return len(large_contours) > 1

    def get_dominant_border_color(self, border_size_px=10):
        """Determines dominant BGR border color from the original image."""
        img_to_sample = self.original_img_cv2_for_color_sampling
        if img_to_sample is None: print("Warning: Orig image missing for border color."); return None
        H, W = img_to_sample.shape[:2]
        eff_border = min(border_size_px, H // 3, W // 3)
        if eff_border <= 0: print("Warning: Image too small for border sample."); return None

        if img_to_sample.shape[2] == 4: img_to_sample = cv2.cvtColor(img_to_sample, cv2.COLOR_BGRA2BGR)
        elif img_to_sample.shape[2] != 3: print("Warning: Unsupported channels for border color."); return None

        borders = []
        if H > eff_border * 2 and W > eff_border * 2:
            borders.append(img_to_sample[0:eff_border, eff_border:W-eff_border].reshape(-1, 3))
            borders.append(img_to_sample[H-eff_border:H, eff_border:W-eff_border].reshape(-1, 3))
            borders.append(img_to_sample[:, 0:eff_border].reshape(-1, 3))
            borders.append(img_to_sample[:, W-eff_border:W].reshape(-1, 3))
        elif H > 0 and W > 0: borders.append(img_to_sample.reshape(-1, 3)) # Fallback: average all
        else: print("Warning: Invalid image dims for border color."); return None

        if not borders: print("Warning: No border pixels collected."); return None
        all_border_pixels = np.concatenate(borders, axis=0)
        if all_border_pixels.size == 0: print("Warning: Border pixel array empty."); return None

        avg_bgr = np.mean(all_border_pixels, axis=0)
        avg_color_int = tuple(int(c) for c in avg_bgr)
        print(f"Determined average border color: BGR {avg_color_int}")
        return avg_color_int
    
    # === Helper Function for Applying Transformations ===
    def apply_transformations(self, input_img, crop_box, pad_vals, target_sz, pad_color_value):
        if input_img is None: return None, False
        img_to_process = input_img.copy()
        processed = False
        try:
            # 1. Crop (if crop_box is defined)
            if crop_box:
                nx1, ny1, nx2, ny2 = crop_box
                if nx1 < nx2 and ny1 < ny2:
                        img_to_process = img_to_process[ny1:ny2, nx1:nx2]
                else: # Should not happen if checks above are done, but safe fallback
                        print("Warning: Invalid crop_box during transformation. Skipping crop.")

            # 2. Pad (if padding is needed)
            if pad_vals and any(p > 0 for p in pad_vals):
                pad_top, pad_bottom, pad_left, pad_right = pad_vals
                img_to_process = cv2.copyMakeBorder(img_to_process, pad_top, pad_bottom, pad_left, pad_right,
                                                    cv2.BORDER_CONSTANT, value=pad_color_value)

            # 3. Resize (always resize to ensure final dimensions)
            H_curr, W_curr = img_to_process.shape[:2]
            if (W_curr, H_curr) != target_sz:
                img_to_process = cv2.resize(img_to_process, target_sz, interpolation=cv2.INTER_AREA)

            processed = True
            return img_to_process, processed
        except Exception as e:
            print(f"Error during apply_transformations: {e}")
            traceback.print_exc()
            return None, False


    def process_image(self, yolo_confidence_threshold=0.7, text_removal_confidence_threshold=50): # Added text_removal_confidence_threshold
        if self.img_cv2 is None:
            print(f"Skipping processing for {self.image_path}: Image could not be loaded.")
            return

        print(f"Processing: {self.image_path}")

        # --- Determine OPAQUE background color for Path 1 padding ---
        padding_color_bgr = self.get_dominant_border_color(border_size_px=15)
        if padding_color_bgr is None:
            print("Could not determine border color, falling back to white.")
            padding_color_bgr = (255, 255, 255) # BGR White

        # --- Basic Checks & Corrections ---
        # (Focus, Illumination, Auto-correct lighting - applied to self.img_cv2)
        focus_ok, focus_val = self.is_image_in_focus()
        illum_ok, brightness = self.is_well_illuminated()
        if not focus_ok: print(f"Focus warning (Value: {focus_val:.2f})")
        if not illum_ok: print(f"Brightness warning (Value: {brightness:.2f})")
        if not focus_ok or not illum_ok:
            print("-- Applying Autocorrect Lighting --")
            self.auto_correct_lighting() # Modifies self.img_cv2

        # --- NEW: Text Removal ---
        # Ensure img_cv2 is not None after auto_correct_lighting before proceeding
        if self.img_cv2 is not None:
            print("-- Attempting to remove text from image --")
            # Try PSM 11 for sparse text, or 7 for a single line of text.
            # You can also add other Tesseract configs here, e.g. '-c tessedit_char_whitelist=0123456789'
            # if you only want to detect numbers.
            tess_config = '--psm 11' 
            self.remove_text_objects(
                confidence_threshold=text_removal_confidence_threshold,
                tesseract_config=tess_config
            ) 
        else:
            print("Skipping text removal as image is None after lighting correction.")


        # --- Quality Checks (Run AFTER Text Removal, BEFORE BG Removal or Cropping/Padding) ---
        bbox = None # Default to None
        confidence = 0.0 # Default to 0.0
        if self.img_cv2 is not None:
            print("Running pre-processing quality checks & detection...")
            if not self.check_angle(): print("Warning: Potential suboptimal angle detected.")
            if not self.check_tidiness(): print("Warning: Potentially messy composition detected.")
            if self.detect_multiple_objects(): print("Warning: Multiple distinct objects detected.")

            # --- Object Detection (Run BEFORE BG Removal) ---
            detected_bbox, detected_confidence = self.detect_object_yolo()
            if detected_bbox is not None and detected_confidence >= yolo_confidence_threshold:
                print(f"YOLO detection accepted (Confidence: {detected_confidence:.2f})")
                bbox = detected_bbox
                confidence = detected_confidence
            elif detected_bbox is not None:
                print(f"YOLO detection REJECTED (Confidence: {detected_confidence:.2f} < {yolo_confidence_threshold})")
            else:
                print("YOLO did not detect any object.")
        else:
            print("Skipping quality checks & detection as image is None.")
            return # Cannot proceed if image became None after corrections

        # --- Store image state AFTER corrections/checks but BEFORE splitting paths ---
        if self.img_cv2 is None: # Check if image became None (e.g. after text removal attempt)
            print(f"Image is None after pre-processing steps for {self.image_path}. Skipping further processing.")
            return
        img_state_before_processing = self.img_cv2.copy()

        # === Define Transformation Parameters ===
        # Calculate these once based on bbox and img_state_before_processing dimensions
        target_ratio = 4/3
        final_size = (1024, 768)
        crop_coords = None # (nx1, ny1, nx2, ny2)
        padding_values = None # (pad_top, pad_bottom, pad_left, pad_right)
        apply_fallback_padding = False

        if bbox:
            x1, y1, x2, y2 = map(int, bbox)
            H_orig, W_orig = img_state_before_processing.shape[:2]
            bbox_w, bbox_h = x2 - x1, y2 - y1
            margin_percent = 5.0
            margin_w, margin_h = int(bbox_w * margin_percent / 100), int(bbox_h * margin_percent / 100)
            nx1, ny1 = max(0, x1 - margin_w), max(0, y1 - margin_h)
            nx2, ny2 = min(W_orig, x2 + margin_w), min(H_orig, y2 + margin_h)

            if nx1 < nx2 and ny1 < ny2:
                crop_coords = (nx1, ny1, nx2, ny2)
                # Calculate padding needed AFTER cropping
                # Dimensions of the potential cropped area (use original state for calculation)
                temp_cropped = img_state_before_processing[ny1:ny2, nx1:nx2]
                ch, cw = temp_cropped.shape[:2]
                if ch > 0 and cw > 0:
                    current_ratio = cw / ch
                    pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
                    if abs(current_ratio - target_ratio) < 1e-6: pass
                    elif current_ratio > target_ratio:
                        target_h = int(cw / target_ratio); pad_v_total = max(0, target_h - ch)
                        pad_top, pad_bottom = pad_v_total // 2, pad_v_total - (pad_v_total // 2)
                    else:
                        target_w = int(ch * target_ratio); pad_h_total = max(0, target_w - cw)
                        pad_left, pad_right = pad_h_total // 2, pad_h_total - (pad_h_total // 2)
                    padding_values = (pad_top, pad_bottom, pad_left, pad_right)
                else:
                     print("Warning: Calculated crop area is empty. Will use fallback.")
                     apply_fallback_padding = True
            else:
                print(f"Warning: Invalid crop dims calculated. Will use fallback.")
                apply_fallback_padding = True
        else:
            apply_fallback_padding = True # No bbox means use fallback

        if apply_fallback_padding:
            print("No reliable object detected or crop failed. Calculating fallback padding/resizing.")
            crop_coords = None # Ensure no cropping if fallback
            H_orig, W_orig = img_state_before_processing.shape[:2]
            if H_orig > 0 and W_orig > 0:
                 current_ratio = W_orig / H_orig
                 if abs(current_ratio - target_ratio) > 1e-6 or (W_orig, H_orig) != final_size:
                     pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
                     if current_ratio > target_ratio:
                         target_h = int(W_orig / target_ratio); pad_v_total = max(0, target_h - H_orig)
                         pad_top, pad_bottom = pad_v_total // 2, pad_v_total - (pad_v_total // 2)
                     else:
                         target_w = int(H_orig * target_ratio); pad_h_total = max(0, target_w - W_orig)
                         pad_left, pad_right = pad_h_total // 2, pad_h_total - (pad_h_total // 2)
                     padding_values = (pad_top, pad_bottom, pad_left, pad_right)
                 else:
                     print("Image already has target ratio/size. No fallback padding needed.")
                     padding_values = (0, 0, 0, 0) # No padding needed
            else:
                 print("Fallback error: Image has zero dimensions.")
                 padding_values = None # Cannot process


        # === Path 1: Process and Save WITH Original Background ===
        print("\n--- Processing for version WITHOUT background removal ---")
        img_processed_original_bg, processed_orig = self.apply_transformations(
            img_state_before_processing,
            crop_coords,
            padding_values,
            final_size,
            padding_color_bgr # Opaque padding
        )
        if processed_orig and img_processed_original_bg is not None:
             output_path_no_bg = f"{self.output_base_path}_bg_original"
             self._save_optimized(img_processed_original_bg, output_path_no_bg, allow_jpg_conversion=True)
        else:
             print("Skipping save for original background version (processing failed).")

        # === Path 2: Process and Save WITH Background Removed ===
        print("\n--- Processing for version WITH background removal ---")
        img_for_bg_removal = img_state_before_processing.copy() # Start from clean state again

        # Apply Background Removal HERE
        # Modify self.img_cv2 temporarily for the remove_background method call
        self.img_cv2 = img_for_bg_removal
        self.remove_background(alpha_matting=True, model="isnet-general-use", fg_threshold=210, bg_threshold=10, erode_size=1) 
        img_bg_removed_raw = self.img_cv2 # Get the BGRA result back

        if img_bg_removed_raw is None or img_bg_removed_raw.shape[2] != 4:
             print("Background removal failed or did not produce BGRA image. Skipping BG removed path.")
        else:
            # Apply SAME transformations but with TRANSPARENT padding
            img_processed_bg_removed, processed_removed = self.apply_transformations(
                img_bg_removed_raw,
                crop_coords,
                padding_values,
                final_size,
                (0, 0, 0, 0) # Transparent padding
            )
            if processed_removed and img_processed_bg_removed is not None:
                 output_path_with_bg = f"{self.output_base_path}_bg_removed"
                 self._save_optimized(img_processed_bg_removed, output_path_with_bg, allow_jpg_conversion=False) # No JPG for transparency
            else:
                 print("Skipping save for background removed version (processing failed).")

        # Reset self.img_cv2 just in case it was modified by remove_background
        self.img_cv2 = img_state_before_processing

        print(f"\nFinished processing {self.image_path}")

class BatchImageProcessor:
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        # Store yolo paths if needed by BatchImageProcessor, or pass to ImageProcessor
        self.yolo_config_path = "yolov3.cfg"
        self.yolo_weights_path = "yolov3.weights"

    def process_folder(self):
        for filename in os.listdir(self.input_folder):
            print(40*"-")
            print(f"Starting file: {filename}")
            print(40*"-")
            input_path = os.path.join(self.input_folder, filename)
            if os.path.isfile(input_path) and self.is_image_file(filename):
                # Generate a BASE output path without extension or final suffix
                output_base_name = self.generate_output_base_name(filename)
                output_base_path = os.path.join(self.output_folder, output_base_name)

                # Pass the input path and the BASE output path to ImageProcessor
                processor = ImageProcessor(input_path, output_base_path)
                # Pass yolo paths to ImageProcessor instance if they were made instance attributes
                # or if ImageProcessor's __init__ is updated to take them.
                # For now, assuming ImageProcessor uses its defaults or finds them.
                # If yolo_config and yolo_weights are global/fixed, this is fine.
                # If they are configurable per batch run, ImageProcessor needs to know them.
                processor.yolo_config_path = self.yolo_config_path
                processor.yolo_weights_path = self.yolo_weights_path
                processor.process_image() # process_image now handles the two saves internally
            else:
                 print(f"Skipping non-image file or directory: {filename}")

    def is_image_file(self, filename):
        extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.heic']
        return any(filename.lower().endswith(ext) for ext in extensions)

    # --- MODIFIED: Generate base name without suffix/extension ---
    def generate_output_base_name(self, input_filename):
        name, _ = os.path.splitext(input_filename)
        # Simple base name, suffixes (_bg_original, _bg_removed) will be added later
        return f"{name}_processed"

if __name__ == '__main__':
    # --- Configuration ---
    yolo_config = "yolov3.cfg"  # Path to YOLO config file
    yolo_weights = "yolov3.weights" # Path to YOLO weights file
    input_dir = 'input_images'     # Folder with input images
    output_dir = 'processed_images' # Folder for processed output images
    # --- End Configuration ---

    # Check if Tesseract is available (pytesseract will raise TesseractNotFoundError if not)
    try:
        pytesseract.get_tesseract_version()
        print(f"Tesseract OCR version {pytesseract.get_tesseract_version()} detected.")
    except pytesseract.TesseractNotFoundError:
        print("Error: Tesseract OCR is not installed or not found in your system's PATH.")
        print("Text removal feature will not work. Please install Tesseract OCR.")
        print("Installation guide: https://tesseract-ocr.github.io/tessdoc/Installation.html")
        # Optionally, exit if Tesseract is critical, or continue with a warning
        # exit() 
    except Exception as e:
        print(f"Could not verify Tesseract version: {e}")


    # Check if YOLO files exist
    if not os.path.exists(yolo_config) or not os.path.exists(yolo_weights):
         print(f"Error: YOLOv3 config ('{yolo_config}') or weights ('{yolo_weights}') not found.")
         print("Please download https://github.com/patrick013/Object-Detection---Yolov3/blob/master/model/yolov3.weights and place them in the correct directory.")
         exit() # Exit if YOLO files are missing

    if not os.path.exists(input_dir):
         print(f"Error: Input directory '{input_dir}' not found. Please create it and add images.")
         exit()

    batch_processor = BatchImageProcessor(input_dir, output_dir)
    # Pass yolo paths to BatchImageProcessor if they are configurable
    batch_processor.yolo_config_path = yolo_config
    batch_processor.yolo_weights_path = yolo_weights
    batch_processor.process_folder()
    print(f"\nProcessing complete. Processed images are in '{output_dir}' folder.")
    print("Look for files ending with '_bg_original.*' and '_bg_removed.*'")