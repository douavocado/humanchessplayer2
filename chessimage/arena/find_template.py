#!/usr/bin/env python3

import cv2
import numpy as np

def find_template(template, image):
    # Check if images are valid
    if template is None or image is None:
        print("Error: Invalid template or image")
        return None
    
    try:
        # Convert both to grayscale for better matching
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Get template dimensions
        h, w = template_gray.shape
        
        # Perform template matching
        result = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        
        # Set a threshold for matches
        threshold = 0.8
        
        # Find locations where match exceeds threshold
        locations = np.where(result >= threshold)
        
        # If no match found
        if len(locations[0]) == 0:
            print("Template not found in the image")
            return None
        
        # Extract coordinates (taking the best match)
        best_match_idx = np.argmax(result)
        best_match_value = np.max(result)
        
        # If best match is below threshold
        if best_match_value < threshold:
            print(f"Best match confidence ({best_match_value:.4f}) below threshold ({threshold})")
            return None
        
        # Convert 1D index to 2D coordinates
        best_match_y, best_match_x = np.unravel_index(best_match_idx, result.shape)
        
        # Calculate bounding box
        top_left = (best_match_x, best_match_y)
        bottom_right = (best_match_x + w, best_match_y + h)
        
        # Calculate centre point
        centre_x = best_match_x + w // 2
        centre_y = best_match_y + h // 2
        
        print(f"Match found with confidence: {best_match_value:.4f}")
        print(f"Bounding box: Top-left {top_left}, Bottom-right {bottom_right}")
        print(f"Centre point: ({centre_x}, {centre_y})")
        
        # Optional: Draw bounding box on image
        result_image = image.copy()
        cv2.rectangle(result_image, top_left, bottom_right, (0, 255, 0), 2)
        cv2.circle(result_image, (centre_x, centre_y), 5, (0, 0, 255), -1)
        
        # Save result image
        # cv2.imwrite("match_result.png", result_image)
        
        return (centre_x, centre_y)
    
    except Exception as e:
        print(f"Error during template matching: {e}")
        return None

if __name__ == "__main__":
    template_path = "back_to_tournament.png"
    image_path = "screen_shot_test.png"
    
    # Load images
    template = cv2.imread(template_path)
    image = cv2.imread(image_path)
    
    if template is None or image is None:
        print("Error: Could not load one or both images")
        exit(1)
    
    print("Searching for back_to_tournament.png in screen_shot_test.png...")
    centre = find_template(template, image)
    
    if centre:
        print(f"Found template at centre coordinates: {centre}")
    else:
        print("Template not found or matching failed") 