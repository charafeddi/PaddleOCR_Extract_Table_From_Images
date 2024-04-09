import cv2
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
import pandas as pd
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import logging



def detect_and_save_table(image_path):
    # Read the image
    image = cv2.imread(image_path)
    output_path= r'D:/2023-2024/final project/charaf/OCR model/assets/img/img2.png'
    if image is None:
        raise ValueError("Could Not read the Image.")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Line detection using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    # Create an empty image to draw lines
    line_image = np.zeros_like(image)

    # Draw lines on the image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 3)

    # Convert line image to grayscale
    gray_line_image = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)

    # Find contours - these will be the tables
    contours, _ = cv2.findContours(gray_line_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the contour with the largest area (and some additional heuristics) is the table
    max_area = 0
    best_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Add additional heuristics if necessary, e.g., aspect ratio
        if area > max_area:
            max_area = area
            best_cnt = cnt
    
    if best_cnt is not None:
        x, y, w, h = cv2.boundingRect(best_cnt)
        # Crop the table area
        cropped_table = image[y:y+h, x:x+w]
        cv2.imwrite(output_path, cropped_table)
    else:
        print("No table found.")
        
    return output_path
    
def intersection (box1, box2):
    return [ box2[0], box1[1], box2[2], box1[3]]

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter =  abs(max((x2 - x1, 0)) * max((y2 - y1), 0))
    if inter == 0 :
         return 0

    box_1_area = abs((box1[2] - box1[0]) * (box1[3] - box1[1]))
    box_2_area = abs((box2[2] - box2[0]) * (box2[3] - box2[1]))

    return inter / float(box_1_area + box_2_area - inter)
    
def extracting_table(): 
    
    # Initialize the PaddleOCR model with English, Arabic, and French languages
    image_path = r'D:/2023-2024/final project/charaf/OCR model/assets/img/img2.png'
    image_cv = cv2.imread(image_path)
    image_height= image_cv.shape[0]
    image_width = image_cv.shape[1]
    output_ocr = ocr_model.ocr(image_path)
    # Initialize lists to store texts, probabilities, and boxes
    
    texts = []
    probabilities = []
    boxes = []
    
    # Loop through the output to extract information
    for item in output_ocr:
        for element in item:
            box = element[0]  # The bounding box
            text = element[1][0]  # The detected text
            probability = element[1][1]  # The probability
    
            # Append the extracted information to the respective lists
            boxes.append(box)
            texts.append(text)
            probabilities.append(probability)
    
    image_boxes = image_cv.copy()

    for box, text in zip(boxes, texts):
        # Extract the top-left and bottom-right coordinates
        top_left = (int(box[0][0]), int(box[0][1]))
        bottom_right = (int(box[2][0]), int(box[2][1]))
    
        # Draw the rectangle on the image
        cv2.rectangle(image_boxes, top_left, bottom_right, (0, 255, 0), 2)  # Green color, thickness 2
        cv2.putText(image_boxes, text, (int(box[0][0]), int(box[0][1])),cv2.FONT_HERSHEY_SIMPLEX,1,(222,0,0),1)

    cv2.imwrite(r'D:/2023-2024/final project/charaf/OCR model/assets/img/detections.jpg', image_boxes)

    image_line = image_cv.copy()
    heriz_boxes = []
    vert_boxes = []
    for box in boxes:
        x_h, x_v = 0, int (box[0][0])
        y_h, y_v = int(box[0][1]),0
        width_h, width_v = image_width, int(box[2][0] - box[0][0])
        height_h,height_v = int(box[2][1] - box[0][1]), image_height
    
        heriz_boxes.append([x_h,y_h,x_h+width_h, y_h + height_h])
        vert_boxes.append([x_v,y_v,x_v+width_v, y_v + height_v])
    
        cv2.rectangle(image_line, (x_h, y_h), (x_h+width_h, y_h + height_h),(255,0,0),1)
        cv2.rectangle(image_line, (x_v, y_v), (x_v+width_v, y_v + height_v),(0,255,0),1)

    # Display the image with drawn boxes
    cv2.imwrite(r'D:/2023-2024/final project/charaf/OCR model/assets/image with Boxes.jpg', image_line)

    horiz_out = tf.image.non_max_suppression(
        heriz_boxes,
        probabilities,
        max_output_size =1000,
        iou_threshold =0.1,
        score_threshold = float('-inf'),
        name=None
    )

    img_vert_horiz_lines =  image_cv.copy()
    horiz_lines = np.sort(np.array(horiz_out))

    for val in horiz_lines:
        # Extract the top-left and bottom-right coordinates
        top_left = (int(heriz_boxes[val][0]), int(heriz_boxes[val][1]))
        bottom_right = (int(heriz_boxes[val][2]), int(heriz_boxes[val][3]))
        cv2.rectangle(img_vert_horiz_lines, top_left, bottom_right, (0, 255, 0), 2)  # Green color, thickness 2

    # Display the image with drawn boxes
    cv2.imshow(r'D:/2023-2024/final project/charaf/OCR model/assets/Image  with horizlines.jpg', img_vert_horiz_lines)

    vert_out = tf.image.non_max_suppression(
        vert_boxes,
        probabilities,
        max_output_size =1000,
        iou_threshold =0.1,
        score_threshold = float('-inf'),
        name=None
    )

    vert_lines = np.sort(np.array(vert_out))

    for val in vert_lines:
        # Extract the top-left and bottom-right coordinates
        top_left = (int(vert_boxes[val][0]), int(vert_boxes[val][1]))
        bottom_right = (int(vert_boxes[val][2]), int(vert_boxes[val][3]))
        cv2.rectangle(img_vert_horiz_lines, top_left, bottom_right, (255, 255, 0), 2)  # Green color, thickness 2
    
    # Display the image with drawn boxes
    cv2.imshow(r'D:/2023-2024/final project/charaf/OCR model/assets/Image  with vertlines.jpg', img_vert_horiz_lines)
    
    #convert to CSV 

    out_array = [['' for i in range(len(vert_lines))] for j in range(len(horiz_lines))]

    unoredered_boxes = []

    for i in vert_lines:
        #print(vert_boxes[i])
        unoredered_boxes.append(vert_boxes[i][0])
        
    ordered_boxes = np.argsort(unoredered_boxes)

    for i in range(len(horiz_lines)):
        for j in range(len(vert_lines)):
            resultant = intersection (heriz_boxes[horiz_lines[i]], vert_boxes[vert_lines[ordered_boxes[j]]])
            for b in range(len(boxes)):
                the_box = [boxes[b][0][0],boxes[b][0][1], boxes[b][2][0], boxes[b][2][1]]
                if(iou(resultant,the_box) >0.1):
                    out_array[i][j] = texts[b]
                    
    pd.DataFrame(out_array).to_csv(r'D:\2023-2024\final project\charaf\OCR model\assets\output.csv')
    return out_array


app = Flask(__name__)

# Load PaddleOCR once
ocr_model = None

with app.app_context():
#    global ocr_model
    try:
        print("Starting to load PaddleOCR model...")
        ocr_model = PaddleOCR(use_angle_cls=True, lang='fr', use_gpu=False)
        print("PaddleOCR model loaded successfully.")
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")

# Set up a route for OCR processing
@app.route('/pp', methods=['POST'])
def process_image():
    app.logger.info("Processing image...")
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        img_path = os.path.join('D:/2023-2024/final project/charaf/OCR model/assets/img', filename)
        file.save(img_path)  # Corrected here

        # Now apply your OCR function here, for example:
        # Process the image and find table cells
        detect_and_save_table(img_path)
        # extracting table 
        result = extracting_table()

        # Clean up and remove the image if you want
        os.remove(img_path)

        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5002)
