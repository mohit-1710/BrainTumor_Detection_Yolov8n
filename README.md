# **Brain Tumor Detection Using YOLOv8n**

## **Overview**
This project focuses on the detection of brain tumors from MRI images using the YOLOv8n (You Only Look Once) object detection model. The aim is to provide a reliable and efficient method for identifying tumors in MRI scans, aiding early diagnosis and treatment. The model was trained on a dataset of 19,000 images and achieves high performance in precision and accuracy. An interactive interface allows users to upload images for real-time tumor detection.

## **Project Motivation**

This project was undertaken as a part of the entrance task for the **Drone Club of Scaler School of Technology**. The club emphasizes innovative solutions, and this challenge presented an opportunity to apply deep learning to a significant real-world problem: brain tumor detection in MRI scans.

Brain tumors can be life-threatening if not detected early. Manual analysis of MRI images by radiologists is time-consuming and prone to human error. By utilizing deep learning, we can assist in speeding up the detection process while improving accuracy. This project represents both a technical challenge and an opportunity to contribute to healthcare advancements, which aligns with the Drone Club’s focus on innovation and impact.

## **Data Source**
The dataset used in this project was sourced from **Kaggle**, containing **19,000 MRI brain images** annotated for tumor detection. The dataset was structured to work with the YOLOv8n model.

- **Image Format**: JPEG
- **Annotations**: Provided in YOLO format (bounding boxes for tumor regions)
- **Dataset Split**:
  - **Training Set**: Used to train the model.
  - **Validation Set**: Used during training to fine-tune the model.
  - **Test Set**: Used for final evaluation and performance metrics.

## **Environment Setup**

### Step 1: Python Installation
The project was built using **Python 3.x**. Make sure Python is installed on your system.

### Step 2: Required Libraries
Install the following Python libraries to manage the dataset, train the model, visualize results, and create the user interface:

```bash
pip install ultralytics
pip install opencv-python
pip install matplotlib
pip install pandas
pip install ipywidgets
```

- **Ultralytics**: For YOLOv8 model training and inference.
- **OpenCV**: For image processing and displaying results.
- **Matplotlib**: For plotting graphs and visualizing model performance.
- **Pandas**: To structure data and generate tables.
- **Ipywidgets**: For creating interactive widgets like file upload buttons.

### Step 3: Setting Up Jupyter Notebook
To train the model and visualize the results, **Jupyter Notebook** was used. Install it as follows:

```bash
pip install notebook
```

Once installed, launch Jupyter Notebook to train the model and monitor progress.

## **Model Training**

### Why YOLOv8n?
The **YOLOv8n** model was selected for its speed and accuracy, offering an efficient real-time object detection solution, especially suitable for identifying brain tumors in MRI images.

### Training Process
The model was trained on 19,000 MRI images using an **NVIDIA RTX 4050 GPU** to speed up the process. After many hours of training, the model was optimized and achieved excellent results in tumor detection.

## **Model Performance Evaluation**

After training, the model was evaluated on a custom test set. The following metrics were recorded:

- **Accuracy**: 0.9202
- **Precision**: 1.0
- **Recall**: 0.9202
- **F1-Score**: 0.9584

### Performance Visualization

To visualize the model’s performance, a table and bar chart were created to display the evaluation metrics.

#### **Table of Metrics**:

| Metric      | Value       |
|-------------|-------------|
| Accuracy    | 0.9202      |
| Precision   | 1.0         |
| Recall      | 0.9202      |
| F1-Score    | 0.9584      |

#### **Bar Chart**:

```python
import matplotlib.pyplot as plt

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [0.9202, 1.0, 0.9202, 0.9584]

plt.bar(metrics, values, color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('YOLOv8n Model Performance')
plt.show()
```

This chart visually demonstrates the model's high accuracy and precision.

## **User Interface for Tumor Detection**

A user-friendly upload button was created, allowing users to upload their own MRI images. The YOLOv8n model processes the images in real-time, detects any tumors present, and draws a **bounding box** around the detected tumor. The confidence score is displayed above the bounding box.

### Upload Functionality:

```python
import ipywidgets as widgets
from IPython.display import display
from ultralytics import YOLO
import cv2

# Load trained YOLOv8 model
model = YOLO('best.pt')

# Function to detect tumor
def detect(image_path):
    img = cv2.imread(image_path)
    results = model.predict(img)
    for result in results:
        if result.boxes:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = [int(b) for b in box]
                confidence = result.boxes.conf[0].item()
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img, f"Confidence: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    cv2.imshow('Detected Tumor', img)

# File upload widget
upload_btn = widgets.FileUpload()

def on_upload_change(change):
    for filename, file_info in upload_btn.value.items():
        with open(filename, 'wb') as f:
            f.write(file_info['content'])
        detect(filename)

upload_btn.observe(on_upload_change, names='value')
display(upload_btn)
```

### How It Works:
- **Upload Image**: Users can upload MRI images through the widget.
- **Real-Time Detection**: The model processes the image, detects tumors, and draws bounding boxes around them.
- **Confidence Score**: The confidence score for detection is displayed above the bounding box.

## **Conclusion**

The YOLOv8n model provides an effective solution for detecting brain tumors in MRI scans. With **92% accuracy** and **100% precision**, it demonstrates the potential to assist radiologists by quickly identifying tumor regions. The user-friendly interface further adds value, allowing for real-time tumor detection.

### **Future Improvements**:
- **Model Fine-Tuning**: Training the model with larger datasets could improve its performance even further.
- **Multi-Abnormality Detection**: Expanding the model to detect other brain conditions could make it a comprehensive diagnostic tool.
- **Web Application**: Future steps may include deploying this project as a web application to increase accessibility for medical professionals.
