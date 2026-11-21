# Gender and Age Detection Using Deep Learning

A real-time **Gender and Age Range Detection System** built using Python, OpenCV, PyTorch, and a pretrained Vision Transformer (ViT) model from Hugging Face.

The application detects multiple faces in images and live webcam streams, predicts approximate age groups and gender, and displays the results with bounding boxes.

## Features

- **Multi-Face Detection:** Detect multiple faces in a single image or webcam frame.
- **Age Range Prediction:** Predict approximate age groups instead of exact ages.
- **Gender Prediction:** Classify detected faces as Male or Female.
- **Gender Confidence:** Display the model's gender prediction confidence.
- **Image Analysis:** Upload and analyze images.
- **Live Webcam Detection:** Perform real-time face detection and prediction.
- **Fullscreen Webcam:** Display webcam detection in fullscreen mode.
- **Annotated Output:** Draw bounding boxes and prediction labels around detected faces.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Face detection and image processing |
| PyTorch | Deep learning model execution |
| Hugging Face Transformers | Vision Transformer integration |
| ViT (Vision Transformer) | Age and gender prediction |

## Model Information

**Age and Gender Model**

- Model: `abhilash88/age-gender-prediction`
- Source: Hugging Face
- Architecture: Vision Transformer (ViT)
- Framework: PyTorch

**Face Detection Model**

- OpenCV DNN face detector
- Model files: `opencv_face_detector_uint8.pb`
- Configuration: `opencv_face_detector.pbtxt`

The face detector identifies faces, while the ViT model predicts age and gender for each detected face.

## Age Categories

The predicted age is displayed in one of the following approximate ranges:

| Predicted Age | Displayed Range |
|---|---|
| 0–14 | 0-14 |
| 15–20 | 15-20 |
| 21–25 | 21-25 |
| 26–30 | 26-30 |
| 31–40 | 31-40 |
| 41–50 | 41-50 |
| 51–60 | 51-60 |
| 61–100 | 61+ |

## Project Structure

```text
Gender-and-Age-Detection/
│
├── main.py
│   └── Main application
│
├── age_gender_model.py
│   └── ViT age and gender prediction
│
├── opencv_face_detector.pbtxt
│   └── Face detector configuration
│
├── opencv_face_detector_uint8.pb
│   └── Face detection model
│
├── requirements.txt
│   └── Project dependencies
│
├── README.md
│   └── Project documentation
│
├── photo.jpg
├── jude.jpg
├── profile.jpg
│   └── Sample input images
│
└── result_vit.jpg
    └── Sample prediction output
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Gender-and-Age-Detection-OpenCV-Python-Source-Code
```

### 2. Create a virtual environment

```powershell
py -m venv .venv
```

### 3. Activate the environment

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

## How to Run

### 1. Analyze an Image

```powershell
python main.py --image photo.jpg
```

### 2. Save the Prediction Result

```powershell
python main.py --image photo.jpg --output result.jpg --no-display
```

The annotated image will be saved to the specified output path.

### 3. Start Live Webcam Detection

```powershell
python main.py
```

The application opens the webcam in fullscreen mode.

Press **q** to exit the webcam window.

### 4. Use Another Camera

```powershell
python main.py --camera 1
```

## Sample Output

The application displays:

- Green bounding boxes around detected faces.
- Gender classification with confidence percentage.
- Predicted age range.

Example:

```text
Female 87%
21-25

Male 98%
0-14
```

The application supports detecting multiple faces in the same frame.

## Important Notes

- An internet connection is required during the first model download unless the model is already cached locally.
- The face detector model files must remain in the same folder as `main.py`.
- Age predictions are approximate, not exact ages.
- Gender predictions may be incorrect.
- Gender confidence is a model score, not a guarantee of prediction accuracy.
- Results can vary depending on lighting, face angle, image quality, and facial visibility.
- Use images and webcam footage only with appropriate permission.

## Future Improvements

- Improve prediction accuracy through model evaluation and fine-tuning.
- Add face tracking across video frames.
- Improve performance for low-light conditions.
- Add a graphical user interface.
- Generate prediction reports.

## Author

Developed as a Deep Learning and Computer Vision project using Python, OpenCV, PyTorch, and Hugging Face Transformers.
