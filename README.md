# face-mask-detector-live-alert
Real-time face mask detection system using OpenCV and TensorFlow with live webcam feed and alert functionality. Trained on a custom dataset using CNN and deployed locally for instant monitoring. Ideal for smart surveillance in public spaces.

# 😷 Face Mask Detector with Live Alert System

This is a real-time face mask detection project built using **Python**, **TensorFlow/Keras**, and **OpenCV**. It detects whether a person is wearing a mask using a webcam and raises an **instant alert** if a mask is not detected. Ideal for smart surveillance systems in public areas such as offices, malls, hospitals, and schools.

---

## 📌 Project Highlights

- ✅ Real-time webcam feed using OpenCV
- ✅ CNN-based custom trained model
- ✅ Face detection using Haar Cascades
- ✅ Audio alert for “No Mask” detection (optional)
- ✅ High accuracy detection using Keras/TensorFlow
- ✅ Can be extended to web UI using Streamlit or Flask

---

## 🧠 Tech Stack

| Layer            | Tools Used                             |
|------------------|-----------------------------------------|
| Programming      | Python 3.7+                             |
| ML/DL Framework  | TensorFlow, Keras                       |
| Image Processing | OpenCV, Haar Cascades                   |
| Visualization    | Matplotlib                             |
| Sound Alert      | Playsound (optional)                    |
| Deployment       | Flask / Streamlit (optional)            |

---

## 📁 Folder Structure

face-mask-detector-live-alert/
├── dataset/
│ ├── with_mask/
│ └── without_mask/
├── mask_detector_model.h5 # Saved trained model
├── train_mask_detector.py # Model training script
├── realtime_mask_detector.py # Real-time webcam detection
├── requirements.txt # All dependencies
├── alert.wav # Optional audio alert
└── README.md

---

## 📦 Requirements

Install the dependencies using:
```bash
pip install -r requirements.txt
requirements.txt should contain:

bash
Copy
Edit
tensorflow>=2.9.0
keras>=2.9.0
opencv-python
numpy
matplotlib
playsound        # Optional for alerts
streamlit        # Optional for deployment
flask            # Optional for deployment
🧠 Model Training — train_mask_detector.py
Load dataset from Kaggle – https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset/data

Resize and normalize images

Use a CNN architecture (Conv2D, MaxPooling, Flatten, Dense)

Compile and train model with Keras

Save as mask_detector_model.h5
🧠 Model Training — train_mask_detector.py
Load dataset from Kaggle – https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset/data

Resize and normalize images

Use a CNN architecture (Conv2D, MaxPooling, Flatten, Dense)

Compile and train model with Keras

Save as mask_detector_model.h5

📈 Add optional training visualization:

python
Copy
Edit
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.show()
🎥 Real-Time Detection — realtime_mask_detector.py
Load webcam feed using cv2.VideoCapture(0)

Use Haar Cascade to detect faces

Resize face images to 100x100

Use trained model (.h5) to predict mask or no mask

Draw bounding box:

Green: "Mask"

Red: "No Mask"

Optional: play alert sound using playsound('alert.wav')

Press q to quit

🚀 How to Run the Project
🔹 Step 1: Clone the Repository
bash
Copy
Edit
git clone 
cd face-mask-detector-live-alert
🔹 Step 2: Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
🔹 Step 3: Train the Model (optional if model is provided)
bash
Copy
Edit
python train_mask_detector.py
🔹 Step 4: Run Real-Time Detection
bash
Copy
Edit
python realtime_mask_detector.py
🧪 Sample Output
👤 Face with Mask:

Green box with label: Mask

🚫 Face without Mask:

Red box with label: No Mask

Optional: Sound alert using alert.wav

🎯 Future Enhancements
 Deploy as Streamlit or Flask Web App

 Add email/SMS alert integration

 Support for thermal scanning

 Multi-face detection with distance estimation

 Add confidence score and log reports

🖼️ Demo (Add Screenshots or GIF)
With Mask	Without Mask

🔗 Dataset Used
Kaggle Dataset: https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset/data

💼 Real-World Applications
Office security systems

Hospital entry gate scanners

Retail stores

University campuses

Airports and transportation hubs

📜 License
This project is licensed under the MIT License.

🙌 Author
Shivain Gupta
📧 shivaingupta999@gmail.com
🔗 LinkedIn :  https://www.linkedin.com/in/shivain-gupta-827772346/
    GitHub   :  https://github.com/Shivain-codes/face-mask-detector-live-alert
    
