# Lip Reading using Deep Learning (CNN + LSTM)

This project implements a **visual speech recognition (lip reading) system** that predicts spoken text directly from lip movements in video — without using audio.

The goal of this project was to gain hands-on experience with **computer vision, sequence modeling, and deep learning pipelines**, and to understand how real-world ML systems are built end-to-end.

---

## 🚀 Project Overview

The system works as follows:

1. Input video of a speaker
2. Extract frames from video
3. Detect facial landmarks and crop the lip region
4. Convert lip frames into fixed-length sequences
5. Train a CNN + LSTM model using **CTC loss**
6. Predict spoken text from unseen videos

---

## 🧠 Model Architecture

- **CNN (TimeDistributed)** for spatial feature extraction
- **LSTM layers** for temporal sequence modeling
- **CTC Loss** for alignment-free sequence prediction

Input → CNN → LSTM → LSTM → Dense → CTC Loss


---

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- MediaPipe
- NumPy
- CNN + LSTM
- CTC Loss
- Sequence Padding & Encoding

---

## 📂 Project Structure



LipReadingProject/
├── preprocessing/
├── model/
├── data/
├── saved_model/
├── requirements.txt
└── README.md


---

## ⚙️ How to Run

### 1️⃣ Create virtual environment
```bash
python -m venv venv

2️⃣ Activate environment
# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt

🧪 Training the Model
python model/train.py


The model is trained using CTC loss for sequence prediction.

🔮 Prediction

To predict text from a lip video:

python model/predict.py <video_name>


Example:

python model/predict.py bbaf2n


Output:

Predicted Text: bin blue at two

📈 Key Learnings

End-to-end deep learning pipeline design

Lip region extraction using facial landmarks

Handling temporal data with CNN-LSTM

Training sequence models with CTC loss

Debugging performance and data issues

Structuring ML projects like production systems

🔮 Future Improvements

Improve accuracy with larger datasets

Use 3D CNNs or Transformers

Add beam search decoding

Real-time webcam inference

Speaker-independent training

📌 Disclaimer

This project is built for learning and experimentation purposes to understand visual speech recognition and deep learning systems.

👤 Author

Asish Samiraju
Front-End Developer | ML & Data Analytics Learner