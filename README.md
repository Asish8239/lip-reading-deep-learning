🧠 Lip Reading from Silent Videos using Deep Learning

This project focuses on automatic lip reading from silent video clips, converting visual mouth movements into readable text using deep learning techniques.
The system processes video frames, extracts lip regions, learns spatio-temporal patterns, and predicts spoken sentences without using audio.

🚀 Project Highlights

🎥 Lip reading from silent videos only

🧠 Deep Learning model using CNN + LSTM

🧾 CTC Loss for sequence-to-sequence prediction

📐 Automatic lip region extraction

🔁 End-to-end pipeline: video → frames → lips → text

🧪 Trained and tested on real-world dataset samples

🛠️ Technologies & Tools Used
Programming & Frameworks

Python

TensorFlow / Keras

NumPy

OpenCV

Computer Vision

MediaPipe – Face & lip landmark detection

Frame extraction and preprocessing

Deep Learning Architecture

CNN (Convolutional Neural Network) – spatial feature extraction

LSTM (Long Short-Term Memory) – temporal sequence modeling

CTC Loss (Connectionist Temporal Classification) – alignment-free sequence learning

Dataset

GRID Corpus (speaker-wise video and alignment files)

Dataset not included in the repository due to size constraints.

📂 Project Structure
LipReadingProject/
│
├── model/
│   ├── cnn_lstm_model.py
│   ├── train.py
│   └── predict.py
│
├── preprocessing/
│   ├── extract_frames.py
│   ├── lip_extraction.py
│   ├── data_loader.py
│   └── label_parser.py
│
├── utils/
│   └── text_encoder.py
│
├── .gitignore
├── README.md
└── requirements.txt

🔄 Workflow Overview

Video Input

Silent video clips are taken as input.

Frame Extraction

Each video is split into fixed-length frame sequences.

Lip Region Extraction

MediaPipe is used to detect facial landmarks.

Lip regions are cropped and resized.

Data Preparation

Lip frames are normalized and stacked.

Alignment files (.align) are parsed and encoded.

Model Training

CNN extracts spatial features from each frame.

LSTM learns temporal dependencies.

CTC Loss aligns predictions with variable-length text labels.

Prediction

Given a new silent video, the model predicts the spoken sentence.

🧪 Example Prediction
python model/predict.py bbaf2n


Output:

Predicted Text: bin blue at two

⚠️ Notes

Due to large size, datasets, extracted frames, and trained models are excluded from the repository.

The project is designed to be scalable to multiple speakers and vocabularies.

Training on CPU is slow; GPU is recommended for faster experimentation.

📌 Key Learnings

Practical implementation of sequence learning with CTC

Handling visual-only speech recognition

Efficient preprocessing for video-based deep learning

End-to-end ML project structuring and deployment readiness

📈 Future Improvements

Add Transformer-based architectures

Improve accuracy with data augmentation

Real-time lip reading support

Web interface for live predictions

👤 Author

Asish Samiraju
Aspiring ML Engineer | Deep Learning & Computer Vision
🔗 GitHub: https://github.com/Asish8239

⭐ Acknowledgements

GRID Corpus Dataset

TensorFlow & MediaPipe Teams

Open-source deep learning community