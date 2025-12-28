🎵INSTRUNET AI CNN-BASED MUSICAL INSTRUMENT RECOGNITION SYSTEM
🎯PROJECT OVERVIEW InstruNet AI is a deep learning–based system that automatically identifies musical instruments from audio files. The system converts audio signals into Mel-spectrogram images and uses a Convolutional Neural Network (CNN) to classify different musical instruments.

This project demonstrates an end-to-end machine learning pipeline including data preprocessing, CNN model development, fine-tuning, evaluation, and analysis.

🎯OBJECTIVES

Convert audio files into Mel-spectrogram representations
Build and train a CNN for musical instrument classification
Improve performance through fine-tuning techniques
Evaluate the model using confusion matrix and class-wise metrics
Analyze classification errors and record final observations
⚙️DATASET Dataset Used: IRMAS (Instrument Recognition in Musical Audio Signals)

Instrument Classes (11):

Cello
Clarinet
Flute
Acoustic Guitar
Electric Guitar
Organ
Piano
Saxophone
Trumpet
Violin
Human Voice
Each instrument is stored in a separate folder and treated as a class label.

PROJECT STRUCTURE InstruNet_AI/ | |-- data/ | |-- raw_audio/ (IRMAS audio files – not pushed to GitHub) | |-- spectrograms/ (Generated spectrogram images – not pushed) | |-- src/ | |-- preprocessing.py | |-- train_cnn.py | |-- train_cnn_finetuned.py | |-- evaluate_model.py | |-- model/ | |-- instrunet_cnn.h5 | |-- instrunet_cnn_finetuned.h5 | |-- requirements.txt |-- .gitignore |-- README.txt

⚙️TECHNOLOGY STACK Programming Language : Python 3 Deep Learning : TensorFlow / Keras Audio Processing : Librosa Visualization : Matplotlib, Seaborn Evaluation : Scikit-learn IDE : VS Code Version Control : Git & GitHub

📊EVALUATION METRICS

Accuracy
Precision
Recall
F1-score
Confusion Matrix
Misclassifications mainly occurred between spectrally similar instruments such as acoustic vs electric guitar and voice vs saxophone.

HOW TO RUN THE PROJECT

Create virtual environment python -m venv venv source venv/Scripts/activate

Install dependencies pip install -r requirements.txt

Generate spectrograms python src/preprocessing.py

Train baseline CNN python src/train_cnn.py

Fine-tune the model python src/train_cnn_finetuned.py

Evaluate final model python src/evaluate_model.py

FINAL STATUS

Dataset prepared successfully
CNN trained and fine-tuned
Final model selected
Evaluation and error analysis completed
Documentation completed
AUTHOR

Akshaya CNN-Based Musical Instrument Recognition System

LICENSE

This project is intended for academic and educational purposes only.