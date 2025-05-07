# Text Emotion Detection

## Overview

**Text Emotion Detection** is a machine learning-based application designed to classify emotions expressed in textual data. The dataset contains text samples labeled with emotions such as **joy**, **sadness**, **anger**, **fear**, **surprise**, **neutral**, **disgust**, and **shame**.  

The project uses Python libraries such as **Pandas**, **NumPy**, **Seaborn**, **Scikit-learn**, and **Joblib** for preprocessing, training, and saving the model. It is suitable for use cases like sentiment analysis, customer feedback processing, and social media monitoring.

---

## Table of Contents

- [Features](#features)  
- [Dataset](#dataset)  
- [Requirements](#requirements)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Project Structure](#project-structure)  
- [Model Training](#model-training)  
- [Saving and Loading the Model](#saving-and-loading-the-model)  
- [Results](#results)  
- [Contributing](#contributing)  
- [License](#license)  
- [Contact](#contact)  

---

## Features

- **Emotion Classification**: Classifies text into one of eight emotions: *joy, sadness, anger, fear, surprise, neutral, disgust, shame*  
- **Data Preprocessing**: Cleans and prepares text for model training  
- **Machine Learning Pipeline**: Uses Scikit-learn pipeline for TF-IDF feature extraction and classification  
- **Model Persistence**: Saves the trained model using Joblib for future use  
- **Visualization**: Includes Seaborn-based bar plots for emotion distribution  

---

## Dataset

The dataset used is `emotion_dataset_raw.csv`, containing two columns:

- `Text`: The text expressing emotion  
- `Emotion`: The emotion label (e.g., joy, sadness, etc.)

### Emotion Distribution:

| Emotion   | Samples |
|-----------|---------|
| Joy       | 11,045  |
| Sadness   | 6,722   |
| Fear      | 5,410   |
| Anger     | 4,297   |
| Surprise  | 4,062   |
| Neutral   | 2,254   |
| Disgust   | 856     |
| Shame     | 146     |

A bar plot visualizing this distribution is generated using Seaborn in the notebook.

---

## Requirements

You will need the following libraries:

- Python 3.9+  
- pandas  
- numpy  
- seaborn  
- scikit-learn  
- joblib  
- matplotlib  

Install via the provided `requirements.txt`.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/text-emotion-detection.git
cd text-emotion-detection

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
## Dataset Note:
Place emotion_dataset_raw.csv in the root directory. If not available, you may use a similar dataset.

## Usage
Run the Notebook
Open the notebook:
jupyter notebook Text_Emotion_Detection.ipynb
Run the cells to:

Load and preprocess the data

Train the machine learning model

Save the model for later use

Predict Emotions
After training, the model is saved as text_emotion.pkl. You can use it as follows:
import joblib

model = joblib.load("text_emotion.pkl")
text = ["I am so happy today!"]
prediction = model.predict(text)
print(prediction)  # Output: ['joy']
Visualize Results
Seaborn bar plots are included to visualize emotion distribution.

Project Structure
bash
Copy
Edit
text-emotion-detection/
│
├── Text_Emotion_Detection.ipynb   # Main notebook
├── emotion_dataset_raw.csv        # Dataset (not included, must be sourced)
├── text_emotion.pkl               # Saved trained model
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
Model Training
The model training process includes:

Loading the Dataset: Using pandas.read_csv()

EDA:

df.head() for data preview

df['Emotion'].value_counts() and Seaborn for emotion distribution

Preprocessing & Training:

Likely uses tokenization and TF-IDF vectorization

Classifier (e.g., Logistic Regression) in a pipeline

Can be extended with:

Special character removal

Stopword filtering

Train-test split

Evaluation (accuracy, precision, recall)

Hyperparameter tuning

Saving and Loading the Model
python
Copy
Edit
# Save the model
import joblib
pipeline_file = open("text_emotion.pkl", "wb")
joblib.dump(pipe_lr, pipeline_file)
pipeline_file.close()

# Load the model
model = joblib.load("text_emotion.pkl")
Results
The dataset is imbalanced, with "joy" being most frequent and "shame" least.

The trained model can predict emotion for new text input.

Evaluation metrics (accuracy, precision, recall, F1) should be added for better assessment.

Contributing
Contributions are welcome!

To contribute:

Fork the repository

Create a feature branch:

bash
Copy
Edit
git checkout -b feature-branch
Make changes and commit:

bash
Copy
Edit
git commit -m "Add feature"
Push the branch:

bash
Copy
Edit
git push origin feature-branch
Open a Pull Request

Please follow the code style and add tests where applicable.
