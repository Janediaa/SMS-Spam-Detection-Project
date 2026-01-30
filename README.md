# SMS Spam Detection System

A machine learning-based web application for detecting spam SMS messages using multiple classification models with ensemble predictions.

## Features
- Multi-model ensemble: Naive Bayes, SVM, Random Forest, XGBoost
- Real-time spam/ham classification
- Responsive web interface

## Installation
Follow these steps to set up the project on your local machine.

### Prerequisites
- Python 3.8+
- pip

### Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd SMS-Spam-Detection-Project
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Train Models
```bash
cd backend
python train_models.py
```

### Run the Application
```bash
cd backend
python app.py
```
Access the app at `http://localhost:5000`.

## Project Structure
```
SMS-Spam-Detection-Project/
├── backend/       # Flask backend
├── frontend/      # Frontend files
├── models/        # Trained models
├── data/          # Dataset
├── requirements.txt
└── GUIDE.md
```

Have fun !!
