# FastAPI Banknote Authentication Classifier

A simple FastAPI application that serves a machine learning model to classify whether a banknote is real or fake based on its features.

##  Features

- Built with **FastAPI**
- Serves a trained **scikit-learn classifier**
- Accepts JSON input and returns prediction
- Includes GET, POST, PUT endpoints

## Model Input

The model expects a JSON input with the following features:

```json
{
  "variance": 2.3,
  "skewness": 3.1,
  "curtosis": 1.2,
  "entropy": -1.0
}


## How to Run:
* 1. Create a virtual environment:
python -m venv apivenv
Windows: apivenv\Scripts\activate

* 2. Install dependencies:
pip install fastapi uvicorn scikit-learn pandas numpy

* 3. Run the app:
uvicorn app:app --reload

