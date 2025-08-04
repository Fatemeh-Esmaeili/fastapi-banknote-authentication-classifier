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
```

## How to Run:
* 1. Create a virtual environment:
```
python -m venv apivenv
On Windows: apivenv\Scripts\activate
```

* 2. Install dependencies:
```
pip install fastapi uvicorn scikit-learn pandas numpy
```
or using the requirements.txt file
```
pip install -r requirements.txt
```
This file is generated with the following code: 
```
pip freeze > requirements.txt
```

* 3. Run the app:
```
uvicorn FileName:ObjectName --reload
uvicorn app2:application --reload
uvicorn app:app --reload
```
* 4. Access the API docs:

Swagger UI: http://127.0.0.1:8000/docs

Redoc: http://127.0.0.1:8000/redoc

## API Endpoints

| Method | Endpoint           | Description                         |
|--------|--------------------|-------------------------------------|
| GET    | `/`                | Welcome message                     |
| GET    | `/{name}`          | Personalized message                |
| POST   | `/predict`         | Predict with banknote features      |
| PUT    | `/update_predict`  | Update banknote features and predict|



