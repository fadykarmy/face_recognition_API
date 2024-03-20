import face_recognition
import cv2
import numpy as np
from fastapi import FastAPI
import uvicorn
import joblib

#Load model
model=joblib.load("face_recognition_model.pkl")

#initioalize an app
app =FastAPI(debug=True)


#routing
@app.get('/predict')
def home():
    return {'hello'}

#run through terminal
if __name__=='__main__':
    uvicorn.run(app,host='127.0.0.1',port=8000)