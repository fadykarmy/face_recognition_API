from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
import face_recognition
import numpy as np
from PIL import Image
import io
import os

app = FastAPI()

# Directory where known face images are stored
KNOWN_FACES_DIR = "img/"

# Load known face encodings and their names
known_face_encodings = []
known_face_names = []

def load_known_faces():
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            name = os.path.splitext(filename)[0]
            image_path = os.path.join(KNOWN_FACES_DIR, filename)
            face_image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(face_image)[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(name)

load_known_faces()

@app.post("/recognize/")
async def recognize_faces(image: UploadFile = File(...)):
    try:
        # Read image file
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        img = np.array(img)

        # Find all faces in the image
        face_locations = face_recognition.face_locations(img)
        face_encodings = face_recognition.face_encodings(img, face_locations)

        # Recognize faces
        recognized_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            recognized_names.append(name)

        return recognized_names

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
