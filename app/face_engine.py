import json
import numpy as np
import face_recognition
import cv2

def extract_face_encoding(image_path: str) -> str:
    """Extract 128D encoding from a saved image file and return as JSON string."""
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    if not encodings:
        raise ValueError("No facial features detected in the uploaded image.")
    return json.dumps(encodings[0].tolist())

def match_face_in_frame(frame: np.ndarray, known_encoding_json: str, tolerance: float = 0.5) -> bool:
    """Detect and match a student's face encoding within a given video frame."""
    known_encoding = np.array(json.loads(known_encoding_json))
    
    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces([known_encoding], face_encoding, tolerance=tolerance)
        if True in matches:
            return True
    return False