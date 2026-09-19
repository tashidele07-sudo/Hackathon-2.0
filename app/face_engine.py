# app/face_engine.py
import json
import numpy as np
import face_recognition
import cv2

def extract_face_encoding(media_path: str, media_type: str = "image") -> str:
    """Extract 128D encoding from either a .jpg photo or a .mp4 video clip."""
    if media_type == "video" or media_path.endswith(".mp4"):
        cap = cv2.VideoCapture(media_path)
        if not cap.isOpened():
            raise ValueError(f"Could not read video file at {media_path}")

        encodings = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % 5 != 0 and frame_count != 1:
                continue
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_frame)
            if encodings:
                break
        cap.release()
    else:
        image = face_recognition.load_image_file(media_path)
        encodings = face_recognition.face_encodings(image)

    if not encodings:
        raise ValueError(f"No facial features detected in: {media_path}")
        
    return json.dumps(encodings[0].tolist())

def match_face_in_frame(frame: np.ndarray, known_encoding_json: str, tolerance: float = 0.5) -> bool:
    """Detect and match a student's face encoding within a given video frame."""
    known_encoding = np.array(json.loads(known_encoding_json))
    
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces([known_encoding], face_encoding, tolerance=tolerance)
        if True in matches:
            return True
    return False