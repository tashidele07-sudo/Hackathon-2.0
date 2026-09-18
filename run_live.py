from app.realtime_recognizer import run_live_opencv_recognition

if __name__ == "__main__":
    print("Starting Real-Time OpenCV Face Recognition...")
    print("Press 'q' on the camera window to quit.")
    
    # 0 uses your computer's default webcam
    run_live_opencv_recognition(video_source=0)