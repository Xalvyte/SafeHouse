from picamera2 import Picamera2
import cv2
from camera.recognition import recognize_faces

# Init camera
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration({"size": (1280, 720)}))
picam2.start()

def generate_frames():
    while True:
        frame = picam2.capture_array()
        frame = recognize_faces(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
