import asyncio
import socket
import websockets
import cv2
import numpy as np
import pickle
import time
import threading
import queue
import face_recognition
from flask import Flask, Response

# Load preprocessed face encodings
ENCODINGS_FILE = "face_encodings.pkl"

with open(ENCODINGS_FILE, "rb") as f:
    data = pickle.load(f)
    known_face_encodings = data["encodings"]
    known_face_names = data["names"]

frame_queue = queue.Queue()
face_locations = []
face_encodings = []
lock = threading.Lock()

ENCODE_INTERVAL = 5
frame_count = 0
running = True  # Global flag to control threads

# Flask app for video streaming
app = Flask(__name__)

def process_frames():
    global face_locations, face_encodings, frame_count, running
    while running:
        if frame_queue.empty():
            time.sleep(0.01)  # Avoid busy waiting
            continue

        frame = frame_queue.get()
        start_time = time.time()

        # Convert to RGB for face recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Update face locations periodically
        if frame_count % ENCODE_INTERVAL == 0:
            with lock:
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, model="large")

        # Process each detected face
        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.45)
            name = "Unknown"
            color = (0, 255, 255)  # Yellow for unknown

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                color = (0, 255, 0) if name.endswith("G") else (0, 0, 255)  # Green/Red based on suffix

            # Draw bounding box and label
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        frame_count += 1

        # Calculate and log FPS
        fps = 1 / max(1e-5, (time.time() - start_time))
        print(f"FPS: {int(fps)}")  # Print FPS instead of showing image

        # For Render, we can't display the frame with cv2.imshow()
        # Save the processed frames to the queue
        frame_queue.put(frame)

# MJPEG Streaming Function for Flask
def generate_video():
    while running:
        if not frame_queue.empty():
            frame = frame_queue.get()
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

async def handle_websocket(websocket):
    print("New connection established")
    try:
        while running:
            message = await websocket.recv()
            if isinstance(message, bytes):
                np_array = np.frombuffer(message, dtype=np.uint8)
                frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
                if frame is not None:
                    frame_queue.put(frame)
    except websockets.exceptions.ConnectionClosed:
        print("Connection closed")

async def main():
    local_ip = get_local_ip()
    server = await websockets.serve(handle_websocket, local_ip, 8888)
    print(f"WebSocket server started on ws://{local_ip}:8888")
    try:
        while running:
            await asyncio.sleep(1)
    finally:
        server.close()
        await server.wait_closed()

if __name__ == "__main__":
    process_thread = threading.Thread(target=process_frames)
    process_thread.start()

    # Run Flask app in a separate thread
    flask_thread = threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5000))
    flask_thread.start()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        running = False
    finally:
        running = False
        process_thread.join()
        flask_thread.join()
        cv2.destroyAllWindows()
