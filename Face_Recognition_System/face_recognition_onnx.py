import cv2
#import threading
import time
import numpy as np
import tensorflow as tf
from imutils.video import VideoStream
from mtcnn.mtcnn import MTCNN
#from multiiiiithread import CameraStream  # Import Camera class
import serial
import onnxruntime as ort  # NEW for ONNX
import matplotlib.pyplot as plt

class FaceRecognition:
    def __init__(self, camera):
        self.RTSP_URL = 0  # Using webcam (change to RTSP if needed)
        self.camera = camera
        self.running = False  # Wait for UART command
        self.known_face_start_time = None  
        self.use_onnx = True

        # UART Setup
        self.SERIAL_PORT = "/dev/usb-stm"
        self.BAUD_RATE = 115200
        # UART commented for testing
        # try:
        #     self.uart = serial.Serial(self.SERIAL_PORT, self.BAUD_RATE, timeout=1)
        #     print(f"Connected to {self.SERIAL_PORT} at {self.BAUD_RATE} baud.")
        # except serial.SerialException as e:
        #     print(f"Error: Unable to open serial port {self.SERIAL_PORT}: {e}")
        #     exit()

        # Load FaceNet model
        if self.use_onnx:
            start_time = time.time()
            self.ort_session = ort.InferenceSession("E:/algoAsignments/facenet.onnx")
            print(f"ONNX model load time: {time.time() - start_time:.2f} seconds")

        data = np.load("E:/algoAsignments/[TEMPLATE]/models/linux_lectures/detected_faces_facenet_new.npz", allow_pickle=True)
        self.stored_embeddings = data['faces']
        self.labels = data['labels']
        self.stored_embeddings /= np.linalg.norm(self.stored_embeddings, axis=1, keepdims=True)

        # Initialize MTCNN detector
        self.detector = MTCNN()

    def preprocess_image(self, image):
    # Resize to 160x160
            img = cv2.resize(image, (160, 160))

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # CLAHE (contrast limited adaptive histogram equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_eq = clahe.apply(gray)

            # Gaussian Blur
            gray_blur = cv2.GaussianBlur(gray_eq, (3, 3), 0)

            # Sharpening filter
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharp = cv2.filter2D(gray_blur, -1, kernel)

            # Convert grayscale back to 3-channel BGR
            img_final = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

            # Normalize and expand dims for model input
            img_final_normalized = img_final.astype('float32') / 255.0
            img_final_normalized = np.expand_dims(img_final_normalized, axis=0)

            # === Display all preprocessing steps ===
            #plt.figure(figsize=(15, 3))
            #plt.subplot(1, 5, 1)
            #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            #plt.title('Original'); plt.axis('off')

            #plt.subplot(1, 5, 2)
            #plt.imshow(gray, cmap='gray')
            #plt.title('Grayscale'); plt.axis('off')

            #plt.subplot(1, 5, 3)
            #plt.imshow(gray_eq, cmap='gray')
            #plt.title('CLAHE'); plt.axis('off')

            #plt.subplot(1, 5, 4)
            #plt.imshow(gray_blur, cmap='gray')
            #plt.title('Blurred'); plt.axis('off')

            #plt.subplot(1, 5, 5)
            #plt.imshow(cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB))
            #plt.title('Preprocessed Face'); plt.axis('off')

            #plt.tight_layout()
            #plt.show()

            return img_final_normalized


    def get_face_embedding_onnx(self, face_image):
        preprocessed = self.preprocess_image(face_image)
        ort_inputs = {self.ort_session.get_inputs()[0].name: preprocessed}
        embedding = self.ort_session.run(None, ort_inputs)[0][0]
        l2_norm = np.linalg.norm(embedding)
        embedding /= l2_norm
        return embedding, l2_norm

    def verify_face(self, l2_norm, min_threshold=7, max_threshold=15):
        return min_threshold < l2_norm < max_threshold

    def recognize_face(self, face_embedding, threshold=0.9):
        start_time = time.time()
        min_distance = float("inf")
        best_match = "Unknown"
        for idx, stored_embedding in enumerate(self.stored_embeddings):
            distance = np.linalg.norm(face_embedding - stored_embedding)
            if distance < min_distance and distance < threshold:
                min_distance = distance
                best_match = self.labels[idx]
        print(f"Time to recognize face: {time.time() - start_time:.4f} seconds")        
        return best_match, min_distance

    def process_frames(self):
        print("[INFO] Processing frames for face recognition...")
        self.running = True
        while self.running:
            frame = self.camera.read()
            if frame is None:
                print("[DEBUG] Waiting for camera frame...")
                time.sleep(0.1)
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces_detected = self.detector.detect_faces(frame_rgb)
            recognition_result = "0"
            face_recognized = False  

            for face in faces_detected:
                x, y, w, h = face['box']
                confidence = face['confidence']
                if confidence < 0.80:
                    continue

                x2, y2 = x + w, y + h
                face_crop = frame_rgb[y:y2, x:x2]

                if face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
                    print("[ERROR] Invalid face crop. Skipping.")
                    continue

                embedding, l2_norm = self.get_face_embedding_onnx(face_crop)
               
                if self.verify_face(l2_norm):
                    recognized_name, min_distance = self.recognize_face(embedding, threshold=0.8)
                    print("min_distance: " + str(min_distance))
                    print(f"[INFO] Recognized: {recognized_name} (Distance: {min_distance:.2f})")

                    color = (0, 255, 0) if recognized_name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
                    label_text = f"{recognized_name}: {min_distance:.2f}"
                    cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    frame_filename = f"E:/algoAsignments/frames_camera/frame_{int(time.time())}.jpg"
                    cv2.imwrite(frame_filename, frame)
                    print(f"Saved {frame_filename}")
                    if recognized_name != "Unknown":
                        if self.known_face_start_time is None:
                            self.known_face_start_time = time.time()
                        elif time.time() - self.known_face_start_time >= 1:
                            print(f"[INFO] Known face {recognized_name} recognized for 1 second. Stopping stream.")
                            self.running = False  # Stop processing frames
                        face_recognized = True
                    else:
                        self.known_face_start_time = None  

            if not face_recognized:
                self.known_face_start_time = None  

            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        cv2.destroyAllWindows()

    def run(self):
        self.process_frames()  

    def stop(self):
        self.running = False
        cv2.destroyAllWindows()
        print("[INFO] Stopping face recognition...")

if __name__ == "__main__":
    vs = VideoStream(src=0, resolution=(160, 160), framerate=20).start()
    face_recog = FaceRecognition(vs) 
    try:
        face_recog.run()  
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        face_recog.stop()
