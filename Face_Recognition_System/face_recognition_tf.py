import cv2
#import threading
import time
import numpy as np
import tensorflow as tf
from imutils.video import VideoStream
from mtcnn.mtcnn import MTCNN
#from multiiiiithread import CameraStream  # Import Camera class
import serial

class FaceRecognition:
    def __init__(self, camera):
        self.RTSP_URL = 0  # Using webcam (change to RTSP if needed)
        #print(f"[INFO] Initializing Face Recognition on Thread: {threading.get_ident()}")
        self.camera = camera
        self.running = False  # Change from True to False (Wait for UART command)
        self.known_face_start_time = None  

        # UART Setup
        self.SERIAL_PORT = "/dev/usb-stm"
        self.BAUD_RATE = 115200
        #try:
            #self.uart = serial.Serial(self.SERIAL_PORT, self.BAUD_RATE, timeout=1)
            #print(f"Connected to {self.SERIAL_PORT} at {self.BAUD_RATE} baud.")
        #except serial.SerialException as e:
            #print(f"Error: Unable to open serial port {self.SERIAL_PORT}: {e}")
            #exit()

        # Load FaceNet model
        #tensorflow_model_dir = "/home/pi/Desktop/amal/facenet-tensorflow-tensorflow2-default-v2"
        start_time = time.time()
        self.interpreter = tf.lite.Interpreter(model_path="E:/algoAsignments/facenet.tflite")
        self.interpreter.allocate_tensors()
        print(f"Model load time: {time.time() - start_time:.2f} seconds")
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        #self.model = tf.saved_model.load(tensorflow_model_dir)
        #self.infer = self.model.signatures['serving_default']
        # Load stored embeddings & labels
        #data = np.load("/home/pi/Desktop/amal/detected_faces_facenet_4_classes.npz", allow_pickle=True)
        data = np.load("E:/algoAsignments/detected_faces_facenet_solve_lighting.npz", allow_pickle=True)
        self.stored_embeddings = data['faces']
        self.labels = data['labels']
        self.stored_embeddings /= np.linalg.norm(self.stored_embeddings, axis=1, keepdims=True)

        # Initialize MTCNN detector
        self.detector = MTCNN()

    #def receive_uart_command(self):
        #"""Wait for a command from UART before starting face detection."""
        #print("[INFO] Waiting for command from UART...")
        #while True:
            #if self.uart.in_waiting > 0:  # Check if data is available
                #command = self.uart.readline().decode('utf-8').strip()
                #if command:  
                    #print(f"[INFO] Received command from UART: {command}")
                    #return  # Exit loop and proceed to face detection

    #def send_by_uart(self, data):
        #self.uart.write(data.encode())

    def preprocess_image(self, image):
        """ Resize and normalize face image for FaceNet """
        img = cv2.resize(image, (160, 160))
        img = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        return img


    def get_face_embedding(self, face_image):
        """ Generate face embedding using FaceNet model """
        #face_tensor = tf.convert_to_tensor(self.preprocess_image(face_image), dtype=tf.float32)
        #embedding = self.infer(face_tensor)['Bottleneck_BatchNorm'].numpy().flatten()
        #l2_norm = np.linalg.norm(embedding)
        #embedding /= l2_norm
        #return embedding, l2_norm
        preprocessed = self.preprocess_image(face_image).astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed)
        self.interpreter.invoke()
        embedding = self.interpreter.get_tensor(self.output_details[0]['index']).flatten()
        l2_norm = np.linalg.norm(embedding)
        embedding /= l2_norm
        return embedding, l2_norm

    def verify_face(self, l2_norm, min_threshold=7, max_threshold=15):
        """ Check if face embedding norm is within an acceptable range """
        return min_threshold < l2_norm < max_threshold

    def recognize_face(self, face_embedding, threshold=0.9):
        """ Compare face embedding with stored embeddings """
        start_time = time.time()  # Start time for recognition
        min_distance = float("inf")
        best_match = "Unknown"
        for idx, stored_embedding in enumerate(self.stored_embeddings):
            distance = np.linalg.norm(face_embedding - stored_embedding)
            #print("distance: "+str(distance))
            if distance < min_distance and distance < threshold:
                min_distance = distance
                best_match = self.labels[idx]
        end_time = time.time()  # End time after recognition is done
        print(f"Time to recognize face: {end_time - start_time:.4f} seconds")        
        return best_match, min_distance

    def process_frames(self):
        """ Process camera frames after receiving UART command """
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
                #print("confidence: "+str(confidence))
                if confidence < 0.90:
                    continue

                x2, y2 = x + w, y + h
                face_crop = frame_rgb[y:y2, x:x2]

                if face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
                    print("[ERROR] Invalid face crop. Skipping.")
                    continue
     
                embedding, l2_norm = self.get_face_embedding(face_crop)

                if self.verify_face(l2_norm):
                    recognized_name, min_distance = self.recognize_face(embedding, threshold=0.8)
                    print("min_distance: "+str(min_distance))
                    #recognition_result = "1" if recognized_name != "Unknown" else "0"
                    #self.send_by_uart(recognition_result)  # Send result to UART

                    print(f"[INFO] Recognized: {recognized_name} (Distance: {min_distance:.2f})")

                    color = (0, 255, 0) if recognized_name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
                    label_text = f"{recognized_name}: {min_distance:.2f}"
                    cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                     # Save the frame
                    frame_filename = f"E:/algoAsignments/frames_camera/frame_{int(time.time())}.jpg"
                    cv2.imwrite(frame_filename, frame)
                    print(f"Saved {frame_filename}")
                    if recognized_name != "Unknown":
                        #if self.known_face_start_time is None:
                         #   self.known_face_start_time = time.time()
                        #elif time.time() - self.known_face_start_time >= 1:
                         #   print(f"[INFO] Known face {recognized_name} recognized for 1 second. Stopping stream.")
                          #  self.running = False  # Stop processing frames
                        return
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
        """ Main function to start face recognition after receiving a UART command """
        #self.receive_uart_command()  # Wait for UART command
        self.process_frames()  # Start processing frames
        
    def stop(self):
        """ Stop face recognition """
        self.running = False
        cv2.destroyAllWindows()
        print("[INFO] Stopping face recognition...")

if __name__ == "__main__":
    #camera=CameraStream("/dev/video-car")
    vs = VideoStream(src=0, resolution=(320, 320), framerate=30).start()
    face_recog = FaceRecognition(vs) 
    try:
        face_recog.run()  
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        face_recog.stop()
