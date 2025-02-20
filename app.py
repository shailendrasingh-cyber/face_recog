import cv2
import face_recognition
import os
import numpy as np

def load_and_encode_images(images_path):
    known_encodings = []
    known_names = []
    
    if not os.path.exists(images_path):
        print(f"Error: The folder '{images_path}' does not exist.")
        return known_encodings, known_names
    
    for filename in os.listdir(images_path):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(images_path, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            
            if encodings:  # Ensure the image contains at least one face
                known_encodings.append(encodings[0])
                known_names.append(os.path.splitext(filename)[0])
            else:
                print(f"Warning: No face found in {filename}, skipping...")
    
    return known_encodings, known_names

def recognize_faces(known_encodings, known_names):
    video_capture = cv2.VideoCapture(0) 
    
    if not video_capture.isOpened():
        print("Error: Could not access the webcam.")
        return
    
    print("Press 'q' to exit the face recognition window.")
    
    while True: 
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture video. Exiting...")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"
            
            if matches:
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]
            
            top, right, bottom, left = [v * 4 for v in face_location]  # Scale back up
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    images_path = "images"  
    known_encodings, known_names = load_and_encode_images(images_path)
    print(f"Loaded encodings for {len(known_names)} people.")
    if known_encodings:
        recognize_faces(known_encodings, known_names)
    else:
        print("No valid face encodings found. Exiting...")
