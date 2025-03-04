# Face Recognition System

This project is a simple real-time face recognition system using OpenCV and the `face_recognition` library in Python. It loads images from a specified folder, encodes the faces, and then detects and recognizes faces in a live video feed.

## Installation

Ensure you have Python installed on your system, then install the required dependencies:

```sh
pip install opencv-python face-recognition numpy
```

## Usage

1. **Add Images:** Place images of known people in the `images` folder. The filename (excluding the extension) will be used as the name for that person when detected.
2. **Run the Script:** Execute the following command in the terminal:

   ```sh
   python app.py
   ```

3. **Detection:** The script will open a webcam feed and try to match detected faces with those stored in the `images` folder.
4. **Exit:** Press 'q' to close the face recognition window.

## Folder Structure

```
project-folder/
│── images/       # Folder containing known images
│   ├── person1.jpg
│   ├── person2.png
│── app.py  # Main script
│── README.md    # Documentation
```

## Notes
- Ensure the images have clear and visible faces.
- The system scales the video frames to improve performance.
- It matches faces based on encodings and displays the person's name (filename without extension) if recognized.

