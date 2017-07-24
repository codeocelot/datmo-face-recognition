import os
import face_recognition

image_path = os.path.join(os.environ.get("DATA_DIR"), 'test', 'test_image.jpg')

image = face_recognition.load_image_file(image_path)
face_locations = face_recognition.face_locations(image)
print(face_locations)