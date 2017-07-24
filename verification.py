import face_recognition
import os

donald_image_path = os.path.join(os.environ.get("DATA_DIR"), 'putin', '4.jpg')
unknown_image_path = os.path.join(os.environ.get("DATA_DIR"), 'test', 'putin_snl.jpg')
known_image = face_recognition.load_image_file(donald_image_path)
unknown_image = face_recognition.load_image_file(unknown_image_path)

donald_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([donald_encoding], unknown_encoding)

print(results)