import urllib, cStringIO
import scipy 
import pickle
import os
import numpy as np
import json
import face_recognition
from flask import Flask, request, jsonify


filename = os.path.join(os.environ['SNAPSHOT_DIR'],'model.dat')
clf = pickle.load( open(filename , "rb" ) )
filename = os.path.join(os.environ['SNAPSHOT_DIR'],'face_names.pkl')
face_names = np.array(pickle.load(open(filename, 'rb')))

def add(params):
    return params['a'] + params['b']


def recognition(params):
    """
    Loads an image url (.jpg, .png, etc) into a numpy array
    :param url: image url to load
    :return: face recognition over image url
    """
    image_file = cStringIO.StringIO(urllib.urlopen(params['url']).read())
    image = scipy.misc.imread(image_file, mode='RGB')
    # read the image file in a numpy array
    list_encoding = face_recognition.face_encodings(image)
    test_pred = []
    test_preds = []
    if list_encoding:
        for encoding in list_encoding:
            test_pred = face_names[clf.predict([encoding])][0]
            test_preds.append(test_pred)
    return list(test_preds)

functions_list = [add, recognition]

app = Flask(__name__)

@app.route('/<func_name>', methods=['POST'])
def api_root(func_name):
    for function in functions_list:
        if function.__name__ == func_name:
            try:
                json_req_data = request.get_json()
                if json_req_data:
                    res = function(json_req_data)
                else:
                    return jsonify({"error": "error in receiving the json input"})
            except Exception as e:
                return jsonify({"error": "error while running the function"})
            return jsonify({"result": res})
    output_string = 'function: %s not found' % func_name
    return jsonify({"error": output_string})

if __name__ == '__main__':
    app.run(host='0.0.0.0')