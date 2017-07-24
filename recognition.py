import face_recognition
import os
import pandas as pd
import numpy as np
import json
import pickle
from glob import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_fscore_support
np.random.seed(5)

data = []
target = []
face_names = []
num_faces = len(face_names)
# Training load data
faces_selected = set()

with open(os.environ['INPUT_DIR']+'/config.json') as f:
     config = json.load(f)

class_names = config['class_names']
n_jobs = int(config['n_jobs'])
split_prob = config['split_prob']

for file_path in glob(os.environ.get("DATA_DIR")+"/*/*"):
    class_name = file_path.split('/')[-2]
    if class_name in class_names and class_name != 'test':
        if class_name not in faces_selected:
            faces_selected.add(class_name)
            face_names.append(class_name)
        load_image = face_recognition.load_image_file(file_path)
        list_encoding = face_recognition.face_encodings(load_image)
        if len(list_encoding) > 0:
            face_encoding = list_encoding[0]
            data.append(face_encoding)
            target.append(face_names.index(class_name))

face_names = np.array(face_names)
data = np.asarray(data)
dimensions = range(len(face_encoding))
filename = os.path.join(os.environ['OUTPUT_DIR'],'face_names.pkl')
pickle.dump(face_names, open(filename, 'wb'))


df = pd.DataFrame(data, columns=dimensions)
df['is_train'] = np.random.uniform(0, 1, len(df)) <= split_prob
df['face'] = pd.Categorical.from_codes(target, face_names)
df.head()

train, test = df[df['is_train']==True], df[df['is_train']==False]

features = df.columns[:len(face_encoding)]
clf = RandomForestClassifier(n_jobs=n_jobs)
y, _ = pd.factorize(train['face'])
clf.fit(train[features], y)
preds = face_names[np.array(clf.predict(test[features]))]
cross_validation = pd.crosstab(test['face'], preds, rownames=['actual'], colnames=['preds'])
y_true = np.array(list(test['face']))
y_pred = np.array(preds)

p_r_f1 = precision_recall_fscore_support(y_true, y_pred, average='macro')
precision = p_r_f1[0]
recall = p_r_f1[1]
f1_score = p_r_f1[2]
stats = {'label': 'random_forest'}
stats['precision'] = precision
stats['recall'] = recall
stats['f1_score'] = f1_score
print stats

model_filename = os.path.join(os.environ['OUTPUT_DIR'],'model.dat')
pickle.dump(clf, open(model_filename, 'wb'))
stats_filename = os.path.join(os.environ['OUTPUT_DIR'],'stats.json')
with open(stats_filename, 'wb') as f:
    f.write(json.dumps(stats))