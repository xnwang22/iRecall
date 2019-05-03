import cv2
import os
import sys
import numpy as np
import uuid

algorithms = ['basic', 'lbh', 'eigen', 'fisher']
models = {'lbh': cv2.face.LBPHFaceRecognizer_create(), 'eigen': cv2.face_EigenFaceRecognizer.create(),
          'fisher': cv2.face_FisherFaceRecognizer.create()}



def get_key_by_value(dictionary, val):
    for key, value in dictionary.items():
        if val == value[0]:
            return key
    return "key doesn't exist"

def process_image(filename, dictionary, root_folder, maxConfidence):
    countTotal = 0
    countRecognized = 0
    countUnkown = 0
    faces = get_faces(filename)
    for face in faces:
        if dictionary is not None:
            (person_id, Confidence) = recognize_face(face, root_folder)
            if Confidence < maxConfidence:
                print("Label: %s, Confidence: %.2f" % (get_key_by_value(dictionary,person_id), Confidence))
                save_face(face, get_key_by_value(dictionary,person_id), root_folder)
                countRecognized += 1
            else:
                save_face(face, 'unknown', root_folder)
                countUnkown += 1
        else:
            save_face(face, 'unknown', root_folder)
            countUnkown += 1
        countTotal += 1
    print("Found "+str(countTotal)+" faces, "+str(countRecognized)+" known faces, "+str(countUnkown)+" unknown faces")

def get_faces(filename):
    templatename = 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(templatename)
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    gray_faces = []
    c = 0
    for (x,y,w,h) in faces:
        #img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        gray_faces.append(cv2.resize(gray[y:y + h, x:x + w], (200, 200), interpolation=cv2.INTER_LINEAR))
        c += 1
    return gray_faces

def recognize_face(face, root_folder):
    try:
        model = cv2.face_EigenFaceRecognizer.create()
        model.read(root_folder + '/model_eigen.tst')
        #gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        params = model.predict(face)
        return params
    except cv2.error as e:
        print("iRecall error:", e)
    except:
        print("recognize_face Error:", sys.exc_info()[0])
    return (-1, 1000000)

def save_face(face, folder_name, root_folder):
    try:
        person_folder = os.path.join(root_folder, folder_name)
        try:
            os.stat(person_folder)
        except:
            os.mkdir(person_folder)

        filename = os.path.join(person_folder, uuid.uuid4().hex + '.pgm')
        #filename = str(root_folder + '/' + folder_name + '/' + uuid.uuid4().hex() + '.pgm')
        cv2.imwrite(filename, face)
    except cv2.error as e:
        print("cv2 error:", e)
    except:
        print("Unexpected error:", sys.exc_info()[0])


def save_dictionary(dic, root_folder):
    np.save(root_folder + '/dictionary.npy',dic)

def load_dictionary(root_folder):
#    dic = None
    try:
        dictionary_path = os.path.join(root_folder,'dictionary.npy')
        dic = np.load(dictionary_path, encoding = 'latin1').item()
        return dic
    except:
        print("Dictionary doesn't exist, all faces will be placed in the unknown folder:", sys.exc_info()[0])
        return None

# read images in all subfolders of the root folder
# each subfolder name is a name of a person, his/her name also stored in dictionary
# and id form dictionary used for model training
# y - name X - image of face

def train_model(root_folder):
    dictionary = {}
    count_names = 0
    c = 0
    X, y = [], []
    image_names = []
    #collect training material
    for subdir, dirs, filenames1 in os.walk(root_folder):
        for dir in dirs:
            if dir == 'unknown':
                continue
            print(os.path.join(root_folder, dir))
            for d1,d2,filenames in os.walk(os.path.join(root_folder, dir)):
                dictionary[dir]=[]
                dictionary[dir].append(count_names)
                count_names += 1
                for filename in filenames:
                    try:
                        filepath = os.path.join(d1,filename)
                        im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                        # resize to given size (if given)
                        X.append(np.asarray(im, dtype=np.uint8))
                        y.append(dictionary[dir])

                        c = c + 1
                        image_names.append(filename)
                    except:
                        print("Unexpected error:", sys.exc_info()[0])
                        raise

    if c == 0:
        print("no training material found in the folder " + root_folder + " cannot create model and dictionary")
        return
    else:
        y = np.asarray(y, dtype=np.int32)
        model = None
        model = cv2.face_EigenFaceRecognizer.create()
        model.train(np.asarray(X), np.asarray(y))
        np.save(root_folder + '/dictionary.npy', dictionary)
        model.save(root_folder + '/model_eigen.tst')
        dictionary = np.load(root_folder + '/dictionary.npy')
