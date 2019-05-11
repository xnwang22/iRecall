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


def align(gray, left_eye_x,left_eye_y, right_eye_x, right_eye_y, center_x, center_y):

    # compute the angle between the eye centroids
    dY = right_eye_y - left_eye_y
    dX = right_eye_x - left_eye_x
    angle = np.degrees(np.arctan2(dY, dX))# - 180

    center_x = dX
    center_y = dY

    # compute the desired right eye x-coordinate based on the
    # desired x-coordinate of the left eye
    #desiredRightEyeX = 1.0 - left_eye_x

    # determine the scale of the new resulting image by taking
    # the ratio of the distance between eyes in the *current*
    # image to the ratio of distance between eyes in the
    # *desired* image
#    dist = np.sqrt((dX ** 2) + (dY ** 2))
#    desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
#    desiredDist *= self.desiredFaceWidth
#    scale = desiredDist / dist

    # compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image

    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)

    # update the translation component of the matrix
    #tX = 200 * 0.5
    #tY = 200 * desiredRightEyeX
    M[0, 2] += (center_x)
    M[1, 2] += (center_y)

    # apply the affine transformation
    (w, h) = (200, 200)
    output = cv2.warpAffine(gray, M, (w, h))

    # return the aligned face
    return output

def get_faces(filename):
    templatename = 'haarcascade_frontalface_alt2.xml'#''haarcascade_frontalface_default.xml'
    eye_templatename = 'haarcascade_eye.xml'
    left_eye_templatename = 'haarcascade_lefteye_2splits.xml'
    right_eye_templatename = 'haarcascade_righteye_2splits.xml'
    face_cascade = cv2.CascadeClassifier(templatename)
    eye_cascade = cv2.CascadeClassifier(eye_templatename)
    right_eye_cascade = cv2.CascadeClassifier(right_eye_templatename)
    left_eye_cascade = cv2.CascadeClassifier(left_eye_templatename)
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 3)
    gray_faces = []
    c = 0
    for (x,y,w,h) in faces:
        #img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        center = (x+w//2, y+h//2)
        radius = (w+h)//4

        face_image = cv2.resize(gray[y:y + h, x:x + w], (200, 200), interpolation=cv2.INTER_LINEAR)

        #find eyes
#        eyes = eye_cascade.detectMultiScale(face_image, 1.2, 3)
#        if (len(eyes) == 2):
#            for (ex, ey, ew, eh) in eyes:
#                #cv2.rectangle(face_image, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
#                eye_center = (ex + ew // 2, ey + eh // 2)
#                eye_radius = (ew + eh) // 4
#                cv2.circle(face_image, eye_center, eye_radius, (128, 128, 0), 2)

        #face_image_color = cv2.resize(img[y:y + h, x:x + w], (200, 200), interpolation=cv2.INTER_LINEAR)

#            face_image = align(face_image, left_eye_x,left_eye_y, right_eye_x, right_eye_y, center[0],center[1])
        gray_faces.append(face_image)
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
