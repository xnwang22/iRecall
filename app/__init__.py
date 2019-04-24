import cv2
import os
import sys
import numpy
#import exceptions


def saveFace(filename):
    templatename = 'haarcascade_frontalface_default.xml'

    face_cascade = cv2.CascadeClassifier(templatename)

    img = cv2.imread(filename)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    count=0
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        f = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
        cv2.imwrite('./data/unknown_%s.pgm' % str(count), f)
        count = count +1
    return count

def showFace(filename):
    templatename = 'haarcascade_frontalface_default.xml'

    face_cascade = cv2.CascadeClassifier(templatename)

    img = cv2.imread(filename)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    count=0
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        count = count +1
    cv2.namedWindow('Faces')
    cv2.imshow('Face Detected!', img)
#    cv2.imwrite('./face.jpg', img)
    cv2.waitKey(0)
    return count

#y - name X - image of face
def memorizeFaces(foldername):
    c = 0
    X, y = [], []
    names = []
    for dirname, dirnames, filenames in os.walk(foldername):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    if (filename == ".directory"):
                        continue
                    filepath = os.path.join(subject_path, filename)
                    im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                    # resize to given size (if given)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                    names.append(filename)
#                except IOError, (errno, strerror):
#                   print ("I/O error({0}): {1}".format(errno, strerror))
                except:
                    print ("Unexpected error:", sys.exc_info()[0])
                    raise
            c = c + 1

    y = np.asarray(y, dtype=np.int32)

    model = cv2.face.createEigenFaceRecognizer()
    model.train(np.asarray(X), np.asarray(y))
    return model

def recognizeFaces(filename, model, names):
    templatename = 'haarcascade_frontalface_default.xml'

    face_cascade = cv2.CascadeClassifier(templatename)

    img = cv2.imread(filename)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    count = 0
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count = count + 1
        try:
            roi = cv2.resize(img, (200, 200), interpolation=cv2.INTER_LINEAR)
            params = model.predict(roi)
            print ("Label: %s, Confidence: %.2f" % (params[0], params[1]))
            cv2.putText(img, params[0], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        except:
            continue
        cv2.imshow("camera", img)
        if cv2.waitKey(1000 / 12) & 0xff == ord("q"):
            break
    cv2.destroyAllWindows()


    cv2.namedWindow('Faces')
    cv2.imshow('Face Detected!', img)
    #    cv2.imwrite('./face.jpg', img)
    cv2.waitKey(0)
    return count



if __name__ == "__main__":
    saveFace('test.jpg')
    showFace('test.jpg')
#    model = memorizeFaces('./data')
#    recognizeFaces('test.jpg', model)
