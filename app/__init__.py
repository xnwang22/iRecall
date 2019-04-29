import cv2
import os
import sys
import numpy
#import exceptions
recignized_names=['aaron', 'bad', 'callie', 'karen', 'doug', 'mike', 'nick', 'mark', 'opera', 'sophia', 'unknown']
algorithms = ['basic', 'lbh', 'eigen', 'fisher']
models = {'lbh': cv2.face.LBPHFaceRecognizer_create(), 'eigen': cv2.face_EigenFaceRecognizer.create(),
          'fisher': cv2.face_FisherFaceRecognizer.create()}

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
        cv2.imwrite('./data/{}.pgm'.format(recignized_names[count]), f)
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
    image_names = []
    for dirname, dirnames, filenames in os.walk(foldername):
 #       for subdirname in dirnames:
 #           subject_path = os.path.join(dirname, subdirname)
        for filename in filenames:
            try:
                if (filename == ".directory"):
                    continue
                filepath = os.path.join(foldername, filename)
                im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                # resize to given size (if given)
                X.append(numpy.asarray(im, dtype=numpy.uint8))
                y.append(c)
                c = c + 1
                image_names.append(filename)
#                except IOError, (errno, strerror):
#                   print ("I/O error({0}): {1}".format(errno, strerror))
            except:
                print ("Unexpected error:", sys.exc_info()[0])
                raise

    y = numpy.asarray(y, dtype=numpy.int32)

    model0 = None
    for i in range(1,4):
        model = models[algorithms[i]]
        model.train(numpy.asarray(X), numpy.asarray(y))
        model.save('model_{}.tst'.format(algorithms[i]))
    return models

def recognizeFaces(filename, i):

    model = models[algorithms[i]]
    model.read('model_{}.tst'.format(algorithms[i]))

    templatename = 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(templatename)

    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    count = 0
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        img1 = cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count = count + 1
        try:
            roi = cv2.resize(img1[y:y + h, x:x + w], (200, 200), interpolation=cv2.INTER_LINEAR)
            params = model.predict(roi)
            print("Label: %s, Confidence: %.2f" % (params[0], params[1]))
            cv2.putText(img, recignized_names[params[0]], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        except cv2.error as e:
            print("cv2 error:", e)
            continue
        except:
            print("Unexpected error:", sys.exc_info()[0])
            continue
    cv2.namedWindow('Faces')
    cv2.imshow('Face Detected!', img)
    cv2.imwrite('./face_{}.jpg'.format(algorithms[i]), img)
    # cv2.waitKey(0)
    return count


if __name__ == "__main__":
    saveFace('test.jpg')
#    showFace('test.jpg')

    model = memorizeFaces('./data')
    for i in range(1,4):
        recognizeFaces('test.jpg', i)
