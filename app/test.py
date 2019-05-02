import iRecall
import os

if __name__ == "__main__":
    root_folder = './data/'
    try:
        os.stat(root_folder)
    except:
        os.mkdir(root_folder)

    dictionary = iRecall.load_dictionary(root_folder)
    maxConfidence = 10000
    iRecall.process_image('test.jpg', dictionary, root_folder, maxConfidence)
