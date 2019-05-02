import iRecall
import os

if __name__ == "__main__":
    root_folder = './data/'
    try:
        os.stat(root_folder)
    except:
        os.mkdir(root_folder)
    iRecall.train_model(root_folder)

