import iRecall
import os

def checkDir(dir):
    try:
        os.stat(dir)
    except:
        os.mkdir(dir)

# reads all images in the folder image_folder,
# tries to find and recognise  faces
# move processed images to the bak_folder
if __name__ == "__main__":
    image_folder = './images/'
    bak_folder = './bak/'
    root_folder = './data/'
    checkDir(root_folder)
    checkDir(image_folder)
    checkDir(bak_folder)

    dictionary = iRecall.load_dictionary(root_folder)
    # ok confidence value so far ...
    maxConfidence = 7500
    # for each image run process image function
    # iRecall.process_image('test.jpg', dictionary, root_folder, maxConfidence)

    count=0
    for dirpath, subdirs, files in os.walk(image_folder):
        for x in files:
            if x.lower().endswith(".jpg") or x.lower().endswith(".png"):
                iRecall.process_image(os.path.join(dirpath, x), dictionary, root_folder, maxConfidence)

                path_from = os.path.join(dirpath, x)

                path_to = os.path.join(bak_folder, x)
                try:
                    os.stat(path_to)
                    os.remove(path_from)
                except:
                    os.rename(path_from, path_to)


                count = count + 1

    print('processed files:', count)
