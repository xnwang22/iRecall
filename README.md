# iRecall

test image recognition with opencv or tensor flow


Here are the steps how to use it:

1. Place image file (jpg) to the folder images
2. execute folder_scanner.py tool - it will try to find faces in each image file, create folder data and folder unknown. All the "faces" will be placed into folder unknown, processed image files will be moved to the folder bak, if file with this name already exists, original image file will be removed
3. execute app/web/webadmin.py tool. it will start web server on port 5000
4. Open URL <local server>:5000/list_names and create new names, open folder unknown and move all face images to the corresponding persons folder
5. execute train.py tool, it will create dictionary and model trained on the entered information images and pictures of faces
4. Execute folder_scanner.py again (make sure image or mages added to the folder images); it will recognize faces, print their names in console output and create images of captured faces place them into each person't folder or if not recognized it will create them in the folder unknown


