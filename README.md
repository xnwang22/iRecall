# iRecall
test image recognition with tensor flow


Here are the steps how to test it:

1. execute test.py - it will find faces in test.jpg file, create folder data and folder unknown. All the "faces" will be placed into folder unknown
2. create folder for each person (/data/John, /data/Tom, etc) review captured faces and move them from folder unknown to the corresponding folder
3. execute train.py, it will create dictionary and model in the data folder
4. Execute train.py again; it will recognize faces, print their names in console output and will add dictionary and place them into each person't folder


to open web administration tool start app/web/webadmin.py and use following URL:
localhost:5000/list_names
