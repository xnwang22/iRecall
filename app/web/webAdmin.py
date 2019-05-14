from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import socket
from PIL import Image
import os
from flask import send_file
import io

BASE_DIR = '../data'

app = Flask(__name__)

class PersonForm(Form):
    person_name = TextAreaField('',[validators.DataRequired()])

@app.route('/')
def index():
    form = PersonForm(request.form)
    return render_template('unknown.html', form=form)
#    return render_template('unknown.html')

def listdirs(folder):
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

#list folder names
@app.route('/list_names')
def dir_listing():
    # Show directory contents
    #files = os.listdir(BASE_DIR)
    files = listdirs(BASE_DIR)
    return render_template('folders.html', files=files)

#list face images in a folder names
@app.route('/list_images/<folder>')
def list_images(folder):

    # Joining the base and the requested path
    abs_path = os.path.join(BASE_DIR, folder)

    # Return 404 if path doesn't exist
    #if not os.path.exists(abs_path):
    #    return abort(404)

    # Check if path is a file and serve
    #if os.path.isfile(abs_path):
    #    return send_file(abs_path)
    folders = os.listdir(BASE_DIR)

    # Show directory contents
    files = os.listdir(abs_path)
    return render_template('faces.html', files=files, name=folder, folders=folders)

@app.route('/get_image/<name>/<file>')
def get_image(name, file):
    abs_path = os.path.join(BASE_DIR, name)
    abs_path = os.path.join(abs_path, file)
    im = Image.open(abs_path)
    return serve_pil_image(im)

def serve_pil_image(pil_img):
    img_io = io.BytesIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

@app.route('/new_person')
def new_person():
    form = PersonForm(request.form)
    return render_template('new_person.html', form=form)

@app.route('/new_person', methods=['POST'])
def new_person_post():
    form = PersonForm(request.form)
    if request.method == 'POST' and form.validate():
        name = request.form['person_name']
        person_folder = os.path.join(BASE_DIR, name)
        try:
            os.stat(person_folder)
        except:
            os.mkdir(person_folder)
        return dir_listing()
        #return render_template('ok.html', name=name)
    return render_template('unknown.html', form=form)


@app.route('/move_face/<folder_from>/<folder_to>/<file_name>')
def move_face(folder_from,folder_to, file_name):
    path_from = os.path.join(BASE_DIR, folder_from)
    path_from = os.path.join(path_from, file_name)
    path_to = os.path.join(BASE_DIR, folder_to)
    path_to = os.path.join(path_to, file_name)
    if (path_to != path_from):
        os.rename(path_from, path_to)
    return list_images(folder_from)

@app.route('/delete_face/<folder_from>/<file_name>')
def delete_face(folder_from, file_name):
    path_from = os.path.join(BASE_DIR, folder_from)
    path_from = os.path.join(path_from, file_name)
    os.remove(path_from)
    return list_images(folder_from)

if __name__ == '__main__':
    #create base folder if doesnt exist
    try:
        os.stat(BASE_DIR)
    except:
        os.mkdir(BASE_DIR)
    # name = socket.gethostbyname(socket.gethostname())
    print('open home page: http://' + socket.gethostname() + ' post:5000/list_names')
    app.run(debug=True, host='0.0.0.0')