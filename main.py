import os
import urllib.request
from config import app
from flask import flash, request, Response, redirect, render_template, abort, render_template_string, send_from_directory
from PIL import Image
from io import StringIO
from werkzeug.utils import secure_filename
from app import iRecall

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_file():
	if request.method == 'POST':
		# check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			flash('No file selected for uploading')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			fname = request.form.get('fname')
			lname = request.form.get('lname')
			filename = secure_filename(file.filename)
			folder = os.path.join(app.config['UPLOAD_FOLDER'], '{}_{}'.format(lname, fname))
			if not os.path.exists(folder):
				os.mkdir(folder)
			file.save(os.path.join(folder, filename))
			flash('File(s) successfully uploaded to {}'.format(folder))
			root_folder = app.config['UPLOAD_FOLDER']
			dictionary = iRecall.load_dictionary(root_folder)
			maxConfidence = 10000
			# iRecall.process_image('app/test.jpg', dictionary, root_folder, maxConfidence)
			return redirect('/')

@app.route('/download')
def download_file():
	url = 'https://codeload.github.com/fogleman/Minecraft/zip/master'
	response = urllib.request.urlopen(url)
	data = response.read()
	response.close()

WIDTH = 1000
HEIGHT = 800

TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
	<title></title>
	<meta charset="utf-8" />
	<style>
body {
	margin: 0;
	background-color: #333;
}
.image {
	display: block;
	margin: 2em auto;
	background-color: #444;
	box-shadow: 0 0 10px rgba(0,0,0,0.3);
}
img {
	display: block;
}
	</style>
	<script src="https://code.jquery.com/jquery-1.10.2.min.js" charset="utf-8"></script>
	<script src="http://luis-almeida.github.io/unveil/jquery.unveil.min.js" charset="utf-8"></script>
	<script>
$(document).ready(function() {
	$('img').unveil(1000);
});
	</script>
</head>
<body>
	{% for image in images %}
		<a class="image" href="{{ image.src }}" style="width: {{ image.width }}px; height: {{ image.height }}px">
			<img src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" data-src="{{ image.src }}?w={{ image.width }}&amp;h={{ image.height }}" width="{{ image.width }}" height="{{ image.height }}" />
		</a>
	{% endfor %}
</body>
'''

@app.route('/<path:filename>')
def image(filename):
	try:
		w = int(request.args['w'])
		h = int(request.args['h'])
	except (KeyError, ValueError):
		return send_from_directory('.', filename)

	try:
		im = Image.open(filename)
		im.thumbnail((w, h), Image.ANTIALIAS)
		io = StringIO.StringIO()
		im.save(io, format='JPEG')
		return Response(io.getvalue(), mimetype='image/jpeg')

	except IOError:
		abort(404)

	return send_from_directory('.', filename)

@app.route('/list')
def index():
	images = []
	for root, dirs, files in os.walk('./app/data'):
		for filename in [os.path.join(root, name) for name in files]:
			if not filename.endswith('.jpg'):
				continue
			im = Image.open(filename)
			w, h = im.size
			aspect = 1.0*w/h
			if aspect > 1.0*WIDTH/HEIGHT:
				width = min(w, WIDTH)
				height = width/aspect
			else:
				height = min(h, HEIGHT)
				width = height*aspect
			images.append({
				'width': int(width),
				'height': int(height),
				'src': filename
			})

	return render_template_string(TEMPLATE, **{
		'images': images
	})


if __name__ == "__main__":
	app.run(
		host="0.0.0.0",
		port=int("8888"),
		debug=True
	)