import os
from flask import Flask, request
from werkzeug.utils import secure_filename
from flask_cors import CORS
from services.model_creator import ModelCreator

from preprocessing.data_preprocessor import DataPreprocessor

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploaded_files')
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
CORS(app)

app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
data_preprocessor = DataPreprocessor(UPLOAD_FOLDER)
model_creator = ModelCreator()

@app.route("/")
def main():
    models = data_preprocessor.get_all_models()
    return {"data": models}, 200

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/populate', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'filesLength' not in request.form:
            return {"data":"No file part"}, 400

        filesLength = int(request.form['filesLength'])
        if filesLength <= 0:
            return {"data":"No selected files"}, 400

        for i in range(filesLength):
            file = request.files['file[' + str(i) + ']']
            if file.filename == '':
                return {"data":"Empty file uploaded"}, 400
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        data_preprocessor.process_data()
        return {"data": "OK"}, 200

@app.route('/api/train', methods=['POST'])
def model_training():
    if request.method == 'POST':
        dateFrom = request.form['dateFrom'].split('-')
        dateTo = request.form['dateTo'].split('-')

        if dateFrom[0] == dateTo[0] and dateFrom[1] == dateTo[1] and dateFrom[2] == dateTo[2]:
            model_creator.start_model_training(0,0,0,0,0,0)
        else:
            model_creator.start_model_training(int(dateFrom[0]),int(dateFrom[1]),int(dateFrom[2]),
                                               int(dateTo[0]),int(dateTo[1]),int(dateTo[2]))
        return {"data": "OK"}, 200

@app.route('/api/test', methods=['POST'])
def test():
    if request.method == 'POST':
        model_creator.set_path('{}'.format(request.form['model']))
        pred_days = request.form['days']
        pred_date = request.form['date'].split('-')

        return model_creator.predict(int(pred_days), int(pred_date[0]), int(pred_date[1]), int(pred_date[2]), request.form['model'])

@app.route('/api/csv', methods=['GET'])
def csv():
    if request.method == 'GET':
        model_creator.get_csv()
        return {"data": "OK"}, 200
    
if __name__ == "__main__":
    app.run(debug = True)