# Import Dependencies
from flask import Flask, render_template, request, redirect, flash
import os
from werkzeug.utils import secure_filename
from main import getPrediction

#################################################
# Flask Setup
#################################################

# Use relative path to "static" folder inside the project
UPLOAD_FOLDER = os.path.join(os.getcwd(), "static")

app = Flask(__name__)                    
app.secret_key = '8662747133'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route to HTML    
@app.route('/')
def index():
    return render_template('index.html')

@app.route("/", methods=['POST']) 
def submit_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    # Error message if no file submitted
    if file.filename == '':
        flash('No file selected for uploading')
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Save file into static/ folder
        file.save(filepath)

        # Run prediction
        answer, probability_results, filename = getPrediction(filename)

        # Flash results to frontend
        flash(answer)
        flash(probability_results)  # accuracy
        flash(filename)

        return redirect('/')

if __name__ == "__main__":
    app.run(debug=True)
