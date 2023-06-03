import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('food_pred_models.pkl', 'rb'))

@app.route('/')
def home():
    ordered =''
    return render_template('index.html', output_prediction=ordered)


@app.route('/predict', methods=['POST','GET'])
def predict():
    favorite_bahan = float(request.form['favorite_bahan'])
    gender = float(request.form['gender'])
    umur = float(request.form['umur'])
    alamat = float(request.form['alamat'])
    rasa = float(request.form['rasa'])
    occupation = float(request.form['occupation'])
    fave_drink = float(request.form['fave_drink'])
    menu_ordered = float(request.form['menu_ordered'])
    result = model.predict([[favorite_bahan, gender, umur, alamat, rasa, occupation, fave_drink, menu_ordered]])[0]
    return render_template('index.html', result=result)



if __name__ == "__main__":
    app.run(debug=True)