
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('weather.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    prediction=model.predict(final_features)
    output=prediction[0]

    if output>(0.5):
        return render_template('two.html',prediction_text='Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output))
    else:
        return render_template('weather.html',prediction_text='Your Forest is safe.\n Probability of fire occuring is {}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)

if __name__ == "__main__":
    app.run(debug=True)