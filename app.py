import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)

app.config['SECRET_KEY'] = 'mysecretkey'

model_reg=pickle.load(open('model_reg.pkl','rb'))
model_clf=pickle.load(open('model_clf.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_api_reg')
def predict_api_reg():
    return render_template('regression.html')

@app.route('/predict_reg',methods=['GET','POST'])
def predict_reg():

    data=[float(x) for x in request.form.values()]
    final_features = [np.array(data)]
    print(data)
    
    output=model_reg.predict(final_features)[0]
    print(output)
    return render_template('regression.html', prediction_text="Temperature is  {}".format(output))

@app.route('/predict_api_clf')
def predict_api_clf():
    return render_template('classification.html')

@app.route('/predict_clf',methods=['GET','POST'])
def predict_clf():
    
    data=[float(x) for x in request.form.values()]
    final_features = [np.array(data)]
    print(data)
    
    output=model_clf.predict(final_features)[0]
    print(output)
    return render_template('classification.html', prediction_text="Hence there will  {}".format(output))

if __name__=="__main__":
    app.run(debug=True)

