import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
model2 = pickle.load(open('model2.pkl', 'rb'))
model3 = pickle.load(open('model3.pkl', 'rb'))

@app.route('/')
def welcome():
    return render_template('welcome.html')
@app.route('/home')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    fert=['Agroblen','DAP','GROMOR(28-28)','NPK(14-35-14)','NPK(20-20)']
    fr="Urea"
    count=0
    c=""
    for i in range(0,5):
        if(prediction[0][i]==1):
            c=fert[i]
            count=count+1
            break
        i=i+1
            
    
    
    if count==1:
        
        crop=c
    else:
        crop=fr

    #output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Fertilizer is {}'.format(crop))

@app.route('/home2')
def home2():
    return render_template('index2.html')

@app.route('/predict_crop',methods=['POST'])
def predict_crop():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    prediction2 = model2.predict_proba(final_features)
    
    probabilities = np.ravel(prediction2)
    classes = model2.classes_
    d = {k:v for k,v in zip(classes,probabilities)}
    
    from operator import itemgetter
    sort = sorted(d.items(), key=itemgetter(1),reverse = True)
    crop = []
    prob = []
    for i,v in sort:
        crop.append(i.lower())
        prob.append(v)
    final_d = {k:v for k,v in zip(crop,prob)}
    return render_template('index2.html', prediction_text='Crop that is best to grow is {}'.format(crop[0]),prediction_text2='Second best crop to grow is {}'.format(crop[1]),prediction_text3='3rd best crop to grow is {}'.format(crop[2]))

@app.route('/home3')
def home3():
    return render_template('index3.html')

@app.route('/predict_price',methods=['POST'])
def predict_price():
    inp=request.values.get("Date") 
    
    # Get forecast 70 steps ahead in future
    pred_uc = model3.get_forecast(steps=70)

# Get confidence intervals of forecasts
    pred_ci = pred_uc.conf_int()
    
    modified = pred_ci.reset_index()
    modified.rename(columns = {'index':'Date'}, inplace = True)
    
    l=modified.loc[modified['Date'] == inp]
    
    low=abs(l['lower Modal_Price'].values[0])
    upp=abs(l['upper Modal_Price'].values[0])
    
    
    return render_template('index3.html', prediction_text='LOWER MODAL PRICE is {}'.format(low),prediction_text2="UPPER MODAL PRICE is {}".format(upp))
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/index11')
def index11():
    return render_template('index11.html')

@app.route('/index21')
def index21():
    return render_template('index21.html')

@app.route('/index31')
def index31():
    return render_template('index31.html')


if __name__ == "__main__":
    
    app.run(debug=True)