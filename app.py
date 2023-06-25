from flask import Flask,render_template,request
import pickle
import numpy as np

model = pickle.load(open('model1.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_price():
    bedrooms = float(request.form.get('bedrooms'))
    bathrooms = float(request.form.get('bathrooms'))
    sqft_living = int(request.form.get('sqft_living'))
    sqft_lot = int(request.form.get('sqft_lot'))
    floors = float(request.form.get('floors'))
    waterfront = int(request.form.get('waterfront'))
    view = int(request.form.get('view'))
    condition = int(request.form.get('condition'))
    sqft_above = int(request.form.get('sqft_above'))
    sqft_basement = int(request.form.get('sqft_basement'))
    yr_built = int(request.form.get('yr_built'))
    yr_renovated = int(request.form.get('yr_renovated'))
    street = (request.form.get('street'))
    city = (request.form.get('city'))
    statezip = (request.form.get('statezip'))
    age_of_house = int(request.form.get('age_of_house'))
    total_sqft = int(request.form.get('total_sqft'))
    cost_persqft = float(request.form.get('cost_persqft'))
    result = model.predict(np.array([bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, sqft_above, sqft_basement,yr_built, yr_renovated, street, city, statezip, age_of_house, total_sqft, cost_persqft]).reshape(1, 3))
    return result



if __name__ == '__main__':
    app.run(debug=True)
