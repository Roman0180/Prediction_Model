from flask import Flask, request
import pickle
import numpy as np
import sys

app = Flask(__name__)
app.config["DEBUG"] = True

# file path to the saved model
model_filepath = 'prediction_model.sav'
model = pickle.load(open(model_filepath, 'rb'))


@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Get parameters for temperature
        prediction_attributes = np.array([], dtype='float')
        params = ["OP_CARRIER","OP_CARRIER_FL_NUM","ORIGIN","DEST","CRS_DEP_TIME","DEP_TIME","TAXI_OUT","WHEELS_OFF","WHEELS_ON","TAXI_IN","CRS_ARR_TIME","ARR_TIME","ARR_DELAY"]
        str_cols = ["OP_CARRIER", "ORIGIN", "DEST"]
        for param in params: 
            if param in str_cols: 
                str_val = request.args.get(param)
                # convert the string to float
                convert_to_float = lambda x: sum(ord(c) for c in x)
                prediction_attributes = np.append(prediction_attributes, convert_to_float(str_val))
            else:
                prediction_attributes = np.append(prediction_attributes, float(request.args.get((param))))
        features = [prediction_attributes]
        prediction = model.predict(features)
        output = round(prediction[0], 2)
        if output < 0: 
            return {'status' : 'early', 'pred_val': output}
        elif output < 2.9 and output > 0: 
            return {'status' : 'on time', 'pred_val': output}
        else:
            return {'status' : 'delayed', 'pred_val': output}
    except Exception as e:
        print(e)
        return 'Calculation Error', 500


app.run()