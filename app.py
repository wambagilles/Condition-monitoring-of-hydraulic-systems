from sensor_dataset import compute_apriori_vars
import pandas as pd 
from utils import downscale_sample_rate, NumpyEncoder
import numpy as np
from model import AutoLagNet
import torch
from flask import Flask, request, jsonify, render_template, url_for, flash, redirect
import json
from make_inference import make_inference


# Create a Flask application
app = Flask('Condition_Prediction')
app.config['SECRET_KEY'] = '1ca6614d8c88d67a9aab6fe9196f742ea4c1d583b1ca9f07'


predictions = []

@app.route('/')
def index():
    return render_template('index.html', predictions=predictions)


# Define an endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    device = 'cpu'
    n_var_in = 2 
    n_neurons_out = 2
    n_var_a_priori = 8
    MODEL_PATH = "checkpoints/model_20241007_170626_13"

    render_template('make_prediction.html')

    # Get cycle ID data from the request in JSON format
    cycle_id = request.get_json()["cycle_id"]

    pressure = pd.read_csv("data_subset/ps2.txt", header=None, sep='\t')
    pressure = pd.DataFrame(pressure.apply(downscale_sample_rate, axis=1).tolist())
    volume_flow = pd.read_csv("data_subset/fs1.txt", header=None, sep='\t')

    pressure = pressure.iloc[cycle_id].to_numpy()
    volume_flow = volume_flow.iloc[cycle_id].to_numpy()
    encoder_input = np.array([volume_flow, pressure])
    a_priori_vars = np.array([compute_apriori_vars(volume_flow, pressure)])

    model = AutoLagNet( n_var_in, n_neurons_out, n_var_a_priori)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    model.to(device)
    

    with torch.no_grad():    
        pred = model(torch.from_numpy(encoder_input).unsqueeze(0).float(), torch.from_numpy(a_priori_vars).unsqueeze(0).float())
        # Prepare the result in JSON format
        result = {
            'condition_probability': pred.detach().to('cpu').numpy()[0],
            'condition': "Optimal" if np.argmax(pred) == 1 else "Non Optimal"
        }

        dumped = json.dumps(result, cls=NumpyEncoder)

       
        print(result)
        return dumped


@app.route('/make_prediction/', methods=('GET', 'POST'))
def make_prediction():
    if request.method == 'POST':
        cycle_id = request.form['cycle_id']
        pred = make_inference(cycle_id)

        if not cycle_id:
            flash('Cycle ID is required!')
        else:
            predictions.append({'cycle_id': cycle_id, "prediction": pred})
            
            return redirect(url_for('index'))
            

    return render_template('make_prediction.html')        

    
if __name__ == "__main__":
 app.run(debug=True, host='0.0.0.0', port=9696)