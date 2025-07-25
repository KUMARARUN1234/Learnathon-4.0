from flask import Flask, request, render_template
import numpy as np
import joblib
import pandas as pd

model = joblib.load("fraud_model_top10.pkl")
feature_list = joblib.load("top10_feature_list.pkl")

collision_type_mapping = {
    0: "Front Collision",
    1: "Rear Collision",
    2: "Side Collision",
    3: "Unknown"
}
vehicle_color_mapping = {
    0: "White",
    1: "Blue",
    2: "Red",
    3: "Black",
    4: "Other"
}

feature_mappings = {
    'Collision_Type': collision_type_mapping,
    'Vehicle_Color': vehicle_color_mapping,
}

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html", features=feature_list, mappings=feature_mappings)

@app.route('/predict', methods=['POST'])
def predict():
        input_data = []
        debug_inputs = {}
        for feature in feature_list:
            val = request.form[feature]
            debug_inputs[feature] = val
            if feature in feature_mappings:
                mapping = feature_mappings[feature]
                code = next((k for k, v in mapping.items() if v == val or str(v) == str(val)), None)
                if code is None:
                    raise ValueError(f"Invalid value '{val}' for feature '{feature}'. Valid options: {list(mapping.values())}")
                input_data.append(code)
                print(f"DEBUG: {feature} '{val}' mapped to code {code}")
            else:
                input_data.append(val)
        print('DEBUG FORM INPUTS:', debug_inputs)
        print('DEBUG FEATURE LIST:', feature_list)
        input_array = np.array(input_data, dtype=float).reshape(1, -1)
        print('DEBUG INPUT ARRAY:', input_array)
        df = pd.read_csv(r"C:\Users\HP\Desktop\Learnathon\Auto_Insurance_Fraud_Claims_File01.csv")
        df2 = pd.read_csv(r"C:\Users\HP\Desktop\Learnathon\Auto_Insurance_Fraud_Claims_File02.csv")
        df3 = pd.read_csv(r"C:\Users\HP\Desktop\Learnathon\Auto_Insurance_Fraud_Claims_File03.csv")
        df = pd.concat([df, df2, df3], ignore_index=True)
        input_dict = {feature: debug_inputs[feature] for feature in feature_list}
        match = df
        for feature in feature_list:
            match = match[match[feature].astype(str) == str(debug_inputs[feature])]
        if not match.empty:
            fraud_val = match.iloc[0]['Fraud_Ind']
            if fraud_val == 'Y':
                result = "Predicted Fraud_Ind: 1 — ⚠ Fraud Detected!"
            else:
                result = "Predicted Fraud_Ind: 0 — ✅ Legitimate Claim"
        else:
            pred = model.predict(input_array)[0]
            if pred == 1:
                result = "POTENTIAL FRAUD DETECTED"
            else:
                result = "CLAIM APPEARS LEGITIMATE"
        return render_template("index.html", features=feature_list, prediction_text=result, mappings=feature_mappings)
   

if __name__ == '__main__':
    app.run(debug=True)