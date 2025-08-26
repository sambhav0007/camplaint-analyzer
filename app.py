from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# --- Load All Models ---
MODELS_DIR = 'models'
try:
    category_model = joblib.load(os.path.join(MODELS_DIR, 'category_model.pkl'))
    priority_model = joblib.load(os.path.join(MODELS_DIR, 'priority_model.pkl'))
    type_model = joblib.load(os.path.join(MODELS_DIR, 'type_model.pkl'))
    department_model = joblib.load(os.path.join(MODELS_DIR, 'department_model.pkl')) # Naya model load karein
    print("All 4 models loaded successfully.")
except FileNotFoundError:
    print("Error: Model files not found. Please run train.py first.")
    exit()

# Rule-based department_map ki ab zaroorat nahi hai, humne use hata diya hai.

@app.route('/analyze', methods=['POST'])
def analyze_complaint():
    try:
        data = request.get_json(force=True)
        complaint_text = data.get('complaint', '')

        if not complaint_text:
            return jsonify({'error': 'Complaint text cannot be empty'}), 400

        # --- Predictions from all models ---
        predicted_category = category_model.predict([complaint_text])[0]
        predicted_priority = priority_model.predict([complaint_text])[0]
        predicted_type = type_model.predict([complaint_text])[0]
        
        # Naye model se department predict karein (YAHAN CODE UPDATE HUA HAI)
        assigned_department = department_model.predict([complaint_text])[0]
        
        # Confidence score (example ke liye abhi sirf category ka use kar rahe hain)
        category_probas = category_model.predict_proba([complaint_text])
        confidence = round(category_probas.max() * 100, 2)

        # Frontend ko bhejne ke liye response taiyaar karein
        response = {
            'complaintText': complaint_text,
            'category': predicted_category,
            'priority': predicted_priority,
            'type': predicted_type,
            'assignedDepartment': assigned_department, # Model se aayi hui prediction
            'aiConfidence': confidence
        }
        
        return jsonify(response)

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': 'An error occurred during analysis.'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
