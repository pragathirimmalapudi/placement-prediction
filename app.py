from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the Random Forest model
clff = joblib.load('random_forest_model.joblib')

# Mapping for the 'stream' variable
stream_mapping = {"CSE": 0, "ECE": 1, "IT": 2, "MECH": 3, "Civil": 4, "EC": 5}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get user input
            age = float(request.form['age'])
            gender = request.form['gender']
            internships = float(request.form['internships'])
            cgpa = float(request.form['cgpa'])
            hostel = int(request.form['hostel'])
            backlogs = int(request.form['backlogs'])
            stream = request.form['stream']

            # Convert categorical variables to numerical format
            stream = stream_mapping.get(stream, -1)  # Use -1 if the stream is not found in mapping

            if stream == -1:
                return render_template('index.html', error_message='Invalid stream selected. Please try again.')

            # Encode gender as a numerical value
            gender_mapping = {"Male": 0, "Female": 1}
            gender = gender_mapping.get(gender, -1)

            if gender == -1:
                return render_template('index.html', error_message='Invalid gender selected. Please try again.')

            # Make prediction using the Random Forest model
            user_data = [[age, gender, internships, cgpa, hostel, backlogs, stream]]  # adjust as needed
            prediction_rf = clff.predict(user_data)

            # Map numerical prediction to "Yes" or "No"
            prediction_text = "Yes" if prediction_rf[0] == 1 else "No"

            return render_template('result.html', prediction_rf=prediction_text)

        except Exception as e:
            print(e)
            return render_template('index.html', error_message='Error processing the request. Please try again.')

    # Handle other cases or render the form again if needed
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
