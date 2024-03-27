from flask import Flask, render_template, request
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the trained model
nb_classifier, tfidf_vectorizer = joblib.load('phishing_detection_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    subject_line = request.form['subject']
    
    # Vectorize the user input
    user_input_tfidf = tfidf_vectorizer.transform([subject_line])

    # Predict whether the input is phishing or not
    prediction = nb_classifier.predict(user_input_tfidf)[0]

    if prediction == 1:
        result = "The entered email subject line is likely to be a phishing email."
    else:
        result = "The entered email subject line is likely to be a legitimate email."

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
