from flask import Flask, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("loan_model.pkl", "rb"))

@app.route("/")
def home():
    return '''
    <h2>Loan Approval Prediction System</h2>
    <form action="/predict" method="post">
        Gender (Male=1, Female=0): <input type="text" name="Gender"><br><br>
        Married (Yes=1, No=0): <input type="text" name="Married"><br><br>
        Dependents (0/1/2/3): <input type="text" name="Dependents"><br><br>
        Education (Graduate=1, Not Graduate=0): <input type="text" name="Education"><br><br>
        Self Employed (Yes=1, No=0): <input type="text" name="Self_Employed"><br><br>
        Applicant Income: <input type="text" name="ApplicantIncome"><br><br>
        Coapplicant Income: <input type="text" name="CoapplicantIncome"><br><br>
        Loan Amount: <input type="text" name="LoanAmount"><br><br>
        Loan Amount Term: <input type="text" name="Loan_Amount_Term"><br><br>
        Credit History (1/0): <input type="text" name="Credit_History"><br><br>
        Property Area (Urban=2, Semiurban=1, Rural=0): <input type="text" name="Property_Area"><br><br>
        <input type="submit" value="Predict">
    </form>
    '''

@app.route("/predict", methods=["POST"])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    result = "Loan Approved" if prediction[0] == 1 else "Loan Rejected"
    return f"<h2>Prediction Result: {result}</h2>"

if __name__ == "__main__":
    app.run(debug=True)
