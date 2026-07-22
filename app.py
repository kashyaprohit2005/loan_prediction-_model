import os
import joblib
import gradio as gr

# Load Model
try:
    model = joblib.load("loan_prediction_model.pkl")
except:
    model = None


def predict_loan_status(dep, edu, self_emp, income, loan, term, cibil, res, com, lux, bank):
    values = [dep, edu, self_emp, income, loan, term, cibil, res, com, lux, bank]

    if any(v is None or str(v).strip() == "" for v in values):
        return "❌ Please fill all fields."

    try:
        dep = int(dep)
        edu = int(edu)
        self_emp = int(self_emp)
        income = float(income)
        loan = float(loan)
        term = int(term)
        cibil = int(cibil)
        res = float(res)
        com = float(com)
        lux = float(lux)
        bank = float(bank)
    except:
        return "❌ Invalid input."

    if any(v < 0 for v in [dep, income, loan, term, cibil, res, com, lux, bank]):
        return "❌ Negative values are not allowed."

    if not (300 <= cibil <= 900):
        return "❌ CIBIL Score must be between 300-900."

    if model is None:
        return "❌ Model not loaded."

    pred = model.predict([[dep, edu, self_emp, income, loan, term, cibil, res, com, lux, bank]])

    if pred[0] == 1:
        return """🟢 LOAN APPROVED

✅ Congratulations!

The applicant satisfies the loan eligibility criteria."""
    else:
        return """🔴 LOAN REJECTED

❌ Sorry!

The applicant does not satisfy the loan eligibility criteria."""


description = """
<div style="text-align:center;padding:20px;border-radius:15px;
background:linear-gradient(135deg,#0f4c81,#1e88e5);color:white;">
<h1>🏦 Loan Approval Prediction</h1>
<p>AI Powered Loan Eligibility Assessment using Random Forest</p>
</div>
"""

developer = """
## 👨‍💻 Developer

**Rohit Kashyap**

📧 rohitkashyaprohit03456@gmail.com

### 🚀 Technologies
- Python
- Scikit-Learn
- Gradio
- Random Forest
- Render
"""

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="sky",
    radius_size="lg"
)

css = """
.gradio-container{max-width:1050px!important;margin:auto;}
footer{visibility:hidden;}
"""

interface = gr.Interface(
    fn=predict_loan_status,
    inputs=[
        gr.Number(label="👨 Dependents"),
        gr.Dropdown([("Graduate",1),("Not Graduate",0)],label="🎓 Education"),
        gr.Dropdown([("Yes",1),("No",0)],label="💼 Self Employed"),
        gr.Number(label="💰 Annual Income"),
        gr.Number(label="🏦 Loan Amount"),
        gr.Number(label="📅 Loan Term"),
        gr.Number(label="📈 CIBIL Score"),
        gr.Number(label="🏠 Residential Assets"),
        gr.Number(label="🏢 Commercial Assets"),
        gr.Number(label="💎 Luxury Assets"),
        gr.Number(label="🏛 Bank Assets"),
    ],
    outputs=gr.Textbox(label="📊 Prediction", lines=7, show_copy_button=True),
    title="🏦 Loan Approval Prediction System",
    description=description,
    article=developer,
    theme=theme,
    css=css,
    submit_btn="🔍 Predict",
    clear_btn="🗑 Reset",
    examples=[
        [2,1,0,850000,250000,12,780,350000,120000,90000,150000],
        [4,0,1,250000,600000,36,480,90000,20000,10000,25000]
    ]
)

if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT",7860))
    )
