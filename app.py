# app.py

import os
import joblib
import gradio as gr

# ==========================================================
# Load the trained model
# ==========================================================
try:
    deployed_rf = joblib.load("loan_prediction_model.pkl")
except Exception as e:
    print(f"Warning: Model not found or error loading. {e}")
    deployed_rf = None

# ==========================================================
# Prediction Function with Error Handling
# ==========================================================
def predict_loan_status(
    no_of_dependents,
    education,
    self_employed,
    income_annum,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value,
):
    # --- CODE BLOCK: INPUT CAPTURE & VALIDATION ---
    values = [
        no_of_dependents, education, self_employed, income_annum, 
        loan_amount, loan_term, cibil_score, residential_assets_value, 
        commercial_assets_value, luxury_assets_value, bank_asset_value
    ]

    # 1. Empty input check
    if any(v is None or str(v).strip() == "" for v in values):
        return "❌ Please fill in all the input fields."

    # 2. Type casting
    try:
        no_of_dependents = int(no_of_dependents)
        education = int(education) # From Dropdown
        self_employed = int(self_employed) # From Dropdown
        income_annum = float(income_annum)
        loan_amount = float(loan_amount)
        loan_term = int(loan_term)
        cibil_score = int(cibil_score)
        residential_assets_value = float(residential_assets_value)
        commercial_assets_value = float(commercial_assets_value)
        luxury_assets_value = float(luxury_assets_value)
        bank_asset_value = float(bank_asset_value)
    except (ValueError, TypeError):
        return "❌ Please enter valid numeric values."

    # 3. Negative value check
    if any(v < 0 for v in values):
        return "❌ Negative values are not allowed for financial metrics."

    # 4. Specific Range Validations
    if not (300 <= cibil_score <= 900):
        return "❌ CIBIL score must be between 300 and 900."
    
    if no_of_dependents > 20:
        return "❌ Number of dependents seems unusually high (Max 20)."
    # ----------------------------------------------

    # --- CODE BLOCK: MODEL EXECUTION ---
    if deployed_rf is None:
        return "❌ Model failed to load. Please check your .pkl file."

    try:
        # Array strictly ordered to match the X dataframe provided
        input_data = [[
            no_of_dependents,
            education,
            self_employed,
            income_annum,
            loan_amount,
            loan_term,
            cibil_score,
            residential_assets_value,
            commercial_assets_value,
            luxury_assets_value,
            bank_asset_value
        ]]

        prediction = deployed_rf.predict(input_data)

        # Assuming 1 = Approved, 0 = Rejected based on standard loan datasets
        if prediction[0] == 1:
            return (
                "🟢 Prediction Result\n\n"
                "Loan Status: APPROVED\n\n"
                "The applicant meets the criteria for this loan."
            )
        else:
            return (
                "🔴 Prediction Result\n\n"
                "Loan Status: REJECTED\n\n"
                "The applicant does not meet the criteria."
            )

    except Exception as e:
        return f"❌ Prediction failed.\n\nError: {str(e)}"
    # -----------------------------------

# ==========================================================
# Description & Footer
# ==========================================================
# --- CODE BLOCK: UI TEXT CONFIGURATION ---
DESCRIPTION = """
<div style="text-align:center;padding:20px;border-radius:15px;
background:linear-gradient(135deg,#0f4c81,#1e88e5);
color:white;">

<h1>🏦 Loan Approval Prediction System</h1>

<h3>AI Powered Loan Eligibility Assessment</h3>

<p style="font-size:16px;">
Predict whether a loan application is likely to be
<b>Approved</b> or <b>Rejected</b> using a trained
<b>Random Forest Machine Learning Model</b>.
</p>

</div>
"""
developer_info = """
---

# 👨‍💻 Developer

### Rohit Kashyap

📧 **Email:** rohitkashyaprohit03456@gmail.com

💻 **Machine Learning Developer | Python | Data Science**

---

## 🚀 Technologies Used

- 🐍 Python
- 🤖 Scikit-Learn
- 🌲 Random Forest Classifier
- 🎨 Gradio
- ☁️ Render

---

⭐ Thank you for using this project.
"""
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="sky",
    neutral_hue="slate",
    radius_size="lg",
    text_size="lg",
)
custom_css = """
.gradio-container{
    max-width:1100px !important;
    margin:auto;
}

footer{
    visibility:hidden;
}

.gr-button{
    border-radius:12px !important;
    font-size:18px !important;
    font-weight:bold;
}

textarea{
    font-size:17px !important;
}

input{
    border-radius:10px !important;
}

"""
gr.Number(label="👨 Number of Dependents")

gr.Dropdown(
    choices=[("Graduate",1),("Not Graduate",0)],
    label="🎓 Education"
)

gr.Dropdown(
    choices=[("Yes",1),("No",0)],
    label="💼 Self Employed"
)

gr.Number(label="💰 Annual Income")

gr.Number(label="🏦 Loan Amount")

gr.Number(label="📅 Loan Term")

gr.Number(label="📈 CIBIL Score (300-900)")

gr.Number(label="🏠 Residential Assets")

gr.Number(label="🏢 Commercial Assets")

gr.Number(label="💎 Luxury Assets")

gr.Number(label="🏛 Bank Assets")
outputs=gr.Textbox(
label="📊 Prediction Result",
lines=8,
show_copy_button=True
)
interface = gr.Interface(
    fn=predict_loan_status,
    inputs=[...],
    outputs=...,
    title="🏦 Loan Approval Prediction System",
    description=DESCRIPTION,
    article=developer_info,
    theme=theme,
    css=custom_css,
)
return f"""
🟢 LOAN APPROVED

━━━━━━━━━━━━━━━━━━━━━━

✅ Status : APPROVED

🎉 Congratulations!

The applicant satisfies the eligibility criteria according to the trained Random Forest model.

━━━━━━━━━━━━━━━━━━━━━━
"""
examples=[
    [2,1,0,850000,250000,12,780,350000,120000,90000,150000],
    [4,0,1,250000,600000,36,480,90000,20000,10000,25000],
]
submit_btn="🔍 Predict Loan Status",
clear_btn="🗑 Reset"

# --------------------------------------------------------

# ==========================================================
# Launch
# ==========================================================
if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
