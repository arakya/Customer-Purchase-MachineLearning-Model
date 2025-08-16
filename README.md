# Customer Purchase Status — Stacking Ensemble (Streamlit App)

Predicts whether a customer will make a purchase using a **stacking ensemble** (Random Forest, SVM, Logistic Regression, Naive Bayes).  
Ships with a **Streamlit app** for live predictions and a clean, reproducible setup.

---

## ▶️ Live Demo
Try it here: [Streamlit App](https://customer-purchase-machinelearning-model-5uckm63eaodt6si2ymgb66.streamlit.app/)

---

## 📸 Demo Screenshot
![Streamlit Demo](screenshot.png)

---

## What’s inside
- End-to-end workflow from training (notebook) to deployable artifacts.
- `app.py` — Streamlit UI for inference.
- `predict.py` — programmatic inference helper.
- `stacked_model.pkl` + `scaler.pkl` — trained artifacts.
- Reproducible environment via `requirements.txt`.

---

## Repository structure
```
CustomerPurchaseStatus/
├── app.py
├── predict.py
├── stacked_model.pkl
├── scaler.pkl
├── CustomerPurchase.ipynb
├── CustomerPurchase.html
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1) Create & activate a virtual environment (recommended)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Launch the Streamlit app locally
```bash
streamlit run app.py
```
A browser window will open. Enter feature values (e.g., Age, Income, Purchases, Time on Site) and you’ll see:
- **Predicted class**: 0 = no purchase, 1 = purchase  
- **Probability** (confidence score)

---

## Programmatic inference (no UI)

```python
# predict_example.py
import pickle, numpy as np

with open("stacked_model.pkl","rb") as f: model = pickle.load(f)
with open("scaler.pkl","rb") as f: scaler = pickle.load(f)

# replace with your feature vector in the SAME order used during training
x_raw = np.array([
    # Age, AnnualIncome, NumberOfPurchases, TimeSpentOnWebsite, ...
])

x_scaled = scaler.transform([x_raw])
proba = model.predict_proba(x_scaled)[0,1]
pred  = int(proba >= 0.5)

print(f"Probability of Purchase: {proba:.3f}")
print(f"Predicted Class: {pred}")
```

Run it:
```bash
python predict_example.py
```

---

## Regenerating `scaler.pkl` (if you retrain)
```python
import pickle
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)   # X_train must match feature order
with open("scaler.pkl","wb") as f:
    pickle.dump(scaler, f)
```

⚠️ Important: if you add/remove/reorder features, retrain the model and refit the scaler before inference.

---

## Notes
- Metrics reported in the notebook for the stacked model: Accuracy 95%, Precision 97%, Recall 91%, F1 94%.  
- `CustomerPurchase.html` is an export of the notebook for quick viewing without Jupyter.

---

## License
Educational/portfolio use. Add a specific license if you plan to distribute.
