
import tkinter as tk
from tkinter import messagebox, ttk, Scrollbar, Canvas
import joblib
import numpy as np
from database import init_db, save_result

# Load model and init DB
model = joblib.load('breast_cancer_model.pkl')
init_db()

# All 30 feature names
features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Main GUI
app = tk.Tk()
app.title("Breast Cancer Detection (30-Feature)")
app.geometry("500x800")


canvas = Canvas(app,  bg="#1e3d59")
scroll_y = Scrollbar(app, orient="vertical", command=canvas.yview)

frame = tk.Frame(canvas, bg="#1e3d59")
canvas.create_window((0, 0), window=frame, anchor="nw")
canvas.configure(yscrollcommand=scroll_y.set)

tk.Label(frame, text="Patient Name",font=("Arial", 12), fg="#ffff66", bg="#1e3d59").pack()
entry_name = tk.Entry(frame)
entry_name.pack(pady=5)

entries = {}
for feat in features:
    tk.Label(frame, text=feat, font=("Arial", 12), 
                   fg="#ffff66", bg="#1e3d59").pack()
    entry = tk.Entry(frame)
    entry.pack()
    entries[feat] = entry

def predict():
    try:
        name = entry_name.get()
        values = [float(entries[f].get()) for f in features]
        result = "Malignant" if model.predict([values])[0] == 1 else "Benign"
        lbl_result.config(text=f"Result: {result}")
        save_result(name, result)
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {e}")

tk.Button(frame, text="Predict", command=predict).pack(pady=10)
lbl_result = tk.Label(frame, text="Result: ", font=("Arial", 14))
lbl_result.pack(pady=10)

frame.update_idletasks()
canvas.configure(scrollregion=canvas.bbox("all"))
canvas.pack(fill="both", expand=True, side="left")
scroll_y.pack(fill="y", side="right")

app.mainloop()