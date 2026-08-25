import tkinter as tk
from tkinter import messagebox, ttk, Scrollbar, Canvas
import joblib
import numpy as np
from database import init_db, save_result

# ----------------------------------------------------------------------
# Load model and init DB
# ----------------------------------------------------------------------
model = joblib.load('breast_cancer_model.pkl')
init_db()

# ----------------------------------------------------------------------
# Pink theme palette
# ----------------------------------------------------------------------
COLOR_BG          = "#fff0f5"   # lavender blush (main background)
COLOR_PANEL       = "#ffe1ec"   # soft pink panel
COLOR_HEADER      = "#d6006d"   # deep pink header
COLOR_ACCENT      = "#ff4da6"   # hot pink accent
COLOR_ACCENT_DARK = "#c2185b"   # pressed / hover state
COLOR_TEXT        = "#7a0245"   # deep magenta text
COLOR_LABEL       = "#ad1457"   # section labels
COLOR_ENTRY_BG    = "#ffffff"
COLOR_BOT_BUBBLE  = "#ffd1e6"   # AI assistant chat bubble
COLOR_BENIGN      = "#2e7d32"   # keep green for "good news"
COLOR_MALIGNANT   = "#c62828"   # keep red for "warning"

# All 30 feature names, grouped for a nicer layout
FEATURE_GROUPS = {
    "Mean Values": [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
        'fractal_dimension_mean'
    ],
    "Standard Error Values": [
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
        'fractal_dimension_se'
    ],
    "Worst Values": [
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst',
        'fractal_dimension_worst'
    ],
}
features = [f for group in FEATURE_GROUPS.values() for f in group]

# ----------------------------------------------------------------------
# Main window
# ----------------------------------------------------------------------
app = tk.Tk()
app.title("Breast Cancer AI Assistant")
app.geometry("1400x900")
app.configure(bg=COLOR_BG)

# ---- Header -------------------------------------------------------
header = tk.Frame(app, bg=COLOR_HEADER, height=70)
header.pack(fill="x", side="top")
header.pack_propagate(False)

tk.Label(
    header, text="🎀  Breast Cancer AI Assistant",
    font=("Segoe UI", 16, "bold"), fg="white", bg=COLOR_HEADER
).pack(pady=6)
tk.Label(
    header, text="30-Feature Diagnostic Model  •  95.61% Accuracy",
    font=("Segoe UI", 9), fg="#ffe1ec", bg=COLOR_HEADER
).pack()

# ---- AI Assistant chat bubble (feedback area) ----------------------
assistant_frame = tk.Frame(app, bg=COLOR_BG)
assistant_frame.pack(fill="x", padx=15, pady=(12, 4))

tk.Label(
    assistant_frame, text="🤖 AI Assistant", font=("Segoe UI", 10, "bold"),
    fg=COLOR_HEADER, bg=COLOR_BG, anchor="w"
).pack(fill="x")

bubble = tk.Frame(assistant_frame, bg=COLOR_BOT_BUBBLE, bd=0)
bubble.pack(fill="x", pady=(4, 0))

lbl_assistant_msg = tk.Label(
    bubble,
    text="Hi! Enter the patient's cell nuclei measurements below and press "
         "\"Run Prediction\" — I'll analyze the values and let you know what "
         "the model finds.",
    font=("Segoe UI", 10), fg=COLOR_TEXT, bg=COLOR_BOT_BUBBLE,
    wraplength=480, justify="left", padx=12, pady=10
)
lbl_assistant_msg.pack(fill="x")

# ---- Scrollable form area ------------------------------------------
canvas_frame = tk.Frame(app, bg=COLOR_BG)
canvas_frame.pack(fill="both", expand=True, padx=10, pady=5)

canvas = Canvas(canvas_frame, bg=COLOR_BG, highlightthickness=0)
scroll_y = Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
frame = tk.Frame(canvas, bg=COLOR_BG)

frame_window = canvas.create_window((0, 0), window=frame, anchor="n")

def _on_frame_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

def _on_canvas_configure(event):
    # Keep the inner frame centered and matched to the canvas width
    canvas.itemconfig(frame_window, width=event.width)

frame.bind("<Configure>", _on_frame_configure)
canvas.bind("<Configure>", _on_canvas_configure)
canvas.configure(yscrollcommand=scroll_y.set)

def _on_mousewheel(event):
    canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

canvas.bind_all("<MouseWheel>", _on_mousewheel)

# Three even columns for the feature groups, centered as a whole
frame.grid_columnconfigure(0, weight=1, uniform="col")
frame.grid_columnconfigure(1, weight=1, uniform="col")
frame.grid_columnconfigure(2, weight=1, uniform="col")

# ---- Patient name (centered, spans all columns) ----------------------
name_outer = tk.Frame(frame, bg=COLOR_BG)
name_outer.grid(row=0, column=0, columnspan=3, sticky="ew", padx=8, pady=(6, 10))
name_outer.grid_columnconfigure(0, weight=1)

name_card = tk.Frame(name_outer, bg=COLOR_PANEL, padx=16, pady=12)
name_card.grid(row=0, column=0)

tk.Label(
    name_card, text="Patient Name", font=("Segoe UI", 11, "bold"),
    fg=COLOR_HEADER, bg=COLOR_PANEL
).pack(anchor="w")
entry_name = tk.Entry(
    name_card, font=("Segoe UI", 10), bg=COLOR_ENTRY_BG,
    fg=COLOR_TEXT, relief="flat", highlightthickness=1,
    highlightbackground=COLOR_ACCENT, highlightcolor=COLOR_ACCENT_DARK,
    width=40
)
entry_name.pack(fill="x", ipady=4, pady=(4, 0))

# ---- Feature entry fields, grouped in pink cards, side by side -------
entries = {}
for col, (group_name, group_features) in enumerate(FEATURE_GROUPS.items()):
    card = tk.Frame(frame, bg=COLOR_PANEL, padx=12, pady=10)
    card.grid(row=1, column=col, sticky="new", padx=8, pady=6)

    tk.Label(
        card, text=group_name, font=("Segoe UI", 11, "bold"),
        fg=COLOR_HEADER, bg=COLOR_PANEL
    ).pack(anchor="w", pady=(0, 6))

    for feat in group_features:
        row = tk.Frame(card, bg=COLOR_PANEL)
        row.pack(fill="x", pady=2)

        tk.Label(
            row, text=feat, font=("Segoe UI", 9), fg=COLOR_LABEL,
            bg=COLOR_PANEL, width=18, anchor="w"
        ).pack(side="left")

        entry = tk.Entry(
            row, font=("Segoe UI", 9), bg=COLOR_ENTRY_BG, fg=COLOR_TEXT,
            relief="flat", highlightthickness=1,
            highlightbackground="#ffb6d5", highlightcolor=COLOR_ACCENT_DARK
        )
        entry.pack(side="left", fill="x", expand=True, ipady=3)
        entries[feat] = entry

# ----------------------------------------------------------------------
# Prediction logic
# ----------------------------------------------------------------------
def predict():
    try:
        name = entry_name.get().strip() or "Unnamed patient"
        values = [float(entries[f].get()) for f in features]

        prediction = model.predict([values])[0]
        result = "Malignant" if prediction == 1 else "Benign"

        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([values])[0]
            confidence = float(proba[int(prediction)]) * 100

        # Update result label
        color = COLOR_MALIGNANT if result == "Malignant" else COLOR_BENIGN
        conf_text = f" ({confidence:.1f}% confidence)" if confidence is not None else ""
        lbl_result.config(text=f"Result: {result}{conf_text}", fg=color)

        # Conversational AI assistant message
        if result == "Benign":
            msg = (
                f"Good news for {name} — the model classifies these measurements "
                f"as Benign{conf_text}. This is not a diagnosis; please confirm "
                f"with a qualified clinician."
            )
        else:
            msg = (
                f"Heads up — for {name}, the model classifies these measurements "
                f"as Malignant{conf_text}. Please treat this as a screening signal "
                f"only and follow up with a medical professional promptly."
            )
        lbl_assistant_msg.config(text=msg)

        save_result(name, result, confidence)

    except Exception as e:
        lbl_assistant_msg.config(
            text=f"I couldn't run that prediction — please check the inputs. "
                 f"({e})"
        )
        messagebox.showerror("Error", f"Invalid input: {e}")

def clear_fields():
    entry_name.delete(0, tk.END)
    for entry in entries.values():
        entry.delete(0, tk.END)
    lbl_result.config(text="Result: ", fg=COLOR_TEXT)
    lbl_assistant_msg.config(
        text="Fields cleared. Enter new measurements whenever you're ready."
    )

# ---- Action buttons (centered, spans all columns) ---------------------
btn_outer = tk.Frame(frame, bg=COLOR_BG)
btn_outer.grid(row=2, column=0, columnspan=3, sticky="ew", padx=8, pady=(16, 4))
btn_outer.grid_columnconfigure(0, weight=1)

btn_frame = tk.Frame(btn_outer, bg=COLOR_BG)
btn_frame.grid(row=0, column=0)

btn_predict = tk.Button(
    btn_frame, text="Run Prediction", command=predict,
    font=("Segoe UI", 11, "bold"), fg="white", bg=COLOR_ACCENT,
    activebackground=COLOR_ACCENT_DARK, activeforeground="white",
    relief="flat", cursor="hand2", padx=28, pady=10
)
btn_predict.pack(side="left", padx=(0, 10))

btn_clear = tk.Button(
    btn_frame, text="Clear", command=clear_fields,
    font=("Segoe UI", 11, "bold"), fg=COLOR_HEADER, bg=COLOR_PANEL,
    activebackground="#ffc2dd", activeforeground=COLOR_HEADER,
    relief="flat", cursor="hand2", padx=28, pady=10
)
btn_clear.pack(side="left", padx=(10, 0))

# ---- Result (centered, spans all columns) ------------------------------
result_outer = tk.Frame(frame, bg=COLOR_BG)
result_outer.grid(row=3, column=0, columnspan=3, sticky="ew", padx=8, pady=(4, 24))
result_outer.grid_columnconfigure(0, weight=1)

lbl_result = tk.Label(
    result_outer, text="Result: ", font=("Segoe UI", 14, "bold"),
    fg=COLOR_TEXT, bg=COLOR_BG
)
lbl_result.grid(row=0, column=0)

canvas.pack(fill="both", expand=True, side="left")
scroll_y.pack(fill="y", side="right")

app.mainloop()
