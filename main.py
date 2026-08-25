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
# Offline FAQ knowledge base — Q&A restricted to this project only.
# No external API calls; pure keyword matching against a fixed answer set.
# ----------------------------------------------------------------------
FAQ_KB = [
    {
        "keywords": ["what is this", "what does this app", "what does this do",
                     "purpose", "about this project", "what is the project"],
        "answer": (
            "This app predicts whether a breast tumor is likely Benign or "
            "Malignant from 30 numeric measurements of cell nuclei, using a "
            "Logistic Regression model trained on the Wisconsin Breast Cancer "
            "dataset."
        ),
    },
    {
        "keywords": ["accuracy", "how accurate", "how good is the model", "performance"],
        "answer": (
            "The model reaches about 95.61% accuracy on the held-out test "
            "split (20% of the dataset) using all 30 features."
        ),
    },
    {
        "keywords": ["algorithm", "model type", "which model", "what model",
                     "logistic regression", "how does the model work"],
        "answer": (
            "It uses scikit-learn's LogisticRegression classifier "
            "(max_iter=10000), trained on 30 numeric features per sample."
        ),
    },
    {
        "keywords": ["dataset", "data.csv", "training data", "where does the data come from",
                     "what data"],
        "answer": (
            "The model is trained on the classic Wisconsin Diagnostic Breast "
            "Cancer dataset (data.csv), with the 'id' and unnamed columns "
            "dropped and the diagnosis mapped to Malignant=1 / Benign=0."
        ),
    },
    {
        "keywords": ["mean value", "mean values", "what are mean"],
        "answer": (
            "The 'Mean Values' column holds the average of each cell-nuclei "
            "measurement (radius, texture, perimeter, area, smoothness, "
            "compactness, concavity, concave points, symmetry, fractal "
            "dimension) across all cells in the sample image."
        ),
    },
    {
        "keywords": ["standard error", "se value", "se values", "what is se"],
        "answer": (
            "The 'Standard Error Values' column measures how much each "
            "feature varies across the cells in the sample — a small SE "
            "means the cells were fairly uniform."
        ),
    },
    {
        "keywords": ["worst value", "worst values", "what is worst"],
        "answer": (
            "The 'Worst Values' column records the largest (most extreme) "
            "value seen for each feature among all cells in the sample, "
            "which tends to be a strong signal for malignancy."
        ),
    },

    # ---- All 30 individual features, explained in plain language ----
    {"keywords": ["radius_mean"], "answer": "radius_mean: the average distance from the center of a cell nucleus to its edge, averaged across all cells in the sample. Bigger nuclei on average can point toward malignancy."},
    {"keywords": ["radius_se"], "answer": "radius_se: how much the nucleus radius varies from cell to cell in the sample. A higher number means the cells are less consistent in size."},
    {"keywords": ["radius_worst"], "answer": "radius_worst: the largest nucleus radius found among all the cells in the sample — an outlier/extreme value rather than an average."},
    {"keywords": ["texture_mean"], "answer": "texture_mean: the average roughness of the cell surface, measured from variation in grayscale pixel shades in the image."},
    {"keywords": ["texture_se"], "answer": "texture_se: how much that surface roughness (texture) varies between cells in the sample."},
    {"keywords": ["texture_worst"], "answer": "texture_worst: the roughest (highest texture) value found among all cells in the sample."},
    {"keywords": ["perimeter_mean"], "answer": "perimeter_mean: the average length of the boundary line around a cell nucleus."},
    {"keywords": ["perimeter_se"], "answer": "perimeter_se: how much that boundary length varies between cells in the sample."},
    {"keywords": ["perimeter_worst"], "answer": "perimeter_worst: the longest nucleus boundary found among all cells in the sample."},
    {"keywords": ["area_mean"], "answer": "area_mean: the average area (size) of the cell nuclei in the sample. Larger area is one of the stronger indicators used by the model."},
    {"keywords": ["area_se"], "answer": "area_se: how much the nucleus area varies from cell to cell."},
    {"keywords": ["area_worst"], "answer": "area_worst: the largest nucleus area found among all cells in the sample."},
    {"keywords": ["smoothness_mean"], "answer": "smoothness_mean: on average, how smooth or jagged the edge of each nucleus is — think of it as tracing the outline and feeling for bumps."},
    {"keywords": ["smoothness_se"], "answer": "smoothness_se: how much that edge-smoothness varies between cells."},
    {"keywords": ["smoothness_worst"], "answer": "smoothness_worst: the bumpiest (least smooth) nucleus edge found in the sample."},
    {"keywords": ["compactness_mean"], "answer": "compactness_mean: on average, how tightly packed/circular the nucleus shape is versus stretched out or irregular. Higher values mean less circular."},
    {"keywords": ["compactness_se"], "answer": "compactness_se: how much that shape-compactness varies between cells."},
    {"keywords": ["compactness_worst"], "answer": "compactness_worst: the least-circular (most irregular) compactness value found in the sample."},
    {"keywords": ["concavity_mean"], "answer": "concavity_mean: on average, how deep the inward 'dents' in the nucleus boundary are — more pronounced dents are associated with malignancy."},
    {"keywords": ["concavity_se"], "answer": "concavity_se: how much the depth of those dents varies between cells."},
    {"keywords": ["concavity_worst"], "answer": "concavity_worst: the deepest inward dent found among all cells in the sample."},
    {"keywords": ["concave points_mean", "concave point mean"], "answer": "concave points_mean: on average, how many distinct inward dents (not just how deep, but how many) appear on the nucleus boundary."},
    {"keywords": ["concave points_se", "concave point se"], "answer": "concave points_se: how much the number of inward dents varies between cells."},
    {"keywords": ["concave points_worst", "concave point worst"], "answer": "concave points_worst: the highest number of inward dents found on any single cell in the sample — often one of the most predictive features."},
    {"keywords": ["symmetry_mean"], "answer": "symmetry_mean: on average, how symmetric (evenly balanced) each nucleus shape is. Less symmetric shapes lean toward malignancy."},
    {"keywords": ["symmetry_se"], "answer": "symmetry_se: how much that symmetry varies between cells."},
    {"keywords": ["symmetry_worst"], "answer": "symmetry_worst: the least symmetric (most lopsided) nucleus shape found in the sample."},
    {"keywords": ["fractal_dimension_mean"], "answer": "fractal_dimension_mean: on average, how complex or 'jagged' the nucleus boundary looks — like measuring a coastline, a more irregular edge gives a higher value."},
    {"keywords": ["fractal_dimension_se"], "answer": "fractal_dimension_se: how much that boundary complexity varies between cells."},
    {"keywords": ["fractal_dimension_worst"], "answer": "fractal_dimension_worst: the most complex/irregular boundary found among all cells in the sample."},

    # ---- General fallback per base metric (catches shorter questions) ----
    {"keywords": ["radius"], "answer": "radius_* (mean/se/worst) measures the distance from the center of a cell nucleus to its edge — larger radius values tend to correlate with malignant samples. Ask about radius_mean, radius_se, or radius_worst specifically for more detail."},
    {"keywords": ["texture"], "answer": "texture_* (mean/se/worst) measures how rough or varied the cell surface looks in the image. Ask about texture_mean, texture_se, or texture_worst specifically for more detail."},
    {"keywords": ["perimeter"], "answer": "perimeter_* (mean/se/worst) is the length of the boundary line around a cell nucleus, closely related to radius and area."},
    {"keywords": ["area"], "answer": "area_* (mean/se/worst) is the measured size of the cell nucleus — larger, more irregular nuclei tend to push predictions toward Malignant."},
    {"keywords": ["smoothness"], "answer": "smoothness_* (mean/se/worst) captures how jagged or smooth the nucleus boundary is."},
    {"keywords": ["compactness"], "answer": "compactness_* (mean/se/worst) is higher for irregular, non-circular nucleus shapes, and lower for round, compact ones."},
    {"keywords": ["concavity"], "answer": "concavity_* (mean/se/worst) measures how deep the inward-curving 'dents' in the nucleus boundary are."},
    {"keywords": ["concave point"], "answer": "concave points_* (mean/se/worst) counts how many inward dents appear on the nucleus boundary — more concave points often indicate malignancy."},
    {"keywords": ["symmetry"], "answer": "symmetry_* (mean/se/worst) measures how evenly balanced the nucleus shape is — malignant cells are often less symmetric."},
    {"keywords": ["fractal"], "answer": "fractal_dimension_* (mean/se/worst) approximates the 'coastline complexity' of the nucleus boundary — higher values mean a more irregular, jagged outline."},

    {
        "keywords": ["benign", "what does benign mean"],
        "answer": (
            "Benign means the model's prediction leans toward a non-cancerous "
            "tumor based on the entered measurements. It's a model output, "
            "not a medical diagnosis."
        ),
    },
    {
        "keywords": ["malignant", "what does malignant mean"],
        "answer": (
            "Malignant means the model's prediction leans toward a cancerous "
            "tumor based on the entered measurements. Treat it as a "
            "screening signal, not a medical diagnosis."
        ),
    },
    {
        "keywords": ["confidence", "probability", "how sure"],
        "answer": (
            "The confidence percentage comes from the model's "
            "predict_proba() output — it's the probability the model "
            "assigns to whichever class (Benign/Malignant) it predicted."
        ),
    },
    {
        "keywords": ["how do i use", "how to use", "how does this work", "instructions"],
        "answer": (
            "Enter the patient's name and all 30 measurements, then click "
            "'Run Prediction'. The result and confidence appear at the "
            "bottom, and each prediction is saved to the local database. "
            "Use 'Clear' to reset all fields."
        ),
    },
    {
        "keywords": ["database", "records.db", "sqlite", "where is data saved", "save"],
        "answer": (
            "Every prediction (patient name, result, confidence, timestamp) "
            "is saved locally to a SQLite database file called records.db "
            "via database.py."
        ),
    },
    {
        "keywords": ["run prediction button", "predict button"],
        "answer": "The 'Run Prediction' button reads all 30 entered values, feeds them to the trained model, and displays the Benign/Malignant result with confidence.",
    },
    {
        "keywords": ["clear button"],
        "answer": "The 'Clear' button empties the patient name and all 30 input fields and resets the result display.",
    },
    {
        "keywords": ["is this a diagnosis", "is this real", "can i trust", "medical advice",
                     "doctor", "disclaimer", "accurate diagnosis"],
        "answer": (
            "No — this tool is a machine learning demo, not a medical "
            "diagnosis. Always confirm results with a qualified healthcare "
            "professional."
        ),
    },
    {
        "keywords": ["how many features", "30 features", "number of features"],
        "answer": "The model uses all 30 numeric features from the dataset: 10 measurements, each captured as a mean, standard error, and worst value.",
    },

    # ---- General breast cancer education (factual, not medical advice) ----
    {
        "keywords": ["what is breast cancer", "define breast cancer"],
        "answer": (
            "Breast cancer happens when cells in breast tissue grow and "
            "divide out of control, forming a tumor. It can start in "
            "different parts of the breast (most often the ducts or "
            "lobules) and, if untreated, can spread to other parts of the "
            "body. It's one of the most common cancers affecting women, "
            "though men can develop it too."
        ),
    },
    {
        "keywords": ["symptom", "warning sign", "signs of breast cancer"],
        "answer": (
            "Common signs include a new lump or thickening in the breast or "
            "underarm, a change in breast size or shape, skin dimpling or "
            "puckering, nipple discharge (especially bloody), nipple "
            "turning inward, or redness/scaling of the skin. Many lumps "
            "turn out to be benign, but any new change is worth having a "
            "doctor check."
        ),
    },
    {
        "keywords": ["risk factor", "who gets breast cancer", "causes of breast cancer"],
        "answer": (
            "Key risk factors include being female and older age, a family "
            "history or inherited gene mutations (like BRCA1/BRCA2), early "
            "menstruation or late menopause, dense breast tissue, obesity, "
            "alcohol use, and prior radiation exposure. Having risk factors "
            "doesn't mean someone will get breast cancer, and many people "
            "diagnosed have no major risk factors at all."
        ),
    },
    {
        "keywords": ["screening", "mammogram", "early detection", "self exam", "self-exam"],
        "answer": (
            "Screening tools include mammograms (X-rays of the breast), "
            "clinical breast exams by a doctor, and breast self-awareness "
            "(getting familiar with how your breasts normally look and "
            "feel so you notice changes). Regular screening helps catch "
            "cancer earlier, when it's usually easier to treat. Screening "
            "guidelines vary by age and risk, so it's best to ask a doctor "
            "what schedule fits your situation."
        ),
    },
    {
        "keywords": ["stage", "staging"],
        "answer": (
            "Breast cancer stages (0 to IV) describe how far the cancer has "
            "spread: Stage 0 is non-invasive/contained to ducts, Stages I–III "
            "reflect growing tumor size and spread to nearby lymph nodes, "
            "and Stage IV means it has spread to distant organs (metastatic). "
            "Earlier stages generally have better outcomes."
        ),
    },
    {
        "keywords": ["treatment", "how is it treated", "chemotherapy", "radiation therapy",
                     "surgery for breast cancer"],
        "answer": (
            "Common treatments include surgery (lumpectomy or mastectomy), "
            "radiation therapy, chemotherapy, hormone therapy, and targeted "
            "or immunotherapy drugs. The right combination depends on the "
            "cancer type, stage, and the patient's overall health — that's "
            "a decision for an oncologist, not something this app can "
            "advise on."
        ),
    },
    {
        "keywords": ["difference between benign and malignant", "benign vs malignant",
                     "tumor types", "what is a tumor"],
        "answer": (
            "In general (not just in this app), a benign tumor grows "
            "slowly, stays in one place, and doesn't invade nearby tissue "
            "or spread elsewhere in the body. A malignant tumor is "
            "cancerous — it can grow more aggressively, invade surrounding "
            "tissue, and spread (metastasize) through the blood or lymph "
            "system to other organs."
        ),
    },
    {
        "keywords": ["prevention", "reduce risk", "how to avoid breast cancer"],
        "answer": (
            "No approach guarantees prevention, but factors linked to lower "
            "risk include maintaining a healthy weight, staying physically "
            "active, limiting alcohol, not smoking, and — for those at high "
            "genetic risk — genetic counseling and closer screening. This "
            "is general health information, not personalized medical advice."
        ),
    },
    {
        "keywords": ["survival rate", "prognosis", "outcome"],
        "answer": (
            "Prognosis depends heavily on the cancer's type, stage at "
            "diagnosis, and how it responds to treatment — early-stage, "
            "localized breast cancer generally has a much higher survival "
            "rate than cancer that has spread. Exact statistics vary by "
            "source and population, so a doctor is the right person to "
            "discuss an individual prognosis."
        ),
    },
]

OUT_OF_SCOPE_MSG = (
    "I can only help with questions about this project and breast cancer "
    "in general — things like the model, the 30 features, the dataset, "
    "accuracy, how to use the app, or general breast cancer facts "
    "(symptoms, risk factors, screening, stages, treatment). Try asking "
    "about one of those!"
)

def answer_faq(question: str) -> str:
    q = question.lower().strip()
    if not q:
        return "Ask me something about this project — e.g. \"what does confidence mean?\" or \"how accurate is the model?\""

    best_match = None
    best_score = 0
    for entry in FAQ_KB:
        # Score by the length of the longest matching keyword, so specific
        # multi-word phrases win over short generic words (e.g. "difference
        # between benign and malignant" beats a bare "benign" match).
        matched_lengths = [len(kw) for kw in entry["keywords"] if kw in q]
        score = max(matched_lengths) if matched_lengths else 0
        if score > best_score:
            best_score = score
            best_match = entry

    if best_match and best_score > 0:
        return best_match["answer"]
    return OUT_OF_SCOPE_MSG

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

# ---- Ask the AI Assistant (offline project FAQ) ----------------------
qa_frame = tk.Frame(app, bg=COLOR_BG)
qa_frame.pack(fill="x", padx=15, pady=(12, 4))

tk.Label(
    qa_frame, text="🤖 AI Assistant", font=("Segoe UI", 10, "bold"),
    fg=COLOR_HEADER, bg=COLOR_BG, anchor="w"
).pack(fill="x", pady=(0, 4))

qa_input_row = tk.Frame(qa_frame, bg=COLOR_BG)
qa_input_row.pack(fill="x")

entry_question = tk.Entry(
    qa_input_row, font=("Segoe UI", 10), bg=COLOR_ENTRY_BG,
    fg=COLOR_TEXT, relief="flat", highlightthickness=1,
    highlightbackground=COLOR_ACCENT, highlightcolor=COLOR_ACCENT_DARK
)
entry_question.insert(0, "Ask about this project (e.g. \"how accurate is the model?\")")
entry_question.config(fg="#c2185b")

def _clear_placeholder(event):
    if entry_question.get().startswith("Ask about this project"):
        entry_question.delete(0, tk.END)
        entry_question.config(fg=COLOR_TEXT)

entry_question.bind("<FocusIn>", _clear_placeholder)
entry_question.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 8))

def ask_question(event=None):
    question = entry_question.get().strip()
    if not question or question.startswith("Ask about this project"):
        return
    answer = answer_faq(question)
    lbl_assistant_msg.config(text=answer)

entry_question.bind("<Return>", ask_question)

btn_ask = tk.Button(
    qa_input_row, text="Ask", command=ask_question,
    font=("Segoe UI", 10, "bold"), fg="white", bg=COLOR_ACCENT,
    activebackground=COLOR_ACCENT_DARK, activeforeground="white",
    relief="flat", cursor="hand2", padx=18, pady=4
)
btn_ask.pack(side="left")

# ---- AI Assistant answer bubble (below the question box) -------------
bubble = tk.Frame(app, bg=COLOR_BOT_BUBBLE, bd=0)
bubble.pack(fill="x", padx=15, pady=(8, 10))

lbl_assistant_msg = tk.Label(
    bubble,
    text="Hi! I can answer questions about this app, the 30 features, and "
         "general breast cancer facts (symptoms, risk factors, screening, "
         "stages, treatment). Ask me something above, or enter measurements "
         "and press \"Run Prediction\" to get a result.",
    font=("Segoe UI", 10), fg=COLOR_TEXT, bg=COLOR_BOT_BUBBLE,
    wraplength=1000, justify="left", padx=12, pady=10
)
lbl_assistant_msg.pack(fill="x")

def _update_bubble_wrap(event):
    lbl_assistant_msg.config(wraplength=max(event.width - 24, 200))

bubble.bind("<Configure>", _update_bubble_wrap)

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
