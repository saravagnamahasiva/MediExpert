from datetime import datetime
from difflib import get_close_matches
from functools import wraps
from io import BytesIO
import base64
import json
import os
import sqlite3
import random
import secrets
import socket

from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from flask_cors import CORS
import joblib
import pandas as pd
import qrcode
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from werkzeug.security import check_password_hash, generate_password_hash


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "static", "data")
MODEL_PATH = os.path.join(BASE_DIR, "model", "disease_model.pkl")
DISEASE_INFO_PATH = os.path.join(DATA_DIR, "disease_info.json")

app = Flask(__name__)
CORS(app)
app.secret_key = os.environ.get("MEDIEXPERT_SECRET_KEY", "change-this-secret-before-deployment")
os.makedirs(app.instance_path, exist_ok=True)
DB_PATH = os.environ.get("MEDIEXPERT_DB_PATH", os.path.join(app.instance_path, "mediexpert.db"))


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                profile_photo TEXT,
                report_token TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS diagnoses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                symptoms TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                severity TEXT,
                diagnosis TEXT NOT NULL,
                confidence REAL,
                description TEXT,
                medicines TEXT,
                precautions TEXT,
                advice TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS symptom_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                symptoms TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                text TEXT NOT NULL,
                time TEXT NOT NULL,
                days TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS activities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                activity_type TEXT NOT NULL,
                details TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );

            INSERT INTO activities (user_id, activity_type, details, created_at)
            SELECT u.id, 'signup', 'Account existed before activity tracking', u.created_at
            FROM users u
            WHERE NOT EXISTS (
                SELECT 1
                FROM activities a
                WHERE a.user_id = u.id AND a.activity_type = 'signup'
            );
            """
        )
        user_columns = {row["name"] for row in db.execute("PRAGMA table_info(users)").fetchall()}
        if "report_token" not in user_columns:
            db.execute("ALTER TABLE users ADD COLUMN report_token TEXT")
        if "profile_photo" not in user_columns:
            db.execute("ALTER TABLE users ADD COLUMN profile_photo TEXT")
        users_without_tokens = db.execute(
            "SELECT id FROM users WHERE report_token IS NULL OR report_token = ''"
        ).fetchall()
        for row in users_without_tokens:
            db.execute(
                "UPDATE users SET report_token = ? WHERE id = ?",
                (secrets.token_urlsafe(18), row["id"]),
            )


def now_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped


def current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    with get_db() as db:
        return db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def ensure_report_token(user_id):
    with get_db() as db:
        row = db.execute("SELECT report_token FROM users WHERE id = ?", (user_id,)).fetchone()
        if row and row["report_token"]:
            return row["report_token"]
        token = secrets.token_urlsafe(18)
        db.execute("UPDATE users SET report_token = ? WHERE id = ?", (token, user_id))
        return token


def user_by_report_token(token):
    if not token:
        return None
    with get_db() as db:
        return db.execute("SELECT * FROM users WHERE report_token = ?", (token,)).fetchone()


def log_activity(user_id, activity_type, details=""):
    if not user_id:
        return
    with get_db() as db:
        db.execute(
            "INSERT INTO activities (user_id, activity_type, details, created_at) VALUES (?, ?, ?, ?)",
            (user_id, activity_type, details, now_text()),
        )


@app.context_processor
def inject_user():
    return {"current_user": current_user()}


def load_json(path, fallback):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return fallback


disease_info = load_json(DISEASE_INFO_PATH, {})
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
feature_names = list(getattr(model, "feature_names_in_", [])) if model else []
if not feature_names and os.path.exists(os.path.join(BASE_DIR, "datasets", "training_data.csv")):
    sample = pd.read_csv(os.path.join(BASE_DIR, "datasets", "training_data.csv"), nrows=1)
    feature_names = [col for col in sample.columns if col and col != "prognosis" and not col.startswith("Unnamed")]


def normalize_key(value):
    return " ".join(str(value).replace("_", " ").split()).strip().lower()


disease_info_lookup = {normalize_key(key): value for key, value in disease_info.items()}
feature_lookup = {normalize_key(feature): feature for feature in feature_names}

SYMPTOM_ALIASES = {
    "fever": "high_fever",
    "temperature": "high_fever",
    "high temperature": "high_fever",
    "cold": "continuous_sneezing",
    "sneezing": "continuous_sneezing",
    "stomach ache": "stomach_pain",
    "stomachache": "stomach_pain",
    "belly pain": "abdominal_pain",
    "tummy pain": "abdominal_pain",
    "loose motions": "diarrhoea",
    "loose motion": "diarrhoea",
    "sore throat": "throat_irritation",
    "throat pain": "throat_irritation",
    "breathing problem": "breathlessness",
    "breathing difficulty": "breathlessness",
    "shortness of breath": "breathlessness",
    "bp": "palpitations",
    "high bp": "palpitations",
    "urine smell": "foul_smell_of urine",
    "burning urine": "burning_micturition",
    "painful urination": "burning_micturition",
    "skin spots": "red_spots_over_body",
    "rash": "skin_rash",
    "body pain": "muscle_pain",
    "body pains": "muscle_pain",
    "weak": "fatigue",
    "weakness": "fatigue",
    "tired": "fatigue",
    "tiredness": "fatigue",
    "dizzy": "dizziness",
    "yellow eyes": "yellowing_of_eyes",
    "yellow skin": "yellowish_skin",
    "fast heart beat": "fast_heart_rate",
    "fast heartbeat": "fast_heart_rate",
}


def symptom_to_feature(symptom):
    normalized_text = normalize_key(symptom)
    if not normalized_text:
        return None
    if normalized_text in SYMPTOM_ALIASES:
        return SYMPTOM_ALIASES[normalized_text]
    if normalized_text in feature_lookup:
        return feature_lookup[normalized_text]
    underscored = normalized_text.replace(" ", "_")
    if underscored in feature_names:
        return underscored
    close = get_close_matches(normalized_text, list(feature_lookup.keys()), n=1, cutoff=0.78)
    if close:
        return feature_lookup[close[0]]
    return underscored


def parse_symptoms(symptoms):
    parsed = []
    for symptom in symptoms:
        text = str(symptom).strip()
        if not text:
            continue
        feature = symptom_to_feature(text)
        parsed.append(
            {
                "input": text,
                "feature": feature,
                "matched": bool(feature and feature in feature_names),
                "label": titleize(feature) if feature else text.title(),
            }
        )
    return parsed


def titleize(value):
    return value.replace("_", " ").strip().title()


def info_for_diagnosis(diagnosis, symptoms):
    candidates = [diagnosis] + [titleize(s) for s in symptoms]
    for candidate in candidates:
        normalized = normalize_key(candidate)
        if normalized in disease_info_lookup:
            info = disease_info_lookup[normalized]
            break
    else:
        info = {}

    medicines = []
    dosage = info.get("dosage", {})
    for med in info.get("medicines", []):
        medicines.append({"name": med, "dosage": dosage.get(med, "Consult a doctor for dosage.")})

    return {
        "description": info.get(
            "description",
            "This AI result is a screening suggestion based on selected symptoms, not a confirmed diagnosis.",
        ),
        "medicines": medicines,
        "precautions": info.get(
            "precautions",
            ["Track symptoms", "Stay hydrated", "Consult a qualified clinician if symptoms worsen"],
        ),
        "advice": info.get(
            "advice",
            "Please consult a doctor for severe, persistent, or emergency symptoms.",
        ),
        "source": "static/data/disease_info.json" if info else "fallback",
    }


def available_diseases():
    model_diseases = [str(name).strip() for name in getattr(model, "classes_", [])]
    json_diseases = [str(name).strip() for name in disease_info.keys()]
    combined = sorted({name for name in model_diseases + json_diseases if name}, key=lambda item: item.lower())
    return combined


def build_diagnosis_payload(diagnosis, confidence=0.0, matched=None, unmatched=None, normalized=None, top_predictions=None):
    info = info_for_diagnosis(diagnosis, [])
    return {
        "disease": diagnosis,
        "confidence": round(confidence * 100, 2) if confidence <= 1 else round(confidence, 2),
        "matched_symptoms": [titleize(s) for s in (matched or [])],
        "unmatched_symptoms": unmatched or [],
        "normalized_symptoms": normalized or [],
        "top_predictions": top_predictions or [],
        **info,
    }


def predict_disease(symptoms):
    parsed = parse_symptoms(symptoms)
    matched = []
    for item in parsed:
        if item["matched"] and item["feature"] not in matched:
            matched.append(item["feature"])

    if not model or not feature_names or not matched:
        diagnosis = titleize(symptoms[0]) if symptoms else "General Health Concern"
        return diagnosis, 0.0, matched, []

    row = {feature: 0 for feature in feature_names}
    for symptom in matched:
        row[symptom] = 1

    frame = pd.DataFrame([row], columns=feature_names)
    diagnosis = str(model.predict(frame)[0])
    confidence = 0.0
    top_predictions = []
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(frame)[0]
        classes = list(model.classes_)
        ranked = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)[:3]
        top_predictions = [{"disease": str(name), "confidence": round(float(prob) * 100, 2)} for name, prob in ranked]
        confidence = float(ranked[0][1]) if ranked else 0.0
    return diagnosis, confidence, matched, top_predictions


def diagnosis_records(user_id, limit=None):
    sql = "SELECT * FROM diagnoses WHERE user_id = ? ORDER BY created_at DESC"
    params = [user_id]
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    with get_db() as db:
        rows = db.execute(sql, params).fetchall()
    records = []
    for row in rows:
        meds = json.loads(row["medicines"] or "[]")
        precautions = json.loads(row["precautions"] or "[]")
        records.append(
            {
                "date": row["created_at"].split(" ")[0],
                "symptoms": row["symptoms"],
                "diagnosis": row["diagnosis"],
                "medicine": ", ".join(m["name"] for m in meds) if meds else "Consult doctor",
                "precautions": ", ".join(precautions),
                "severity": row["severity"] or "",
            }
        )
    return records


def symptom_log_records(user_id, limit=None):
    sql = "SELECT symptoms, created_at FROM symptom_logs WHERE user_id = ? ORDER BY created_at DESC"
    params = [user_id]
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    with get_db() as db:
        return db.execute(sql, params).fetchall()


def reminder_records(user_id, limit=None):
    sql = "SELECT text, time, days, created_at FROM reminders WHERE user_id = ? ORDER BY time ASC"
    params = [user_id]
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    with get_db() as db:
        rows = db.execute(sql, params).fetchall()
    reminders = []
    for row in rows:
        try:
            days = ", ".join(json.loads(row["days"] or "[]"))
        except json.JSONDecodeError:
            days = row["days"] or ""
        reminders.append({"text": row["text"], "time": row["time"], "days": days, "created_at": row["created_at"]})
    return reminders


def report_context_for_user(user):
    return {
        "user": user,
        "records": diagnosis_records(user["id"]),
        "symptom_logs": symptom_log_records(user["id"]),
        "reminders": reminder_records(user["id"]),
        "activities": activity_records(user["id"], limit=20),
        "timeline_events": user_timeline(user["id"], limit=30),
    }


def make_qr_data_url(value):
    qr_code = qrcode.QRCode(version=1, box_size=10, border=4)
    qr_code.add_data(value)
    qr_code.make(fit=True)
    img = qr_code.make_image(fill_color="black", back_color="white")
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def make_qr_png(value):
    qr_code = qrcode.QRCode(version=1, box_size=10, border=4)
    qr_code.add_data(value)
    qr_code.make(fit=True)
    img = qr_code.make_image(fill_color="black", back_color="white")
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    buffered.seek(0)
    return buffered


def local_network_host():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return None


def report_scan_url(token):
    public_base = os.environ.get("MEDIEXPERT_PUBLIC_BASE_URL", "").strip().rstrip("/")
    if public_base:
        return f"{public_base}{url_for('shared_report', token=token)}"

    host = request.host.split(":", 1)[0]
    if host in {"127.0.0.1", "localhost", "0.0.0.0"}:
        network_host = local_network_host()
        if network_host and not network_host.startswith("127."):
            port = request.host.split(":", 1)[1] if ":" in request.host else "5000"
            return f"{request.scheme}://{network_host}:{port}{url_for('shared_report', token=token)}"
    return url_for("shared_report", token=token, _external=True)


def read_profile_photo(upload):
    if not upload or not upload.filename:
        return None
    allowed_types = {"image/jpeg", "image/png", "image/webp"}
    if upload.mimetype not in allowed_types:
        raise ValueError("Please upload a JPG, PNG, or WEBP photo.")
    data = upload.read()
    if len(data) > 2 * 1024 * 1024:
        raise ValueError("Please choose a photo smaller than 2 MB.")
    return f"data:{upload.mimetype};base64,{base64.b64encode(data).decode('utf-8')}"


def activity_records(user_id, limit=30):
    with get_db() as db:
        return db.execute(
            """
            SELECT activity_type, details, created_at
            FROM activities
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()


ACTIVITY_LABELS = {
    "signup": ("Account Created", "Your MediExpert+ account was created."),
    "login": ("Signed In", "You signed in to MediExpert+."),
    "diagnosis": ("Diagnosis Search", ""),
    "symptom_log": ("Symptom Logged", ""),
    "reminder_added": ("Reminder Added", ""),
    "reminder_updated": ("Reminder Updated", ""),
    "reminder_deleted": ("Reminder Deleted", ""),
    "chatbot": ("MediBot Chat", ""),
    "report_viewed": ("Health Report Viewed", "You opened your health report."),
    "report_downloaded": ("Health Report Downloaded", "You downloaded your health report."),
    "qr_generated": ("QR Report Generated", ""),
    "profile_updated": ("Profile Updated", "Your profile details were updated."),
    "feedback": ("Feedback Submitted", "You submitted feedback."),
    "restart_plan": ("Plan Restarted", ""),
    "delete_all": ("Health Data Reset", ""),
}


def format_timeline_time(value):
    dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    return {
        "sort_key": value,
        "date": dt.strftime("%d %b %Y"),
        "day": dt.strftime("%A"),
        "time": dt.strftime("%I:%M %p"),
    }


def user_timeline(user_id, limit=80):
    events = []
    with get_db() as db:
        activities = db.execute(
            """
            SELECT activity_type, details, created_at
            FROM activities
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
        diagnoses = db.execute(
            """
            SELECT symptoms, diagnosis, severity, created_at
            FROM diagnoses
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()

    for row in activities:
        timing = format_timeline_time(row["created_at"])
        title, fallback = ACTIVITY_LABELS.get(row["activity_type"], (row["activity_type"].replace("_", " ").title(), ""))
        details = row["details"] or fallback
        events.append({
            **timing,
            "title": title,
            "details": details,
            "category": row["activity_type"],
        })

    for row in diagnoses:
        timing = format_timeline_time(row["created_at"])
        events.append({
            **timing,
            "title": "Diagnosis Result",
            "details": f"{row['symptoms']} -> {row['diagnosis']}",
            "category": "diagnosis_result",
            "meta": row["severity"] or "",
        })

    events.sort(key=lambda event: event["sort_key"], reverse=True)
    return events[:limit]


def latest_report_text(user_id):
    user = current_user()
    records = diagnosis_records(user_id, limit=1)
    if records:
        latest = records[0]
        return "\n".join(
            [
                f"Name: {user['name']}",
                f"Age: {user['age'] or 'Not set'}",
                f"Gender: {user['gender'] or 'Not set'}",
                f"Date: {latest['date']}",
                f"Symptoms: {latest['symptoms']}",
                f"Severity: {latest['severity'] or 'Not set'}",
                f"Diagnosis: {latest['diagnosis']}",
                f"Medicines: {latest['medicine']}",
                f"Precautions: {latest['precautions']}",
            ]
        )
    return "\n".join(
        [
            f"Name: {user['name'] if user else 'Not set'}",
            f"Age: {user['age'] if user and user['age'] else 'Not set'}",
            f"Gender: {user['gender'] if user and user['gender'] else 'Not set'}",
            "No diagnosis records yet.",
        ]
    )


@app.route("/")
def home():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        with get_db() as db:
            user = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            log_activity(user["id"], "login", "User logged in")
            flash("Welcome back!", "success")
            return redirect(url_for("dashboard"))
        flash("Invalid email or password.", "danger")
    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        age = request.form.get("age", "").strip()
        gender = request.form.get("gender", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not name or not email or not age or not gender or not password:
            flash("Please fill all signup fields.", "danger")
            return render_template("signup.html")

        try:
            age_value = int(age)
            if age_value <= 0 or age_value > 120:
                raise ValueError
        except ValueError:
            flash("Please enter a valid age.", "danger")
            return render_template("signup.html")

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template("signup.html")

        try:
            with get_db() as db:
                cur = db.execute(
                    """
                    INSERT INTO users (name, email, password_hash, age, gender, report_token, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (name, email, generate_password_hash(password), age_value, gender, secrets.token_urlsafe(18), now_text()),
                )
            log_activity(cur.lastrowid, "signup", "Account created")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("An account with this email already exists.", "danger")
    return render_template("signup.html")


@app.route("/dashboard")
@login_required
def dashboard():
    records = diagnosis_records(session["user_id"], limit=3)
    with get_db() as db:
        reminder_count = db.execute(
            "SELECT COUNT(*) FROM reminders WHERE user_id = ?",
            (session["user_id"],),
        ).fetchone()[0]
        symptom_count = db.execute(
            "SELECT COUNT(*) FROM symptom_logs WHERE user_id = ?",
            (session["user_id"],),
        ).fetchone()[0]
    return render_template(
        "dashboard.html",
        user=current_user(),
        recent_records=records,
        reminder_count=reminder_count,
        symptom_count=symptom_count,
    )


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully!", "info")
    return redirect(url_for("login"))


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    data = request.get_json(silent=True) or {}
    symptoms = [s for s in data.get("symptoms", []) if str(s).strip()]
    selected_disease = str(data.get("selected_disease", "")).strip()
    if not symptoms and not selected_disease:
        return jsonify({"error": "Please enter at least one symptom."}), 400

    if selected_disease:
        parsed_symptoms = parse_symptoms(symptoms)
        diagnosis = selected_disease
        confidence = 0.0
        matched = [item["feature"] for item in parsed_symptoms if item["matched"]]
        top_predictions = [{"disease": selected_disease, "confidence": 100.0}]
        result = build_diagnosis_payload(
            diagnosis,
            confidence=100.0,
            matched=matched,
            unmatched=[item["input"] for item in parsed_symptoms if not item["matched"]],
            normalized=parsed_symptoms,
            top_predictions=top_predictions,
        )
    else:
        parsed_symptoms = parse_symptoms(symptoms)
        diagnosis, confidence, matched, top_predictions = predict_disease(symptoms)
        result = build_diagnosis_payload(
            diagnosis,
            confidence=confidence,
            matched=matched,
            unmatched=[item["input"] for item in parsed_symptoms if not item["matched"]],
            normalized=parsed_symptoms,
            top_predictions=top_predictions,
        )

    with get_db() as db:
        db.execute(
            """
            INSERT INTO diagnoses
            (user_id, symptoms, age, gender, severity, diagnosis, confidence, description, medicines, precautions, advice, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session["user_id"],
                ", ".join(symptoms),
                data.get("age"),
                data.get("gender"),
                data.get("severity"),
                diagnosis,
                confidence,
                result["description"],
                json.dumps(result["medicines"]),
                json.dumps(result["precautions"]),
                result["advice"],
                now_text(),
            ),
        )
    log_activity(session["user_id"], "diagnosis", f"{', '.join(symptoms)} -> {diagnosis}")

    return jsonify(result)


@app.route("/symptom_diagnosis")
@login_required
def symptom_diagnosis():
    return render_template("symptom_diagnosis.html", user=current_user())


@app.route("/diagnosis_history")
@login_required
def diagnosis_history_api():
    return jsonify(diagnosis_records(session["user_id"], limit=5))


@app.route("/symptoms_catalog")
@login_required
def symptoms_catalog():
    return jsonify([titleize(symptom) for symptom in feature_names])


@app.route("/diseases_catalog")
@login_required
def diseases_catalog():
    return jsonify(available_diseases())


@app.route("/disease_details")
@login_required
def disease_details():
    disease = request.args.get("disease", "").strip()
    if not disease:
        return jsonify({"error": "Please select a disease."}), 400
    return jsonify(build_diagnosis_payload(disease, confidence=100.0, top_predictions=[{"disease": disease, "confidence": 100.0}]))


@app.route("/timeline")
@login_required
def timeline():
    return render_template("health_timeline.html", events=user_timeline(session["user_id"]), user=current_user())


@app.route("/health_timeline")
@login_required
def health_timeline():
    return redirect(url_for("timeline"))


@app.route("/log_symptom", methods=["GET", "POST"])
@login_required
def log_symptom():
    if request.method == "POST":
        symptoms = request.form.get("symptoms", "").strip()
        if symptoms:
            with get_db() as db:
                db.execute(
                    "INSERT INTO symptom_logs (user_id, symptoms, created_at) VALUES (?, ?, ?)",
                    (session["user_id"], symptoms, now_text()),
                )
            log_activity(session["user_id"], "symptom_log", symptoms)
        return redirect(url_for("timeline"))
    return render_template("log_symptom.html")


@app.route("/chatbot")
@login_required
def chatbot():
    log_activity(session["user_id"], "chatbot", "Opened MediBot")
    return render_template("chatbot.html")


@app.route("/chatbot_api", methods=["POST"])
@login_required
def chatbot_api():
    user_msg = (request.json or {}).get("message", "").lower()
    responses = {
        "hi": "Hello! How can I assist you today?",
        "hello": "Hi there! Ask me anything related to health or fitness.",
        "what is fever": "Fever is a temporary increase in your body temperature, often due to infection or illness.",
        "what is covid": "COVID-19 is a respiratory illness caused by the SARS-CoV-2 virus. Common symptoms include cough, fever, and loss of taste or smell.",
        "what is diabetes": "Diabetes affects how your body turns food into energy and can lead to high blood sugar.",
        "what causes headache": "Headaches can be caused by stress, dehydration, eye strain, migraines, or other illnesses.",
        "how to reduce stress": "Try deep breathing, meditation, light activity, sleep, and talking to someone you trust.",
        "how much water to drink": "Many adults aim for about 2 to 2.5 liters daily, but needs vary by health, activity, and climate.",
        "diet for weight loss": "Focus on vegetables, fruits, lean proteins, whole grains, and fewer sugary or processed foods.",
        "first aid for burns": "Cool the burn under running water for 10-20 minutes, avoid ice, and cover with a clean dressing.",
        "what is high bp": "High blood pressure means blood pressure is consistently elevated, which can strain organs over time.",
        "how to control bp": "Reduce salt, stay active, manage stress, avoid tobacco, and take medicines as prescribed.",
        "how to boost immunity": "Sleep well, eat balanced meals, stay active, keep vaccines current, and manage stress.",
        "symptoms of anemia": "Fatigue, pale skin, shortness of breath, and dizziness are common anemia symptoms.",
        "what is asthma": "Asthma narrows and inflames airways, making breathing harder for some people.",
        "what is bmi": "BMI estimates body size using height and weight; it is a screening measure, not a full health diagnosis.",
        "how to improve sleep": "Keep a regular schedule, reduce screens before bed, avoid late caffeine, and make the room calm.",
        "bye": "Take care. I am here when you need health guidance again.",
    }
    for keyword, reply in responses.items():
        if keyword in user_msg:
            log_activity(session["user_id"], "chatbot", f"Asked: {user_msg[:120]}")
            return jsonify({"reply": reply})
    log_activity(session["user_id"], "chatbot", f"Asked: {user_msg[:120]}")
    return jsonify({"reply": random.choice([
        "I am still learning. Try asking about symptoms, fitness, diet, or first aid.",
        "Please rephrase that as a health question and I will do my best.",
        "For urgent symptoms, contact a doctor or emergency service right away.",
    ])})


@app.route("/tips")
@login_required
def tips():
    return render_template("health_tips.html")


@app.route("/reminders")
@login_required
def show_reminders():
    return render_template("reminders.html")


@app.route("/get_reminders")
@login_required
def get_reminders():
    with get_db() as db:
        rows = db.execute(
            "SELECT * FROM reminders WHERE user_id = ? ORDER BY time",
            (session["user_id"],),
        ).fetchall()
    return jsonify([
        {"id": row["id"], "text": row["text"], "time": row["time"], "days": json.loads(row["days"])}
        for row in rows
    ])


@app.route("/add_reminder", methods=["POST"])
@login_required
def add_reminder():
    data = request.get_json(silent=True) or {}
    with get_db() as db:
        cur = db.execute(
            "INSERT INTO reminders (user_id, text, time, days, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                session["user_id"],
                data.get("text", "").strip(),
                data.get("time", ""),
                json.dumps(data.get("days", [])),
                now_text(),
            ),
        )
    log_activity(session["user_id"], "reminder_added", data.get("text", "").strip())
    return jsonify({"message": "Reminder added.", "id": cur.lastrowid})


@app.route("/delete_reminder/<int:reminder_id>", methods=["DELETE"])
@login_required
def delete_reminder(reminder_id):
    with get_db() as db:
        db.execute("DELETE FROM reminders WHERE id = ? AND user_id = ?", (reminder_id, session["user_id"]))
    log_activity(session["user_id"], "reminder_deleted", f"Reminder #{reminder_id}")
    return jsonify({"message": "Reminder deleted."})


@app.route("/update_reminder/<int:reminder_id>", methods=["PUT"])
@login_required
def update_reminder(reminder_id):
    data = request.get_json(silent=True) or {}
    with get_db() as db:
        db.execute(
            "UPDATE reminders SET text = ?, time = ?, days = ? WHERE id = ? AND user_id = ?",
            (
                data.get("text", "").strip(),
                data.get("time", ""),
                json.dumps(data.get("days", [])),
                reminder_id,
                session["user_id"],
            ),
        )
    log_activity(session["user_id"], "reminder_updated", data.get("text", "").strip())
    return jsonify({"message": "Reminder updated."})


@app.route("/tracker", methods=["GET", "POST"])
@login_required
def tracker():
    if request.method == "POST":
        feeling = request.form.get("feeling", "").strip()
        energy = request.form.get("energy", "").strip()
        note = request.form.get("note", "").strip()
        check_in = ", ".join(item for item in [f"Feeling: {feeling}" if feeling else "", f"Energy: {energy}" if energy else "", note] if item)
        if check_in:
            with get_db() as db:
                db.execute(
                    "INSERT INTO symptom_logs (user_id, symptoms, created_at) VALUES (?, ?, ?)",
                    (session["user_id"], check_in, now_text()),
                )
            log_activity(session["user_id"], "symptom_log", check_in)
            flash("Your check-in was saved.", "success")
        return redirect(url_for("tracker"))
    return render_template(
        "symptom_tracker.html",
        user=current_user(),
        records=diagnosis_records(session["user_id"]),
        symptom_logs=symptom_log_records(session["user_id"], limit=6),
    )


@app.route("/trends")
@login_required
def community_trends():
    with get_db() as db:
        rows = db.execute(
            """
            SELECT diagnosis, COUNT(*) AS total
            FROM diagnoses
            GROUP BY diagnosis
            ORDER BY total DESC, diagnosis ASC
            LIMIT 6
            """
        ).fetchall()
    labels = [row["diagnosis"] for row in rows] or ["No reports yet"]
    totals = [row["total"] for row in rows] or [0]
    return render_template("community_trends.html", trend_labels=labels, trend_totals=totals, trends=rows)


@app.route("/language")
@login_required
def language():
    return "<h2>Language switching is available on the diagnosis and reminders pages.</h2>"


@app.route("/qr", methods=["GET", "POST"])
@login_required
def qr():
    token = ensure_report_token(session["user_id"])
    scan_url = report_scan_url(token)
    qr_code_data = make_qr_data_url(scan_url)
    if request.method == "POST":
        log_activity(session["user_id"], "qr_generated", "Generated health report QR")
    context = report_context_for_user(current_user())
    return render_template(
        "qr_generator.html",
        qr_code_data=qr_code_data,
        report_url=scan_url,
        report_view_url=url_for("shared_report", token=token),
        report_download_url=url_for("download_shared_report", token=token),
        qr_download_url=url_for("download_report_qr", token=token),
        **context,
    )


@app.route("/download_report_qr/<token>")
def download_report_qr(token):
    user = user_by_report_token(token)
    if not user:
        abort(404)
    qr_png = make_qr_png(report_scan_url(token))
    return send_file(qr_png, as_attachment=True, download_name="mediexpert_health_report_qr.png", mimetype="image/png")


@app.route("/export")
@login_required
def export_report():
    log_activity(session["user_id"], "report_viewed", "Opened health report")
    user = current_user()
    token = ensure_report_token(session["user_id"])
    return render_template(
        "export_report.html",
        report_url=url_for("shared_report", token=token, _external=True),
        **report_context_for_user(user),
    )


def build_health_report_pdf(user, records, symptom_logs, reminders, activities, timeline_events):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=28, leftMargin=28, topMargin=28, bottomMargin=28)
    styles = getSampleStyleSheet()
    elements = []

    try:
        if user["profile_photo"]:
            photo_header, photo_data = user["profile_photo"].split(",", 1)
            img = Image(BytesIO(base64.b64decode(photo_data)), width=74, height=74)
        else:
            img = Image(os.path.join(BASE_DIR, "static", "patinlogo.png"), width=70, height=70)
        img.hAlign = "CENTER"
        elements.append(img)
        elements.append(Spacer(1, 10))
    except Exception:
        pass

    elements.append(Paragraph("<b>MediExpert+ Complete Health Report</b>", styles["Title"]))
    elements.append(Paragraph("<b>Powered by MediExpert+</b>", styles["Heading3"]))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(f"<b>Name:</b> {user['name']}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Email:</b> {user['email']}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Age:</b> {user['age'] or 'Not set'}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Gender:</b> {user['gender'] or 'Not set'}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Generated:</b> {now_text()}", styles["Normal"]))
    elements.append(Spacer(1, 14))

    def paragraph(value):
        return Paragraph(str(value or "-"), styles["BodyText"])

    diagnosis_data = [["Date", "Symptoms", "Diagnosis", "Medicines", "Precautions"]]
    for record in records:
        diagnosis_data.append([
            paragraph(record["date"]),
            paragraph(record["symptoms"]),
            paragraph(record["diagnosis"]),
            paragraph(record["medicine"]),
            paragraph(record["precautions"]),
        ])
    if len(diagnosis_data) == 1:
        diagnosis_data.append(["-", paragraph("No diagnosis records yet"), "-", "-", "-"])

    elements.append(Paragraph("<b>Diagnosis History</b>", styles["Heading2"]))
    diagnosis_table = Table(diagnosis_data, colWidths=[62, 112, 98, 122, 128], repeatRows=1)
    diagnosis_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f8a9d")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b7c8ce")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f4fbfc")),
    ]))
    elements.append(diagnosis_table)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b>Symptom Logs</b>", styles["Heading2"]))
    log_data = [["Date & Time", "Symptoms"]]
    for row in symptom_logs:
        log_data.append([paragraph(row["created_at"]), paragraph(row["symptoms"])])
    if len(log_data) == 1:
        log_data.append(["-", paragraph("No symptom logs yet")])
    log_table = Table(log_data, colWidths=[120, 390], repeatRows=1)
    log_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#177245")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c6d8cf")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(log_table)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b>Medicine Reminders</b>", styles["Heading2"]))
    reminder_data = [["Time", "Days", "Reminder"]]
    for row in reminders:
        reminder_data.append([paragraph(row["time"]), paragraph(row["days"]), paragraph(row["text"])])
    if len(reminder_data) == 1:
        reminder_data.append(["-", "-", paragraph("No reminders yet")])
    reminder_table = Table(reminder_data, colWidths=[70, 140, 300], repeatRows=1)
    reminder_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2457a6")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c7d2e6")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(reminder_table)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b>Health Timeline</b>", styles["Heading2"]))
    activity_data = [["Date", "Day & Time", "Activity", "Details"]]
    for event in timeline_events:
        activity_data.append([
            paragraph(event["date"]),
            paragraph(f"{event['day']} {event['time']}"),
            paragraph(event["title"]),
            paragraph(event["details"]),
        ])
    if len(activity_data) == 1:
        activity_data.append(["-", "-", paragraph("No recent activity"), "-"])
    activity_table = Table(activity_data, colWidths=[68, 115, 120, 210], repeatRows=1)
    activity_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#5a4a9f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d2cde7")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(activity_table)
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("This report is AI-assisted and should be reviewed by a qualified clinician.", styles["Italic"]))
    elements.append(Paragraph("<b>Powered by MediExpert+</b>", styles["Normal"]))
    doc.build(elements)
    buffer.seek(0)
    return buffer


@app.route("/download_user_report")
@login_required
def download_user_report():
    log_activity(session["user_id"], "report_downloaded", "Downloaded PDF report")
    context = report_context_for_user(current_user())
    buffer = build_health_report_pdf(**context)
    return send_file(buffer, as_attachment=True, download_name="mediexpert_health_report.pdf", mimetype="application/pdf")


@app.route("/shared_report/<token>")
def shared_report(token):
    user = user_by_report_token(token)
    if not user:
        abort(404)
    return render_template(
        "shared_report.html",
        public=True,
        download_url=url_for("download_shared_report", token=token),
        **report_context_for_user(user),
    )


@app.route("/download_shared_report/<token>")
def download_shared_report(token):
    user = user_by_report_token(token)
    if not user:
        abort(404)
    buffer = build_health_report_pdf(**report_context_for_user(user))
    return send_file(buffer, as_attachment=True, download_name="mediexpert_health_report.pdf", mimetype="application/pdf")


@app.route("/edit_profile", methods=["GET", "POST"])
@login_required
def edit_profile():
    if request.method == "POST":
        try:
            profile_photo = read_profile_photo(request.files.get("profile_photo"))
        except ValueError as exc:
            flash(str(exc), "danger")
            return redirect(url_for("edit_profile"))
        with get_db() as db:
            params = [
                request.form.get("name", "").strip(),
                request.form.get("age") or None,
                request.form.get("gender"),
            ]
            sql = "UPDATE users SET name = ?, age = ?, gender = ?"
            if profile_photo:
                sql += ", profile_photo = ?"
                params.append(profile_photo)
            sql += " WHERE id = ?"
            params.append(session["user_id"])
            db.execute(sql, params)
        session["user_name"] = request.form.get("name", "").strip()
        log_activity(session["user_id"], "profile_updated", "Updated profile details")
        flash("Profile updated.", "success")
        return redirect(url_for("edit_profile"))
    return render_template("edit_profile.html", user=current_user())


@app.route("/restart_plan")
@login_required
def restart_plan():
    return render_template("restart_plan.html")


@app.route("/restart_plan", methods=["POST"])
@login_required
def restart_plan_post():
    with get_db() as db:
        db.execute("DELETE FROM diagnoses WHERE user_id = ?", (session["user_id"],))
        db.execute("DELETE FROM symptom_logs WHERE user_id = ?", (session["user_id"],))
        db.execute("DELETE FROM reminders WHERE user_id = ?", (session["user_id"],))
    log_activity(session["user_id"], "restart_plan", "Cleared diagnosis, symptoms, and reminders")
    flash("Your plan and tracking data were restarted.", "success")
    return redirect(url_for("dashboard"))


@app.route("/view_history")
@login_required
def view_history():
    return render_template(
        "view_history.html",
        records=diagnosis_records(session["user_id"], limit=20),
        activities=activity_records(session["user_id"], limit=30),
    )


@app.route("/feedback", methods=["GET", "POST"])
@login_required
def feedback():
    if request.method == "POST":
        with get_db() as db:
            db.execute(
                "INSERT INTO feedback (user_id, name, email, message, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    session["user_id"],
                    request.form.get("name", "").strip(),
                    request.form.get("email", "").strip(),
                    request.form.get("message", "").strip(),
                    now_text(),
                ),
            )
        flash("Thank you for your feedback.", "success")
        log_activity(session["user_id"], "feedback", "Submitted feedback")
        return redirect(url_for("feedback"))
    return render_template("feedback.html", user=current_user())


@app.route("/privacy")
@login_required
def privacy():
    return render_template("privacy.html")


@app.route("/delete_all", methods=["POST"])
@login_required
def delete_all():
    with get_db() as db:
        db.execute("DELETE FROM diagnoses WHERE user_id = ?", (session["user_id"],))
        db.execute("DELETE FROM symptom_logs WHERE user_id = ?", (session["user_id"],))
        db.execute("DELETE FROM reminders WHERE user_id = ?", (session["user_id"],))
    log_activity(session["user_id"], "delete_all", "Deleted diagnosis, symptoms, and reminders")
    return "", 200


init_db()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=os.environ.get("FLASK_DEBUG", "0") == "1")
