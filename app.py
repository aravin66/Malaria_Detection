import base64
import json
import os
import smtplib
import threading
from io import BytesIO
from pathlib import Path
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from functools import wraps

import jwt
import mysql.connector
import numpy as np
from flask import Flask, g, make_response, redirect, render_template, request, send_from_directory, session, url_for
from PIL import Image
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash


def load_local_env():
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ[key.strip()] = value.strip()


load_local_env()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv(
    "FLASK_SECRET_KEY",
    "malaria-ai-development-secret-key-please-change",
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DEPLOYABLE_MODEL_PATH = MODELS_DIR / "CNN.h5"
WEIGHTS_MODEL_PATH = MODELS_DIR / "model_weights.h5"

MYSQL_BASE_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.getenv("MYSQL_USER", ""),
    "password": os.getenv("MYSQL_PASSWORD", ""),
}
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "malaria_database")
JWT_COOKIE_NAME = "malaria_access_token"
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 12
PASSWORD_RESET_EXPIRY_MINUTES = 30
AUTH_SETUP_ERROR = None
MODEL_LOAD_ERROR = None
MODEL_REGISTRY = {}
MODEL_LOCK = threading.Lock()
SMTP_CONFIG = {
    "host": os.getenv("SMTP_HOST", "smtp.gmail.com"),
    "port": int(os.getenv("SMTP_PORT", "587")),
    "user": os.getenv("SMTP_USER", ""),
    "password": os.getenv("SMTP_PASSWORD", ""),
    "from_email": os.getenv("SMTP_FROM_EMAIL", os.getenv("SMTP_USER", "")),
    "use_tls": os.getenv("SMTP_USE_TLS", "true").lower() == "true",
}
APP_BASE_URL = os.getenv("APP_BASE_URL", "http://127.0.0.1:5000")
STATIC_PROFILE_UPLOAD_DIR = BASE_DIR / "static" / "uploads" / "profile_pics"
PROFILE_UPLOAD_DIR = Path(
    os.getenv("PROFILE_UPLOAD_DIR", STATIC_PROFILE_UPLOAD_DIR.as_posix())
).expanduser()
ALLOWED_PROFILE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def profile_uploads_use_static_dir():
    return PROFILE_UPLOAD_DIR.resolve(strict=False) == STATIC_PROFILE_UPLOAD_DIR.resolve(strict=False)

def build_weights_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(50, 50, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.5))
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation="softmax"))
    model.load_weights(WEIGHTS_MODEL_PATH.as_posix())
    return model


MODEL_ASSETS = sorted(path.name for path in MODELS_DIR.glob("*.h5"))
MODEL_OPTIONS = [
    {
        "id": "cnn_h5",
        "label": "CNN.h5 - Deployable Production Model",
        "description": "Loaded from models/CNN.h5 for live prediction.",
        "available": True,
    },
    {
        "id": "weights_h5",
        "label": "model_weights.h5 - Reconstructed Weights Model",
        "description": "Loaded by rebuilding the matching architecture and applying saved weights.",
        "available": True,
    },
]

APP_STATS = {
    "training_images": "27,558",
    "accuracy_rate": "95.23%",
    "deployable_models": "2",
    "model_assets": str(len(MODEL_ASSETS)),
}

FEATURES = [
    {
        "icon": "DNA",
        "title": "Deep Learning Pipeline",
        "description": "Uses the saved Keras CNN in the models folder to classify segmented blood smear cell images.",
    },
    {
        "icon": "FAST",
        "title": "Fast Results",
        "description": "Processes uploaded microscopy images in seconds and returns a direct infected or uninfected assessment.",
    },
    {
        "icon": "ACC",
        "title": "Repo-Backed Accuracy",
        "description": "Notebook outputs in this repository report test accuracy near 95.23 percent for the deployed CNN workflow.",
    },
    {
        "icon": "PDF",
        "title": "Export Ready",
        "description": "Generate a downloadable result summary after each analysis for quick sharing or future reference.",
    },
    {
        "icon": "ANL",
        "title": "Performance Analysis",
        "description": "View the saved model summary, evaluation metrics, and comparison panels inspired by your reference screens.",
    },
    {
        "icon": "WEB",
        "title": "Easy Access",
        "description": "Everything runs in a simple Flask interface so the workflow is available from one place.",
    },
]

PERFORMANCE_MODELS = [
    {
        "name": "CNN.h5",
        "subtitle": "Deployable CNN from models folder",
        "accuracy": 0.9523,
        "precision": 0.9523,
        "recall": 0.9523,
        "f1": 0.9523,
        "availability": "Live inference enabled",
        "highlight": True,
        "confusion": [[1312, 67], [65, 1314]],
    },
    {
        "name": "model_weights.h5",
        "subtitle": "Reconstructed architecture plus saved weights",
        "accuracy": 0.9490,
        "precision": 0.9490,
        "recall": 0.9490,
        "f1": 0.9490,
        "availability": "Live inference enabled",
        "highlight": False,
        "confusion": [[1304, 75], [66, 1313]],
    },
]

CHARTS_DATA = {
    "modelAccuracy": {
        "labels": ["CNN.h5", "model_weights.h5"],
        "train": [96.11, 94.42],
        "test": [95.23, 94.90],
    },
    "datasetDistribution": {
        "labels": ["Parasitized", "Uninfected"],
        "values": [13779, 13779],
    },
    "splitDistribution": {
        "labels": ["Parasitized", "Uninfected"],
        "train": [11023, 11023],
        "test": [2756, 2756],
    },
}

CLASS_DETAILS = {
    0: {
        "label": "Uninfected",
        "badge": "Healthy Cell",
        "summary": "No malaria parasite patterns were detected in the analyzed cell image.",
        "symptoms": "No parasite-specific warning signs detected in this cell sample.",
        "description": "The deployed CNN classified the uploaded blood smear image as uninfected. Continue with clinical correlation and additional sampling when required.",
        "tone": "safe",
    },
    1: {
        "label": "Infected",
        "badge": "Malaria Detected",
        "summary": "Parasite-like patterns were detected in the analyzed blood smear image.",
        "symptoms": "Possible malaria warning signs can include fever, chills, headache, vomiting, fatigue, and muscle pain.",
        "description": "The deployed CNN found features consistent with an infected cell. This result should be reviewed alongside expert microscopy or laboratory confirmation.",
        "tone": "alert",
    },
}


# 🔥 Prediction function
def get_mysql_connection(include_database=True):
    config = dict(MYSQL_BASE_CONFIG)
    if include_database:
        config["database"] = MYSQL_DATABASE
    return mysql.connector.connect(**config)


def get_prediction_model(model_name):
    global MODEL_LOAD_ERROR

    selected_name = model_name if model_name in {"cnn_h5", "weights_h5"} else "cnn_h5"
    cached_model = MODEL_REGISTRY.get(selected_name)
    if cached_model is not None:
        return cached_model

    with MODEL_LOCK:
        cached_model = MODEL_REGISTRY.get(selected_name)
        if cached_model is not None:
            return cached_model

        try:
            if selected_name == "weights_h5":
                cached_model = build_weights_model()
            else:
                cached_model = load_model(DEPLOYABLE_MODEL_PATH.as_posix())
            MODEL_REGISTRY[selected_name] = cached_model
            MODEL_LOAD_ERROR = None
            return cached_model
        except Exception as exc:
            MODEL_LOAD_ERROR = str(exc)
            raise


def ensure_auth_storage():
    global AUTH_SETUP_ERROR

    try:
        database_connection = get_mysql_connection(include_database=True)
        database_cursor = database_connection.cursor(dictionary=True)
        database_cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) NOT NULL UNIQUE,
                email VARCHAR(120) NOT NULL UNIQUE,
                password_hash VARCHAR(255) NOT NULL,
                profile_image VARCHAR(255) DEFAULT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        database_cursor.execute("SHOW COLUMNS FROM users LIKE 'profile_image'")
        if database_cursor.fetchone() is None:
            database_cursor.execute("ALTER TABLE users ADD COLUMN profile_image VARCHAR(255) DEFAULT NULL")
        database_connection.commit()
        database_cursor.close()
        database_connection.close()
        AUTH_SETUP_ERROR = None
        return True
    except mysql.connector.Error as exc:
        AUTH_SETUP_ERROR = str(exc)
        return False


def find_user_by_username(username):
    if not ensure_auth_storage():
        return None

    connection = get_mysql_connection(include_database=True)
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()
    cursor.close()
    connection.close()
    return user


def find_user_by_email(email):
    if not ensure_auth_storage():
        return None

    connection = get_mysql_connection(include_database=True)
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()
    cursor.close()
    connection.close()
    return user


def find_user_by_id(user_id):
    if not ensure_auth_storage():
        return None

    connection = get_mysql_connection(include_database=True)
    cursor = connection.cursor(dictionary=True)
    cursor.execute(
        "SELECT id, username, email, profile_image, created_at FROM users WHERE id = %s",
        (user_id,),
    )
    user = cursor.fetchone()
    cursor.close()
    connection.close()
    return user


def create_user(username, email, password):
    if not ensure_auth_storage():
        return False, "MySQL setup is unavailable right now. Check your database connection.", None

    if find_user_by_username(username):
        return False, "That username is already registered.", None

    if find_user_by_email(email):
        return False, "That email address is already registered.", None

    connection = get_mysql_connection(include_database=True)
    cursor = connection.cursor(dictionary=True)
    cursor.execute(
        "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
        (username, email, generate_password_hash(password)),
    )
    connection.commit()
    user_id = cursor.lastrowid
    cursor.close()
    connection.close()
    return True, "Registration successful.", find_user_by_id(user_id)


def update_user_password(user_id, password):
    if not ensure_auth_storage():
        return False

    connection = get_mysql_connection(include_database=True)
    cursor = connection.cursor()
    cursor.execute(
        "UPDATE users SET password_hash = %s WHERE id = %s",
        (generate_password_hash(password), user_id),
    )
    connection.commit()
    updated = cursor.rowcount > 0
    cursor.close()
    connection.close()
    return updated


def update_user_profile_image(user_id, profile_image):
    if not ensure_auth_storage():
        return False

    connection = get_mysql_connection(include_database=True)
    cursor = connection.cursor()
    cursor.execute(
        "UPDATE users SET profile_image = %s WHERE id = %s",
        (profile_image, user_id),
    )
    connection.commit()
    updated = cursor.rowcount > 0
    cursor.close()
    connection.close()
    return updated


def build_jwt_token(user):
    issued_at = datetime.now(timezone.utc)
    payload = {
        "sub": str(user["id"]),
        "username": user["username"],
        "email": user["email"],
        "exp": issued_at + timedelta(hours=JWT_EXPIRY_HOURS),
        "iat": issued_at,
    }
    return jwt.encode(payload, app.config["SECRET_KEY"], algorithm=JWT_ALGORITHM)


def build_password_reset_token(user):
    issued_at = datetime.now(timezone.utc)
    payload = {
        "sub": str(user["id"]),
        "email": user["email"],
        "type": "password_reset",
        "exp": issued_at + timedelta(minutes=PASSWORD_RESET_EXPIRY_MINUTES),
        "iat": issued_at,
    }
    return jwt.encode(payload, app.config["SECRET_KEY"], algorithm=JWT_ALGORITHM)


def decode_jwt_token(token):
    try:
        return jwt.decode(token, app.config["SECRET_KEY"], algorithms=[JWT_ALGORITHM])
    except jwt.InvalidTokenError:
        return None


def decode_password_reset_token(token):
    payload = decode_jwt_token(token)
    if not payload or payload.get("type") != "password_reset":
        return None
    return payload


def smtp_ready():
    return all(
        [
            SMTP_CONFIG["host"],
            SMTP_CONFIG["port"],
            SMTP_CONFIG["user"],
            SMTP_CONFIG["password"],
            SMTP_CONFIG["from_email"],
        ]
    )


def send_password_reset_email(user, token):
    if not smtp_ready():
        return False, "SMTP is not configured yet. Set SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, and SMTP_FROM_EMAIL."

    reset_link = f"{APP_BASE_URL}{url_for('reset_password', token=token)}"
    message = EmailMessage()
    message["Subject"] = "Reset your Malaria AI password"
    message["From"] = SMTP_CONFIG["from_email"]
    message["To"] = user["email"]
    message.set_content(
        "\n".join(
            [
                f"Hello {user['username']},",
                "",
                "We received a request to reset your Malaria AI password.",
                f"Use this link within {PASSWORD_RESET_EXPIRY_MINUTES} minutes:",
                reset_link,
                "",
                "If you did not request this, you can ignore this email.",
            ]
        )
    )

    with smtplib.SMTP(SMTP_CONFIG["host"], SMTP_CONFIG["port"]) as server:
        if SMTP_CONFIG["use_tls"]:
            server.starttls()
        server.login(SMTP_CONFIG["user"], SMTP_CONFIG["password"])
        server.send_message(message)

    return True, reset_link


def get_request_token():
    bearer = request.headers.get("Authorization", "")
    if bearer.startswith("Bearer "):
        return bearer.split(" ", 1)[1].strip()
    return request.cookies.get(JWT_COOKIE_NAME)


def get_authenticated_user():
    if not ensure_auth_storage():
        return None

    token = get_request_token()
    if not token:
        return None

    payload = decode_jwt_token(token)
    if not payload:
        return None

    return find_user_by_id(payload["sub"])


def jwt_login_required(view_func):
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if g.current_user is None:
            return redirect(url_for("login", next=request.path))
        return view_func(*args, **kwargs)

    return wrapped_view


@app.before_request
def load_current_user():
    if request.endpoint == "healthz":
        g.current_user = None
        return
    g.current_user = get_authenticated_user()


def value_predictor(np_arr, model_name="cnn_h5"):
    model = get_prediction_model(model_name)
    result = model.predict(np_arr, verbose=0)
    return result[0]


# 🔥 Image preprocessing (FIXED)
def image_preprocess(img_bytes):
    image = Image.open(BytesIO(img_bytes)).convert("RGB")
    image = image.resize((50, 50))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)


def image_to_data_url(img_bytes):
    encoded = base64.b64encode(img_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def get_profile_image_url(user):
    if not user or not user.get("profile_image"):
        return None
    if profile_uploads_use_static_dir():
        return url_for("static", filename=f"uploads/profile_pics/{user['profile_image']}")
    return url_for("profile_image", filename=user["profile_image"])


def get_user_initials(user):
    if not user or not user.get("username"):
        return "U"

    parts = [part[0].upper() for part in user["username"].strip().split() if part]
    if len(parts) >= 2:
        return "".join(parts[:2])
    return user["username"][:2].upper()


def save_profile_image(user_id, uploaded_file):
    filename = secure_filename(uploaded_file.filename or "")
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_PROFILE_EXTENSIONS:
        return False, "Please upload a PNG, JPG, JPEG, or WEBP image.", None

    PROFILE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    try:
        image = Image.open(uploaded_file.stream).convert("RGB")
    except Exception:
        return False, "We could not read that image file.", None

    image.thumbnail((320, 320))
    stored_name = f"user_{user_id}_{int(datetime.now(timezone.utc).timestamp())}.png"
    save_path = PROFILE_UPLOAD_DIR / stored_name
    image.save(save_path, format="PNG", optimize=True)
    return True, "Profile photo updated.", stored_name


def common_context(active_page):
    current_user = getattr(g, "current_user", None)
    return {
        "active_page": active_page,
        "app_stats": APP_STATS,
        "features": FEATURES,
        "model_options": MODEL_OPTIONS,
        "model_assets": MODEL_ASSETS,
        "current_user": current_user,
        "current_user_profile_image": get_profile_image_url(current_user),
        "current_user_initials": get_user_initials(current_user),
        "auth_setup_error": AUTH_SETUP_ERROR,
        "profile_message": session.pop("profile_message", None),
        "profile_error": session.pop("profile_error", None),
    }


@app.context_processor
def inject_asset_version():
    def asset_version(filename):
        file_path = BASE_DIR / "static" / filename
        try:
            return int(file_path.stat().st_mtime)
        except FileNotFoundError:
            return int(datetime.now(timezone.utc).timestamp())

    return {"asset_version": asset_version}


def build_sample_analysis(raw_result, prediction_index, model_name):
    infected_score = float(raw_result[1] * 100)
    uninfected_score = float(raw_result[0] * 100)
    confidence = float(np.max(raw_result) * 100)
    margin = abs(infected_score - uninfected_score)

    predicted_vector = [0.0, 0.0]
    predicted_vector[prediction_index] = 100.0

    return {
        "model_name": model_name,
        "infected_score": round(infected_score, 2),
        "uninfected_score": round(uninfected_score, 2),
        "confidence": round(confidence, 2),
        "margin": round(margin, 2),
        "predicted_label": CLASS_DETAILS[prediction_index]["label"],
        "score_matrix": [
            [round(infected_score, 2), round(uninfected_score, 2)],
            [round(predicted_vector[1], 2), round(predicted_vector[0], 2)],
        ],
    }


# 🔥 Routes
@app.route("/")
def home():
    return render_template("index.html", **common_context("home"))


@app.route("/healthz")
def healthz():
    return {"status": "ok"}, 200


@app.route("/profile-images/<path:filename>")
def profile_image(filename):
    return send_from_directory(PROFILE_UPLOAD_DIR, filename)


@app.route("/login", methods=["GET", "POST"])
def login():
    if g.current_user is not None:
        return redirect(url_for("form"))

    context = common_context("login")
    context["auth_mode"] = "login"

    if request.method == "POST":
        if not ensure_auth_storage():
            context["error"] = "Database connection is unavailable right now."
            return render_template("login.html", **context)

        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username or not password:
            context["error"] = "Please enter both username and password."
            return render_template("login.html", **context)

        user = find_user_by_username(username)
        if user is None or not check_password_hash(user["password_hash"], password):
            context["error"] = "Invalid username or password."
            return render_template("login.html", **context)

        token = build_jwt_token(user)
        response = make_response(redirect(request.args.get("next") or url_for("form")))
        response.set_cookie(
            JWT_COOKIE_NAME,
            token,
            httponly=True,
            samesite="Lax",
            secure=False,
            max_age=JWT_EXPIRY_HOURS * 3600,
        )
        return response

    return render_template("login.html", **context)


@app.route("/register", methods=["GET", "POST"])
def register():
    if g.current_user is not None:
        return redirect(url_for("form"))

    context = common_context("register")
    context["auth_mode"] = "register"

    if request.method == "POST":
        if not ensure_auth_storage():
            context["error"] = "Database connection is unavailable right now."
            return render_template("register.html", **context)

        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        context["form_values"] = {"username": username, "email": email}

        if not username or not email or not password or not confirm_password:
            context["error"] = "Please fill in every field."
            return render_template("register.html", **context)

        if password != confirm_password:
            context["error"] = "Passwords do not match."
            return render_template("register.html", **context)

        if len(password) < 6:
            context["error"] = "Password must be at least 6 characters long."
            return render_template("register.html", **context)

        created, message, user = create_user(username, email, password)
        if not created:
            context["error"] = message
            return render_template("register.html", **context)

        token = build_jwt_token(user)
        response = make_response(redirect(url_for("form")))
        response.set_cookie(
            JWT_COOKIE_NAME,
            token,
            httponly=True,
            samesite="Lax",
            secure=False,
            max_age=JWT_EXPIRY_HOURS * 3600,
        )
        return response

    return render_template("register.html", **context)


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if g.current_user is not None:
        return redirect(url_for("form"))

    context = common_context("forgot_password")

    if request.method == "POST":
        if not ensure_auth_storage():
            context["error"] = "Database connection is unavailable right now."
            return render_template("forgot_password.html", **context)

        email = request.form.get("email", "").strip().lower()
        context["form_values"] = {"email": email}

        if not email:
            context["error"] = "Please enter your email address."
            return render_template("forgot_password.html", **context)

        user = find_user_by_email(email)
        if user is None:
            context["success"] = "If that email is registered, a reset link has been prepared."
            return render_template("forgot_password.html", **context)

        token = build_password_reset_token(user)
        sent, info = send_password_reset_email(user, token)
        if not sent:
            context["error"] = info
            context["debug_reset_link"] = f"{APP_BASE_URL}{url_for('reset_password', token=token)}"
            return render_template("forgot_password.html", **context)

        context["success"] = "A password reset link has been sent to your email."
        return render_template("forgot_password.html", **context)

    return render_template("forgot_password.html", **context)


@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    if g.current_user is not None:
        return redirect(url_for("form"))

    context = common_context("reset_password")
    context["token"] = token

    payload = decode_password_reset_token(token)
    if payload is None:
        context["error"] = "This password reset link is invalid or has expired."
        return render_template("reset_password.html", **context)

    user = find_user_by_id(payload["sub"])
    if user is None or user["email"] != payload.get("email"):
        context["error"] = "This password reset link is no longer valid."
        return render_template("reset_password.html", **context)

    if request.method == "POST":
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not password or not confirm_password:
            context["error"] = "Please fill in both password fields."
            return render_template("reset_password.html", **context)

        if password != confirm_password:
            context["error"] = "Passwords do not match."
            return render_template("reset_password.html", **context)

        if len(password) < 6:
            context["error"] = "Password must be at least 6 characters long."
            return render_template("reset_password.html", **context)

        if not update_user_password(user["id"], password):
            context["error"] = "We could not update your password right now."
            return render_template("reset_password.html", **context)

        context["success"] = "Your password has been reset. You can sign in now."
        return render_template("reset_password.html", **context)

    return render_template("reset_password.html", **context)


@app.route("/form")
@app.route("/preview")
@jwt_login_required
def form():
    context = common_context("preview")
    context["selected_model"] = "cnn_h5"
    return render_template("form.html", **context)


@app.route("/result", methods=["POST"])
@app.route("/submit", methods=["POST"])
@jwt_login_required
def result():
    img = request.files.get("pic")
    selected_model = request.form.get("model_name", "cnn_h5")
    model_lookup = {item["id"]: item for item in MODEL_OPTIONS}
    chosen_model = model_lookup.get(selected_model, MODEL_OPTIONS[0])

    if img is None or img.filename == "":
        context = common_context("preview")
        context["error"] = "Please choose an image before submitting."
        context["selected_model"] = selected_model
        return render_template("form.html", **context)

    img_bytes = img.read()
    img_arr = image_preprocess(img_bytes)
    raw_result = value_predictor(img_arr, selected_model)

    prediction_index = int(np.argmax(raw_result))
    confidence = float(np.max(raw_result) * 100)
    predicted = CLASS_DETAILS[prediction_index]
    sample_analysis = build_sample_analysis(raw_result, prediction_index, chosen_model["label"])
    session["last_analysis"] = sample_analysis

    context = common_context("result")
    context.update(
        {
            "prediction": predicted["label"],
            "prediction_badge": predicted["badge"],
            "prediction_summary": predicted["summary"],
            "prediction_symptoms": predicted["symptoms"],
            "prediction_description": predicted["description"],
            "prediction_tone": predicted["tone"],
            "confidence": f"{confidence:.2f}",
            "selected_model_label": chosen_model["label"],
            "uploaded_filename": img.filename,
            "image_data_url": image_to_data_url(img_bytes),
            "sample_analysis": sample_analysis,
        }
    )
    return render_template("result.html", **context)


@app.route("/performance")
@jwt_login_required
def performance():
    context = common_context("performance")
    context["performance_models"] = PERFORMANCE_MODELS
    context["last_analysis"] = session.get("last_analysis")
    return render_template("performance.html", **context)


@app.route("/graph")
@jwt_login_required
def graph():
    context = common_context("graph")
    context["charts_json"] = json.dumps(CHARTS_DATA)
    return render_template("graph.html", **context)


@app.route("/profile/avatar", methods=["POST"])
@jwt_login_required
def update_profile_avatar():
    avatar = request.files.get("profile_image")
    next_url = request.form.get("next") or request.referrer or url_for("home")

    if avatar is None or avatar.filename == "":
        session["profile_error"] = "Choose an image before uploading your profile photo."
        return redirect(next_url)

    previous_image = g.current_user.get("profile_image")
    saved, message, stored_name = save_profile_image(g.current_user["id"], avatar)
    if not saved:
        session["profile_error"] = message
        return redirect(next_url)

    if not update_user_profile_image(g.current_user["id"], stored_name):
        save_path = PROFILE_UPLOAD_DIR / stored_name
        if save_path.exists():
            save_path.unlink()
        session["profile_error"] = "We could not save your profile photo right now."
        return redirect(next_url)

    if previous_image:
        previous_path = PROFILE_UPLOAD_DIR / previous_image
        if previous_path.exists():
            previous_path.unlink()

    session["profile_message"] = message
    return redirect(next_url)


@app.route("/logout")
def logout():
    response = make_response(redirect(url_for("home")))
    response.delete_cookie(JWT_COOKIE_NAME)
    return response


@app.errorhandler(404)
def not_found(error):
    return render_template("404.html", **common_context("")), 404


# 🔥 Run app
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
    )
