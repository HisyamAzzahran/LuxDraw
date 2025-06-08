from flask import Blueprint, render_template, request, redirect, url_for
from drawing_logic import DrawingLogic
from scripts.hand_tracking import hand_tracking
from models.user_model import User
from app import db
from werkzeug.utils import secure_filename

app_routes = Blueprint('app_routes', __name__)

@app_routes.route("/")
def index():
    return render_template("dashboard.html")

@app_routes.route("/result")
def result():
    # Jalankan fungsi hand tracking
    result_data = hand_tracking(save_canvas=True)

    # Cek jika ada error dari fungsi hand_tracking
    if "error" in result_data:
        return render_template("error.html", error=result_data["error"])

    # Ambil data hasil OCR dan path canvas
    detected_text = result_data.get("detected_text", "No text detected")
    status_text = result_data.get("status_text", "No status available")
    canvas_path = result_data.get("canvas_path", None)

    return render_template(
        "result.html",
        detected_text=detected_text,
        status_text=status_text,
        canvas_path=canvas_path
    )

@app_routes.route("/dashboard")
def dashboard():
    total_drawings = 10
    total_hand_tracks = 15
    return render_template("dashboard.html", drawings=total_drawings, tracks=total_hand_tracks)

@app_routes.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            return redirect(url_for("app_routes.dashboard"))
        return "Invalid credentials!"
    return render_template("login.html")

@app_routes.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for("app_routes.login"))
    return render_template("register.html")

@app_routes.route("/save_drawing", methods=["POST"])
def save_drawing():
    if "image" in request.files:
        image = request.files["image"]
        filename = secure_filename(image.filename)
        image.save(f"outputs/{filename}")
        return "Image saved successfully!", 200
    return "No image provided!", 400
