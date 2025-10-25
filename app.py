from flask import Flask, render_template, Response, redirect, url_for, flash, request, jsonify, has_request_context, has_app_context, Blueprint
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from database.db import db
from database.models import User, Log
from datetime import datetime, timedelta
from flask_migrate import Migrate
import os, time, threading, smtplib, ssl, cv2
import shutil
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import pytz

try:
    import gpiod
    from gpiod.line import Direction, Value
    from picamera2 import Picamera2
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("[WARN] GPIO/PiCamera2 modules not available. Running in Windows/mock mode.")

import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
from camera.recognition import recognize_faces

# ===== FLASK APP =====
app = Flask(__name__)
app.secret_key = "supersecretkey"

# ===== Dataset & Faces =====
DATASET_PATH = "faces"
os.makedirs(DATASET_PATH, exist_ok=True)
recognizer = recognize_faces()

# ===== DATABASE =====
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL")
db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

UNKNOWN_COOLDOWN = 60  # seconds
last_unknown_email_time = 0

migrate = Migrate(app, db)

logs_bp = Blueprint("logs", __name__)
manila_tz = pytz.timezone("Asia/Manila")

# ===== EMAIL NOTIFIER =====
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")  # App password from Gmail

def get_all_emails():
    """Fetch all user emails from DB"""
    with app.app_context():
        users = User.query.filter(User.email.isnot(None)).all()
        return [u.email for u in users if u.email]

def send_email(subject, body):
    """Send email to all registered users"""
    recipients = get_all_emails()
    if not recipients:
        print("⚠️ No recipients found, skipping email.")
        return
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, recipients, msg.as_string())

        print(f"📧 Email sent: {subject} -> {len(recipients)} users")
    except Exception as e:
        print(f"❌ Email failed: {e}")

# ===== Load Users =====
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ===== Helpers =====
def get_people():
    people = {}
    if os.path.exists(DATASET_PATH):
        for person_name in os.listdir(DATASET_PATH):
            person_folder = os.path.join(DATASET_PATH, person_name)
            if os.path.isdir(person_folder):
                images = [f for f in os.listdir(person_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                people[person_name] = images
    return people

def _insert_log(message: str, user):
    log = Log(event=message, user=user, timestamp=datetime.utcnow())
    db.session.add(log)
    db.session.commit()

def add_log(message: str, user=None):
    if user is None and has_request_context():
        try:
            if current_user.is_authenticated:
                user = current_user
        except Exception:
            user = None
    if has_app_context():
        _insert_log(message, user)
    else:
        with app.app_context():
            _insert_log(message, user)

# ===== GPIO Setup =====
if GPIO_AVAILABLE:
    CHIP_NAME = "/dev/gpiochip0"
    SOLENOID_LINE = 23
    PIR_LINE = 24

    LINES = {
        SOLENOID_LINE: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.ACTIVE),
        PIR_LINE: gpiod.LineSettings(direction=Direction.INPUT),
    }

    chip = gpiod.Chip(CHIP_NAME)
    gpio_request = gpiod.request_lines(CHIP_NAME, consumer="safehouse", config=LINES)
    solenoid = SOLENOID_LINE
    pir = PIR_LINE

    # Camera setup
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
    picam2.configure(config)
    try:
        picam2.set_controls({"AwbEnable": True})
        print("[INFO] Manual white balance applied")
    except Exception as e:
        print(f"[WARN] Could not set WB: {e}")
    picam2.start()
else:
    picam2 = None
    def capture_array():
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        return frame

# ===== Load EMBEDDINGS =====
EMBEDDINGS_PATH = "embeddings.npy"
LABELS_PATH = "labels.npy"

if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(LABELS_PATH):
    embeddings = np.load(EMBEDDINGS_PATH)
    labels = np.load(LABELS_PATH)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(embeddings, labels)
    print("[INFO] Loaded cached embeddings.")
else:
    embeddings, labels = [], []
    knn = KNeighborsClassifier(n_neighbors=3)

facenet = InceptionResnetV1(pretrained='vggface2').eval()
rf = recognize_faces()  # global recognizer

# ===== PIR Loop =====
CONF_THRESHOLD = 0.30  # confidence threshold to unlock

def unlock_and_relock(label, conf):
    try:
        gpio_request.set_value(solenoid, Value.INACTIVE)
        add_log(f"Door unlocked for {label} (confidence {conf:.2f})")
        send_email("SafeHouse Alert", f"Door unlocked for {label} (confidence {conf:.2f})")
        print(f"[INFO] Door unlocked for {label} ({conf:.2f})")
        time.sleep(4)
        gpio_request.set_value(solenoid, Value.ACTIVE)
        add_log(f"Door locked again after 4 seconds for {label}")
    except Exception as e:
        print(f"[ERROR] Unlock thread failed: {e}")

# ===== Shared Motion State =====
motion_detected = False
last_motion_time = 0
MOTION_DURATION = 5  # seconds to keep recognition active
last_unlock_times = {}
UNLOCK_COOLDOWN = 5  # seconds before same person can trigger unlock again

# ===== PIR Loop =====
def pir_loop():
    global motion_detected, last_motion_time
    while True:
        try:
            pir_val = gpio_request.get_value(pir)
            if pir_val == Value.ACTIVE:
                motion_detected = True
                last_motion_time = time.time()
            else:
                if time.time() - last_motion_time > MOTION_DURATION:
                    motion_detected = False
            time.sleep(0.2)
        except Exception as e:
            print(f"[WARN] PIR loop error: {e}")
            time.sleep(0.5)

threading.Thread(target=pir_loop, daemon=True).start()

def log_retention_cleaner():
    while True:
        try:
            with app.app_context():
                cutoff = datetime.utcnow() - timedelta(days=7)
                Log.query.filter(Log.timestamp < cutoff).delete(synchronize_session=False)
                db.session.commit()
        except Exception as e:
            print(f"[WARN] Log retention cleanup failed: {e}")
        time.sleep(3600)  # Run every hour


threading.Thread(target=log_retention_cleaner, daemon=True).start()

# ===== Camera Stream =====
def generate_frames():
    while True:
        frame_np = picam2.capture_array() if GPIO_AVAILABLE else capture_array()
        frame_with_boxes = frame_np

        if motion_detected:
            frame_with_boxes, detections = rf.recognize_frame(frame_np)
            for det in detections:
                label = det['label']
                conf = det['confidence']

                if label != "Unknown" and conf >= CONF_THRESHOLD:
                    now = time.time()
                    last_time = last_unlock_times.get(label, 0)
                    if now - last_time > UNLOCK_COOLDOWN:
                        last_unlock_times[label] = now
                        threading.Thread(target=unlock_and_relock, args=(label, conf), daemon=True).start()

                elif label == "Unknown":
                    now = time.time()
                    global last_unknown_email_time
                    if now - last_unknown_email_time > UNKNOWN_COOLDOWN:
                        last_unknown_email_time = now

                        print(f"[WARN] Unknown face detected ({conf:.2f})")

                        # Capture 3 snapshots
                       
                        snapshots = []
                        for i in range(3):
                            frame_snapshot = picam2.capture_array() if GPIO_AVAILABLE else capture_array()
                            img_path = f"static/unknown/{int(time.time())}_{i}.jpg"
                            os.makedirs(os.path.dirname(img_path), exist_ok=True)
                            cv2.imwrite(img_path, frame_snapshot)
                            snapshots.append(img_path)

                            time.sleep(1)


                        # Store snapshots in logs
                        unknown_log = Log(
                            event="Unknown face detected",
                            user=None,
                            timestamp=datetime.utcnow(),
                            snapshots=",".join(snapshots)
                        )

                        # ✅ Wrap DB operations inside app.app_context()
                        with app.app_context():
                            db.session.add(unknown_log)
                            db.session.commit()

                        # Send email with snapshots
                        def send_unknown_email(snapshots):
                            recipients = get_all_emails()
                            if not recipients:
                                return
                            msg = MIMEMultipart()
                            msg["From"] = EMAIL_ADDRESS
                            msg["To"] = ", ".join(recipients)
                            msg["Subject"] = "⚠️ Unknown Face Detected"
                            msg.attach(MIMEText(f"An unknown face was detected at {datetime.utcnow()}", "plain"))

                            for path in snapshots:
                                abs_path = os.path.join(path)
                                with open(abs_path, "rb") as f:
                                    part = MIMEImage(f.read(), name=os.path.basename(path))
                                    msg.attach(part)

                            context = ssl.create_default_context()
                            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                                server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                                server.sendmail(EMAIL_ADDRESS, recipients, msg.as_string())
                            print(f"[INFO] Unknown face email sent to {len(recipients)} users")

                        threading.Thread(target=send_unknown_email, args=(snapshots,), daemon=True).start()

        ret, buffer = cv2.imencode('.jpg', frame_with_boxes)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ===== ROUTES =====
@app.route('/')
def landing():
    return render_template('landing.html', logged_in=current_user.is_authenticated)

@app.route('/dashboard')
@login_required
def dashboard():
    people = get_people()
    logs = Log.query.order_by(Log.timestamp.asc()).all()
    return render_template("dashboard.html", user=current_user, people=people, logs=logs)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/vidfeed_page')
def videofeed_page():
    return render_template("vidfeed_page.html")

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method=="POST":
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash("Username already exists", "danger")
            return redirect(url_for('register'))
        if User.query.filter_by(email=email).first():
            flash("Email already registered", "danger")
            return redirect(url_for('register'))
        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        flash("Registration successful! Please log in.", "success")
        return redirect(url_for('login'))
    return render_template("register.html")

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method=='POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash,password):
            login_user(user)
            flash("Logged in successfully!","success")
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid username or password.","danger")
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Logged out successfully.", "success")
    return redirect(url_for('landing'))

@app.route('/profile', methods=['GET','POST'])
@login_required
def profile():
    if request.method=="POST":
        username = request.form.get('username','').strip()
        email = request.form.get('email','').strip()
        password = request.form.get('password','').strip()
        existing_user = User.query.filter(User.username==username, User.id!=current_user.id).first()
        if existing_user:
            flash("Username already taken.","danger")
            return redirect(url_for('profile'))
        current_user.username = username
        if email:
            current_user.email = email
        if password:
            current_user.password_hash = generate_password_hash(password)
        db.session.commit()
        flash("Profile updated successfully.","success")
        return redirect(url_for('profile'))
    return render_template("profile.html", user=current_user)

@app.route('/add_user', methods=['GET','POST'])
@login_required
def add_user():
    if request.method=="POST":
        name = request.form['name'].strip()
        images = request.files.getlist('images')

        # Require between 10 and 20 images
        if len(images) < 10 or len(images) > 20:
            flash("Please upload between 10 and 20 images.","danger")
            return redirect(url_for('add_user'))

        person_dir = os.path.join("faces", name)
        os.makedirs(person_dir, exist_ok=True)

        image_paths = []
        for i, img in enumerate(images):
            img_path = os.path.join(person_dir, f"{i+1}.jpg")
            img.save(img_path)
            image_paths.append(img_path)

        rf.add_new_user(name, image_paths)
        add_log(f"{name} added successfully by {current_user.username}")
        flash(f"{name} added successfully with {len(images)} images.","success")
        return redirect(url_for('add_user'))

    return render_template("add_user.html", user=current_user)
    

    
@app.route("/manage_users")
def manage_users():
    static_faces_dir = "static/faces"
    people = {}

    # Ensure sync so static/ is always up to date
    recognizer.sync_faces_to_static()

    if os.path.exists(static_faces_dir):
        for person in sorted(os.listdir(static_faces_dir)):
            person_dir = os.path.join(static_faces_dir, person)
            if os.path.isdir(person_dir):
                images = sorted(os.listdir(person_dir))
                people[person] = images

    return render_template("manage_users.html", people=people)



@app.route("/delete_user/<username>", methods=["POST"])
def delete_user(username):
    faces_dir = "faces"
    static_faces_dir = "static/faces"
    user_dir = os.path.join(faces_dir, username)
    static_user_dir = os.path.join(static_faces_dir, username)

    # Delete from faces/
    if os.path.exists(user_dir):
        try:
            shutil.rmtree(user_dir)
            print(f"[INFO] Deleted {username} from faces/")
        except Exception as e:
            print(f"[WARN] Could not delete {user_dir}: {e}")
            flash(f"Failed to delete {username} from faces folder.", "error")

    # Delete from static/faces/
    if os.path.exists(static_user_dir):
        try:
            shutil.rmtree(static_user_dir)
            print(f"[INFO] Deleted {username} from static/faces/")
        except Exception as e:
            print(f"[WARN] Could not delete {static_user_dir}: {e}")
            flash(f"Failed to delete {username} from static folder.", "error")

    # Rebuild embeddings so knn stays in sync
    recognizer.rebuild_cache()
    flash(f"User {username} deleted successfully and model retrained.", "success")

    return redirect(url_for("manage_users"))


@app.route('/unlock')
@login_required
def unlock():
    if GPIO_AVAILABLE:
        gpio_request.set_value(SOLENOID_LINE, Value.INACTIVE)
        time.sleep(4)
        gpio_request.set_value(SOLENOID_LINE, Value.ACTIVE)

    msg = f"Door was unlocked by {current_user.username}"
    add_log(msg)
    send_email("🔓 Door Unlocked", msg)   # ✅ send email
    return jsonify({"status":"unlocked"})


@app.route('/lock')
@login_required
def lock():
    gpio_request.set_value(SOLENOID_LINE, Value.ACTIVE)

    msg = f"Door was locked by {current_user.username}"
    add_log(msg)
    send_email("🔒 Door Locked", msg)   # ✅ send email
    return jsonify({"status":"locked"})


@app.route('/logs_json')
@login_required
def logs_json():
    logs = Log.query.order_by(Log.timestamp.asc()).all()
    return jsonify([{
        "event": log.event,
        "user": log.user.username if log.user else None,
        "timestamp": log.manila_time().strftime("%d-%m-%y (%I:%M %p)")
    } for log in logs])

@app.route("/delete_logs_range/<int:start>/<int:end>")
@login_required
def delete_logs_range(start,end):
    Log.query.filter(Log.id.between(start,end)).delete(synchronize_session=False)
    db.session.commit()
    return jsonify({"status":f"Deleted logs {start} to {end}"})

@app.route('/users_list')
@login_required
def users_list():
    users = rf.list_users()
    return jsonify({"users":users})

@app.route("/set_wb")
def set_wb():
    red = float(request.args.get("red",1.5))
    blue = float(request.args.get("blue",1.5))
    try:
        picam2.set_controls({"AwbEnable": False, "ColourGains": (red,blue)})
        return jsonify({"status":f"WB set to Red={red}, Blue={blue}"})
    except Exception as e:
        return jsonify({"error": str(e)})

from sqlalchemy import func

@app.route("/logs_by_date/<date_str>")
@login_required
def logs_by_date(date_str):
    try:
        query_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        logs = Log.query.filter(func.date(Log.timestamp) == query_date).order_by(Log.timestamp.asc()).all()

        log_data = []
        for log in logs:
            ts = log.manila_time().strftime("%Y-%m-%d %H:%M:%S") if log.manila_time() else "N/A"
            log_data.append({
                "id": log.id,
                "event": log.event,
                "timestamp": ts,
                "snapshots": log.snapshots.split(",") if log.snapshots else [],  # <-- FIXED
                "user": log.user.username if log.user else None,
            })

        return jsonify(log_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    
# ==== Email ==== 

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from threading import Thread
import time

from database.models import User

# === EMAIL CONFIG ===
EMAIL_ADDRESS = "thesis.raspberrypi@gmail.com"
EMAIL_PASSWORD = "ckzu fwhu aviv jsqc"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465


def get_all_emails():
    """Fetch all user emails from the database inside Flask app context."""
    try:
        with app.app_context():   # ✅ add this
            users = User.query.all()
            return [u.email for u in users if u.email]
    except Exception as e:
        print(f"❌ Failed to fetch users: {e}")
        return []


def send_email(subject, body):
    """Send email to all registered users."""
    try:
        with app.app_context():   # ✅ wrap again
            recipients = get_all_emails()
            if not recipients:
                print("⚠️ No recipients found.")
                return

        msg = MIMEMultipart()
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, recipients, msg.as_string())

        print(f"📧 Sent email: {subject} → {recipients}")

    except Exception as e:
        print(f"❌ Failed to send email: {e}")



# === EVENTS HOOKS ===
def notify_unlock(user):
    """Call this when the door is unlocked."""
    send_email("🔓 Door Unlocked", f"The door was unlocked by {user}.")


def notify_unknown_face():
    """Call this when an unknown face is detected for >3s."""
    send_email("⚠️ Unknown Face Detected", "An unknown face was detected for more than 3 seconds.")

def snapshot_retention_cleaner():
    """Delete snapshots older than 7 days from static/unknown/."""
    folder = os.path.join("static", "unknown")
    while True:
        try:
            if os.path.exists(folder):
                cutoff = time.time() - (7 * 24 * 3600)  # 7 days in seconds
                for filename in os.listdir(folder):
                    filepath = os.path.join(folder, filename)
                    if os.path.isfile(filepath):
                        mtime = os.path.getmtime(filepath)
                        if mtime < cutoff:
                            try:
                                os.remove(filepath)
                                print(f"[CLEANUP] Deleted old snapshot: {filepath}")
                            except Exception as e:
                                print(f"[WARN] Could not delete {filepath}: {e}")
        except Exception as e:
            print(f"[WARN] Snapshot retention cleanup failed: {e}")
        time.sleep(3600)  # run every hour

# Start the cleanup thread
threading.Thread(target=snapshot_retention_cleaner, daemon=True).start()


# === HEARTBEAT EVERY 5 MINUTES ===
def heartbeat_notifier():
    while True:
        time.sleep(300)  # 5 minutes
        send_email("✅ System Heartbeat", "SafeHouse system is running normally.")


def start_notifier_thread():
    t = Thread(target=heartbeat_notifier, daemon=True)
    t.start()


# === STARTUP HOOK ===
@app.before_request
def before_first_request():
    """Start background notifier thread once when app runs."""
    if not hasattr(app, "notifier_started"):
        app.notifier_started = True
        start_notifier_thread()
        send_email("🚀 SafeHouse Started", "SafeHouse prototype has started successfully.")


# ===== Cleanup =====
import atexit
def cleanup():
    try: gpio_request.release()
    except: pass
    try: chip.close()
    except: pass
    print("[INFO] GPIO released. Exiting.")
atexit.register(cleanup)

# ===== Run App =====
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=False)
