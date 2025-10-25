# 🏠 SafeHouse — Smart Facial Recognition Door Lock System

**SafeHouse** is an intelligent door access system that integrates **facial recognition**, **PIR motion detection**, and **relay-based solenoid locking**.  
It is built using **Python**, **Flask**, **PostgreSQL**, and **Docker** — designed for modularity, security, and reproducibility.

---

## 🚀 Features

- Real-time **face detection** and **recognition** using **MTCNN + FaceNet + KNN/SVM**
- **PIR motion sensor** triggers recognition sequence automatically
- **Relay + Solenoid lock** control for secure door access
- **Flask web dashboard** with login, user management, and live video feed
- **Event logging** via PostgreSQL (user actions, detections, unlock events)
- **Email notification system** via Gmail SMTP (optional)
- Fully containerized with **Docker Compose**

---

## 🧱 Tech Stack

| Component | Technology |
|------------|-------------|
| **Backend** | Flask (Python) |
| **Frontend** | HTML + TailwindCSS |
| **Database** | PostgreSQL |
| **Machine Learning** | FaceNet, MTCNN, KNN/SVM |
| **Hardware Integration** | GPIO (gpiod) + PiCamera2 |
| **Deployment** | Docker & Docker Compose |

---

## ⚙️ Hardware Requirements

- **Raspberry Pi 4** (recommended)  
- **Raspberry Pi Camera Module** (or USB webcam)  
- **PIR Motion Sensor**  
- **5V Relay + 12V Solenoid Door Lock + 12V Adapter**

**Wiring Overview:**

| Component | GPIO Pin | Description |
|------------|-----------|-------------|
| PIR Sensor (OUT) | GPIO 17 | Motion detection trigger |
| Relay (IN) | GPIO 27 | Solenoid control |
| Camera | CSI or USB | Video input |
| Power | External 12V | Solenoid power source |

---

## 🧩 Software Prerequisites

- **Docker** (v20.10+)
- **Docker Compose Plugin**
- **Python 3.9+** (if running without Docker)
- **Raspberry Pi OS / Ubuntu**

---

## 🐳 Quick Start (Docker)

### 1️⃣ Clone Repository

```bash
git clone https://github.com/<your-username>/safehouse.git
cd safehouse
```


2️⃣ Create .env File
In the root folder, create a .env file:

```bash
# Email credentials (for alerts and verification)
EMAIL_ADDRESS=youremail@gmail.com
EMAIL_PASSWORD=your_app_password

# PostgreSQL connection
POSTGRES_USER=postgres
POSTGRES_PASSWORD=thesispassword
POSTGRES_DB=safehouse_db

# SQLAlchemy connection string
DATABASE_URL=postgresql://postgres:thesispassword@db:5432/safehouse_db
```

💡 Create an App Password in your Google Account (under Security → App Passwords)
Regular Gmail passwords won’t work for SMTP.

3️⃣ Build and Run Containers

```bash
sudo docker compose build
sudo docker compose up -d
This starts two containers:

safehouse_web — Flask app

safehouse_db — PostgreSQL database
```

Check logs:
```bash
sudo docker compose logs -f
```
Stop:
```bash
sudo docker compose down
```

4️⃣ Access Web Dashboard

Visit:
👉 http://localhost:8000 (if local)
👉 http://<raspberrypi-ip>:8000 (from LAN)

---

## Default features available:

Register / Login

Add User (upload face images)

Live camera feed

Logs viewer

Manual Lock / Unlock buttons

---

## 🧠 Facial Recognition Pipeline

1. PIR sensor detects motion
2. Camera activates for ~10 seconds
3. MTCNN detects face bounding boxes
4. FaceNet generates embeddings (512D vectors)
5. KNN or SVM classifier compares to saved embeddings
6. If confidence > threshold → unlock solenoid (4s)
7. Log recognition event to PostgreSQL
This ensures minimal CPU usage by limiting recognition runs and caching embeddings.

---

##📂 Project Structure

```bash
safehouse/
│
├── app.py                  # Main Flask app entry point
├── camera/
│   ├── recognition.py      # Face detection & recognition logic
│   └── ...
├── database/
│   ├── db.py               # SQLAlchemy setup
│   ├── models.py           # User and Log models
│
├── templates/              # HTML templates (Flask)
├── static/                 # CSS, JS, and assets
│
├── Dockerfile              # Flask container build instructions
├── docker-compose.yml      # Multi-container definition
├── requirements.txt        # Python dependencies
├── .env                    # Environment config (excluded from git)
└── README.md
```

---

## 🧾 Environment Variables Summary
```bash
Variable	             Description
EMAIL_ADDRESS	      -  Gmail address used for sending alerts
EMAIL_PASSWORD	      -  App-specific Gmail password
DATABASE_URL	      -  SQLAlchemy DB connection string
POSTGRES_USER	      -  PostgreSQL user
POSTGRES_PASSWORD	  -  PostgreSQL password
POSTGRES_DB	          -  PostgreSQL database name
```

⚡ Common Docker Commands
```bash
Command	Description
sudo docker compose up -d	Start containers in detached mode
sudo docker compose down	Stop all running containers
sudo docker compose logs -f	View live logs
sudo docker exec -it safehouse_web bash	Enter Flask container shell
sudo docker exec -it safehouse_db psql -U postgres	Access Postgres CLI
```
##🔒 Security Notes

.env file must never be committed to GitHub.
Docker isolates both app and database from the host network.
Only port 8000 (web) is exposed.
Database (db) is only accessible within Docker network, not public.
Email credentials are injected via environment variables at runtime.

##🧪 Troubleshooting

```bash
Issue	Fix
docker compose not found	-    Install plugin: sudo apt install docker-compose-plugin -y
Port 5432 in use	        -    Stop system Postgres: sudo systemctl stop postgresql
Camera feed blank           -    Enable Pi camera via sudo raspi-config
PIR always triggered	    -    Check wiring or adjust threshold sensitivity
Module build errors	        -    Run sudo apt install build-essential libcap-dev -y before build
Email not sending	        -    Ensure App Password is valid and less-secure apps disabled
```

---

## 🧰 Manual Run (Without Docker)
Useful for debugging directly on Raspberry Pi.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
Access at http://localhost:8000
```

---

## 📜 License & Credits
## Developers: Rhod Railey De Vera, France Jethrayne A. Miclat, Wyatt Mathew N. Yatco, Justine Lusung
## Year: 2025

---

Libraries used:

facenet-pytorch

torch, opencv-python

flask, flask-login

sqlalchemy, psycopg2

gpiod, picamera2

🧠 SafeHouse combines local ML inference, sensor-based automation, and secure containerization — a complete end-to-end IoT security system prototype.
