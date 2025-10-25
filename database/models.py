from database.db import db
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import pytz

# Manila timezone
manila_tz = pytz.timezone("Asia/Manila")


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True)
    password_hash = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    @property
    def display_name(self):
        return self.username

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class Log(db.Model):
    __tablename__ = "logs"

    id = db.Column(db.Integer, primary_key=True)
    event = db.Column(db.String(160), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    snapshots = db.Column(db.Text, nullable=True)   # relative paths: "unknown/file1.jpg;unknown/file2.jpg"
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)

    user = db.relationship("User", backref=db.backref("logs", lazy="dynamic"))

    def manila_time(self):
        if self.timestamp:
            return self.timestamp.replace(tzinfo=pytz.UTC).astimezone(manila_tz)
        return None

    def snapshot_list(self):
        """Return list of snapshot relative paths."""
        if not self.snapshots:
            return []
        return self.snapshots.split(";")

    def __repr__(self):
        ts = self.manila_time().strftime("%Y-%m-%d %H:%M:%S") if self.manila_time() else "N/A"
        if self.user:
            return f"<Log {ts} | {self.event} by {self.user.username}>"
        return f"<Log {ts} | {self.event}>"
