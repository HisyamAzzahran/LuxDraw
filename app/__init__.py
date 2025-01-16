from flask import Flask
from app.extensions import db  # Import db dari extensions
from app.routes import app_routes

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///trinity.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Inisialisasi ekstensi
    db.init_app(app)

    # Registrasi blueprint
    app.register_blueprint(app_routes)

    return app
