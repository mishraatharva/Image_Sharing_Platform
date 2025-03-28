from flask import Flask, g
from src.image_sharing_plateform.web_app.routes.main_routes import main
from src.image_sharing_plateform.web_app.routes.authentication_routes import auth
from src.image_sharing_plateform.web_app.routes.activity_routes import activity
from flask_pymongo import PyMongo
from src.image_sharing_plateform.pipeline.stage_05_model_prediction_pipeline import PredictionPipeline
from src.image_sharing_plateform.constants import *
from tensorflow.keras.applications import VGG16
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__, template_folder="src/image_sharing_plateform/web_app/templates")


app.config["SECRET_KEY"] = "this_is_a_secret_key"
app.config["MONGO_URI"] = "mongodb+srv://mishraatharva825:5IXJ8vQXqr80qANX@image-sharing-platform.5kejl.mongodb.net/image-sharing-platform-database?retryWrites=true&w=majority&appName=image-sharing-platform"

mongo = PyMongo(app)


with app.app_context():
    g.prediction_pipeline = PredictionPipeline(PARAMS_FILE_PATH)
    g.model = g.prediction_pipeline.return_trained_model()
    g.base_model = VGG16(input_shape=(500, 500, 3), include_top=False, weights='imagenet')
    g.vectorizer = g.prediction_pipeline.load_vectorizer()


def load_model_vectorizer():
    """Ensure model and vectorizer are available per request."""
    if not hasattr(g, "prediction_pipeline"):
        g.prediction_pipeline = PredictionPipeline(PARAMS_FILE_PATH)

    if not hasattr(g, "model") or not hasattr(g, "base_model"):
        g.model = g.prediction_pipeline.return_trained_model()
        g.base_model = VGG16(input_shape=(500, 500, 3), include_top=False, weights='imagenet')

    if not hasattr(g, "vectorizer"):
        g.vectorizer = g.prediction_pipeline.load_vectorizer()


@app.before_request
def before_request():
    """Ensure resources are loaded before handling requests."""
    g.mongo = mongo  # Store MongoDB instance in `g`
    load_model_vectorizer()


app.register_blueprint(main)
app.register_blueprint(auth)
app.register_blueprint(activity)

if __name__ == "__main__":
    app.run(debug=True)
