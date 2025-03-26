from src.image_sharing_plateform.web_app.form.forms import LoginForm
from src.image_sharing_plateform.web_app.database.user_activity_functionality import upload_user_image_caption , get_all_user_data
from datetime import datetime
import numpy as np
import gridfs
from flask import (
    Flask,
    render_template,
    redirect,
    url_for,
    flash,
    Blueprint,
    request,
    session,
    g
)
activity = Blueprint("activity", __name__)


@activity.route("/dashboard", methods=["GET", "POST"])
def dash_board():
    login_form = LoginForm()
    if 'user' in session:
        print("--------------------------------------------------------------------------------------")
        print(session.keys())
        user_data = get_all_user_data()
        print("--------------------------------------------------------------------------------------")

        return render_template("dashboard.html" ,title="User Dashboard", user_data=user_data)
    else:
        flash("login first...")
        return redirect(url_for("auth.login"))


# @activity.route("/dashboard", methods=["GET", "POST"])
# def dash_board():
#     user_data = get_all_user_data()
#     return render_template("dashboard.html", user_data=user_data)


@activity.route("/upload_image", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        flash("No file uploaded")
        return redirect(request.url)

    file = request.files["file"]
    upload_user_image_caption(file)
    user_data = get_all_user_data()

    # return redirect(request.url, user_data)
    return render_template("dashboard.html", user_data=user_data)