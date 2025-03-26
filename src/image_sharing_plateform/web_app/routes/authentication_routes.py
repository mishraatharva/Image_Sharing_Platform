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
from src.image_sharing_plateform.web_app.form.forms import LoginForm,SignupForm
from werkzeug.security import check_password_hash, generate_password_hash
from src.image_sharing_plateform.web_app.database.authentication_functionality import register_user
from datetime import datetime
auth = Blueprint("auth", __name__)


# mongo = PyMongo(auth)

@auth.route("/signup", methods=["GET", "POST"])
def signup():
    signup_form = SignupForm()
    login_form = LoginForm()
    if signup_form.validate_on_submit():
        from app import mongo  # Import `mongo` from app.py

        existing_user = g.mongo.db.user_credentials.find_one({"email": signup_form.email.data})
        if existing_user:
            flash("Email already registered!", "danger")
            return redirect(url_for("auth.signup"))

        # Hash password securely
        if signup_form.password.data == signup_form.confirm_password.data:
            hashed_pw = generate_password_hash(signup_form.password.data, method='pbkdf2:sha256')
        dob = datetime.strptime(str(signup_form.dob.data), "%Y-%m-%d")

        # Save user to MongoDB
        user_data = {"username": signup_form.username.data, "email": signup_form.email.data, "gender": signup_form.gender.data,"password": hashed_pw, "dob" : dob}
        username = register_user(mongo, user_data)

        flash(f"Successfully Registered {username}!", "success")
        return redirect(url_for("auth.login"))

    return render_template("signup.html", title="SignUp", form=signup_form)


@auth.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        from app import mongo
        user = mongo.db.user_credentials.find_one({"email": form.email.data})

        if user and check_password_hash(user["password"], form.password.data):
            session["user"] = user["email"]  # Store user session
            flash("Logged in Successfully!", "success")
            return redirect(url_for("activity.dash_board"))
        else:
            flash("Incorrect email or password", "danger")

    return render_template("login.html", title="Login", form=form)


@auth.route("/logout")
def logout():
    session.pop("user", None)
    flash("Logged out successfully!", "info")
    return redirect(url_for("auth.login"))


# from flask import session

# @app.route("/")
# def home():
#     if "user" in session:
#         return f"Welcome, {session['user']}! <a href='/logout'>Logout</a>"
#     return "Please <a href='/login'>login</a>."
