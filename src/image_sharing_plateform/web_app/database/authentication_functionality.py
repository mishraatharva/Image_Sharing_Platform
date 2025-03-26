def register_user(mongo,user_data: dict) -> str:
    mongo.db.user_credentials.insert_one(user_data)
    return user_data["username"]

def get_user_login_data(mongo) -> str:
    pass