from flask import g
import gridfs
from datetime import datetime
import base64
import numpy as np

def upload_user_image_caption(file):
    fs = gridfs.GridFS(g.mongo.db, "fs")
    file_data = file.read()
    
    image_feature = g.prediction_pipeline.extract_features(filename=file, base_model=g.base_model)
    generated_caption = g.prediction_pipeline.generate_caption(g.model, image_feature , g.vectorizer)

    file_id = fs.put(file_data, filename=f"profile_pic_{datetime.utcnow().timestamp()}", metadata={"email_address": "atharva@gmail.com"},content_type=file.content_type)

    image_metadata = {
        "timestamp": datetime.utcnow(),
        "email_address": "atharva@gmail.com",
        "image_name": file.filename,
        "gridfs_id": file_id,
        "caption": generated_caption
        }
    g.mongo.db.user_uploaded_images.insert_one(image_metadata)

    
def get_all_user_data():
    fs = gridfs.GridFS(g.mongo.db)
    user_data = []

    metadata_docs = g.mongo.db.user_uploaded_images.find({"email_address": "atharva@gmail.com"})
    try:
        for doc in metadata_docs:
            file_id = doc["gridfs_id"]
            caption = doc["caption"]
            image_data = fs.get(file_id).read()
            
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            image_src = f"data:image/jpeg;base64,{image_base64}"

            user_data.append({"image_src": image_src, "caption": caption})

        return user_data
    except Exception as e:
        print("Error retrieving user data:", e)
        return []
