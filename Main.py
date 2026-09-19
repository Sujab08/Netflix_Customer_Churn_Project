from fastapi import FastAPI
from pydantic import BaseModel 
import joblib
app=FastAPI()
model=joblib.load("Prediction_Model.joblib")
# ['customer_id', 'age', 'gender', 'subscription_type', 'watch_hours',
#        'last_login_days', 'region', 'device', 'monthly_fee', 'churned',
#        'payment_method', 'number_of_profiles', 'avg_watch_time_per_day',
#        'favorite_genre']
class pred(BaseModel):
    age:int
    subscription_type:float
    watch_hours:float
    last_login_days:int
    monthly_fee:float
    number_of_profiles:int
    avg_watch_time_per_day:float
    gender:str
    region:str
    device:str
    payment_method:str
    favorite_genre:str

feature_order = [
    "age",
    "subscription_type",
    "watch_hours",
    "last_login_days",
    "monthly_fee",
    "number_of_profiles",
    "avg_watch_time_per_day",
    "gender_Female",
    "gender_Male",
    "gender_Other",
    "region_Africa",
    "region_Asia",
    "region_Europe",
    "region_North America",
    "region_Oceania",
    "region_South America",
    "device_Desktop",
    "device_Laptop",
    "device_Mobile",
    "device_TV",
    "device_Tablet",
    "payment_method_Credit Card",
    "payment_method_Crypto",
    "payment_method_Debit Card",
    "payment_method_Gift Card",
    "payment_method_PayPal",
    "favorite_genre_Action",
    "favorite_genre_Comedy",
    "favorite_genre_Documentary",
    "favorite_genre_Drama",
    "favorite_genre_Horror",
    "favorite_genre_Romance",
    "favorite_genre_Sci-Fi"
]

def create_feature(data):
    features={
        "age": data.age,
        "subscription_type": data.subscription_type,
        "watch_hours": data.watch_hours,
        "last_login_days": data.last_login_days,
        "monthly_fee": data.monthly_fee,
        "number_of_profiles": data.number_of_profiles,
        "avg_watch_time_per_day": data.avg_watch_time_per_day,

        "gender_Female": 0,
        "gender_Male": 0,
        "gender_Other": 0,

        "region_Africa": 0,
        "region_Asia": 0,
        "region_Europe": 0,
        "region_North America": 0,
        "region_Oceania": 0,
        "region_South America": 0,

        "device_Desktop": 0,
        "device_Laptop": 0,
        "device_Mobile": 0,
        "device_TV": 0,
        "device_Tablet": 0,

        "payment_method_Credit Card": 0,
        "payment_method_Crypto": 0,
        "payment_method_Debit Card": 0,
        "payment_method_Gift Card": 0,
        "payment_method_PayPal": 0,

        "favorite_genre_Action": 0,
        "favorite_genre_Comedy": 0,
        "favorite_genre_Documentary": 0,
        "favorite_genre_Drama": 0,
        "favorite_genre_Horror": 0,
        "favorite_genre_Romance": 0,
        "favorite_genre_Sci-Fi": 0
    }
    features["gender_" + data.gender] = 1
    features["region_" + data.region] = 1
    features["device_" + data.device] = 1
    features["payment_method_" + data.payment_method] = 1
    features["favorite_genre_" + data.favorite_genre] = 1

    return [features[column] for column in feature_order]

@app.get("/")
def home():
    return {"Message":"ML projrct is working"}


@app.post("/predict")
def predict(data:pred):
    features=create_feature(data)
    prediction=model.predict([features])

    return {"prediction":int(prediction[0])}
    