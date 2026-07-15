from fastapi import FastAPI, Form, File,UploadFile
from typing import Annotated
from fastapi.responses import HTMLResponse
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pymongo import MongoClient
from passlib.context import CryptContext
from starlette.middleware.sessions import SessionMiddleware
from joblib import load
import os
import numpy as np
import pandas as pd
import face_recognition


app = FastAPI()
app.add_middleware(SessionMiddleware,secret_key="assesseionsecretkeys222")

app.mount("/static",StaticFiles(directory="static"), name="static")

# Models
heart_model = load("Build Models/Heart_model.joblib")
kidney_model = load("Build Models/ckd_model.joblib")
stroke_model = load("Build Models/stroke_model.joblib")
copd_model = load("Build Models/Copd.joblib")
tb_model = load("Build Models/Copd.joblib")


templates = Jinja2Templates(directory="templates")
pass_context = CryptContext(schemes=["argon2"])



# Show HTML page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
    request=request,
    name="index.html"
    )
@app.get("/home")
async def home(request:Request):
    email = request.session.get("email")
    user =  user_collection.find_one({"email":email})

    return templates.TemplateResponse(
        request=request,
        name="home.html",
        context={
            "name": user["name"],
            "email": user["email"]
        }
    )

@app.get("/login_page", response_class=HTMLResponse)
async def read_login(request: Request):

    return templates.TemplateResponse(name="login.html",request=request)

@app.get ("/face_login", response_class=HTMLResponse)
async def read_login_with_face(request:Request):

    return templates.TemplateResponse(name="face_login.html",request=request)

@app.get("/register_page",response_class=HTMLResponse)
async def read_register(request:Request):

    return templates.TemplateResponse(name="register.html",request=request)

@app.get("/disease_prediction", response_class=HTMLResponse)
async def disease_prediction(request:Request):

    return templates.TemplateResponse(name="dissease_prediction.html",request=request)

@app.get("/heart_disease",response_class=HTMLResponse)
async def heart_disease(request:Request):

    return templates.TemplateResponse(name="heart_disease.html",request=request,context={"prediction": None})

@app.get("/kidney_disease",response_class=HTMLResponse)
async def kidney_disease(request:Request):

    return templates.TemplateResponse(name="kidney_disease.html",request=request,context={"prediction":None})

@app.get("/stroke", response_class=HTMLResponse)
async def stroke(requerst:Request):

    return templates.TemplateResponse(name="stroke.html",request=requerst,context={"prediction":None})
# MongoDB Connection

client = MongoClient("mongodb://localhost:27017")
db = client["medical_ai_database"]
user_collection = db["usersDataCollection"]

# Handle form submission
@app.post("/")
def welcome():
    return "hello "

@app.post("/register")
async def register_data(
    username: str = Form(...),
    email: str = Form(...),
    password: str =Form(...),
    file: UploadFile = File(...),
):
    
    image = face_recognition.load_image_file(file.file)
    face_detect = face_recognition.face_locations(image)
    if len(face_detect) <= 0:
        return "face not find"
    elif len(face_detect) > 1:
        return "there are many faces"
    else :
        encoded_face_data = face_recognition.face_encodings(image,face_detect)
        
        encoded_face_data = encoded_face_data[0].tolist()
        check_email = await user_collection.find_one({"email":email})

        if check_email:
            return "emial is already existed"
        
        hash_pass = pass_context.hash(password)
        
        user_data = {
            "name":username,
            "email":email,
            "password":hash_pass,
            "face_encoding":encoded_face_data
        }

        await user_collection.insert_one(user_data)

        return "done"
    
@app.post("/login")
async def login(request:Request, email: str = Form(...),password: str = Form(...)):
    check_email =  user_collection.find_one({"email":email})
    if check_email:
        valid_password =  pass_context.verify(password,check_email["password"])
        if valid_password:
            request.session["email"] = check_email["email"]
            return RedirectResponse("/home", status_code=303)
        else:
            return "passowrd is not correct"
    else:
        return "invalid email"
        


    
@app.post("/face_login")

async def face_data_login(
    file : UploadFile = File(...)
):
    image = face_recognition.load_image_file(file.file)

    encode = face_recognition.face_encodings(image)
    if len(encode) ==0:
        return {
            "match" : False,
            "message": "no face detected"
        }
    
    unknown_encode = encode[0]
    users =  user_collection.find()

    best_match = None
    min_distance = 999

    for user in users:

        db_encoding = np.array(
            user["face_encoding"],
            dtype=np.float64
        )

        distance = face_recognition.face_distance(
            [db_encoding],
            unknown_encode
        )[0]

        if distance < min_distance:
            min_distance = distance
            best_match = user

    if best_match and min_distance < 0.7:

        return {
            "match": True,
            "user_id": str(best_match["_id"]),
            "name": best_match["name"]
        }

    return {
        "match": False,
        "message": "Face not recognized"
    }

def convertion_yes_no(data):
    if data.lower() == "yes":
        return 1
    elif data.lower() == "no":
        return 0
    



@app.post("/predict_heart_disease")
async def predict_heart_disease(
    request:Request,
    Chest_Pain: str = Form(...),
    Shortness_of_Breath: str = Form(...),
    Fatigue: str =Form(...),
    Palpitations: str = Form(...),
    Dizziness: str = Form(...),
    Swelling: str =Form(...),
    Pain_Arms_Jaw_Back: str = Form(...),
    Cold_Sweats_Nausea: str = Form(...),
    High_BP: str =Form(...),
    High_Cholesterol: str = Form(...),
    Diabetes: str = Form(...),
    Smoking: str =Form(...),
    Obesity: str = Form(...),
    Sedentary_Lifestyle: str = Form(...),
    Family_History: str =Form(...),
    Chronic_Stress: str = Form(...),
    Gender: str = Form(...),
    Age: int =Form(...),
    
):
    if Gender == "male":
        Gender = 1
    else:
        Gender = 0

    Chest_Pain = convertion_yes_no(Chest_Pain)
    Shortness_of_Breath = convertion_yes_no(Shortness_of_Breath)
    Fatigue = convertion_yes_no(Fatigue)
    Palpitations = convertion_yes_no(Palpitations)
    Dizziness = convertion_yes_no(Dizziness)
    Swelling = convertion_yes_no(Swelling)
    Pain_Arms_Jaw_Back = convertion_yes_no(Pain_Arms_Jaw_Back)
    Cold_Sweats_Nausea = convertion_yes_no(Cold_Sweats_Nausea)
    High_BP = convertion_yes_no(High_BP)
    High_Cholesterol = convertion_yes_no(High_Cholesterol)
    Diabetes = convertion_yes_no(Diabetes)
    Smoking = convertion_yes_no(Smoking)
    Obesity = convertion_yes_no(Obesity)
    Sedentary_Lifestyle = convertion_yes_no(Sedentary_Lifestyle)
    Family_History = convertion_yes_no(Family_History)
    Chronic_Stress = convertion_yes_no(Chronic_Stress)

    columns = ['Chest_Pain', 'Shortness_of_Breath', 'Fatigue', 'Palpitations', 'Dizziness', 'Swelling', 'Pain_Arms_Jaw_Back', 'Cold_Sweats_Nausea', 'High_BP', 'High_Cholesterol', 'Diabetes', 'Smoking', 'Obesity', 'Sedentary_Lifestyle', 'Family_History', 'Chronic_Stress', 'Gender', 'Age']

    user_data = pd.DataFrame([[Chest_Pain,Shortness_of_Breath,Fatigue,Palpitations,Dizziness,Swelling,Pain_Arms_Jaw_Back,Cold_Sweats_Nausea,High_BP,High_Cholesterol,Diabetes,Smoking,Obesity,Sedentary_Lifestyle,Family_History,Chronic_Stress,Gender,Age]], columns=columns)
    prediction = heart_model.predict(user_data)[0]
    print((prediction))
    return templates.TemplateResponse(
        name = "heart_disease.html",
        request= request,
        context={  
            "prediction": prediction
        }
    )

@app.post("/predict_ckd")
async def predict_ckd(request:Request,
                      age : int = Form(...),
                      gender : str = Form(...),
                      bmi : float = Form(...),
                      systolic_bp : int = Form(...),
                      diastolic_bp : int = Form(...),
                      diabetes : int = Form(...),
                      hypertension : int = Form(...),
                      serum_creatinine : float = Form(...),
                      blood_urea : float = Form(...),
                      egfr : float = Form(...),
                      albumin : int = Form(...),
                      hemoglobin : float = Form(...),
                      sodium : float = Form(...),
                      potassium : float = Form(...),
                      urine_protein : int = Form(...),
                      smoking : int = Form(...),
                      heart_disease : int = Form(...),
                      edema : int = Form(...),
                      fatigue : int = Form(...), 
                      anemia : int = Form(...)
                      ):
    
    if gender.lower() == "male":
        gender = 1
    else:
        gender = 0
    columns = ['age', 'gender', 'bmi', 'systolic_bp', 'diastolic_bp', 'diabetes','hypertension', 'serum_creatinine', 'blood_urea', 'egfr', 'albumin','hemoglobin', 'sodium', 'potassium', 'urine_protein', 'smoking','heart_disease', 'edema', 'fatigue', 'anemia']
    user_data = pd.DataFrame([[age, gender, bmi, systolic_bp, diastolic_bp, diabetes, hypertension, serum_creatinine, blood_urea, egfr, albumin,hemoglobin, sodium, potassium, urine_protein, smoking,heart_disease, edema, fatigue, anemia]],columns=columns)
    prediction = kidney_model.predict(user_data)
    return templates.TemplateResponse(
        name="kidney_disease.html",
        request=request,
        context={
            "prediction": prediction
        }
    )
@app.post("/predict_stroke")
async def predict_stroke(request:Request,
                        Age : int = Form(...),
                        Gender : str = Form(...),
                        BMI : float = Form(...),
                        Hypertension : int = Form(...),
                        Diabetes : int = Form(...),
                        Heart_disease : int = Form(...),
                        Smoking : int = Form(...),
                        Alcohol_use : float = Form(...),
                        Cholesterol : float = Form(...),
                        Blood_glucose : float = Form(...),
                        Oxygen_saturation : int = Form(...),
                        Systolic_BP : float = Form(...),
                        Diastolic_BP : float = Form(...),
                        Arm_weakness : float = Form(...),
                        Leg_weakness : int = Form(...),
                        Speech_difficulty : int = Form(...),
                        Confusion : int = Form(...),
                        Vision_loss : int = Form(...),
                        Severe_headache : int = Form(...), 
                        Family_history : int = Form(...),
                        Previous_TIA : int = Form(...)
                            ):
    
    columns = ['Age', 'Gender', 'BMI', 'Hypertension', 'Diabetes', 'Heart_disease', 'Smoking', 'Alcohol_use', 'Cholesterol', 'Blood_glucose', 'Oxygen_saturation', 'Systolic_BP', 'Diastolic_BP', 'Arm_weakness', 'Leg_weakness', 'Speech_difficulty', 'Confusion', 'Vision_loss','Severe_headache', 'Family_history', 'Previous_TIA']
    user_data = pd.DataFrame([[Age,Gender,BMI,Hypertension,Diabetes,Heart_disease,Smoking,Alcohol_use,Cholesterol,Blood_glucose,Oxygen_saturation,Systolic_BP,Diastolic_BP,Arm_weakness,Leg_weakness,Speech_difficulty,Confusion,Vision_loss,Severe_headache,Family_history,Previous_TIA]],columns=columns)
    prediction = stroke_model.predict(user_data)
    return templates.TemplateResponse(name="stroke.html",request=request,context={"prediction":prediction})




     