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
import tensorflow as tf
import tensorflow.keras.applications as applications
from tensorflow.keras.models import load_model
import Build_Models.clahe as clahe
from joblib import load
import io
import os
import numpy as np
import pandas as pd
import face_recognition
from PIL import Image
import base64



app = FastAPI()
app.add_middleware(SessionMiddleware,secret_key="assesseionsecretkeys222")

app.mount("/static",StaticFiles(directory="static"), name="static")

# Models
heart_model = load("models/ckd_model.joblib")
kidney_model = load("models/ckd_model.joblib")
stroke_model = load("models/stroke_model.joblib")
copd_model = load("models/Copd.joblib")
tb_model = load("models/TB_model.joblib")

medical_or_not_model = load_model('models/medical_or_not_model.keras')
types_of_medical_model = load_model('models/types_of_medical_model.keras')
CT_categorys_model = load_model('models/CT_categors_model.keras')
chest_CT_Model = load_model('models/chest_ct_model.keras')
kidney_CT_Model = load_model('models/kidney_CT_model.keras')
X_ray_categorys_model = load_model('models/X_ray_category_model.keras')
chest_x_ray_model = load_model('models/chest_x_ray_model.keras')
elbow_x_ray_model = load_model('models/elbow_x_ray_model.keras')
foream_x_ray_model = load_model('models/foreaam_x_ray_model.keras')
hand_x_ray_model = load_model('models/hand_model.keras')
humerus_x_raay_model = load_model('models/humerus_model.keras')
brain_mri_model = load_model('models/Brain_tumor_model.keras')



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

@app.get("/copd_disease",response_class=HTMLResponse)
async def copd(request:Request):

    return templates.TemplateResponse(name="copd.html",request=request,context={'prediction':None})

@app.get("/td_disease",response_class=HTMLResponse)
async def td_disease(request:Request):

    return templates.TemplateResponse(request=request,name="TB_disease.html",context={"prediction":None})

@app.get("/image_diagnosis",response_class=HTMLResponse)
async def image_diagnosis(request:Request):

    return templates.TemplateResponse(request=request,name="image_diagnosis.html")
# MongoDB Connection

client = MongoClient("mongodb://localhost:27017")
db = client["medical_ai_database"]
user_collection = db["usersDataCollection"]

# Handle POST DATA

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
        check_email =  user_collection.find_one({"email":email})

        if check_email:
            return "emial is already existed"
        
        hash_pass = pass_context.hash(password)
        
        user_data = {
            "name":username,
            "email":email,
            "password":hash_pass,
            "face_encoding":encoded_face_data
        }

        user_collection.insert_one(user_data)

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
                        Gender : int = Form(...),
                        BMI : float = Form(...),
                        Hypertension : int = Form(...),
                        Diabetes : int = Form(...),
                        Heart_disease : int = Form(...),
                        Smoking : int = Form(...),
                        Alcohol_use : int = Form(...),
                        Cholesterol : float = Form(...),
                        Blood_glucose : float = Form(...),
                        Oxygen_saturation : float = Form(...),
                        Systolic_BP : int = Form(...),
                        Diastolic_BP : int = Form(...),
                        Arm_weakness : int = Form(...),
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

@app.post("/predict_copd")
async def predict_copd(request:Request,
                        Age : int = Form(...),
                        Gender : int = Form(...),
                        BMI : float = Form(...),
                        Smoking_Status : int = Form(...),
                        Pack_Years : int = Form(...),
                        Secondhand_Smoke_Exposure : int = Form(...),
                        Occupational_Dust_Exposure : int = Form(...),
                        Air_Pollution_Exposure : int = Form(...),
                        Family_History_COPD : int = Form(...),
                        Asthma : int = Form(...),
                        Hypertension : int = Form(...),
                        Diabetes : int = Form(...),
                        Chronic_Cough : int = Form(...),
                        Sputum_Production : int = Form(...),
                        Wheezing : int = Form(...),
                        Shortness_of_Breath : int = Form(...),
                        Oxygen_Saturation : int = Form(...),
                        Physical_Activity : int = Form(...),
                        FEV1_Percent_Predicted : float = Form(...)):
    
    columns = ['Age', 'Gender', 'BMI', 'Smoking_Status', 'Pack_Years','Secondhand_Smoke_Exposure', 'Occupational_Dust_Exposure','Air_Pollution_Exposure', 'Family_History_COPD', 'Asthma','Hypertension', 'Diabetes', 'Chronic_Cough', 'Sputum_Production','Wheezing', 'Shortness_of_Breath', 'Oxygen_Saturation','Physical_Activity', 'FEV1_Percent_Predicted']
    user_data = pd.DataFrame([[Age,Gender,BMI,Smoking_Status,Pack_Years,Secondhand_Smoke_Exposure,Occupational_Dust_Exposure,Air_Pollution_Exposure,Family_History_COPD,Asthma,Hypertension,Diabetes,Chronic_Cough,Sputum_Production,Wheezing,Shortness_of_Breath,Oxygen_Saturation,Physical_Activity,FEV1_Percent_Predicted]],columns=columns)
    prediction = copd_model.predict(user_data)
    return templates.TemplateResponse(request=request, name="copd.html", context={"prediction":prediction})

@app.post('/predict_tb')
async def predict_TB(request:Request,
                    Age : int = Form(...),
                    Gender : int = Form(...),
                    BMI : float = Form(...),
                    Smoking : int = Form(...),
                    Alcohol_Use : int = Form(...),
                    HIV : int = Form(...),
                    Diabetes : int = Form(...),
                    Previous_TB : int = Form(...),
                    Household_TB_Contact : int = Form(...),
                    Persistent_Cough : int = Form(...),
                    Cough_Duration_Days : int = Form(...),
                    Blood_in_Sputum : int = Form(...),
                    Chest_Pain : int = Form(...),
                    Fever : int = Form(...),
                    Night_Sweats : int = Form(...),
                    Weight_Loss : int = Form(...),
                    Fatigue : int = Form(...),
                    Appetite_Loss : int = Form(...),
                    Oxygen_Saturation : int = Form(...),
                    ESR : float = Form(...),
                    WBC_Count : float = Form(...),
                    Chest_Xray_Result : int = Form(...),
                    Socioeconomic_Status : int = Form(...),
                    GeneXpert_Result : str = Form(...)
    ):
    encoder = load("models/GeneXpert_Result_encoder.joblib")
    scaler = load("models/TB_scaler.joblib")
    columns = ['Age', 'Gender', 'BMI', 'Smoking', 'Alcohol_Use', 'HIV', 'Diabetes','Previous_TB', 'Household_TB_Contact', 'Persistent_Cough','Cough_Duration_Days', 'Blood_in_Sputum', 'Chest_Pain', 'Fever','Night_Sweats', 'Weight_Loss', 'Fatigue', 'Appetite_Loss','Oxygen_Saturation', 'ESR', 'WBC_Count', 'Chest_Xray_Result','Socioeconomic_Status']
    WBC_Count *= 1000
    encoder_data = encoder.transform(pd.DataFrame([[GeneXpert_Result]],columns=["GeneXpert_Result"]))
    encoder_df = pd.DataFrame(encoder_data,columns=encoder.get_feature_names_out())
    user_data = pd.DataFrame([[Age,Gender,BMI,Smoking,Alcohol_Use,HIV,Diabetes,Previous_TB,Household_TB_Contact,Persistent_Cough,Cough_Duration_Days,Blood_in_Sputum,Chest_Pain,Fever,Night_Sweats,Weight_Loss,Fatigue,Appetite_Loss,Oxygen_Saturation,ESR,WBC_Count,Chest_Xray_Result,Socioeconomic_Status]],columns=columns)
    user_data = pd.concat([user_data,encoder_df],axis=1)
    scaler_data = scaler.transform(user_data)
    prediction = tb_model.predict(scaler_data)
    prediction = (prediction > 0.5).astype(int)

    return templates.TemplateResponse(request=request,name="TB_disease.html",context={"prediction":prediction})

@app.post("/image_diagnosis")
async def image_ditection(
    request : Request,
    imageInput: UploadFile = File(...)
):  
                        # models

    image_bytes = await imageInput.read()

    image_url = (
        "data:"
        + imageInput.content_type
        + ";base64,"
        + base64.b64encode(image_bytes).decode("utf-8")
    )

    image = tf.io.decode_image(image_bytes,channels=3,expand_animations=False)

    mediacl_or_not_image = tf.image.resize(image, (224, 224))

    mediacl_or_not_image = tf.expand_dims(mediacl_or_not_image, axis=0)

    preprocess_img = applications.efficientnet_v2.preprocess_input(mediacl_or_not_image)

    medical_or_not_prob = medical_or_not_model.predict(preprocess_img)

    medical_or_not_pred = (medical_or_not_prob >= 0.5).astype(int)

    if medical_or_not_pred[0][0] == 0:
        predication = types_of_medical_model.predict(preprocess_img)
        predication = np.argmax(predication,axis=1)
        print(predication)

        if predication[0] == 0:
            print('CT')
            CT_prediction = CT_categorys_model.predict(preprocess_img)
            CT_prediction = (CT_prediction >= 0.5).astype(int)
            print(CT_prediction)

            if CT_prediction[0][0] == 0:
                print('Chest_CT')
                pred = chest_CT_Model.predict(preprocess_img)
                probability = float(pred[0][0])
                
                if probability >= 0.5:
                    label = "Normal"
                    confidence = probability

                else:
                    label = "Abnormal"
                    confidence = 1 - probability
                print(confidence,label)

            else:
                print('Kidney_ct')
                pred = kidney_CT_Model.predict(preprocess_img)
                pred = (pred >=0.5).astype(int)
                probability = float(pred[0][0])
                                
                if probability >= 0.5:
                    label = "Normal"
                    confidence = probability

                else:
                    label = "Abnormal"
                    confidence = 1 - probability
                print(confidence,label)


        elif predication[0] == 1:
            print("MRI")

            pred = brain_mri_model.predict(preprocess_img)
            probability = float(pred[0][0])
                            
            if probability >= 0.5:
                label = "Normal"
                confidence = probability

            else:
                label = "Abnormal"
                confidence = 1 - probability
            print(confidence,label)


        else:
            print('X-Ray')
            x_ray_predication = X_ray_categorys_model.predict(preprocess_img)
            x_ray_predication = np.argmax(x_ray_predication,axis=1)

            if x_ray_predication[0] == 0:
                print('chest x-ray')

                mediacl_or_not_image = mediacl_or_not_image.numpy()

                img = clahe.clahe_preprocess(mediacl_or_not_image.squeeze(),applications.efficientnet_v2.preprocess_input)
                img = np.expand_dims(img,axis=0)

                pred = chest_x_ray_model.predict(img)
                probability = float(pred[0][0])

                if probability >= 0.5:
                    label = "Normal"
                    confidence = probability

                else:
                    label = "Abnormal"
                    confidence = 1 - probability
                print(confidence,label)

            elif x_ray_predication[0] == 1:

                print('elbow')

                mediacl_or_not_image = mediacl_or_not_image.numpy()
                
                img = clahe.clahe_preprocess(mediacl_or_not_image.squeeze(),applications.mobilenet_v3.preprocess_input)
                img = np.expand_dims(img,axis=0)

                pred = elbow_x_ray_model.predict(img)
                probability = float(pred[0][0])

                if probability >= 0.5:
                    label = "Normal"
                    confidence = probability

                else:
                    label = "Abnormal"
                    confidence = 1 - probability
                print(confidence,label)

            elif x_ray_predication[0] == 2:
                print('forearm')
                img = tf.image.resize(image, (256,256))
                mediacl_or_not_image = img.numpy()

                img = clahe.clahe_preprocess(mediacl_or_not_image,applications.efficientnet_v2.preprocess_input)
                img = np.expand_dims(img,axis=0)
                pred = foream_x_ray_model.predict(img)
                probability = float(pred[0][0])

                if probability >= 0.5:
                    label = "Normal"
                    confidence = probability

                else:
                    label = "Abnormal"
                    confidence = 1 - probability
                print(confidence,label)

            elif x_ray_predication[0] == 3:
                print('hand')
                mediacl_or_not_image = mediacl_or_not_image.numpy()
                
                img = clahe.clahe_preprocess(mediacl_or_not_image.squeeze(),applications.densenet.preprocess_input)
                img = np.expand_dims(img,axis=0)
                pred = hand_x_ray_model.predict(img)
                probability = float(pred[0][0])

                if probability >= 0.5:
                    label = "Normal"
                    confidence = probability

                else:
                    label = "Abnormal"
                    confidence = 1 - probability
                print(confidence,label)

            elif x_ray_predication[0] == 4:
                print('humerus')
                mediacl_or_not_image = mediacl_or_not_image.numpy()

                img = clahe.clahe_preprocess(mediacl_or_not_image.squeeze(),applications.efficientnet_v2.preprocess_input)
                img = np.expand_dims(img,axis=0)
                pred = humerus_x_raay_model.predict(img)
                probability = float(pred[0][0])

                if probability >= 0.5:
                    label = "Normal"
                    confidence = probability

                else:
                    label = "Abnormal"
                    confidence = 1 - probability
                print(confidence,label)

            else:
                print('shoulder')
                img = clahe.clahe_preprocess(mediacl_or_not_image,applications.efficientnet_v2.preprocess_input)

#             {'chest x-ray': 0,
            #  'elbow': 1,
            #  'forearm': 2,
            #  'hand': 3,
            #  'humerus': 4,
            #  'shoulder': 5}
            # {'CT': 0, 'MRI': 1, 'X-Ray': 2}
        return  templates.TemplateResponse(request=request,name='result.html', context={"result": {
            "medical": True,
            "model2Prediction": label,
            "model2Confidence": confidence * 100
        },
        "image_url": image_url
        })


    else:
        print('Non medical')

        return templates.TemplateResponse(request=request, name = 'result.html', context={"result": {"medical": False},"image_url": image_url} )
    



     