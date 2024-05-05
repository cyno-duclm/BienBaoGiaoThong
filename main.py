from typing import Union
from fastapi import FastAPI
import cv2
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

# from lib import Utils
from lib import traffic_predict_trained
from PIL import Image
import io
import numpy as np

app = FastAPI()
HOST = 'http://localhost:8000'


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    # Render HTML template
    return templates.TemplateResponse("index.html", {"request": request, "answer": ""})


@app.post("/api/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    result_name = "Stop Sign"
    result_percent = 85

    #  # Convert the uploaded file to an image
    image = Image.open(io.BytesIO(await file.read()))
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # # Call the predict function from Utils
    filename = traffic_predict_trained.utils_predict(image)
    result_img_url = f"{HOST}/static/result/{filename}"

    prediction_result = {
        "result_img_url": result_img_url
    }

    # Return the prediction result as JSON response
    return JSONResponse(content=prediction_result)
