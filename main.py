from typing import Union
from fastapi import FastAPI

from fastapi import FastAPI, Request, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

# # Enable CORS (Cross-Origin Resource Sharing) to allow requests from frontend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allow all origins for simplicity, modify as needed
#     allow_credentials=True,
#     allow_methods=["GET", "POST"],
#     allow_headers=["*"],
# )


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    # Render HTML template
    return templates.TemplateResponse("index.html", {"request": request, "answer": ""})


@app.post("/api/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    # Perform prediction here with the uploaded file
    # Replace the following lines with your prediction logic
    result_percent = 85  # Example result percentage
    result_name = "Stop Sign"  # Example result name

    prediction_result = {
        "result_name": result_name,
        "result_percent": result_percent
    }

    # Return the prediction result as JSON response
    return JSONResponse(content=prediction_result)
