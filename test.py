
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()


@app.post("/predict/")
async def predict(muzzle0: UploadFile):
    if file.content_type.startswith("image/"):
        image_bytes = await file.read()
        # predicted_class = predict_image(image_bytes)
        return print("Image recieved");
        
    else:
        return JSONResponse(content={"error": "Invalid file format"}, status_code=400)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)