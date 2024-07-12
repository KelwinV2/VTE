from typing import Annotated
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, HTTPException
from processing.videoProcessing import video_processer
from processing.imageProcessing import image_processer

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/upload")
async def create_upload_file(file: UploadFile = File(...)):
    content_type = file.content_type

    if (content_type[:content_type.index("/")] == "video"):
        try:
            print('processing video........')
            result = video_processer(file)
            print(result)
            return {"result": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Video processing error -> {e}')
    elif (content_type[:content_type.index("/")] == "image"):
        try:
            result = image_processer(file)
            return {"result": result}
        except:
            raise HTTPException(status_code=500, detail="Image processing error.")
    else:
        raise HTTPException(status_code=415, detail="Unsupported media type.")
