from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
import os, shutil
from app.render import run_render_pipeline

app = FastAPI()

app.mount("/outputs", StaticFiles(directory="app/outputs"), name="outputs")

@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    video_path = run_render_pipeline(file_path)
    return {"video_url": f"/outputs/{os.path.basename(video_path)}"}



@app.post("/generate/")
async def generate(youtube_url: str):
    pkl_path = generate_dance_from_youtube(youtube_url)
    video_path = render_dance_to_video(pkl_path)
    return {"video_path": f"/outputs/{video_path}"}