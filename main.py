import os
import uuid
import json
import subprocess
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import anthropic

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

jobs = {}
anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def get_video_duration(video_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
        capture_output=True, text=True, check=True
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def detect_clips_with_claude(duration: float) -> list[dict]:
    prompt = f"""A video is {duration:.0f} seconds long. Generate 3 short-form clip suggestions for TikTok/Reels/Shorts.

Since we don't have the transcript yet, create realistic clip suggestions spread across the video duration.

Respond ONLY with a valid JSON array:
[
  {{
    "title": "Catchy clip title",
    "start": 10.0,
    "end": 45.0,
    "caption": "Hook caption for social media (max 100 chars)",
    "reason": "Why this section likely works as a short",
    "hook": "Opening line to grab attention"
  }}
]

Space the clips evenly across the {duration:.0f} second video. Each clip should be 20-55 seconds."""

    message = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = message.content[0].text.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def cut_clip(video_path: str, start: float, end: float, output_path: str):
    duration = end - start
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", video_path,
        "-t", str(duration),
        "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black",
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-c:a", "aac", "-b:a", "128k",
        output_path
    ], check=True, capture_output=True)


def process_video(job_id: str, video_path: str):
    try:
        jobs[job_id]["status"] = "analyzing"
        duration = get_video_duration(video_path)
        clips = detect_clips_with_claude(duration)

        jobs[job_id]["status"] = "cutting"
        job_output_dir = OUTPUT_DIR / job_id
        job_output_dir.mkdir(exist_ok=True)

        results = []
        for i, clip in enumerate(clips):
            output_filename = f"clip_{i+1}.mp4"
            output_path = str(job_output_dir / output_filename)
            cut_clip(video_path, clip["start"], clip["end"], output_path)
            results.append({
                **clip,
                "duration": round(clip["end"] - clip["start"], 1),
                "download_url": f"/outputs/{job_id}/{output_filename}"
            })

        jobs[job_id]["status"] = "done"
        jobs[job_id]["clips"] = results

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
    finally:
        try:
            os.remove(video_path)
        except:
            pass


@app.get("/")
def root():
    return {"status": "ClipForge API is running"}


@app.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
        raise HTTPException(400, "Unsupported file format.")

    job_id = str(uuid.uuid4())
    video_path = str(UPLOAD_DIR / f"{job_id}_{file.filename}")

    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)

    jobs[job_id] = {"status": "queued", "clips": []}
    background_tasks.add_task(process_video, job_id, video_path)
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]
