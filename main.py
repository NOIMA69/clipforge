import os
import uuid
import json
import subprocess
import tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import anthropic
import httpx
from faster_whisper import WhisperModel

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


def transcribe_video(video_path: str) -> list[dict]:
    """Use faster-whisper to transcribe video and get timestamps."""
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments_iter, _ = model.transcribe(video_path, beam_size=5)
    segments = []
    for seg in segments_iter:
        segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip()
        })
    return segments


def detect_clips_with_claude(segments: list[dict], duration: float) -> list[dict]:
    """Ask Claude to find the best short-form clips from the transcript."""
    transcript_text = "\n".join(
        f"[{s['start']:.1f}s - {s['end']:.1f}s] {s['text']}"
        for s in segments
    )

    prompt = f"""You are an expert video editor specializing in short-form content for TikTok, Instagram Reels, and YouTube Shorts.

Here is a transcript of a video (total duration: {duration:.0f} seconds) with timestamps:

{transcript_text}

Identify the 3-5 BEST moments to turn into short-form clips (15-60 seconds each). Pick moments that are:
- Self-contained and make sense without context
- Emotionally engaging, surprising, funny, or valuable
- Have a clear hook in the first few seconds

Respond ONLY with a valid JSON array like this:
[
  {{
    "title": "Short catchy title for the clip",
    "start": 12.5,
    "end": 45.0,
    "caption": "Hook caption for social media (max 100 chars)",
    "reason": "Why this clip works well as a short",
    "hook": "The first sentence to grab attention"
  }}
]"""

    message = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = message.content[0].text.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def cut_clip(video_path: str, start: float, end: float, output_path: str):
    """Use FFmpeg to cut a clip and convert to vertical 9:16 format."""
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
    """Full pipeline: transcribe → detect clips → cut → return results."""
    try:
        jobs[job_id]["status"] = "transcribing"

        # Get video duration
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
            capture_output=True, text=True, check=True
        )
        duration = float(json.loads(probe.stdout)["format"]["duration"])

        segments = transcribe_video(video_path)

        jobs[job_id]["status"] = "analyzing"
        clips = detect_clips_with_claude(segments, duration)

        jobs[job_id]["status"] = "cutting"
        results = []
        job_output_dir = OUTPUT_DIR / job_id
        job_output_dir.mkdir(exist_ok=True)

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
        # Clean up uploaded file
        try:
            os.remove(video_path)
        except:
            pass


@app.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
        raise HTTPException(400, "Unsupported file format. Use MP4, MOV, AVI, MKV, or WebM.")

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


@app.get("/download/{job_id}/{filename}")
def download_clip(job_id: str, filename: str):
    path = OUTPUT_DIR / job_id / filename
    if not path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(str(path), media_type="video/mp4", filename=filename)
