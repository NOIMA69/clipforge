import os
import uuid
import json
import subprocess
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import anthropic

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

jobs = {}
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def get_video_info(video_path: str) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", video_path],
        capture_output=True, text=True, check=True
    )
    data = json.loads(result.stdout)
    duration = float(data["format"]["duration"])
    return {"duration": duration}


def transcribe_video(video_path: str) -> dict:
    """Extract audio and transcribe using faster-whisper via subprocess."""
    audio_path = video_path.replace(".mp4", ".wav").replace(".mov", ".wav").replace(".mkv", ".wav")
    # Extract audio
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", audio_path
    ], check=True, capture_output=True)

    # Use whisper via python script
    script = f"""
import json, sys
from faster_whisper import WhisperModel
model = WhisperModel("tiny", device="cpu", compute_type="int8")
segments, info = model.transcribe("{audio_path}", beam_size=3)
result = {{"language": info.language, "segments": []}}
for seg in segments:
    result["segments"].append({{"start": seg.start, "end": seg.end, "text": seg.text.strip()}})
print(json.dumps(result))
"""
    result = subprocess.run(["python3", "-c", script], capture_output=True, text=True)
    try:
        os.remove(audio_path)
    except:
        pass
    if result.returncode != 0:
        return {"language": "en", "segments": []}
    return json.loads(result.stdout)


def analyze_with_claude(transcript_data: dict, duration: float) -> list:
    segments = transcript_data.get("segments", [])
    language = transcript_data.get("language", "unknown")

    if segments:
        transcript_text = "\n".join(f"[{s['start']:.1f}s-{s['end']:.1f}s] {s['text']}" for s in segments)
    else:
        transcript_text = f"No transcript available. Video is {duration:.0f} seconds long."

    prompt = f"""You are an expert viral content editor. Analyze this video transcript (language: {language}, duration: {duration:.0f}s) and find the 3 best moments for TikTok/Reels/Shorts.

TRANSCRIPT:
{transcript_text}

For each clip provide:
- A catchy title
- Start/end timestamps (each clip 20-55 seconds)
- A viral hook (first line to grab attention)
- A social media caption with emojis
- Why it will go viral
- Viral score 1-100 based on: hook strength, emotional impact, shareability, clarity

Respond ONLY with valid JSON array:
[
  {{
    "title": "Catchy title",
    "start": 10.0,
    "end": 45.0,
    "hook": "Opening line that grabs attention",
    "caption": "Social caption with emojis (max 100 chars)",
    "reason": "Why this goes viral",
    "viral_score": 87,
    "language": "{language}"
  }}
]"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = message.content[0].text.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def generate_subtitles(segments: list, start: float, end: float) -> list:
    """Filter segments that fall within clip time range and adjust timestamps."""
    clip_segments = []
    for seg in segments:
        if seg["end"] < start or seg["start"] > end:
            continue
        clip_segments.append({
            "start": max(0, seg["start"] - start),
            "end": min(end - start, seg["end"] - start),
            "text": seg["text"]
        })
    return clip_segments


def create_srt(segments: list, output_path: str):
    """Create SRT subtitle file."""
    def format_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{format_time(seg['start'])} --> {format_time(seg['end'])}\n")
            f.write(f"{seg['text']}\n\n")


def cut_clip_with_subtitles(video_path: str, start: float, end: float, output_path: str, segments: list, watermark: bool = False):
    duration = end - start
    srt_path = output_path.replace(".mp4", ".srt")

    # Create SRT file
    create_srt(segments, srt_path)

    # Build filter chain — crop to fill 9:16 (no black bars)
    filters = [
        "scale=720:1280:force_original_aspect_ratio=increase",
        "crop=720:1280"
    ]

    if segments:
        filters.append(f"subtitles={srt_path}:force_style='FontName=Arial,FontSize=18,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline=2,Bold=1,Alignment=2,MarginV=60'")

    if watermark:
        filters.append("drawtext=text='ClipForge':fontcolor=white:fontsize=24:alpha=0.6:x=(w-tw)/2:y=50")

    filter_str = ",".join(filters)

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", video_path,
        "-t", str(duration),
        "-vf", filter_str,
        "-c:v", "libx264", "-crf", "28", "-preset", "ultrafast",
        "-c:a", "aac", "-b:a", "128k",
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    try:
        os.remove(srt_path)
    except:
        pass


def process_video(job_id: str, video_path: str, watermark: bool = False):
    try:
        jobs[job_id]["status"] = "transcribing"
        info = get_video_info(video_path)
        duration = info["duration"]

        # Try transcription
        try:
            transcript_data = transcribe_video(video_path)
        except Exception as e:
            transcript_data = {"language": "en", "segments": []}

        jobs[job_id]["status"] = "analyzing"
        clips = analyze_with_claude(transcript_data, duration)

        jobs[job_id]["status"] = "cutting"
        job_output_dir = OUTPUT_DIR / job_id
        job_output_dir.mkdir(exist_ok=True)

        results = []
        for i, clip in enumerate(clips):
            output_filename = f"clip_{i+1}.mp4"
            output_path = str(job_output_dir / output_filename)

            clip_segments = generate_subtitles(
                transcript_data.get("segments", []),
                clip["start"], clip["end"]
            )

            cut_clip_with_subtitles(
                video_path, clip["start"], clip["end"],
                output_path, clip_segments, watermark
            )

            results.append({
                **clip,
                "duration": round(clip["end"] - clip["start"], 1),
                "download_url": f"/outputs/{job_id}/{output_filename}",
                "has_subtitles": len(clip_segments) > 0
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
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...), watermark: bool = False):
    if not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
        raise HTTPException(400, "Unsupported file format.")

    job_id = str(uuid.uuid4())
    video_path = str(UPLOAD_DIR / f"{job_id}_{file.filename}")

    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)

    jobs[job_id] = {"status": "queued", "clips": []}
    background_tasks.add_task(process_video, job_id, video_path, watermark)
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]
