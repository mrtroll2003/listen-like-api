# api/index.py
import os
import sys
import tempfile
import shutil
import functools
import mimetypes
import asyncio
from pathlib import Path
import logging
import datetime
from pydantic import BaseModel, HttpUrl, Field, field_validator, ValidationError
from typing import Optional, List, Dict, Any
import re
from urllib.parse import urlparse
import json
import wave
import contextlib

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# --- Library Imports ---
try:
    import moviepy.editor as mp
    MOVIEPY_AVAILABLE = True
except ImportError:
    logging.warning("MoviePy library not found. File upload processing will be limited.")
    MOVIEPY_AVAILABLE = False
    mp = None # Placeholder

try:
    import google.generativeai as genai
    from google.api_core import exceptions as google_api_exceptions
    GEMINI_AVAILABLE = True
except ImportError:
    logging.error("Google Generative AI library not found. pip install google-generativeai")
    GEMINI_AVAILABLE = False
    genai = None
    google_api_exceptions = None

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Media Processing API",
    description="API for transcribing and translating video/audio content.",
    version="1.0.0"
)

# --- CORS Setup ---
origins = [
    "http://localhost", # Base localhost for Flutter web dev
    "http://localhost:8080", # Common Flutter web server port
    # Add the specific port Flutter uses when you run `flutter run -d chrome --web-port YOUR_PORT`
    # e.g., "http://localhost:54321" if you see that port in Flutter logs
    "https://listen-like.onrender.com", 
    "https://listen-like.onrender.com/",  # Your PRODUCTION Flutter frontend on Render
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True, # Important if you ever use cookies/auth headers
    allow_methods=["GET", "POST", "OPTIONS"], # Specify methods used
    allow_headers=["*"], # Or be more specific: ["Content-Type", "Authorization"]
)

# --- Google Gemini Setup ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GEMINI_MODEL_NAME = "gemini-1.5-pro" # Or "gemini-1.5-pro-latest" for potentially better results
gemini_model_instance = None

if not GEMINI_AVAILABLE:
    logger.error("Gemini library not available. API functionality will be severely limited.")
elif not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY environment variable not set. Gemini API calls will fail.")
else:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model_instance = genai.GenerativeModel(GEMINI_MODEL_NAME)
        logger.info(f"Google Gemini client configured successfully with model: {GEMINI_MODEL_NAME}.")
    except Exception as e:
        logger.error(f"Failed to configure Google Gemini: {e}", exc_info=True)
        gemini_model_instance = None # Ensure it's None if setup fails


# --- Pydantic Models ---
class TranslationRequest(BaseModel):
    text_to_translate: str
    target_language: str
    is_json_segments: bool = False

class YouTubeTranscribeRequest(BaseModel):
    youtube_url: HttpUrl

    @field_validator('youtube_url')
    @classmethod
    def check_youtube_url(cls, value: HttpUrl):
        host = value.host.lower() if value.host else ''
        is_youtube_domain = host in ['youtube.com', 'www.youtube.com', 'm.youtube.com', 'youtu.be']
        if not is_youtube_domain:
            raise ValueError('URL must be a valid YouTube domain.')
        if host == 'youtu.be' and (not value.path or len(value.path) <= 1):
             raise ValueError('youtu.be URL must have a video ID path.')
        if 'youtube.com' in host and value.path not in ['/watch'] and \
           not (value.path or '').startswith(('/shorts/', '/embed/')):
             if not value.query or 'v' not in value.query_params: # type: ignore
                  raise ValueError('youtube.com URL must be a watch, shorts, or embed link with video ID.')
        return value

class QuestionGenerationRequest(BaseModel):
    transcript_json: str # Expecting JSON string of timestamped segments
    num_questions: int = Field(default=7, ge=3, le=15)
    question_types: Optional[List[str]] = None
    custom_prompt_instructions: Optional[str] = None

# --- Helper Functions ---

async def extract_audio_moviepy(video_path: str, audio_output_path: str) -> None:
    if not MOVIEPY_AVAILABLE or not mp:
        raise RuntimeError("MoviePy library not available for audio extraction.")
    video_clip = None
    audio_clip = None
    loop = asyncio.get_running_loop()
    try:
        logger.info(f"MoviePy: Loading video clip from {video_path}")
        video_clip = await loop.run_in_executor(None, mp.VideoFileClip, video_path)
        if video_clip.audio is None:
             raise ValueError("Video file contains no audio stream.")
        audio_clip = video_clip.audio
        logger.info(f"MoviePy: Writing audio to {audio_output_path} (16kHz mono WAV)")
        write_func_partial = functools.partial(
            audio_clip.write_audiofile, audio_output_path,
            fps=16000, nbytes=2, codec='pcm_s16le',
            ffmpeg_params=["-ac", "1"], logger=None # Suppress moviepy progress bar
        )
        await loop.run_in_executor(None, write_func_partial)
        logger.info("MoviePy: Audio extraction successful.")
    except Exception as e:
        logger.error(f"MoviePy: Error during audio extraction: {e}", exc_info=True)
        raise RuntimeError(f"MoviePy audio extraction failed.")
    finally:
        if audio_clip: await loop.run_in_executor(None, getattr(audio_clip, 'close', lambda: None))
        if video_clip: await loop.run_in_executor(None, getattr(video_clip, 'close', lambda: None))


async def transcribe_via_gemini(source_uri_or_path: str, is_youtube_uri: bool = False) -> str:
    if not gemini_model_instance:
        raise RuntimeError("Gemini API client not configured.")

    input_part: Optional[types.Part] = None
    prompt_text = """Please provide a precise transcription for the following video/audio in JSON format.
The JSON should be an array of objects. Each object MUST have these keys:
- "start": The start time of the spoken segment in SECONDS (float or integer, e.g., 4.672).
- "end": The end time of the spoken segment in SECONDS (float or integer, e.g., 5.832).
- "text": The transcribed text for that segment (string).
Ensure the entire output is ONLY the valid JSON array, starting with '[' and ending with ']'.
Do not include markdown fences like ```json or any other introductory/explanatory text.
Example segment: {"start": 4.672, "end": 5.832, "text": "All pilots, prepare to sortie."}"""

    try:
        if is_youtube_uri:
            logger.info(f"Gemini: Using YouTube URI for transcription: {source_uri_or_path}")
            input_part = types.Part.from_uri(file_uri=source_uri_or_path, mime_type="video/youtube")
        else:
            mime_type, _ = mimetypes.guess_type(source_uri_or_path)
            if not mime_type or not (mime_type.startswith("audio/") or mime_type.startswith("video/")):
                mime_type = "application/octet-stream" # Fallback
            logger.info(f"Gemini: Uploading local file: {source_uri_or_path} (MIME: {mime_type})")
            loop = asyncio.get_running_loop()
            uploaded_file_obj = await loop.run_in_executor(
                None, genai.upload_file, source_uri_or_path, mime_type=mime_type
            )
            input_part = uploaded_file_obj
            logger.info(f"Gemini: Local file uploaded. URI: {getattr(input_part, 'uri', 'N/A')}")

        if not input_part:
            raise RuntimeError("Failed to prepare input part for Gemini.")

        logger.info("Gemini: Sending transcription request...")
        loop = asyncio.get_running_loop()
        response = await gemini_model_instance.generate_content_async(
            contents=[input_part, prompt_text], # Correct order for some models
            # Forcing JSON output if model supports structured output (e.g., Gemini 1.5 Pro)
            # generation_config=genai.GenerationConfig(response_mime_type="application/json")
            # For Flash, relying on prompt for JSON structure.
        )

        if not response.candidates:
            block_reason = getattr(response.prompt_feedback, 'block_reason', None)
            reason = f"Blocked: {block_reason}" if block_reason else "No response candidates"
            raise RuntimeError(f"Transcription failed ({reason})")

        transcription_json_string = response.text.strip()
        logger.info("Gemini: Transcription response received.")
        # logger.debug(f"Raw Gemini Transcription: {transcription_json_string}")

        try:
            cleaned_json = transcription_json_string.removeprefix("```json").removesuffix("```").strip()
            json.loads(cleaned_json) # Validate JSON
            return cleaned_json
        except json.JSONDecodeError as json_err:
            logger.error(f"Gemini response not valid JSON: {json_err}. Text: {transcription_json_string}")
            raise RuntimeError("Transcription model did not return valid JSON.")

    except (google_api_exceptions.PermissionDenied, google_api_exceptions.ResourceExhausted) as e:
        status = "Permission Denied" if isinstance(e, google_api_exceptions.PermissionDenied) else "Quota Exceeded"
        logger.error(f"Gemini API {status}: {e}", exc_info=False) # exc_info=False for less verbose logs
        raise RuntimeError(f"Transcription failed: Gemini API {status.lower()}.")
    except Exception as e:
        logger.error(f"Gemini transcription processing error: {e}", exc_info=True)
        raise RuntimeError(f"Unexpected error during Gemini transcription.")


async def _translate_single_segment(text: str, target_language: str) -> Optional[str]:
    if not gemini_model_instance or not text: return None
    try:
        prompt = f"Translate ONLY the following text into {target_language}. Return nothing but the translated text itself, no extra phrases:\n\n{text}"
        response = await gemini_model_instance.generate_content_async(prompt)
        return response.text.strip() if response.text else None
    except Exception as e:
        logger.error(f"Gemini segment translation error for target '{target_language}': {e}", exc_info=False)
        return None


# --- API Endpoints ---
@app.get("/api", tags=["Status"])
async def api_root():
    return JSONResponse(content={
        "message": "Media Processing API is running!",
        "status": "ok",
        "timestamp_utc": datetime.datetime.utcnow().isoformat(),
        "gemini_configured": gemini_model_instance is not None,
        "moviepy_available": MOVIEPY_AVAILABLE
    })

@app.post("/api/transcribe", tags=["Transcription"])
async def transcribe_media(
    file: Optional[UploadFile] = File(None, description="Video/audio file to transcribe"),
    payload: Optional[YouTubeTranscribeRequest] = Body(None, description="JSON payload with 'youtube_url'")
):
    if not gemini_model_instance:
        raise HTTPException(status_code=503, detail="Transcription service (Gemini) not configured.")
    if file and payload:
         raise HTTPException(status_code=400, detail="Provide EITHER a file OR a youtube_url, not both.")

    source_description = "Unknown"
    transcription_json_str: Optional[str] = None

    if file:
        if not MOVIEPY_AVAILABLE:
            raise HTTPException(status_code=501, detail="File processing (MoviePy) not available on server.")
        if not file.filename:
             raise HTTPException(status_code=400, detail="No filename provided with file upload.")
        source_description = f"File: {file.filename}"
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="api_upload_")
        temp_dir = temp_dir_obj.name
        try:
            safe_filename = Path(file.filename).name
            video_path = os.path.join(temp_dir, safe_filename)
            with open(video_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer) # More robust way to copy file
            await file.close()

            base_name, _ = os.path.splitext(safe_filename)
            audio_path = os.path.join(temp_dir, f"{base_name}_extracted.wav")
            await extract_audio_moviepy(video_path, audio_path)

            if not os.path.exists(audio_path):
                 raise HTTPException(status_code=500, detail="Audio extraction failed to produce a file.")
            transcription_json_str = await transcribe_via_gemini(audio_path, is_youtube_uri=False)
        except (RuntimeError, ValueError) as e:
             raise HTTPException(status_code=500, detail=f"Error processing file: {e}")
        finally:
            temp_dir_obj.cleanup()
            logger.info(f"Cleaned up temp dir for file: {temp_dir}")

    elif payload and payload.youtube_url:
         source_description = f"YouTube URL: {payload.youtube_url}"
         try:
             transcription_json_str = await transcribe_via_gemini(str(payload.youtube_url), is_youtube_uri=True)
         except RuntimeError as e:
             raise HTTPException(status_code=503, detail=f"Transcription service failed for YouTube URL: {e}")

    else:
        raise HTTPException(status_code=400, detail="No input. Send 'file' or JSON 'youtube_url'.")

    if transcription_json_str:
        logger.info(f"Transcription successful for {source_description}.")
        return JSONResponse(content={
            "transcription_json": transcription_json_str,
            "engine": "Google Gemini",
            "model_used": GEMINI_MODEL_NAME
        })
    else: # Should be caught by exceptions above
        raise HTTPException(status_code=500, detail="Transcription failed for unknown reasons.")


@app.post("/api/translate", tags=["Translation"])
async def translate_text_or_segments_api(payload: TranslationRequest):
    if not gemini_model_instance:
        raise HTTPException(status_code=503, detail="Translation service (Gemini) not configured.")
    logger.info(f"Translation to '{payload.target_language}'. Input: {'JSON Segments' if payload.is_json_segments else 'Plain Text'}")

    if payload.is_json_segments:
        try:
            segments = json.loads(payload.text_to_translate)
            if not isinstance(segments, list): raise ValueError("JSON not a list.")
            translated_segments = []
            tasks = [
                _translate_single_segment(seg.get("text", ""), payload.target_language)
                for seg in segments if isinstance(seg, dict)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, res in enumerate(results):
                new_seg = segments[i].copy() # Keep original start/end
                if isinstance(res, Exception) or res is None:
                    new_seg["text"] = segments[i].get("text", "") # Keep original on error
                else:
                    new_seg["text"] = res
                translated_segments.append(new_seg)
            return JSONResponse(content={
                "translated_segments_json": json.dumps(translated_segments, ensure_ascii=False),
                "target_language": payload.target_language, "model_used": GEMINI_MODEL_NAME
            })
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON segment data: {e}")
        except Exception as e:
            logger.error(f"Segmented translation error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Error during segmented translation.")
    else: # Plain text
        try:
            translated_text = await _translate_single_segment(payload.text_to_translate, payload.target_language)
            if translated_text is None: raise RuntimeError("Translation returned no content.")
            return JSONResponse(content={
                "translated_text": translated_text,
                "target_language": payload.target_language, "model_used": GEMINI_MODEL_NAME
            })
        except Exception as e:
            logger.error(f"Plain text translation error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error during plain text translation: {e}")

@app.post("/api/generate_questions", tags=["Question Generation"])
async def generate_questions_api(payload: QuestionGenerationRequest):
    if not gemini_model_instance:
        raise HTTPException(status_code=503, detail="Question generation service (Gemini) not configured.")

    try:
        # First, parse the input transcript_json to extract plain text for the prompt
        parsed_transcript_segments = json.loads(payload.transcript_json)
        if not isinstance(parsed_transcript_segments, list):
            raise ValueError("transcript_json is not a valid list of segments.")

        full_transcript_text = " ".join(
            segment.get("text", "") for segment in parsed_transcript_segments if isinstance(segment, dict)
        ).strip()

        if not full_transcript_text:
            raise HTTPException(status_code=400, detail="Transcript is empty, cannot generate questions.")

    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid transcript_json format: {e}")


    prompt_parts = [
        "You are an expert creator of IELTS Listening test questions.",
        "Based *only* on the following transcript text, generate a set of IELTS Listening-style questions.",
        f"Generate approximately {payload.num_questions} questions.",
        "Include a variety of question types. Ensure questions directly test information from the transcript."
    ]
    if payload.question_types:
        prompt_parts.append(f"Focus on these types: {', '.join(payload.question_types)}.")
    if payload.custom_prompt_instructions:
        prompt_parts.append(f"\nAdditional instructions: {payload.custom_prompt_instructions}")

    prompt_parts.extend([
        "\nFormat the output STRICTLY as a single JSON object with a key 'questions'.",
        "The value of 'questions' should be a JSON array of question objects.",
        "Each question object must have 'type' (string), 'question_text' (string).",
        "For 'Multiple Choice', include an 'options' (array of strings) key.",
        "Optionally include 'answer_guidance' (string, hint for finding answer, NOT the answer itself).",
        "\nTranscript:\n--- START TRANSCRIPT ---",
        full_transcript_text, # Use the extracted plain text here
        "--- END TRANSCRIPT ---\nGenerate JSON now:"
    ])
    final_prompt = "\n".join(prompt_parts)

    try:
        logger.info(f"Generating ~{payload.num_questions} questions...")
        response = await gemini_model_instance.generate_content_async(final_prompt)
        generated_text = response.text.strip()
        logger.info("Question generation response received.")
        # logger.debug(f"Raw Question Gen Response: {generated_text}")

        cleaned_text = generated_text.removeprefix("```json").removesuffix("```").strip()
        questions_data = json.loads(cleaned_text)
        if not isinstance(questions_data, dict) or "questions" not in questions_data or \
           not isinstance(questions_data["questions"], list):
            raise ValueError("Generated response has unexpected structure.")
        return questions_data # FastAPI will convert dict to JSONResponse
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to parse generated questions as JSON: {e}. Raw: {generated_text}")
        raise HTTPException(status_code=500, detail=f"Failed to parse generated questions. Raw: {generated_text}")
    except Exception as e:
        logger.error(f"Question generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during question generation: {e}")