import os
import sys
import tempfile
import shutil
import asyncio
from pathlib import Path
import logging
import datetime
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# --- Library Imports ---
try:
    import moviepy.editor as mp
except ImportError:
    logging.error("MoviePy library not found. pip install moviepy")
    mp = None

try:
    import whisper
except ImportError:
    logging.error("OpenAI Whisper library not found. pip install -U openai-whisper")
    whisper = None

try:
    import google.generativeai as genai
except ImportError:
    logging.error("Google Generative AI library not found. pip install google-generativeai")
    genai = None

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Simplified Media API")

# --- Whisper Setup ---
WHISPER_MODEL_NAME = "tiny" # Using the smallest model
whisper_loaded_model = None
whisper_loading_lock = asyncio.Lock() # Prevent concurrent loading attempts

async def load_whisper_model():
    """Loads the Whisper model (tiny) if not already loaded."""
    global whisper_loaded_model
    if whisper_loaded_model is None:
        async with whisper_loading_lock:
            # Double check after acquiring lock
            if whisper_loaded_model is None:
                if not whisper:
                    raise RuntimeError("Whisper library failed to import.")
                logger.info(f"Loading Whisper model: {WHISPER_MODEL_NAME}...")
                try:
                    # Run model loading in executor as it can be CPU/IO bound
                    loop = asyncio.get_running_loop()
                    whisper_loaded_model = await loop.run_in_executor(
                        None, # Default executor
                        whisper.load_model,
                        WHISPER_MODEL_NAME
                    )
                    logger.info("Whisper model loaded successfully.")
                except Exception as e:
                    logger.error(f"Error loading Whisper model '{WHISPER_MODEL_NAME}': {e}", exc_info=True)
                    raise RuntimeError(f"Failed to load Whisper model: {e}")
    return whisper_loaded_model

# --- Google Gemini Setup ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
gemini_model = None

if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY environment variable not set. Translation endpoint will fail.")
elif not genai:
    logger.warning("Google Generative AI library not loaded. Translation endpoint will fail.")
else:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        # Using gemini-1.5-flash as requested - fast and capable
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        logger.info("Google Gemini client configured successfully with gemini-1.5-flash-latest.")
    except Exception as e:
        logger.error(f"Failed to configure Google Gemini: {e}", exc_info=True)
        gemini_model = None # Ensure it's None if setup fails

# --- Helper Function: Audio Extraction ---
async def extract_audio_moviepy(video_path: str, audio_path: str):
    """Extracts audio using MoviePy, saving as 16kHz mono WAV."""
    if not mp:
        raise RuntimeError("MoviePy library not available.")

    video_clip = None
    audio_clip = None
    try:
        logger.info(f"MoviePy: Loading video clip from {video_path}")
        # Run synchronous moviepy loading in executor
        loop = asyncio.get_running_loop()
        video_clip = await loop.run_in_executor(None, mp.VideoFileClip, video_path)

        if video_clip.audio is None:
             logger.error(f"MoviePy: No audio found in video file {video_path}")
             raise ValueError("Video file contains no audio stream.")

        audio_clip = video_clip.audio
        logger.info(f"MoviePy: Writing audio to {audio_path} (16kHz mono WAV)")

        # Run synchronous audio writing in executor
        await loop.run_in_executor(
            None,
            audio_clip.write_audiofile,
            audio_path,
            {"codec": 'pcm_s16le', # WAV format
             "fps": 16000,         # 16kHz sample rate for Whisper
             "nbytes": 2,          # 16 bits
             "ffmpeg_params": ["-ac", "1"], # Force mono channel
             "logger": 'bar'} # Suppress progress bar logging somewhat
        )
        logger.info("MoviePy: Audio extraction successful.")

    except Exception as e:
        logger.error(f"MoviePy: Error during audio extraction: {e}", exc_info=True)
        raise RuntimeError(f"MoviePy audio extraction failed: {e}")
    finally:
        # Ensure clips are closed to release file handles
        if audio_clip:
            try:
                await loop.run_in_executor(None, audio_clip.close)
            except Exception as e_close:
                logger.warning(f"Moviepy: Error closing audio clip: {e_close}")
        if video_clip:
            try:
                await loop.run_in_executor(None, video_clip.close)
            except Exception as e_close:
                logger.warning(f"Moviepy: Error closing video clip: {e_close}")


# --- Pydantic Models ---
class TranslationRequest(BaseModel):
    text: str
    target_language: str

# --- API Endpoints ---

@app.get("/api", tags=["Status"])
async def api_root():
    """ Root API endpoint providing status. """
    now = datetime.datetime.utcnow().isoformat()
    return JSONResponse(content={
        "message": "Welcome to the Simplified Media API!",
        "status": "ok",
        "timestamp_utc": now,
        "endpoints": ["/api/transcribe", "/api/translate", "/docs"]
    })

@app.post("/api/transcribe", tags=["Transcription"])
async def transcribe_video(file: UploadFile = File(...)):
    """
    Receives video, extracts audio via MoviePy, transcribes via Whisper (tiny).
    """
    if not whisper:
         raise HTTPException(status_code=501, detail="Whisper library not available.")
    if not mp:
        raise HTTPException(status_code=501, detail="MoviePy library not available.")

    if not file.filename:
         raise HTTPException(status_code=400, detail="No filename provided.")

    # Use a temporary directory
    temp_dir_obj = tempfile.TemporaryDirectory(prefix="api_upload_")
    temp_dir = temp_dir_obj.name
    logger.info(f"Created temporary directory: {temp_dir}")

    try:
        # Define paths within the temp directory
        # Sanitize filename slightly to avoid path issues, though tempfile is safer
        safe_filename = Path(file.filename).name
        video_path = os.path.join(temp_dir, safe_filename)
        base_name, _ = os.path.splitext(safe_filename)
        audio_filename = f"{base_name}_extracted_audio.wav"
        audio_path = os.path.join(temp_dir, audio_filename)

        # 1. Save uploaded video file temporarily
        logger.info(f"Saving uploaded video to: {video_path}")
        try:
            with open(video_path, "wb") as buffer:
                while chunk := await file.read(8192): # Read in chunks
                    buffer.write(chunk)
            logger.info(f"Successfully saved video file: {video_path}")
        except Exception as e:
            logger.error(f"Failed to save uploaded file: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Could not save uploaded file: {e}")
        finally:
             # Ensure uploaded file stream is closed
             await file.close()

        # 2. Extract audio using MoviePy
        logger.info(f"Extracting audio using MoviePy to: {audio_path}")
        try:
            await extract_audio_moviepy(video_path, audio_path)
        except (RuntimeError, ValueError) as e:
            # Catch errors from extract_audio_moviepy
            raise HTTPException(status_code=500, detail=f"Audio extraction failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during audio extraction step: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Audio extraction failed unexpectedly: {e}")

        # 3. Transcribe the audio file using Whisper
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
             logger.error(f"Audio file missing or empty after extraction: {audio_path}")
             raise HTTPException(status_code=500, detail="Audio extraction resulted in an empty file.")

        logger.info(f"Loading Whisper model ({WHISPER_MODEL_NAME}) for transcription...")
        try:
            model = await load_whisper_model() # Load/get cached model
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=f"Failed to load transcription model: {e}")

        logger.info(f"Starting Whisper transcription for: {audio_path}")
        try:
            # Run potentially CPU-bound transcription in executor
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                model.transcribe,
                audio_path,
                {"fp16": False} # fp16=False safer on CPU
            )
            transcription = result["text"]
            logger.info("Whisper transcription successful.")
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Transcription process failed: {e}")

        # 4. Return the result
        return JSONResponse(content={
            "filename": file.filename,
            "content_type": file.content_type,
            "transcription": transcription,
            "whisper_model_used": WHISPER_MODEL_NAME
        })

    except HTTPException:
        # Re-raise HTTPExceptions directly
        raise
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"An unexpected error occurred in transcribe_video: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

    finally:
        # 5. Cleanup temporary directory using the context manager's exit
        logger.info(f"Cleaning up temporary directory: {temp_dir}")
        temp_dir_obj.cleanup() # Explicit cleanup just in case


@app.post("/api/translate", tags=["Translation"])
async def translate_text_gemini(payload: TranslationRequest):
    """
    Translates text to the target language using Google Gemini API.
    """
    if not gemini_model:
        raise HTTPException(status_code=501, detail="Gemini API client not configured or library not available.")

    logger.info(f"Received translation request to '{payload.target_language}'")

    # Construct a clear prompt for Gemini
    prompt = f"Translate the following text into {payload.target_language}. Only return the translated text, without any introductory phrases or explanations:\n\n{payload.text}"

    try:
        logger.info("Sending request to Gemini API...")
        # Run synchronous SDK call in executor
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            gemini_model.generate_content,
            prompt
        )

        # Check for safety ratings / blocks if necessary (optional)
        # if response.prompt_feedback.block_reason:
        #     logger.warning(f"Gemini translation blocked. Reason: {response.prompt_feedback.block_reason}")
        #     raise HTTPException(status_code=400, detail=f"Translation request blocked by safety filters: {response.prompt_feedback.block_reason}")

        translated_text = response.text # Access the translated text directly
        logger.info("Gemini translation successful.")

        return JSONResponse(content={
            "original_text": payload.text,
            "target_language": payload.target_language,
            "translated_text": translated_text,
            "model_used": "gemini-2.0-flash" # Or get from model object if possible/needed
        })

    except Exception as e:
        # Catch potential API errors, network issues, etc.
        logger.error(f"Gemini API translation failed: {e}", exc_info=True)
        # Provide a more specific error if possible based on Gemini SDK exceptions
        # e.g., if e is google.api_core.exceptions.PermissionDenied: ...
        raise HTTPException(status_code=503, detail=f"Translation service failed: {e}")


# --- Optional: Add hello endpoint or others back if needed ---
# @app.get("/api/hello", tags=["Greeting"])
# async def hello_endpoint(name: str = "World"):
#     return JSONResponse(content={"greeting": f"Hello, {name}!"})