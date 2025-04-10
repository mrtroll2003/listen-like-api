import os
import sys
import tempfile
import shutil
import subprocess
import asyncio
from pathlib import Path
import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import datetime

try:
    import whisper
except ImportError:
    logging.error("OpenAI Whisper library not found. Please install it: pip install -U openai-whisper")
    # You might want to raise an exception or handle this more gracefully
    # depending on whether the API can function without whisper at all.
    whisper = None # Set to None so later checks fail clearly

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the FastAPI app instance
# Vercel will look for this 'app' object by default.
app = FastAPI()

MODEL_NAME = os.environ.get("WHISPER_MODEL", "base") # Use 'base' by default, configurable via env var
loaded_model = None

def load_whisper_model():
    """Loads the Whisper model if not already loaded."""
    global loaded_model
    if not whisper:
        raise RuntimeError("Whisper library failed to import.")
    if loaded_model is None:
        logger.info(f"Loading Whisper model: {MODEL_NAME}...")
        try:
            loaded_model = whisper.load_model(MODEL_NAME)
            logger.info("Whisper model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading Whisper model '{MODEL_NAME}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to load Whisper model: {e}")
    return loaded_model

# --- Helper Functions ---
async def run_ffmpeg(input_path: str, output_path: str):
    """
    Extracts audio from video using ffmpeg asynchronously.
    """
    # Construct the path to the bundled ffmpeg relative to this script file
    script_dir = Path(__file__).parent.resolve() # api/ directory
    project_root = script_dir.parent          # Project root directory
    ffmpeg_path = project_root / "bin" / "ffmpeg" # Path object to bin/ffmpeg
    if not ffmpeg_path.is_file():
        logger.error(f"Bundled ffmpeg not found at: {ffmpeg_path}")
        raise RuntimeError(f"ffmpeg binary missing at expected location.")
    # You might need to explicitly make it executable again inside the function
    # if permissions get lost, though git *should* preserve them.
    # os.chmod(ffmpeg_path, 0o755) # Gives rwxr-xr-x permissions

    ffmpeg_cmd = [
        "ffmpeg", # Use the specific path as a string
        "-i", input_path,       # Input file
        "-vn",                  # Disable video recording
        "-acodec", "pcm_s16le", # Audio codec (wav)
        "-ar", "16000",         # Sample rate (recommended for Whisper)
        "-ac", "1",             # Number of audio channels (mono)
        "-y",                   # Overwrite output file if it exists
        output_path,
    ]

    logger.info(f"Running ffmpeg command: {' '.join(ffmpeg_cmd)}")
    try:
        process = await asyncio.create_subprocess_exec(
            *ffmpeg_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate() # Wait for command to finish

        if process.returncode != 0:
            error_message = stderr.decode().strip()
            logger.error(f"ffmpeg error (code {process.returncode}): {error_message}")
            raise RuntimeError(f"ffmpeg failed: {error_message}")
        else:
            logger.info("ffmpeg completed successfully.")
            # logger.debug(f"ffmpeg stdout: {stdout.decode().strip()}") # Optional: log stdout
            # logger.debug(f"ffmpeg stderr: {stderr.decode().strip()}") # Optional: log stderr on success
    except FileNotFoundError:
         logger.error("ffmpeg command not found. Ensure ffmpeg is installed and in the system PATH.")
         raise RuntimeError("ffmpeg is not installed or not found in PATH.")
    except Exception as e:
         logger.error(f"An unexpected error occurred during ffmpeg execution: {e}", exc_info=True)
         raise RuntimeError(f"Audio extraction failed: {e}")


# --- API Endpoints ---

@app.get("/api")
async def api_root():
    """ Root API endpoint. """
    now = datetime.datetime.utcnow().isoformat()
    return JSONResponse(content={
        "message": "Welcome to the Transcription API!",
        "status": "ok",
        "timestamp_utc": now,
        "whisper_model_used": MODEL_NAME
    })

@app.post("/api/transcribe")
async def transcribe_video(request: Request, file: UploadFile = File(...)):
    """
    Receives a video file (MP4, AVI, etc.), extracts audio,
    transcribes it using Whisper, and returns the text.
    Handles multipart/form-data uploads.
    """
    # Basic check if whisper library loaded
    if not whisper:
         raise HTTPException(status_code=500, detail="Whisper library not available on the server.")

    # Load model (might be cached)
    try:
        model = load_whisper_model()
    except RuntimeError as e:
         raise HTTPException(status_code=500, detail=f"Failed to load transcription model: {e}")

    # Log request headers for debugging if needed
    # logger.debug(f"Request Headers: {request.headers}")

    if not file.filename:
         raise HTTPException(status_code=400, detail="No filename provided.")

    # Use a temporary directory for robust cleanup
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp(prefix="api_upload_")
        logger.info(f"Created temporary directory: {temp_dir}")

        video_path = os.path.join(temp_dir, file.filename)
        # Extract base name and extension for audio file
        base_name, _ = os.path.splitext(file.filename)
        audio_filename = f"{base_name}.wav" # Standardize to WAV
        audio_path = os.path.join(temp_dir, audio_filename)

        # 1. Save uploaded video file temporarily
        logger.info(f"Saving uploaded video to: {video_path}")
        try:
            # Read the file in chunks to handle large files efficiently
            with open(video_path, "wb") as buffer:
                while chunk := await file.read(8192): # Read 8KB chunks
                    buffer.write(chunk)
            logger.info(f"Successfully saved video file: {video_path}")
            await file.close() # Close the upload file stream
        except Exception as e:
            logger.error(f"Failed to save uploaded file: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Could not save uploaded file: {e}")

        # 2. Extract audio using ffmpeg
        logger.info(f"Extracting audio to: {audio_path}")
        try:
            await run_ffmpeg(video_path, audio_path)
            logger.info("Audio extraction successful.")
        except RuntimeError as e:
            # run_ffmpeg logs details, just raise HTTP exception
            raise HTTPException(status_code=500, detail=f"Audio extraction failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during audio extraction step: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Audio extraction failed unexpectedly: {e}")


        # 3. Transcribe the audio file using Whisper
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
             logger.error(f"Audio file missing or empty after ffmpeg: {audio_path}")
             raise HTTPException(status_code=500, detail="Audio extraction resulted in an empty or missing file.")

        logger.info(f"Starting transcription for: {audio_path}")
        try:
            # Run transcription in executor to avoid blocking event loop if model.transcribe isn't fully async
            # loop = asyncio.get_running_loop()
            # result = await loop.run_in_executor(None, model.transcribe, audio_path)
            # Note: Whisper's transcribe might be CPU-bound. run_in_executor is good practice.
            # However, for simplicity here, we call it directly. If you face blocking issues, use run_in_executor.
            result = model.transcribe(audio_path, fp16=False) # fp16=False can improve compatibility/reduce GPU needs
            transcription = result["text"]
            logger.info("Transcription successful.")
            # logger.debug(f"Transcription result: {result}") # Log full result if needed
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Transcription process failed: {e}")

        # 4. Return the result
        return JSONResponse(content={
            "filename": file.filename,
            "content_type": file.content_type,
            "transcription": transcription,
            "model_used": MODEL_NAME
        })

    except HTTPException:
        # Re-raise HTTPExceptions directly
        raise
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"An unexpected error occurred in transcribe_video: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

    finally:
        # 5. Cleanup temporary files/directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                # Log cleanup error but don't prevent response if transcription succeeded
                logger.error(f"Error cleaning up temporary directory {temp_dir}: {e}", exc_info=True)
        # Ensure file stream is closed even if saving failed early
        if not file.file.closed:
             await file.close()


@app.get("/api/hello")
async def hello_endpoint(name: str = "World"):
    return JSONResponse(content={"greeting": f"Hello, {name}!"})

# You can add more endpoints here following the same pattern
# Example POST endpoint (requires sending JSON data in the request body)
# from pydantic import BaseModel
# class Item(BaseModel):
#     id: int
#     description: str

# @app.post("/api/items")
# async def create_item(item: Item):
#    """ Receives item data and returns it """
#    return JSONResponse(content={"received_item": item.dict()})

# Note: For Vercel, you generally don't run the app with uvicorn directly
# in this file. Vercel handles the serving part based on the 'app' object.
# The uvicorn command is used for local development.