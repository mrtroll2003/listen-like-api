import os
import sys
import tempfile
import shutil
import subprocess
import asyncio
from pathlib import Path
import logging
from typing import Optional # Added for type hinting

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Body # Added Body
from fastapi.responses import JSONResponse
import datetime
from pydantic import BaseModel # Added for request body validation

try:
    import whisper
except ImportError:
    logging.error("OpenAI Whisper library not found. Please install it: pip install -U openai-whisper")
    # You might want to raise an exception or handle this more gracefully
    # depending on whether the API can function without whisper at all.
    whisper = None # Set to None so later checks fail clearly

# --- Translation Setup ---
try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    # Optional: Specify cache directory if needed within Vercel's writable tmp space
    # HUGGINGFACE_CACHE = "/tmp/huggingface_cache"
    # os.environ["HF_HOME"] = HUGGINGFACE_CACHE
    # os.environ["TRANSFORMERS_CACHE"] = HUGGINGFACE_CACHE
    # Path(HUGGINGFACE_CACHE).mkdir(parents=True, exist_ok=True)
except ImportError:
     logging.error("Hugging Face Transformers library not found. Please install it: pip install transformers sentencepiece torch")
     AutoModelForSeq2SeqLM = None
     AutoTokenizer = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the FastAPI app instance
# Vercel will look for this 'app' object by default.
app = FastAPI()

MODEL_NAME = os.environ.get("WHISPER_MODEL", "base") # Use 'base' by default, configurable via env var
loaded_model = None

def load_whisper_model():
    """Loads the Whisper model if not already loaded."""
    global whisper_loaded_model
    if not whisper:
        raise RuntimeError("Whisper library failed to import.")
    if whisper_loaded_model is None:
        logger.info(f"Loading Whisper model: {WHISPER_MODEL_NAME}...")
        try:
            # Consider specifying download_root if needed on Vercel /tmp
            # whisper_loaded_model = whisper.load_model(WHISPER_MODEL_NAME, download_root="/tmp/whisper_cache")
            whisper_loaded_model = whisper.load_model(WHISPER_MODEL_NAME)
            logger.info("Whisper model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading Whisper model '{WHISPER_MODEL_NAME}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to load Whisper model: {e}")
    return whisper_loaded_model

# --- Translation Model Loading ---
# Using a distilled NLLB model for potentially better performance/size balance
# Other options: "facebook/mbart-large-50-many-to-many-mmt"
TRANSLATION_MODEL_NAME = os.environ.get("TRANSLATION_MODEL", "facebook/nllb-200-distilled-600M")
translation_model = None
translation_tokenizer = None

# Simple mapping from common language names/codes to NLLB Flores-200 codes
# See Flores-200 codes: https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200
# Add more mappings as needed
LANGUAGE_CODE_MAP = {
    "English": "eng_Latn",
    "en": "eng_Latn",
    "Spanish": "spa_Latn",
    "es": "spa_Latn",
    "French": "fra_Latn",
    "fr": "fra_Latn",
    "German": "deu_Latn",
    "de": "deu_Latn",
    "Chinese": "zho_Hans", # Simplified Chinese
    "zh": "zho_Hans",
    "Japanese": "jpn_Jpan",
    "ja": "jpn_Jpan",
    "Arabic": "arb_Arab",
    "ar": "arb_Arab",
    "Russian": "rus_Cyrl",
    "ru": "rus_Cyrl",
    "Hindi": "hin_Deva",
    "hi": "hin_Deva",
    "Korean": "kor_Hang",
    "ko": "kor_Hang",
    "Vietnamese": "vie_Latn",
    "vi": "vie_Latn",
    "Indonesian": "ind_Latn",
    "id": "ind_Latn",
}

def get_flores_code(lang_name_or_code: str) -> Optional[str]:
    """Gets the NLLB Flores-200 code for a given language name/code."""
    return LANGUAGE_CODE_MAP.get(lang_name_or_code.lower())

def load_translation_model_and_tokenizer():
    """Loads the translation model and tokenizer if not already loaded."""
    global translation_model, translation_tokenizer
    if not AutoModelForSeq2SeqLM or not AutoTokenizer:
         raise RuntimeError("Transformers library failed to import.")
    if translation_model is None or translation_tokenizer is None:
        logger.info(f"Loading translation model/tokenizer: {TRANSLATION_MODEL_NAME}...")
        try:
            # device_map="auto" might help distribute across resources if available
            translation_model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL_NAME)
            translation_tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_NAME)
            logger.info("Translation model/tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading translation model '{TRANSLATION_MODEL_NAME}': {e}", exc_info=True)
            # Reset globals on failure to allow retry later if applicable
            translation_model = None
            translation_tokenizer = None
            raise RuntimeError(f"Failed to load translation model/tokenizer: {e}")
    return translation_model, translation_tokenizer


# --- Helper Functions ---
async def run_ffmpeg(input_path: str, output_path: str):
    """
    Extracts audio from video using the ffmpeg binary copied during build.
    """
    # Path to the ffmpeg binary relative to this script (index.py)
    # Since build.sh copies it to 'api/ffmpeg' and index.py is in 'api/'
    script_dir = Path(__file__).parent.resolve()
    ffmpeg_path = script_dir / "ffmpeg" # Should resolve to /var/task/api/ffmpeg at runtime

    # Check if the copied binary exists
    if not ffmpeg_path.is_file():
         # If this happens, the copy in build.sh likely failed or the path is wrong
         logger.error(f"Copied ffmpeg binary not found at expected runtime location: {ffmpeg_path}")
         # You could also try checking common system paths as a fallback, but rely on the copied one first
         # ffmpeg_executable_fallback = "ffmpeg" # Try system path
         raise RuntimeError(f"ffmpeg binary missing at expected location: {ffmpeg_path}. Build copy likely failed.")

    # Make sure it's executable at runtime (belt and suspenders)
    try:
        os.chmod(ffmpeg_path, 0o755) # rwxr-xr-x
    except Exception as e:
        logger.warning(f"Could not set execute permission on {ffmpeg_path}: {e}. It might already be set.")


    ffmpeg_cmd = [
        str(ffmpeg_path), # Use the specific path to the copied binary
        "-i", input_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        "-y", output_path,
    ]

    logger.info(f"Running copied ffmpeg command: {' '.join(ffmpeg_cmd)}")
    try:
        process = await asyncio.create_subprocess_exec(
            *ffmpeg_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_message = stderr.decode().strip()
            logger.error(f"Copied ffmpeg error (code {process.returncode}): {error_message}")
            raise RuntimeError(f"Copied ffmpeg failed: {error_message}")
        else:
            logger.info("Copied ffmpeg completed successfully.")

    except FileNotFoundError:
         # This error now specifically means the ffmpeg_path determined above wasn't found/executable
         logger.error(f"Failed to execute ffmpeg command at path: {ffmpeg_path}")
         raise RuntimeError(f"ffmpeg execution failed (FileNotFound at {ffmpeg_path}).")
    except Exception as e:
         logger.error(f"An unexpected error occurred during ffmpeg execution: {e}", exc_info=True)
         raise RuntimeError(f"Audio extraction failed: {e}")

# --- Pydantic Models for Request/Response ---
class TranslationRequest(BaseModel):
    text: str
    target_language: str
    source_language: str = "en" # Default source language is English

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

@app.post("/api/transcribe", tags=["Transcription"])
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

@app.post("/api/translate", tags=["Translation"])
async def translate_text(payload: TranslationRequest):
    """
    Translates text from a source language (default: English) to a target language.
    Requires 'text' and 'target_language' in the JSON request body.
    'target_language' should be a common name (e.g., 'Spanish', 'French') or code (e.g., 'es', 'fr').
    """
    if not AutoModelForSeq2SeqLM or not AutoTokenizer:
        raise HTTPException(status_code=501, detail="Translation library not available on the server.")

    try:
        model, tokenizer = load_translation_model_and_tokenizer()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to load translation model: {e}")

    # --- Get Language Codes ---
    target_lang_code = get_flores_code(payload.target_language)
    if not target_lang_code:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported target language: '{payload.target_language}'. Please use a common name or code (e.g., 'Spanish', 'es', 'French', 'fr'). Supported codes: {list(LANGUAGE_CODE_MAP.keys())}"
        )

    source_lang_code = get_flores_code(payload.source_language)
    if not source_lang_code:
         # Defaulting to English if source is not provided or invalid, could raise error instead
        logger.warning(f"Unsupported source language '{payload.source_language}'. Defaulting to English ('en').")
        source_lang_code = "eng_Latn" # Default NLLB code for English

    logger.info(f"Translation request: From '{source_lang_code}' To '{target_lang_code}'")

    try:
        # --- Prepare input for the model ---
        tokenizer.src_lang = source_lang_code
        # max_length can be adjusted based on expected input/output sizes
        inputs = tokenizer(payload.text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # --- Generate translation ---
        # Get the target language ID for forcing the decoder
        forced_bos_token_id = tokenizer.lang_code_to_id[target_lang_code]

        logger.info("Generating translation...")
        # Use run_in_executor for the potentially CPU-bound generation task
        loop = asyncio.get_running_loop()
        generated_tokens = await loop.run_in_executor(
            None, # Default thread pool executor
            model.generate, # Function to run
            **inputs.to(model.device), # Pass tokenized inputs (move to model device if applicable)
            forced_bos_token_id=forced_bos_token_id, # Force target language
            max_length=512 # Should match or exceed input max_length
        )
        # generated_tokens = model.generate(
        #     **inputs.to(model.device), # Ensure tensors are on the same device as model
        #     forced_bos_token_id=forced_bos_token_id,
        #     max_length=512 # Adjust as needed
        # )

        # --- Decode the result ---
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        logger.info("Translation generation successful.")

        return JSONResponse(content={
            "original_text": payload.text,
            "source_language_code": source_lang_code,
            "target_language_code": target_lang_code,
            "target_language_input": payload.target_language,
            "translated_text": translated_text,
            "model_used": TRANSLATION_MODEL_NAME
        })

    except Exception as e:
        logger.error(f"Translation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Translation process failed: {e}")




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