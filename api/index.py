import os
import sys
import tempfile
import shutil
import functools
import asyncio
from pathlib import Path
import logging
import datetime
from pydantic import BaseModel, HttpUrl, ValidationError
import re # Added for regex
from urllib.parse import urlparse

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# --- Library Imports ---
try:
    import moviepy.editor as mp
except ImportError:
    logging.error("MoviePy library not found. pip install moviepy")
    mp = None

try:
    import speech_recognition as sr
    # Check for internet connection needed for recognize_google()
    # Can't reliably check this at import time in serverless, handle errors later
except ImportError:
    logging.error("SpeechRecognition library not found. pip install SpeechRecognition")
    sr = None

try:
    import google.generativeai as genai
except ImportError:
    logging.error("Google Generative AI library not found. pip install google-generativeai")
    genai = None

try:
    from pytube import YouTube
    from pytube.exceptions import PytubeError, VideoUnavailable, RegexMatchError
except ImportError:
     logging.error("Pytube library not found. pip install pytube")
     YouTube = None
     PytubeError = None # Define base exception for broader catch if needed
     VideoUnavailable = None
     RegexMatchError = None
# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Simplified Media API")


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
        gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        logger.info("Google Gemini client configured successfully with gemini-2.0-flash.")
    except Exception as e:
        logger.error(f"Failed to configure Google Gemini: {e}", exc_info=True)
        gemini_model = None # Ensure it's None if setup fails

# --- Helper Function: Audio Extraction ---
async def extract_audio_moviepy(video_path: str, audio_path: str):
    """Extracts audio using MoviePy, saving as WAV."""
    if not mp:
        raise RuntimeError("MoviePy library not available.")

    video_clip = None
    audio_clip = None
    loop = asyncio.get_running_loop()
    try:
        logger.info(f"MoviePy: Loading video clip from {video_path}")
        video_clip = await loop.run_in_executor(None, mp.VideoFileClip, video_path)

        if video_clip.audio is None:
             logger.error(f"MoviePy: No audio found in video file {video_path}")
             raise ValueError("Video file contains no audio stream.")

        audio_clip = video_clip.audio
        logger.info(f"MoviePy: Writing audio to {audio_path} (WAV)")

        # Create a partial function with all arguments pre-filled
        write_func_partial = functools.partial(
            audio_clip.write_audiofile,
            audio_path,             # 1st positional arg for write_audiofile
            fps=16000,              # Keyword args for write_audiofile
            nbytes=2,
            codec='pcm_s16le',
            ffmpeg_params=["-ac", "1"],
            logger=None
        )

        # Write as standard WAV, SpeechRecognition will handle format internally
        await loop.run_in_executor(
            None,               # Use default executor
            write_func_partial  # Call the pre-configured function
        )
        logger.info("MoviePy: Audio extraction successful.")

    except Exception as e:
        logger.error(f"MoviePy: Error during audio extraction: {e}", exc_info=True)
        # Reraise with a general message, specific error logged
        raise RuntimeError(f"MoviePy audio extraction failed.")
    finally:
        # Ensure clips are closed
        if audio_clip:
            try: await loop.run_in_executor(None, audio_clip.close)
            except Exception as e_close: logger.warning(f"Moviepy: Error closing audio clip: {e_close}")
        if video_clip:
            try: await loop.run_in_executor(None, video_clip.close)
            except Exception as e_close: logger.warning(f"Moviepy: Error closing video clip: {e_close}")


# --- Helper Function: Transcription (SpeechRecognition) ---
async def transcribe_audio_google(audio_path: str) -> str:
    """
    Transcribes audio from a file using SpeechRecognition's Google Web Speech API.
    Runs the synchronous SpeechRecognition code in an executor.
    """
    if not sr:
        raise RuntimeError("SpeechRecognition library not available.")

    recognizer = sr.Recognizer()
    # Note: Add timeout/phrase_time_limit if needed, but Recognizer handles file reading
    # recognizer.pause_threshold = 0.8 # Example adjustment

    loop = asyncio.get_running_loop()
    try:
        transcript = await loop.run_in_executor(
            None, # Default thread pool executor
            _sync_transcribe, # Helper sync function
            recognizer,
            audio_path
        )
        return transcript
    except sr.UnknownValueError:
        logger.warning(f"Google Web Speech could not understand audio: {audio_path}")
        return "Error: Could not transcribe audio (Speech not recognized)"
    except sr.RequestError as e:
        logger.error(f"Google Web Speech API request failed: {e}")
        # Don't expose detailed error potentially containing keys/internal info
        return f"Error: Could not request results from transcription service."
    except Exception as e:
        logger.error(f"Unexpected error during transcription: {e}", exc_info=True)
        return "Error: An unexpected error occurred during transcription."

def _sync_transcribe(recognizer: sr.Recognizer, audio_path: str) -> str:
    """Synchronous helper for transcription to run in executor."""
    audio_file = sr.AudioFile(audio_path)
    with audio_file as source:
        logger.info(f"SpeechRecognition: Recording audio data from {audio_path}")
        # record() loads the whole file here, might be memory intensive for long files
        audio_data = recognizer.record(source)
        logger.info(f"SpeechRecognition: Sending audio data for recognition...")
        # This is the part that makes the network call to Google
        transcript = recognizer.recognize_google(audio_data)
        logger.info(f"SpeechRecognition: Transcription received.")
    return transcript
# Link syntax checker/for regex
YOUTUBE_DOMAINS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
}

def is_valid_youtube_url(url: str) -> bool:
    """Checks if a string is a valid HTTP/HTTPS URL pointing to a known YouTube domain."""
    try:
        # Basic URL structure validation
        parsed_url = urlparse(url)
        if not all([parsed_url.scheme in ["http", "https"], parsed_url.netloc]):
            logger.warning(f"URL validation failed (scheme/netloc): {url}")
            return False

        # Check if the domain is a known YouTube domain
        domain = parsed_url.netloc.lower()
        if domain not in YOUTUBE_DOMAINS:
            # Handle youtu.be separately as netloc is just 'youtu.be'
             if domain == 'youtu.be' and parsed_url.path and len(parsed_url.path) > 1:
                 return True # youtu.be/VIDEO_ID is valid
             logger.warning(f"URL validation failed (domain not YouTube): {domain}")
             return False

        # For youtube.com domains, check for /watch path (basic check)
        if "youtube.com" in domain:
            if not parsed_url.path.startswith("/watch"):
                 logger.warning(f"URL validation failed (path not /watch): {parsed_url.path}")
                 # Allow other valid paths like /shorts/ ? Maybe too complex for basic validation.
                 # return False # Be strict? Or allow other paths? Let's be lenient for now.
                 pass # Allow other paths for now

        # Optional: More sophisticated regex for video ID might be added later
        # For now, domain and basic structure check is sufficient

        return True

    except Exception as e:
        logger.error(f"Error during URL parsing: {e}", exc_info=True)
        return False

# --- Pydantic Models ---
class TranslationRequest(BaseModel):
    text: str
    target_language: str

class YouTubeTranscribeRequest(BaseModel):
    youtube_url: str
    # Future: add language hint, model choice etc.

# --- API Endpoints ---

@app.get("/api", tags=["Status"])
async def api_root():
    """ Root API endpoint providing status. """
    now = datetime.datetime.utcnow().isoformat()
    return JSONResponse(content={
        "message": "Welcome to the Simplified Media API!",
        "status": "ok",
        "timestamp_utc": now,
        "endpoints": ["/api/transcribe", "/api/translate", "/api/transcribe_youtube", "/docs"]
    })

@app.post("/api/transcribe", tags=["Transcription"])
async def transcribe_video(file: UploadFile = File(...)):
    """
    Receives video, extracts audio via MoviePy, transcribes via SpeechRecognition (Google Web Speech API).
    """
    if not sr:
         raise HTTPException(status_code=501, detail="SpeechRecognition library not available.")
    if not mp:
        raise HTTPException(status_code=501, detail="MoviePy library not available.")

    if not file.filename:
         raise HTTPException(status_code=400, detail="No filename provided.")

    # Use a temporary directory for robust cleanup
    temp_dir_obj = tempfile.TemporaryDirectory(prefix="api_upload_")
    temp_dir = temp_dir_obj.name
    logger.info(f"Created temporary directory: {temp_dir}")

    try:
        safe_filename = Path(file.filename).name
        video_path = os.path.join(temp_dir, safe_filename)
        base_name, _ = os.path.splitext(safe_filename)
        # Use a simple name for the single extracted audio file
        audio_filename = f"{base_name}_extracted_audio.wav"
        audio_path = os.path.join(temp_dir, audio_filename)

        # 1. Save uploaded video file temporarily
        logger.info(f"Saving uploaded video to: {video_path}")
        try:
            with open(video_path, "wb") as buffer:
                while chunk := await file.read(8192): buffer.write(chunk)
            logger.info(f"Successfully saved video file: {video_path}")
        except Exception as e:
            logger.error(f"Failed to save uploaded file: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Could not save uploaded file.")
        finally:
            await file.close() # Ensure stream is closed

        # 2. Extract audio using MoviePy
        logger.info(f"Extracting audio using MoviePy to: {audio_path}")
        try:
            await extract_audio_moviepy(video_path, audio_path)
        except (RuntimeError, ValueError) as e:
            raise HTTPException(status_code=500, detail=f"Audio extraction failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during audio extraction: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Audio extraction failed unexpectedly.")

        # 3. Transcribe the extracted audio file
        if not os.path.exists(audio_path):
             logger.error(f"Audio file missing after extraction: {audio_path}")
             raise HTTPException(status_code=500, detail="Audio extraction did not produce a file.")

        logger.info(f"Starting transcription for: {audio_path}")
        transcription = await transcribe_audio_google(audio_path)

        # Check if transcription returned an error message
        if transcription.startswith("Error:"):
            logger.warning(f"Transcription failed with message: {transcription}")
            # Return 502 Bad Gateway if the external service failed, 400 if audio was bad
            status_code = 502 if "service" in transcription else 400 if "Could not transcribe" in transcription else 500
            raise HTTPException(status_code=status_code, detail=transcription)

        logger.info("Transcription successful.")

        # 4. Return the result
        return JSONResponse(content={
            "filename": file.filename,
            "content_type": file.content_type,
            "transcription": transcription,
            "engine": "SpeechRecognition (Google Web Speech API)"
        })

    except HTTPException:
        raise # Re-raise exceptions we already handled
    except Exception as e:
        logger.error(f"An unexpected error occurred in transcribe_video endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")
    finally:
        # Cleanup temporary directory
        logger.info(f"Cleaning up temporary directory: {temp_dir}")
        temp_dir_obj.cleanup()

@app.post("/api/transcribe_youtube", tags=["Transcription"])
async def transcribe_youtube(payload: YouTubeTranscribeRequest):
    """
    Downloads audio from a YouTube URL, extracts audio, and transcribes it.
    """
    if not YouTube:
        raise HTTPException(status_code=501, detail="Pytube library not available.")
    if not sr:
         raise HTTPException(status_code=501, detail="SpeechRecognition library not available.")
    if not mp:
        raise HTTPException(status_code=501, detail="MoviePy library not available.")

    # 1. Validate URL
    if not is_valid_youtube_url(payload.youtube_url):
        raise HTTPException(status_code=400, detail="Invalid or non-YouTube URL provided.")

    temp_dir_obj = tempfile.TemporaryDirectory(prefix="youtube_dl_")
    temp_dir = temp_dir_obj.name
    logger.info(f"Created temporary directory for YouTube download: {temp_dir}")

    try:
        loop = asyncio.get_running_loop()

        # 2. Download Audio using Pytube
        logger.info(f"Attempting to download audio for YouTube URL: {payload.youtube_url}")
        downloaded_file_path = None
        try:
            # Run synchronous pytube operations in executor
            def _sync_download():
                yt = YouTube(payload.youtube_url)
                # Filter for audio-only streams, prefer mp4 (often AAC), order by bitrate desc
                audio_stream = yt.streams.filter(only_audio=True, file_extension='mp4').order_by('abr').desc().first()
                if not audio_stream:
                    # Fallback: Try any audio stream if mp4 isn't available
                    logger.warning("No MP4 audio stream found, trying any audio stream...")
                    audio_stream = yt.streams.filter(only_audio=True).order_by('abr').desc().first()

                if not audio_stream:
                    logger.error("No suitable audio stream found for this YouTube video.")
                    raise ValueError("No suitable audio-only stream found.")

                logger.info(f"Selected audio stream: itag={audio_stream.itag}, abr={audio_stream.abr}, type={audio_stream.mime_type}")
                # Define a consistent filename for the download
                download_filename = "youtube_audio_download" # Extension will be added by pytube
                return audio_stream.download(output_path=temp_dir, filename=download_filename)

            # Execute the download function
            downloaded_file_path = await loop.run_in_executor(None, _sync_download)
            logger.info(f"Successfully downloaded YouTube audio to: {downloaded_file_path}")

        except (VideoUnavailable, RegexMatchError) as e:
            logger.error(f"Pytube error: Video unavailable or URL regex failed for {payload.youtube_url}: {e}", exc_info=True)
            raise HTTPException(status_code=404, detail=f"Video not found or unavailable: {e}")
        except PytubeError as e: # Catch other general pytube errors
             logger.error(f"Pytube error downloading {payload.youtube_url}: {e}", exc_info=True)
             raise HTTPException(status_code=500, detail=f"Failed to download YouTube audio: {e}")
        except ValueError as e: # Catch our specific "no stream" error
             logger.error(f"ValueError during YouTube download: {e}")
             raise HTTPException(status_code=400, detail=str(e))
        except Exception as e: # Catch unexpected errors during download
             logger.error(f"Unexpected error during YouTube download: {e}", exc_info=True)
             raise HTTPException(status_code=500, detail="An unexpected error occurred during download.")


        # 3. Extract/Convert Audio to WAV using MoviePy (for consistency)
        if not downloaded_file_path or not os.path.exists(downloaded_file_path):
             logger.error("Downloaded audio file path is missing after pytube download.")
             raise HTTPException(status_code=500, detail="Audio download succeeded but file is missing.")

        base_name, _ = os.path.splitext(os.path.basename(downloaded_file_path))
        wav_audio_filename = f"{base_name}_extracted.wav"
        wav_audio_path = os.path.join(temp_dir, wav_audio_filename)

        logger.info(f"Extracting/Converting downloaded audio to WAV: {wav_audio_path}")
        try:
            # Use the existing moviepy function
            await extract_audio_moviepy(downloaded_file_path, wav_audio_path)
        except (RuntimeError, ValueError) as e:
            logger.error(f"Audio extraction failed for downloaded YouTube audio: {e}")
            raise HTTPException(status_code=500, detail=str(e)) # Pass detailed error
        except Exception as e:
            logger.error(f"Unexpected error during audio extraction: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Audio extraction failed unexpectedly.")

        # 4. Transcribe the extracted WAV file
        if not os.path.exists(wav_audio_path):
             logger.error(f"WAV audio file missing after extraction: {wav_audio_path}")
             raise HTTPException(status_code=500, detail="Audio conversion did not produce a WAV file.")

        logger.info(f"Starting transcription for YouTube audio: {wav_audio_path}")
        transcription = await transcribe_audio_google(wav_audio_path)

        # Check for transcription errors
        if transcription.startswith("Error:"):
            logger.warning(f"Transcription failed for YouTube audio with message: {transcription}")
            status_code = 502 if "service" in transcription else 400 if "Could not transcribe" in transcription else 500
            raise HTTPException(status_code=status_code, detail=transcription)

        logger.info("YouTube Transcription successful.")

        # 5. Return the result
        return JSONResponse(content={
            "youtube_url": payload.youtube_url,
            "transcription": transcription,
            "engine": "SpeechRecognition (Google Web Speech API)"
        })

    except HTTPException:
        raise # Re-raise exceptions we already handled
    except Exception as e:
        logger.error(f"An unexpected error occurred in transcribe_youtube endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred processing YouTube URL.")
    finally:
        # Cleanup temporary directory
        logger.info(f"Cleaning up YouTube temporary directory: {temp_dir}")
        temp_dir_obj.cleanup()


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