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
import pafy
from pydantic import BaseModel, HttpUrl, ValidationError
import re # Added for regex
from urllib.parse import urlparse

import time #for yt transcribe timeout
import urllib.error #for yt transcribe errcatch


import wave     #For logging/
import contextlib #debugging

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Library Imports ---
try:
    import moviepy.editor as mp
except ImportError:
    logging.error("MoviePy library not found. pip install moviepy")
    mp = None

try:
    import google.generativeai as genai
    from google.api_core import exceptions as google_api_exceptions
except ImportError:
    logging.error("Google Generative AI library not found. pip install google-generativeai")
    genai = None
    google_api_exceptions = None


    # Check if backend yt-dlp is available (optional but good practice)
try:
    import yt_dlp
    logger.info("yt-dlp backend found for pafy.")
except ImportError:
    logger.warning("yt-dlp backend not found. Pafy might use bundled youtube-dl or fail.")



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
        #debug log
        if os.path.exists(audio_path):
            try:
                with contextlib.closing(wave.open(audio_path, 'rb')) as wf:
                    channels = wf.getnchannels()
                    framerate = wf.getframerate()
                    sampwidth = wf.getsampwidth()
                    nframes = wf.getnframes()
                    duration = nframes / float(framerate) if framerate > 0 else 0
                    logger.info(f"WAV Check: Path={audio_path}, Channels={channels}, Rate={framerate}, Width={sampwidth} bytes, Duration={duration:.2f}s")
                # Check if properties match expectations
                if channels != 1: logger.warning("WAV Check: Audio is not mono!")
                if framerate != 16000: logger.warning(f"WAV Check: Sample rate is {framerate}, expected 16000!")
                if sampwidth != 2: logger.warning(f"WAV Check: Sample width is {sampwidth} bytes, expected 2 (16-bit)!")
            except wave.Error as wave_err:
                logger.error(f"WAV Check: Error reading WAV file properties: {wave_err}")
            except Exception as e:
                logger.error(f"WAV Check: Unexpected error checking WAV properties: {e}", exc_info=True)
        else:
            logger.error(f"WAV Check: Audio file missing, cannot check properties: {audio_path}")
            raise HTTPException(status_code=500, detail="Audio extraction did not produce a file.")
        #debug log ends
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


# --- Helper Function: Transcription  ---
async def transcribe_audio_gemini(audio_path: str) -> str:
    """
    Transcribes audio from a file using the Google Gemini API.
    Uploads the file and sends it with a transcription prompt.
    """
    if not genai or not gemini_model:
        raise RuntimeError("Gemini API client not configured or library not available.")

    uploaded_file = None
    try:
        # 1. Determine MIME type
        mime_type, _ = mimetypes.guess_type(audio_path)
        if not mime_type or not mime_type.startswith("audio/"):
            # Default or fallback if guess fails for common types
            if audio_path.lower().endswith(".wav"): mime_type = "audio/wav"
            elif audio_path.lower().endswith(".mp3"): mime_type = "audio/mpeg"
            elif audio_path.lower().endswith(".ogg"): mime_type = "audio/ogg"
            # Add more if needed, or raise error
            else: mime_type = "application/octet-stream" # Generic fallback
            logger.warning(f"Could not guess MIME type for {audio_path}, using {mime_type}")

        # 2. Upload the audio file
        logger.info(f"Gemini: Uploading audio file: {audio_path} (MIME: {mime_type})")
        # Run synchronous SDK call in executor
        loop = asyncio.get_running_loop()
        upload_func_partial = functools.partial(
            genai.upload_file,
            path=audio_path,      # Pass args intended for upload_file here
            mime_type=mime_type
        )
        uploaded_file = await loop.run_in_executor(
            None,
            upload_func_partial
        )
        logger.info(f"Gemini: File uploaded successfully. URI: {uploaded_file.uri}")

        # 3. Generate content with prompt and uploaded file
        prompt = "Please transcribe the following audio file."
        logger.info("Gemini: Sending transcription request...")

        generate_func_partial = functools.partial(
            gemini_model.generate_content,
            [prompt, uploaded_file] # Arguments for generate_content
            # Add other generate_content args here if needed (e.g., generation_config)
        )
        # Make the generate_content call in executor
        response = await loop.run_in_executor(
            None,
            generate_func_partial   # Call the pre-configured function
        )

        # Check for potential blocks or empty responses
        if not response.candidates:
             logger.warning("Gemini: Response has no candidates. Possibly blocked or empty.")
             # Try accessing prompt feedback for block reason
             block_reason = getattr(response.prompt_feedback, 'block_reason', None)
             if block_reason:
                 return f"Error: Transcription request blocked by safety filters: {block_reason}"
             else:
                 return "Error: Transcription failed (No response candidates)"

        # Extract text from the first candidate
        # Assuming the transcription is in response.text or parts
        transcription = response.text
        logger.info("Gemini: Transcription received successfully.")
        return transcription

    except google_api_exceptions.PermissionDenied as e:
         logger.error(f"Gemini API Permission Denied: {e}", exc_info=True)
         return "Error: Gemini API permission denied. Check your API key and permissions."
    except google_api_exceptions.ResourceExhausted as e:
         logger.error(f"Gemini API Quota Exceeded: {e}", exc_info=True)
         return "Error: Gemini API quota limit reached. Please try again later."
    except Exception as e:
        logger.error(f"Gemini transcription failed: {e}", exc_info=True)
        # Provide a general error, specific details logged
        return f"Error: An unexpected error occurred during Gemini transcription ({type(e).__name__})."
    finally:
        # Optional: Delete the uploaded file from Gemini storage if desired
        # This might require tracking the uploaded_file.name and calling genai.delete_file()
        # For simplicity, we omit this now, but be aware files might persist temporarily.
        if uploaded_file:
            logger.debug(f"Gemini: Uploaded file object: name={uploaded_file.name}, uri={uploaded_file.uri}")
            # Consider adding deletion logic here if managing storage is critical


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
        if "youtube.com" in domain: pass
        return True

    except Exception as e:
        logger.error(f"Error during URL parsing: {e}", exc_info=True)
        return False
    
def _sync_download_pafy(youtube_url: str, temp_dir: str) -> str:
    """Synchronous helper to download audio using pafy."""
    if not pafy:
        raise RuntimeError("Pafy library is not available.")
    try:
        logger.info(f"Pafy: Creating video object for {youtube_url}")
        video = pafy.new(youtube_url)

        logger.info(f"Pafy: Getting best audio stream for '{video.title}'")
        best_audio = video.getbestaudio()
        if not best_audio:
            logger.error("Pafy: No suitable audio stream found.")
            raise ValueError("Pafy could not find any audio stream for this video.")

        # Define target filename within temp_dir
        # Use a generic name + pafy's extension
        download_filename = f"youtube_audio_download.{best_audio.extension}"
        download_filepath = os.path.join(temp_dir, download_filename)

        logger.info(f"Pafy: Downloading audio stream {best_audio.bitrate} {best_audio.extension} to {download_filepath}")
        # filepath arg directs the download location
        actual_filepath = best_audio.download(filepath=download_filepath, quiet=True) # quiet=True suppresses console progress
        logger.info(f"Pafy: Download complete. File saved at: {actual_filepath}")

        # Sometimes pafy might return a slightly different path/filename than requested, use the returned one
        if not os.path.exists(actual_filepath):
             logger.error(f"Pafy download reported success but file not found at {actual_filepath}")
             # Check original path too just in case
             if os.path.exists(download_filepath):
                 logger.warning(f"Pafy download found at requested path {download_filepath} despite returning different path.")
                 return download_filepath
             raise FileNotFoundError("Downloaded file not found after pafy download.")

        return actual_filepath # Return the path where the file was actually saved

    except (ValueError, IOError, OSError, KeyError, AttributeError) as e: # Catch common pafy/download issues
        # Pafy doesn't have super specific exceptions documented well, catch broad categories
        logger.error(f"Pafy: Error processing URL {youtube_url}: {e}", exc_info=True)
        # Check if it looks like a 4xx error (like 404 Not Found, 403 Forbidden)
        if "HTTP Error 4" in str(e):
             raise ValueError(f"Video not found or access denied ({type(e).__name__}: {e})") # Raise specific error type if possible
        raise RuntimeError(f"Pafy failed to process video/download audio: {type(e).__name__}: {e}")
    except Exception as e: # Catch any other unexpected error
         logger.error(f"Pafy: Unexpected error for {youtube_url}: {e}", exc_info=True)
         raise RuntimeError(f"Unexpected error during Pafy download: {type(e).__name__}: {e}")


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
    Receives video, extracts audio via MoviePy, transcribes via Gemini.
    """
    if not genai or not gemini_model: raise HTTPException(status_code=501, detail="Gemini API not configured.")
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
        transcription = await transcribe_audio_gemini(audio_path)

        # Check if transcription returned an error message
        if transcription.startswith("Error:"):
            logger.warning(f"Transcription failed with message: {transcription}")
            # Return 502 Bad Gateway if the external service failed, 400 if audio was bad
            status_code = 503 if "quota" in transcription.lower() or "permission" in transcription.lower() else \
                          400 if "blocked" in transcription.lower() else 500 # Default internal error
            raise HTTPException(status_code=status_code, detail=transcription)

        logger.info("Transcription successful.")

        # 4. Return the result
        return JSONResponse(content={
            "filename": file.filename,
            "content_type": file.content_type,
            "transcription": transcription,
            "engine": "Google Gemini"
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
    if not pafy: raise HTTPException(status_code=501, detail="Pafy library not available.")
    if not genai or not gemini_model: raise HTTPException(status_code=501, detail="Gemini API not configured.")
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
            _sync_download_partial = functools.partial(
                _sync_download_pafy, # Call the new pafy helper
                payload.youtube_url,
                temp_dir
        )

            # Execute the download function
            downloaded_file_path = await loop.run_in_executor(None, _sync_download_partial)
            logger.info(f"Successfully downloaded YouTube audio to: {downloaded_file_path}")

        except RuntimeError as e: # Catch the specific retry failure error
            logger.error(f"RuntimeError from YouTube download: {e}")
            raise HTTPException(status_code=503, detail=str(e)) # Service Unavailable
        except FileNotFoundError as e:
            logger.error(f"{e}: File not found")
            raise HTTPException(status_code=404, detail=str(e)) # Service Unavailable
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
        transcription = await transcribe_audio_gemini(wav_audio_path)

        # Check for transcription errors
        if transcription.startswith("Error:"):
            logger.warning(f"Transcription failed for YouTube audio with message: {transcription}")
            status_code = 503 if "quota" in transcription.lower() or "permission" in transcription.lower() else \
                          400 if "blocked" in transcription.lower() else 500
            raise HTTPException(status_code=status_code, detail=transcription)

        logger.info("YouTube Transcription successful.")

        # 5. Return the result
        return JSONResponse(content={
            "youtube_url": payload.youtube_url,
            "transcription": transcription,
            "engine": "Google Gemini API"
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