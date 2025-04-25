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
from pydantic import BaseModel, HttpUrl, ValidationError, Field
from typing import Optional, List # Added List
import re # Added for regex
from urllib.parse import urlparse

import time #for yt transcribe timeout
import urllib.error #for yt transcribe errcatch
import json # Added for JSON parsing


import wave     #For logging/
import contextlib #debugging

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
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

app = FastAPI(title="Simplified Media API")

# --- Cross origin (CORS) import
origins = [
    "http://localhost", # Base localhost
    "http://localhost:8080", # Example default Flutter web server port
    "http://localhost:55363", # The port shown in your flutter run output
    "http://localhost:62812/",
    # Add the URL where your Flutter Web app will be HOSTED eventually
    # e.g., "https://your-flutter-app.vercel.app",
    # e.g., "https://your-flutter-app-on-render.onrender.com"
    # "*" # Allow all origins (use with caution)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # List of allowed origins
    allow_credentials=True, # Allows cookies (if you use them later)
    allow_methods=["*"], # Allow all methods (GET, POST, PUT, etc.)
    allow_headers=["*"], # Allow all headers
)

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
# YOUTUBE_DOMAINS = {
#     "youtube.com",
#     "www.youtube.com",
#     "m.youtube.com",
#     "youtu.be",
# }

# def is_valid_youtube_url(url: str) -> bool:
#     """Checks if a string is a valid HTTP/HTTPS URL pointing to a known YouTube domain."""
#     try:
#         # Basic URL structure validation
#         parsed_url = urlparse(url)
#         if not all([parsed_url.scheme in ["http", "https"], parsed_url.netloc]):
#             logger.warning(f"URL validation failed (scheme/netloc): {url}")
#             return False

#         # Check if the domain is a known YouTube domain
#         domain = parsed_url.netloc.lower()
#         if domain not in YOUTUBE_DOMAINS:
#             # Handle youtu.be separately as netloc is just 'youtu.be'
#              if domain == 'youtu.be' and parsed_url.path and len(parsed_url.path) > 1:
#                  return True # youtu.be/VIDEO_ID is valid
#              logger.warning(f"URL validation failed (domain not YouTube): {domain}")
#              return False

#         # For youtube.com domains, check for /watch path (basic check)
#         if "youtube.com" in domain: pass
#         return True

#     except Exception as e:
#         logger.error(f"Error during URL parsing: {e}", exc_info=True)
#         return False
    
# def _sync_download_pafy(youtube_url: str, temp_dir: str) -> str:
#     """Synchronous helper to download audio using pafy."""
#     if not pafy:
#         raise RuntimeError("Pafy library is not available.")
#     try:
#         logger.info(f"Pafy: Creating video object for {youtube_url}")
#         video = pafy.new(youtube_url)

#         logger.info(f"Pafy: Getting best audio stream for '{video.title}'")
#         best_audio = video.getbestaudio()
#         if not best_audio:
#             logger.error("Pafy: No suitable audio stream found.")
#             raise ValueError("Pafy could not find any audio stream for this video.")

#         # Define target filename within temp_dir
#         # Use a generic name + pafy's extension
#         download_filename = f"youtube_audio_download.{best_audio.extension}"
#         download_filepath = os.path.join(temp_dir, download_filename)

#         logger.info(f"Pafy: Downloading audio stream {best_audio.bitrate} {best_audio.extension} to {download_filepath}")
#         # filepath arg directs the download location
#         actual_filepath = best_audio.download(filepath=download_filepath, quiet=True) # quiet=True suppresses console progress
#         logger.info(f"Pafy: Download complete. File saved at: {actual_filepath}")

#         # Sometimes pafy might return a slightly different path/filename than requested, use the returned one
#         if not os.path.exists(actual_filepath):
#              logger.error(f"Pafy download reported success but file not found at {actual_filepath}")
#              # Check original path too just in case
#              if os.path.exists(download_filepath):
#                  logger.warning(f"Pafy download found at requested path {download_filepath} despite returning different path.")
#                  return download_filepath
#              raise FileNotFoundError("Downloaded file not found after pafy download.")

#         return actual_filepath # Return the path where the file was actually saved

#     except (ValueError, IOError, OSError, KeyError, AttributeError) as e: # Catch common pafy/download issues
#         # Pafy doesn't have super specific exceptions documented well, catch broad categories
#         logger.error(f"Pafy: Error processing URL {youtube_url}: {e}", exc_info=True)
#         # Check if it looks like a 4xx error (like 404 Not Found, 403 Forbidden)
#         if "HTTP Error 4" in str(e):
#              raise ValueError(f"Video not found or access denied ({type(e).__name__}: {e})") # Raise specific error type if possible
#         raise RuntimeError(f"Pafy failed to process video/download audio: {type(e).__name__}: {e}")
#     except Exception as e: # Catch any other unexpected error
#          logger.error(f"Pafy: Unexpected error for {youtube_url}: {e}", exc_info=True)
#          raise RuntimeError(f"Unexpected error during Pafy download: {type(e).__name__}: {e}")


# --- Pydantic Models ---
class TranslationRequest(BaseModel):
    text: str
    target_language: str

# class YouTubeTranscribeRequest(BaseModel):
#     youtube_url: str
#     # Future: add language hint, model choice etc.
class QuestionGenerationRequest(BaseModel):
    transcript: str = Field(..., description="The transcribed text from the audio.")
    num_questions: int = Field(default=7, ge=3, le=15, description="Approximate number of questions to generate.")
    # Optional: Specify desired types, otherwise model chooses variety
    question_types: Optional[List[str]] = Field(default=None, description="Optional list of desired question types (e.g., ['Multiple Choice', 'Sentence Completion']).")
    custom_prompt_instructions: Optional[str] = Field(default=None, description="Optional: Add custom instructions for the generation prompt.")

# --- API Endpoints ---

@app.get("/api", tags=["Status"])
async def api_root():
    """ Root API endpoint providing status. """
    now = datetime.datetime.utcnow().isoformat()
    return JSONResponse(content={
        "message": "Welcome to the Simplified Media API!",
        "status": "ok",
        "timestamp_utc": now,
        "endpoints": ["/api/transcribe", "/api/translate", "/api/generate_question", "/docs"]
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

@app.post("/api/generate_questions", tags=["Question Generation"])
async def generate_ielts_questions(payload: QuestionGenerationRequest):
    """
    Generates IELTS Listening-style questions based on provided transcript text using Gemini.
    """
    if not gemini_model:
        raise HTTPException(status_code=501, detail="Gemini API client not configured.")

    logger.info(f"Received request to generate ~{payload.num_questions} questions.")

    # --- Construct the Prompt ---
    prompt_parts = [
        "You are an expert creator of IELTS Listening test questions.",
        "Based *only* on the following transcript text, generate a set of IELTS Listening-style questions suitable for testing comprehension.",
        f"Generate approximately {payload.num_questions} questions in total.",
        "Include a variety of question types commonly found in IELTS Listening, such as:",
        "  - Multiple Choice (with options A, B, C, etc.)",
        "  - Sentence Completion (provide the sentence with a blank to fill)",
        "  - Short Answer Questions (requiring a brief answer based on the text)",
        "  - Matching (if distinct categories or items are discussed that can be matched)",
        # Note: Map/Diagram Labeling is difficult without visuals, focus on text-based types unless explicitly asked via custom prompt.
        "Ensure the questions directly test information stated or clearly implied in the transcript."
    ]

    # Add optional user-specified types
    if payload.question_types:
        types_str = ", ".join(payload.question_types)
        prompt_parts.append(f"Focus primarily on these question types if possible: {types_str}.")
    else:
        prompt_parts.append("Ensure a good mix of different applicable question types.")

    # Add optional custom instructions (ALLOWS FOR USER PROMPT ENGINEERING)
    if payload.custom_prompt_instructions:
        prompt_parts.append("\nFollow these additional instructions carefully:")
        prompt_parts.append(payload.custom_prompt_instructions)

    # Specify desired output format (JSON)
    prompt_parts.append("\nFormat the output STRICTLY as a single JSON object containing a single key 'questions'.")
    prompt_parts.append("The value of 'questions' should be a JSON array.")
    prompt_parts.append("Each element in the array should be a JSON object representing one question, containing:")
    prompt_parts.append("  - 'type': A string indicating the question type (e.g., 'Multiple Choice', 'Sentence Completion', 'Short Answer', 'Matching').")
    prompt_parts.append("  - 'question_text': The full text of the question (including blanks for completion types).")
    prompt_parts.append("  - 'options': (Optional) An array of strings for multiple choice options.")
    prompt_parts.append("  - 'answer_guidance': (Optional) A brief note indicating where or how the answer can be found in the text (do NOT provide the actual answer itself).")

    prompt_parts.append("\nHere is the transcript:\n--- TRANSCRIPT START ---")
    prompt_parts.append(payload.transcript)
    prompt_parts.append("--- TRANSCRIPT END ---")
    prompt_parts.append("\nGenerate the JSON object now:")

    final_prompt = "\n".join(prompt_parts)
    # logger.debug(f"Gemini Prompt for Question Gen:\n{final_prompt}") # Uncomment to log the full prompt if debugging

    # --- Call Gemini API ---
    try:
        logger.info("Sending request to Gemini API for question generation...")
        loop = asyncio.get_running_loop()

        # Use partial for the potentially long-running call
        generate_func_partial = functools.partial(
            gemini_model.generate_content,
            final_prompt
            # Add generation_config here if needed (e.g., temperature, max_output_tokens)
            # generation_config=genai.types.GenerationConfig(temperature=0.7)
        )

        response = await loop.run_in_executor(None, generate_func_partial)

        # --- Process Response ---
        if not response.candidates:
             logger.warning("Gemini question gen: Response has no candidates.")
             block_reason = getattr(response.prompt_feedback, 'block_reason', None)
             detail = f"Request blocked by safety filters: {block_reason}" if block_reason else "Generation failed (No response candidates)"
             raise HTTPException(status_code=400, detail=detail)

        generated_text = response.text
        logger.info("Gemini question generation successful.")
        # logger.debug(f"Raw Gemini Response Text:\n{generated_text}") # Uncomment for debugging

        # Attempt to parse the response as JSON
        try:
            # Clean potential markdown code fences
            cleaned_text = generated_text.strip().removeprefix("```json").removesuffix("```").strip()
            questions_data = json.loads(cleaned_text)

            # Basic validation of expected structure
            if not isinstance(questions_data, dict) or "questions" not in questions_data or not isinstance(questions_data["questions"], list):
                 logger.error(f"Gemini response parsed but has unexpected structure: {questions_data}")
                 raise HTTPException(status_code=500, detail="Generated response has unexpected format, expected {'questions': [...]} ")

            logger.info(f"Successfully parsed JSON response. Found {len(questions_data['questions'])} questions.")
            return questions_data # Return the full dict {'questions': [...]}

        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to parse Gemini response as JSON: {json_err}")
            logger.error(f"Raw text received: {generated_text}")
            # Return the raw text with a warning if JSON parsing fails
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse generated questions as JSON. Raw response: {generated_text}"
            )

    # --- Error Handling for API call ---
    except google_api_exceptions.PermissionDenied as e:
         if isinstance(e, google_api_exceptions.PermissionDenied):
             logger.error(f"Gemini API Permission Denied: {e}", exc_info=True)
             raise HTTPException(status_code=503, detail="Gemini API permission denied.")
    except google_api_exceptions.ResourceExhausted as e:
         if isinstance(e, google_api_exceptions.ResourceExhausted):
             logger.error(f"Gemini API Quota Exceeded: {e}", exc_info=True)
             raise HTTPException(status_code=503, detail="Gemini API quota limit reached.")
    except Exception as e:
        logger.error(f"Question generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during question generation ({type(e).__name__}).")
# --- Optional: Add hello endpoint or others back if needed ---
# @app.get("/api/hello", tags=["Greeting"])
# async def hello_endpoint(name: str = "World"):
#     return JSONResponse(content={"greeting": f"Hello, {name}!"})