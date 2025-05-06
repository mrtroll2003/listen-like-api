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
from pydantic import BaseModel, HttpUrl, ValidationError, Field, field_validator
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
    "http://localhost:62812",
    "https://listen-like.onrender.com",
    "https://listen-like.onrender.com/",
    "https://listen-like.onrender.com/Result",
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
        gemini_model = genai.GenerativeModel('gemini-2.5-pro-preview-03-25')
        logger.info("Google Gemini client configured successfully with gemini-2.5-pro-preview-03-25.")
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
async def transcribe_via_gemini(source_uri_or_path: str, is_youtube_uri: bool = False) -> str:
    """
    Transcribes audio/video using Gemini.
    Handles either a local file path or a YouTube URI.
    Requests JSON output with timestamps.
    """
    if not genai or not gemini_model:
        raise RuntimeError("Gemini API client not configured or library not available.")

    uploaded_file_part = None
    prompt = """Please provide a transcription for the following video/audio in JSON format.
            The JSON should be an array of objects. Each object must have the following keys:
            - "start": The start time of the spoken segment in seconds (float or integer).
            - "end": The end time of the spoken segment in seconds (float or integer).
            - "text": The transcribed text for that segment (string).
            Example object: {"start": 4.67, "end": 5.83, "text": "All pilots, prepare to sortie."}
            Ensure the entire output is **only** the valid JSON array, starting with '[' and ending with ']'. Do not include markdown fences like ```json or any introductory text."""

    try:
        if is_youtube_uri:
            logger.info(f"Gemini: Using YouTube URI for transcription: {source_uri_or_path}")
            # Directly use the validated YouTube URL as a URI part
            uploaded_file_part = types.Part.from_uri(
                file_uri=source_uri_or_path,
                mime_type="video/youtube" # Inform Gemini it's a YouTube link
            )
        else: # It's a local file path
            # 1. Determine MIME type for local file
            mime_type, _ = mimetypes.guess_type(source_uri_or_path)
            if not mime_type or not (mime_type.startswith("audio/") or mime_type.startswith("video/")):
                # Add specific fallbacks if needed
                if source_uri_or_path.lower().endswith(".mp3"): mime_type = "audio/mpeg"
                elif source_uri_or_path.lower().endswith(".wav"): mime_type = "audio/wav"
                elif source_uri_or_path.lower().endswith(".mp4"): mime_type = "video/mp4"
                # Add others based on expected formats
                else: mime_type = "application/octet-stream"
                logger.warning(f"Could not guess MIME type for local file: {source_uri_or_path}, using {mime_type}")

            # 2. Upload the local file
            logger.info(f"Gemini: Uploading local file: {source_uri_or_path} (MIME: {mime_type})")
            loop = asyncio.get_running_loop()
            upload_func_partial = functools.partial(genai.upload_file, path=source_uri_or_path, mime_type=mime_type)
            uploaded_file_obj = await loop.run_in_executor(None, upload_func_partial)
            logger.info(f"Gemini: Local file uploaded successfully. URI: {uploaded_file_obj.uri}")
            uploaded_file_part = uploaded_file_obj # Use the uploaded file object

        if not uploaded_file_part:
             raise RuntimeError("Failed to prepare input part for Gemini.")

        # 3. Generate content
        logger.info("Gemini: Sending transcription request with JSON format instruction...")
        loop = asyncio.get_running_loop()

        # Prepare contents list correctly
        contents = [uploaded_file_part, prompt] # Send file/uri part AND prompt part

        generate_func_partial = functools.partial(
            gemini_model.generate_content,
            contents=contents
            # Specify JSON mode if model supports it explicitly (check Gemini docs/versions)
            # generation_config=genai.GenerationConfig(response_mime_type="application/json")
        )
        response = await loop.run_in_executor(None, generate_func_partial)

        # Check for potential blocks or empty responses
        if not response.candidates:
             logger.warning("Gemini: Response has no candidates.")
             block_reason = getattr(response.prompt_feedback, 'block_reason', None)
             reason = f"Blocked by safety filters: {block_reason}" if block_reason else "No response candidates"
             raise RuntimeError(f"Transcription failed ({reason})")

        # Extract text - expecting JSON string
        transcription_json_string = response.text
        logger.info("Gemini: Transcription response received.")
        # logger.debug(f"Raw Gemini Transcription Response:\n{transcription_json_string}") # Debug

        # Validate and return the JSON string
        try:
             # Clean potential markdown fences just in case model adds them
             cleaned_json = transcription_json_string.strip().removeprefix("```json").removesuffix("```").strip()
             # Try parsing to ensure it's valid JSON
             json.loads(cleaned_json)
             return cleaned_json # Return the JSON string itself
        except json.JSONDecodeError as json_err:
             logger.error(f"Gemini response was not valid JSON: {json_err}. Response text: {transcription_json_string}")
             raise RuntimeError("Transcription failed: Model did not return valid JSON.")

    except (google_api_exceptions.PermissionDenied, google_api_exceptions.ResourceExhausted) as e:
        # Handle specific Google API errors
        status = "Permission Denied" if isinstance(e, google_api_exceptions.PermissionDenied) else "Quota Exceeded"
        logger.error(f"Gemini API {status}: {e}", exc_info=True)
        raise RuntimeError(f"Transcription failed: Gemini API {status.lower()}. Check key/quota.")
    except Exception as e:
        logger.error(f"Gemini transcription failed: {e}", exc_info=True)
        raise RuntimeError(f"An unexpected error occurred during Gemini transcription ({type(e).__name__}).")
    finally:
        # Optional: Delete the uploaded file from Gemini storage if desired
        # This might require tracking the uploaded_file.name and calling genai.delete_file()
        # For simplicity, we omit this now, but be aware files might persist temporarily.
        if uploaded_file_part:
            logger.debug(f"Gemini: Uploaded file object: name={uploaded_file_part.name}, uri={uploaded_file_part.uri}")
            # Consider adding deletion logic here if managing storage is critical


# --- Pydantic Models ---
class TranslationRequest(BaseModel):
    text: str
    target_language: str

class YouTubeTranscribeRequest(BaseModel):
    youtube_url: HttpUrl # Use Pydantic's HttpUrl for basic validation

    # Add a validator to be more specific about YouTube URLs
    @field_validator('youtube_url')
    @classmethod
    def check_youtube_url(cls, value: HttpUrl):
        host = value.host.lower() if value.host else ''
        is_youtube_domain = host == 'youtube.com' or \
                              host == 'www.youtube.com' or \
                              host == 'm.youtube.com' or \
                              host == 'youtu.be'
        if not is_youtube_domain:
            raise ValueError('URL must be a valid YouTube domain (youtube.com, youtu.be)')
        # Basic structure check (can be enhanced)
        if host == 'youtu.be' and (value.path is None or len(value.path) <= 1):
             raise ValueError('youtu.be URL must have a video ID path')
        if 'youtube.com' in host and value.path != '/watch' and not value.path.startswith('/shorts/') and not value.path.startswith('/embed/'):
             # Allow query param 'v' even if path isn't exactly /watch (less strict)
             if 'v' not in value.query_params:
                  raise ValueError('youtube.com URL must be a watch, shorts, or embed link with video ID')
        return value

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
async def transcribe_video(file: Optional[UploadFile] = File(None), payload: Optional[YouTubeTranscribeRequest] = Body(None)):
    """
    Transcribes video/audio from either a FILE UPLOAD or a YOUTUBE URL.
    Requires EITHER 'file' (multipart/form-data) OR 'youtube_url' (application/json).
    Returns transcription as a JSON string with timestamps.
    """
    if not genai or not gemini_model: raise HTTPException(status_code=501, detail="Gemini API not configured.")
    if not mp:
        raise HTTPException(status_code=501, detail="MoviePy library not available.")

    if not file.filename:
         raise HTTPException(status_code=400, detail="No filename provided.")

    # Use a temporary directory for robust cleanup
    source_description = "Unknown"
    transcription_json = None
    if file and payload:
         raise HTTPException(status_code=400, detail="Provide EITHER a file upload OR a youtube_url in the body, not both.")
    if file:
        if not file.filename:
             raise HTTPException(status_code=400, detail="No filename provided with file upload.")
        if not mp:
             raise HTTPException(status_code=501, detail="MoviePy library (for file processing) not available.")

        source_description = f"File: {file.filename}"
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="api_upload_")
        temp_dir = temp_dir_obj.name
        logger.info(f"Created temporary directory for file upload: {temp_dir}")
        audio_path = None # Path to extracted audio

        try:
            # Save video temporarily
            safe_filename = Path(file.filename).name
            video_path = os.path.join(temp_dir, safe_filename)
            logger.info(f"Saving uploaded video to: {video_path}")
            try:
                with open(video_path, "wb") as buffer:
                    while chunk := await file.read(8192): buffer.write(chunk)
            except Exception as e:
                logger.error(f"Failed to save uploaded file: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Could not save uploaded file.")
            finally:
                await file.close()

            # Extract audio
            base_name, _ = os.path.splitext(safe_filename)
            audio_filename = f"{base_name}_extracted_audio.wav"
            audio_path = os.path.join(temp_dir, audio_filename)
            logger.info(f"Extracting audio using MoviePy to: {audio_path}")
            await extract_audio_moviepy(video_path, audio_path)

            # Transcribe extracted audio
            if not os.path.exists(audio_path):
                 raise HTTPException(status_code=500, detail="Audio extraction did not produce a file.")
            logger.info(f"Starting transcription for extracted audio: {audio_path}")
            transcription_json = await transcribe_via_gemini(audio_path, is_youtube_uri=False)

        except (RuntimeError, ValueError, HTTPException) as e:
             # Catch specific errors from helpers or explicit HTTP exceptions
             detail = getattr(e, 'detail', str(e)) # Get detail if HTTPException
             status_code = getattr(e, 'status_code', 500) # Get status if HTTPException
             logger.error(f"Error processing file upload: {detail}")
             raise HTTPException(status_code=status_code, detail=detail)
        except Exception as e:
             logger.error(f"Unexpected error processing file upload: {e}", exc_info=True)
             raise HTTPException(status_code=500, detail="Internal error during file processing.")
        finally:
            logger.info(f"Cleaning up temporary directory: {temp_dir}")
            temp_dir_obj.cleanup()

    # --- Handle YouTube URL ---
    elif payload and payload.youtube_url:
         # Pydantic already performed basic + custom validation
         source_description = f"YouTube URL: {payload.youtube_url}"
         youtube_uri = str(payload.youtube_url) # Convert HttpUrl back to string for Gemini
         try:
             logger.info(f"Starting transcription for YouTube URL: {youtube_uri}")
             # Directly pass the validated URL string to the Gemini helper
             transcription_json = await transcribe_via_gemini(youtube_uri, is_youtube_uri=True)
         except RuntimeError as e: # Catch errors from transcribe_via_gemini
              logger.error(f"Error processing YouTube URL: {e}")
              # Decide status code based on error type if possible (e.g., 404, 403 from Gemini)
              raise HTTPException(status_code=503, detail=f"Transcription service failed: {e}")
         except Exception as e:
              logger.error(f"Unexpected error processing YouTube URL: {e}", exc_info=True)
              raise HTTPException(status_code=500, detail="Internal error during YouTube processing.")

    # --- No Input Provided ---
    else:
        raise HTTPException(status_code=400, detail="No input provided. Send either a 'file' upload or a JSON body with 'youtube_url'.")

    # --- Return Result ---
    if transcription_json:
        logger.info(f"Transcription successful for {source_description}.")
        # Return raw JSON string, Flutter will parse it
        return JSONResponse(content={
            "transcription_json": transcription_json, # Key name indicates JSON content
            "engine": "Google Gemini"
        })
    else:
        # Should have been caught by exceptions, but as a fallback
        raise HTTPException(status_code=500, detail="Transcription failed for unknown reasons.")

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