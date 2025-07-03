# backend/main.py
import asyncio
import os
import io
import json
import logging
from starlette.websockets import WebSocketState
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import texttospeech_v1 as tts
import google.generativeai as genai # For Gemini LLM
from google.generativeai.types import GenerationConfig
from google.cloud import dialogflow_v2 as dialogflow
from google.api_core.exceptions import GoogleAPIError

from config import settings # Our settings from config.py

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Allow CORS for your frontend
origins = [
    "http://localhost:5173", # Default Vite dev server port
    # Add your deployed frontend URL here when you deploy
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Google Cloud Clients Initialization ---
# Ensure GOOGLE_APPLICATION_CREDENTIALS environment variable is set
# This is handled by config.py, so these clients will pick it up automatically.
try:
    speech_client = speech.SpeechClient()
    tts_client = tts.TextToSpeechClient()
    #gemini_client = get_default_retrying_async_client() # Async client for Gemini
    dialogflow_sessions_client = dialogflow.SessionsClient()
    logger.info("Google Cloud clients initialized successfully.")

    # Configure Gemini API directly
    if settings.GEMINI_API_KEY: # Assuming you have GEMINI_API_KEY in your config.py
        genai.configure(api_key=settings.GEMINI_API_KEY)
        logger.info("Gemini API configured using API key.")
    else:
        logger.warning("GEMINI_API_KEY not found in config.py. Gemini API calls might fail.")
    
except GoogleAPIError as e:
    logger.error(f"Failed to initialize Google Cloud clients: {e}")
    logger.warning("Ensure your GOOGLE_APPLICATION_CREDENTIALS environment variable is set correctly and the service account has the necessary permissions.")
    # Consider raising an exception or having a fallback if this is critical for startup

# Gemini Model
# Using gemini-1.5-pro for general purpose, consider specialized models for specific tasks
gemini_model = genai.GenerativeModel('gemini-1.5-pro')
gemini_config = GenerationConfig(temperature=0.7, top_p=0.9, top_k=40)


# --- HTML for simple testing (optional) ---
# You can use this to quickly test the WebSocket if your React app isn't ready
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>FastAPI WebSocket Chat</title>
    </head>
    <body>
        <h1>WebSocket Chat</h1>
        <button id="startButton">Start Recording</button>
        <button id="stopButton" disabled>Stop Recording</button>
        <div id="transcript"></div>
        <div id="aiResponse"></div>
        <script>
            let ws;
            let mediaRecorder;
            let audioContext;
            let audioInput;
            let processor;

            const startButton = document.getElementById('startButton');
            const stopButton = document.getElementById('stopButton');
            const transcriptDiv = document.getElementById('transcript');
            const aiResponseDiv = document.getElementById('aiResponse');

            startButton.onclick = async () => {
                startButton.disabled = true;
                stopButton.disabled = false;
                transcriptDiv.textContent = '';
                aiResponseDiv.textContent = '';

                ws = new WebSocket("ws://localhost:8000/ws");

                ws.onopen = () => {
                    console.log("WebSocket opened");
                    // Start microphone access
                    navigator.mediaDevices.getUserMedia({ audio: true })
                        .then(stream => {
                            audioContext = new (window.AudioContext || window.webkitAudioContext)();
                            audioInput = audioContext.createMediaStreamSource(stream);
                            processor = audioContext.createScriptProcessor(4096, 1, 1);

                            processor.onaudioprocess = (event) => {
                                const inputData = event.inputBuffer.getChannelData(0);
                                // Convert Float32Array to Int16Array (LINEAR16 for Google STT)
                                const int16Array = new Int16Array(inputData.length);
                                for (let i = 0; i < inputData.length; i++) {
                                    int16Array[i] = Math.max(-1, Math.min(1, inputData[i])) * 0x7FFF;
                                }
                                if (ws.readyState === WebSocket.OPEN) {
                                    ws.send(int16Array.buffer);
                                }
                            };

                            audioInput.connect(processor);
                            processor.connect(audioContext.destination);
                            mediaRecorder = stream; // Store stream to stop it later
                        })
                        .catch(err => console.error("Error accessing microphone:", err));
                };

                ws.onmessage = (event) => {
                    if (typeof event.data === 'string') {
                        // Assuming AI response text
                        aiResponseDiv.textContent += event.data;
                    } else if (event.data instanceof Blob) {
                        // Assuming AI audio response
                        const audioUrl = URL.createObjectURL(event.data);
                        const audio = new Audio(audioUrl);
                        audio.play();
                        audio.onended = () => URL.revokeObjectURL(audioUrl);
                    }
                };

                ws.onclose = () => {
                    console.log("WebSocket closed");
                };

                ws.onerror = (error) => {
                    console.error("WebSocket Error:", error);
                };
            };

            stopButton.onclick = () => {
                startButton.disabled = false;
                stopButton.disabled = true;

                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send("END_OF_SPEECH"); // Signal end of speech to backend
                    ws.close();
                }

                if (mediaRecorder) {
                    mediaRecorder.getTracks().forEach(track => track.stop());
                }
                if (processor) {
                    processor.disconnect();
                    audioInput.disconnect();
                }
                if (audioContext) {
                    audioContext.close();
                }
            };
        </script>
    </body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html)


# --- Core WebSocket Endpoint ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established.")

    # Speech-to-Text configuration
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, # Mono, 16-bit signed little-endian
        sample_rate_hertz=16000, # This needs to match frontend
        language_code="en-US",
        enable_automatic_punctuation=True,
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=False, # Set to True if you want to send partial transcripts back
        single_utterance=True, # Stop processing after user stops speaking (useful for push-to-talk)
    )

    # Dialogflow session ID (unique per user conversation)
    session_id = str(websocket.client.host) + "-" + str(websocket.client.port) # Basic unique ID
    dialogflow_session_path = dialogflow_sessions_client.session_path(
        settings.GOOGLE_CLOUD_PROJECT_ID, session_id
    )
    logger.info(f"Dialogflow Session Path: {dialogflow_session_path}")


    recognize_requests_queue = asyncio.Queue()
    recognize_responses_queue = asyncio.Queue()
    full_transcript = []
    
    # Flag to control streaming
    is_recording_active = True

    async def consume_audio_stream():
        nonlocal is_recording_active
        # This generator sends audio to Google STT
        while is_recording_active:
            try:
                audio_chunk = await websocket.receive_bytes()
                if audio_chunk == b'END_OF_SPEECH': # Custom signal from frontend
                    logger.info("Received END_OF_SPEECH signal from client.")
                    is_recording_active = False
                    break # Exit loop
                recognize_requests_queue.put_nowait(speech.StreamingRecognizeRequest(audio_content=audio_chunk))
            except WebSocketDisconnect:
                logger.info("Client disconnected during audio consumption.")
                is_recording_active = False
                break
            except Exception as e:
                logger.error(f"Error consuming audio stream: {e}")
                is_recording_active = False
                break

        # Signal end of input to Google STT
        recognize_requests_queue.put_nowait(None) # Signal end of stream to the generator

    async def transcribe_speech():
        nonlocal full_transcript
        try:
            # Create an async generator for the Speech-to-Text requests
            async def request_generator():
                while True:
                    request = await recognize_requests_queue.get()
                    if request is None: # Stop signal
                        break
                    yield request

            # Perform the streaming recognition
            responses = speech_client.streaming_recognize(streaming_config, request_generator())
            async for response in responses:
                for result in response.results:
                    if result.alternatives:
                        transcript_text = result.alternatives[0].transcript
                        logger.info(f"STT Transcript: {transcript_text}")
                        # Send transcript back to frontend (optional, for display)
                        await websocket.send_text(f"You: {transcript_text}") # Send to frontend for display
                        full_transcript.append(transcript_text)

                        if result.is_final:
                            # Process final transcript with Dialogflow/LLM
                            await process_final_transcript(transcript_text)
                            recognize_requests_queue.put_nowait(None) # Stop STT processing for this utterance
                            return # Exit transcription loop after final result for single_utterance
            logger.info("STT finished processing.")

        except Exception as e:
            logger.error(f"Error during speech transcription: {e}")
            is_recording_active = False # Ensure recording stops if STT fails

    async def process_final_transcript(transcript_text: str):
        nonlocal is_recording_active
        logger.info(f"Processing final transcript: '{transcript_text}'")

        if not transcript_text.strip():
            logger.info("Empty transcript, skipping LLM/Dialogflow.")
            return

        # --- Option 1: Direct LLM (Gemini) Call ---
        # response_text = await get_gemini_response(transcript_text)

        # --- Option 2: Dialogflow Integration (Recommended for structure) ---
        response_text = await get_dialogflow_response(transcript_text, dialogflow_session_path)


        if response_text:
            logger.info(f"AI Response Text: {response_text}")
            await stream_tts_and_send_to_client(response_text)
        else:
            await websocket.send_text("I'm sorry, I couldn't generate a response.")


    async def get_gemini_response(prompt: str) -> str:
        """Calls the Gemini LLM directly."""
        try:
            # For conversational memory, you would pass previous messages to the LLM
            # For simplicity, this is stateless.
            response = await gemini_model.generate_content_async(
                prompt,
                generation_config=gemini_config,
                # For conversational memory, add history here:
                # history=[
                #     {'role':'user', 'parts': ["Hello"]},
                #     {'role':'model', 'parts': ["Hi there! How can I help you today?"]},
                #     ...
                # ]
            )
            # Access the text from the response safely
            return response.candidates[0].content.parts[0].text
        except Exception as e:
            logger.error(f"Error calling Gemini LLM: {e}")
            return "I'm sorry, I'm having trouble connecting to the AI at the moment."

    async def get_dialogflow_response(text: str, session_path: str) -> str:
        """Sends text to Dialogflow and gets a response."""
        query_input = dialogflow.QueryInput(text=dialogflow.TextInput(text=text, language_code="en-US"))
        try:
            response = await dialogflow_sessions_client.detect_intent(
                session=session_path, query_input=query_input
            )
            logger.info(f"Dialogflow Query Result: {response.query_result.fulfillment_text}")
            # If Dialogflow's fulfillment is just text, use that.
            # If it triggers a webhook to your backend, your webhook logic would handle it.
            return response.query_result.fulfillment_text
        except Exception as e:
            logger.error(f"Error calling Dialogflow: {e}")
            return "I'm sorry, I'm having trouble with my understanding right now."


    async def stream_tts_and_send_to_client(text: str):
        """Generates speech from text and streams audio back to the client."""
        synthesis_input = tts.SynthesisInput(text=text)
        voice = tts.VoiceSelectionParams(
            language_code="en-US",
            ssml_gender=tts.SsmlVoiceGender.NEUTRAL, # Can be MALE, FEMALE, NEUTRAL
            name="en-US-Neural2-C" # A natural sounding voice, check Google TTS docs for options
        )
        audio_config = tts.AudioConfig(
            audio_encoding=tts.AudioEncoding.LINEAR16, # Raw PCM, 16-bit, mono
            sample_rate_hertz=16000 # Important: Match this with frontend audio playback!
        )

        try:
            # StreamingSynthesize is not directly supported by google-cloud-texttospeech for Python.
            # It's usually a synchronous request for a full audio blob.
            # To simulate streaming, we'll synthesize the full text and then send in chunks.
            # For true streaming TTS, you'd typically need a different API or a self-hosted model.
            response = tts_client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )
            audio_content = response.audio_content

            # Send the AI's text response first
            await websocket.send_text(text) # Send the full text of AI's response

            # Chunk the audio and send it
            chunk_size = 4096 # Adjust as needed
            for i in range(0, len(audio_content), chunk_size):
                chunk = audio_content[i:i + chunk_size]
                if websocket.client_state == WebSocketState.CONNECTED: # Ensure client is still connected
                    await websocket.send_bytes(chunk)
                await asyncio.sleep(0.001) # Small delay to yield control
            logger.info("TTS audio streamed to client.")

        except Exception as e:
            logger.error(f"Error during TTS generation/streaming: {e}")
            await websocket.send_text("I'm sorry, I can't speak right now.")

    # Start tasks
    try:
        # Create the audio consumer task
        audio_consumer_task = asyncio.create_task(consume_audio_stream())
        
        # Create the transcription task
        transcribe_task = asyncio.create_task(transcribe_speech())

        # Wait for either task to finish or client to disconnect
        done, pending = await asyncio.wait(
            [audio_consumer_task, transcribe_task],
            return_when=asyncio.FIRST_COMPLETED
        )

        # Ensure all tasks are properly cancelled if one finishes
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True) # Wait for cancellation to complete

        logger.info("WebSocket tasks finished.")

    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket endpoint error: {e}")
    finally:
        is_recording_active = False # Ensure flag is set to false
        await websocket.close()