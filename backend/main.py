# backend/main.py
import asyncio
import os
import io
import json
import logging
import queue # Import queue module for synchronous queue
from starlette.websockets import WebSocketState # Import WebSocketState for connection check
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import speech_v1p1beta1 as speech # Async Speech-to-Text client
from google.cloud import texttospeech_v1 as tts # Async Text-to-Speech client
import vertexai # Import Vertex AI SDK
from vertexai.generative_models import GenerativeModel, GenerationConfig # For Gemini LLM via Vertex AI
from google.cloud import dialogflow_v2 as dialogflow # Async Dialogflow client
from google.api_core.exceptions import GoogleAPIError

from config import settings # Our settings from config.py

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Allow CORS for your frontend
origins = [
    "http://localhost:5173", # Default Vite dev server port
    # Add your deployed frontend URL here when you deploy, e.g., "https://your-app.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Google Cloud Clients Initialization ---
# Ensure GOOGLE_APPLICATION_CREDENTIALS environment variable is set.
# This is handled by config.py, so these clients will pick it up automatically.
try:
    # Initialize Google Cloud clients. These use Application Default Credentials
    # which are picked up from the GOOGLE_APPLICATION_CREDENTIALS env var.
    speech_client = speech.SpeechClient()
    tts_client = tts.TextToSpeechClient()
    dialogflow_sessions_client = dialogflow.SessionsClient()
    logger.info("Google Cloud clients (Speech, TTS, Dialogflow) initialized successfully using Service Account.")

    # Initialize Vertex AI for Gemini models
    # The `project` and `location` are crucial for Vertex AI.
    # 'us-central1' is a common region for Vertex AI.
    vertexai.init(project=settings.GOOGLE_CLOUD_PROJECT_ID, location="us-central1")
    logger.info(f"Vertex AI initialized for project: {settings.GOOGLE_CLOUD_PROJECT_ID}, location: us-central1")

    # Using gemini-1.5-pro for general purpose. For production, consider `gemini-1.5-flash` for lower latency.
    # This model instance will use the credentials from vertexai.init()
    gemini_model = GenerativeModel('gemini-1.5-pro')
    logger.info("Gemini model ('gemini-1.5-pro') initialized via Vertex AI.")

except GoogleAPIError as e:
    logger.error(f"Failed to initialize Google Cloud clients: {e}")
    logger.warning("Ensure your GOOGLE_APPLICATION_CREDENTIALS environment variable is set correctly and the service account has the necessary permissions (Speech-to-Text Admin, Text-to-Speech Admin, Dialogflow API Client, Vertex AI User).")
    # In a production app, you might want to raise an exception or have a more robust fallback.
    # For now, we'll allow the app to start but log the error.
except Exception as e:
    logger.error(f"An unexpected error occurred during client initialization: {e}")
    logger.warning("Please check your environment setup and API configurations.")


# Gemini Generation Configuration
gemini_config = GenerationConfig(temperature=0.7, top_p=0.9, top_k=40)


# --- HTML for simple testing (optional) ---
# This HTML provides a basic interface to test the WebSocket connection and audio streaming
# directly from the browser, without needing the React app initially.
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>FastAPI WebSocket Chat</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            button { padding: 10px 20px; font-size: 1em; margin: 5px; cursor: pointer; }
            #transcript, #aiResponse {
                margin-top: 20px;
                padding: 15px;
                border: 1px solid #ccc;
                border-radius: 8px;
                background-color: #f9f9f9;
                min-height: 50px;
            }
            #aiResponse { border-color: #2980b9; background-color: #e8f6ff; }
            .recording { background-color: #e74c3c; color: white; }
        </style>
    </head>
    <body>
        <h1>WebSocket Chat</h1>
        <button id="startButton">Start Recording</button>
        <button id="stopButton" disabled>Stop Recording</button>
        <div id="transcript"><strong>You:</strong></div>
        <div id="aiResponse"><strong>AI:</strong></div>
        <script>
            let ws;
            let mediaStream; // Renamed from mediaRecorder for clarity
            let audioContext;
            let audioInput;
            let processor;
            let currentSampleRate = 16000; // Default, will be updated by actual AudioContext sampleRate

            const startButton = document.getElementById('startButton');
            const stopButton = document.getElementById('stopButton');
            const transcriptDiv = document.getElementById('transcript');
            const aiResponseDiv = document.getElementById('aiResponse');

            startButton.onclick = async () => {
                startButton.disabled = true;
                stopButton.disabled = false;
                transcriptDiv.innerHTML = '<strong>You:</strong> '; // Reset content
                aiResponseDiv.innerHTML = '<strong>AI:</strong> '; // Reset content
                startButton.classList.add('recording'); // Add recording style

                ws = new WebSocket("ws://localhost:8000/ws");

                ws.onopen = () => {
                    console.log("WebSocket opened");
                    // Start microphone access
                    navigator.mediaDevices.getUserMedia({ audio: true })
                        .then(stream => {
                            mediaStream = stream; // Store stream to stop it later
                            audioContext = new (window.AudioContext || window.webkitAudioContext)();
                            currentSampleRate = audioContext.sampleRate; // Get actual sample rate from browser
                            console.log("AudioContext sample rate:", currentSampleRate);

                            // Send sample rate to backend immediately after opening WebSocket
                            // This is crucial for STT configuration on the backend.
                            if (ws.readyState === WebSocket.OPEN) {
                                ws.send(JSON.stringify({ type: 'sample_rate', value: currentSampleRate }));
                            }

                            audioInput = audioContext.createMediaStreamSource(mediaStream);
                            // Create a ScriptProcessorNode to get raw audio data.
                            // Buffer size (4096), input channels (1), output channels (1).
                            processor = audioContext.createScriptProcessor(4096, 1, 1);

                            processor.onaudioprocess = (event) => {
                                const inputData = event.inputBuffer.getChannelData(0);
                                // Convert Float32Array (from Web Audio API) to Int16Array (LINEAR16 for Google STT)
                                const int16Array = new Int16Array(inputData.length);
                                for (let i = 0; i < inputData.length; i++) {
                                    // Scale float to 16-bit integer range (-32768 to 32767)
                                    int16Array[i] = Math.max(-1, Math.min(1, inputData[i])) * 0x7FFF;
                                }
                                if (ws.readyState === WebSocket.OPEN) {
                                    ws.send(int16Array.buffer); // Send raw audio bytes
                                }
                            };

                            audioInput.connect(processor);
                            processor.connect(audioContext.destination); // Connect to destination to keep processor alive
                        })
                        .catch(err => {
                            console.error("Error accessing microphone:", err);
                            startButton.disabled = false;
                            stopButton.disabled = true;
                            startButton.classList.remove('recording');
                            alert("Microphone access denied or error: " + err.message); // Use alert for critical user feedback
                        });
                };

                ws.onmessage = (event) => {
                    if (typeof event.data === 'string') {
                        // Assuming AI response text or transcript from backend
                        if (event.data.startsWith("You:")) {
                            transcriptDiv.textContent = event.data; // Overwrite for final transcript
                        } else {
                            aiResponseDiv.textContent += event.data; // Append for streaming AI text
                        }
                    } else if (event.data instanceof Blob) {
                        // Assuming AI audio response (Blob)
                        const audioUrl = URL.createObjectURL(event.data);
                        const audio = new Audio(audioUrl);
                        audio.play();
                        audio.onended = () => URL.revokeObjectURL(audioUrl); // Clean up Blob URL after playback
                    }
                };

                ws.onclose = () => {
                    console.log("WebSocket closed");
                    startButton.disabled = false;
                    stopButton.disabled = true;
                    startButton.classList.remove('recording');
                };

                ws.onerror = (error) => {
                    console.error("WebSocket Error:", error);
                    startButton.disabled = false;
                    stopButton.disabled = true;
                    startButton.classList.remove('recording');
                };
            };

            stopButton.onclick = () => {
                startButton.disabled = false;
                stopButton.disabled = true;
                startButton.classList.remove('recording'); // Remove recording style

                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send("END_OF_SPEECH"); // Signal end of speech to backend
                    ws.close(); // Close WebSocket connection
                }

                if (mediaStream) {
                    mediaStream.getTracks().forEach(track => track.stop()); // Stop microphone track
                }
                if (processor) {
                    processor.disconnect();
                }
                if (audioInput) {
                    audioInput.disconnect();
                }
                if (audioContext) {
                    audioContext.close(); // Close audio context
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

    # Variables to hold STT config. `stt_config_set` will signal when sample rate is received.
    stt_sample_rate_hertz = 0 # Will be updated by frontend
    stt_config_set = asyncio.Event() # Event to signal when STT config is ready

    # Dialogflow session ID (unique per user conversation).
    # Using client host and port for a basic unique ID for development.
    session_id = str(websocket.client.host) + "-" + str(websocket.client.port)
    dialogflow_session_path = dialogflow_sessions_client.session_path(
        settings.GOOGLE_CLOUD_PROJECT_ID, session_id
    )
    logger.info(f"Dialogflow Session Path: {dialogflow_session_path}")

    # Synchronous queue for STT requests. This bridges the async WebSocket
    # with the synchronous `speech_client.streaming_recognize` call (run in a thread).
    sync_recognize_requests_queue = queue.Queue()
    
    # Flag to control streaming and task lifecycle
    is_recording_active = True

    async def consume_audio_stream():
        """
        Consumes audio chunks from the WebSocket and puts them into a synchronous queue.
        Also handles receiving the initial sample rate from the frontend.
        """
        nonlocal is_recording_active, stt_sample_rate_hertz
        try:
            while is_recording_active:
                message = await websocket.receive()
                
                # Handle binary (audio) messages
                if message["type"] == "websocket.receive" and "bytes" in message:
                    audio_chunk = message["bytes"]
                    if audio_chunk == b'END_OF_SPEECH': # Custom signal from frontend to stop recording
                        logger.info("Received END_OF_SPEECH signal from client.")
                        is_recording_active = False
                        break # Exit loop
                    
                    # Wait until sample rate is set before putting audio into the queue for STT
                    await stt_config_set.wait()
                    sync_recognize_requests_queue.put_nowait(speech.StreamingRecognizeRequest(audio_content=audio_chunk))
                
                # Handle text (JSON for sample rate, or other text) messages
                elif message["type"] == "websocket.receive" and "text" in message:
                    try:
                        data = json.loads(message["text"])
                        if data.get("type") == "sample_rate":
                            stt_sample_rate_hertz = int(data["value"])
                            logger.info(f"Received sample rate from frontend: {stt_sample_rate_hertz} Hz")
                            stt_config_set.set() # Signal that STT config is ready
                        # You can add more JSON message types here if needed
                    except json.JSONDecodeError:
                        # If it's not JSON, it might be the transcript echo from frontend (for React app)
                        logger.debug(f"Received non-JSON text message: {message['text']}")
                        # If you want to display the user's transcript on the backend, you can do so here.
                        # For the current setup, the frontend displays its own transcript.
        except WebSocketDisconnect:
            logger.info("Client disconnected during audio consumption.")
            is_recording_active = False
        except Exception as e:
            logger.error(f"Error consuming audio stream: {e}")
            is_recording_active = False
        finally:
            # Crucial: Signal end of input to the synchronous generator for STT
            sync_recognize_requests_queue.put_nowait(None)


    def synchronous_request_generator():
        """
        A synchronous generator that pulls from the synchronous queue.
        This generator will be consumed by `speech_client.streaming_recognize`.
        """
        while True:
            # This `get()` call will block the thread until an item is available
            # or the sentinel `None` is put into the queue.
            request = sync_recognize_requests_queue.get()
            if request is None: # Sentinel value to signal end of stream
                break
            yield request

    async def transcribe_speech():
        """
        Handles the Speech-to-Text transcription.
        Runs the synchronous `speech_client.streaming_recognize` in a separate thread.
        """
        try:
            # Wait for the sample rate to be set by the frontend before configuring STT
            await stt_config_set.wait()

            # Create STT config with the received sample rate
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, # Raw PCM, 16-bit, mono
                sample_rate_hertz=stt_sample_rate_hertz, # Use the dynamic sample rate from frontend
                language_code="en-US",
                enable_automatic_punctuation=True,
            )
            streaming_config = speech.StreamingRecognitionConfig(
                config=config,
                interim_results=False, # Set to True if you want partial results
                single_utterance=True, # Stop processing after user stops speaking
            )

            # Perform the streaming recognition in a separate thread using asyncio.to_thread.
            # This is key to avoid blocking the main asyncio event loop while the synchronous
            # `synchronous_request_generator` is consumed by `speech_client.streaming_recognize`.
            responses = await asyncio.to_thread(
                speech_client.streaming_recognize, streaming_config, synchronous_request_generator()
            )

            # Iterate over the async responses from the STT client
            # The `responses` object returned by `streaming_recognize` is an async iterable.
            async for response in responses:
                for result in response.results:
                    if result.alternatives:
                        transcript_text = result.alternatives[0].transcript
                        logger.info(f"STT Final Transcript: {transcript_text}")
                        # Send final transcript back to frontend for display
                        await websocket.send_text(f"You: {transcript_text}")

                        if result.is_final:
                            # Process final transcript with Dialogflow/LLM
                            await process_final_transcript(transcript_text)
                            # After a final result with `single_utterance=True`, the STT stream
                            # on Google's side will close automatically.
                            return # Exit transcription loop after final result
            logger.info("STT finished processing for the current utterance.")

        except Exception as e:
            logger.error(f"Error during speech transcription: {e}")
            # Ensure the audio consumer also stops if transcription fails
            sync_recognize_requests_queue.put_nowait(None) # Signal end to synchronous_request_generator
            is_recording_active = False # Ensure recording stops if STT fails


    async def process_final_transcript(transcript_text: str):
        """
        Processes the final transcribed text by sending it to Dialogflow (or Gemini directly).
        Then triggers TTS to speak the AI's response.
        """
        logger.info(f"Processing final transcript: '{transcript_text}'")

        if not transcript_text.strip():
            logger.info("Empty transcript, skipping LLM/Dialogflow.")
            return

        # --- Option 1: Direct LLM (Gemini) Call (uncomment to use) ---
        # response_text = await get_gemini_response(transcript_text)

        # --- Option 2: Dialogflow Integration (Recommended for structure) ---
        response_text = await get_dialogflow_response(transcript_text, dialogflow_session_path)

        if response_text:
            logger.info(f"AI Response Text: {response_text}")
            await stream_tts_and_send_to_client(response_text)
        else:
            await websocket.send_text("I'm sorry, I couldn't generate a response.")


    async def get_gemini_response(prompt: str) -> str:
        """
        Calls the Gemini LLM directly using the async client from Vertex AI.
        For conversational memory, you would pass previous messages to the LLM's history.
        """
        if not gemini_model:
            logger.error("Gemini model not initialized.")
            return "AI model not initialized."
        try:
            # Using generate_content_async for asynchronous call
            response = await gemini_model.generate_content_async(
                prompt,
                generation_config=gemini_config,
                # Example for conversational memory (add your chat history here):
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
        """
        Sends text to Dialogflow and gets a response.
        """
        query_input = dialogflow.QueryInput(text=dialogflow.TextInput(text=text, language_code="en-US"))
        try:
            # Using detect_intent for asynchronous call
            response = await dialogflow_sessions_client.detect_intent(
                session=session_path, query_input=query_input
            )
            logger.info(f"Dialogflow Query Result: {response.query_result.fulfillment_text}")
            # Dialogflow's fulfillment_text is the response to send to the user.
            # If Dialogflow triggers a webhook to your backend, your webhook logic
            # would handle the more complex response generation.
            return response.query_result.fulfillment_text
        except Exception as e:
            logger.error(f"Error calling Dialogflow: {e}")
            return "I'm sorry, I'm having trouble with my understanding right now."


    async def stream_tts_and_send_to_client(text: str):
        """
        Generates speech from text using Google Cloud Text-to-Speech and streams audio back to the client.
        Note: Google Cloud TTS `synthesize_speech` is a synchronous call that returns the full audio blob.
        We simulate streaming by chunking and sending. For true streaming TTS, a different API/approach is needed.
        """
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
            # Perform synchronous TTS synthesis in a separate thread to avoid blocking the event loop
            response = await asyncio.to_thread(
                tts_client.synthesize_speech, input=synthesis_input, voice=voice, audio_config=audio_config
            )
            audio_content = response.audio_content

            # Send the AI's text response first to the frontend
            await websocket.send_text(text)

            # Chunk the audio and send it over the WebSocket
            chunk_size = 4096 # Adjust as needed for smoother streaming vs. network overhead
            for i in range(0, len(audio_content), chunk_size):
                chunk = audio_content[i:i + chunk_size]
                # Check if the client is still connected before sending
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_bytes(chunk)
                await asyncio.sleep(0.001) # Small delay to yield control and allow other tasks to run
            logger.info("TTS audio streamed to client.")

        except Exception as e:
            logger.error(f"Error during TTS generation/streaming: {e}")
            await websocket.send_text("I'm sorry, I can't speak right now.")

    # --- Main WebSocket Task Orchestration ---
    try:
        # Create the audio consumer task (receives from frontend, puts into sync queue)
        audio_consumer_task = asyncio.create_task(consume_audio_stream())
        
        # Create the transcription task (pulls from sync queue, calls STT, processes response)
        transcribe_task = asyncio.create_task(transcribe_speech())

        # Wait for either task to finish (e.g., client disconnects or STT finishes processing)
        done, pending = await asyncio.wait(
            [audio_consumer_task, transcribe_task],
            return_when=asyncio.FIRST_COMPLETED # Finish when the first task completes
        )

        # Cancel any remaining pending tasks to ensure clean shutdown
        for task in pending:
            task.cancel()
        # Wait for cancelled tasks to complete their cancellation
        await asyncio.gather(*pending, return_exceptions=True)

        logger.info("WebSocket tasks finished.")

    except WebSocketDisconnect:
        logger.info("Client disconnected gracefully.")
    except Exception as e:
        logger.error(f"WebSocket endpoint encountered an unexpected error: {e}")
    finally:
        is_recording_active = False # Ensure flag is set to false for any lingering processes
        await websocket.close() # Ensure WebSocket is closed
