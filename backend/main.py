# backend/main.py
import asyncio
import os
import io
import json
import logging
import queue # Import queue module for synchronous queue
from starlette.websockets import WebSocketState # Import WebSocketState for connection check
from typing import AsyncGenerator, List, Optional

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
            // Get references to DOM elements
            const startButton = document.getElementById('startButton');
            const stopButton = document.getElementById('stopButton');
            const transcriptDiv = document.getElementById('transcript');
            const aiResponseDiv = document.getElementById('aiResponse');

            // WebSocket and Audio API variables
            let ws;
            let mediaStream;
            let audioContext;
            let audioInput;
            let processor;
            let currentSampleRate; // Will be set by AudioContext

            // Variables for TTS Audio Chunk Accumulation
            let currentAudioChunks = [];
            let expectingAudio = false; // True when we are expecting TTS audio chunks
            let audioChunkTimeout;      // To detect end of audio stream implicitly


            // Function to play audio from a Blob
            // This function remains largely the same, but it's now called *after*
            // all chunks are assembled into a single Blob.
            async function playNextAudio(audioBlob) {
                if (!audioBlob || audioBlob.size === 0) {
                    console.warn("Attempted to play empty audio blob.");
                    return;
                }

                console.log(`Received audio Blob of size: ${audioBlob.size} type: ${audioBlob.type}`);
                
                const audioUrl = URL.createObjectURL(audioBlob);
                const audio = new Audio(audioUrl);

                // Add detailed logging for audio playback
                audio.onloadedmetadata = () => {
                    console.log(`Audio metadata loaded. Duration: ${audio.duration} seconds. Ready state: ${audio.readyState}`);
                };
                audio.oncanplaythrough = () => {
                    console.log('Audio can play through without interruption.');
                    // This is a good point to start playing if you want immediate playback
                    // but we'll stick to the current flow for now.
                };
                audio.onplaying = () => {
                    console.log('Audio started playing.');
                };
                audio.onended = () => {
                    console.log('Audio playback ended.');
                    URL.revokeObjectURL(audioUrl); // Clean up the Blob URL after playback
                    // Here, you could potentially trigger the next action or allow new input
                };
                audio.onerror = (e) => {
                    console.error('Audio playback error:', e);
                    if (audio.error) {
                        console.error('Audio error code:', audio.error.code);
                        console.error('Audio error message:', audio.error.message);
                    }
                    URL.revokeObjectURL(audioUrl);
                };

                try {
                    await audio.play();
                } catch (error) {
                    console.error("Error attempting to play audio:", error);
                }
            }


            // --- startButton.onclick remains mostly the same ---
            startButton.onclick = async () => {
                startButton.disabled = true;
                stopButton.disabled = false;
                transcriptDiv.innerHTML = '<strong>You:</strong> '; // Reset content
                aiResponseDiv.innerHTML = '<strong>AI:</strong> '; // Reset content
                startButton.classList.add('recording'); // Add recording style

                // Reset audio accumulation variables for a new turn
                currentAudioChunks = [];
                expectingAudio = false;
                if (audioChunkTimeout) clearTimeout(audioChunkTimeout);


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

                // --- MODIFIED ws.onmessage ---
                ws.onmessage = async (event) => {
                    if (typeof event.data === 'string') {
                        // It's a text message (either STT transcript or AI response text)
                        // Your current backend sends AI text as plain text.
                        // For robust handling, it's better to wrap all messages in JSON,
                        // but we'll adapt to your current backend's text sending for now.
                        if (event.data.startsWith("You:")) {
                            transcriptDiv.textContent = event.data; // Overwrite for final transcript
                            // Optionally, if an AI response was pending audio, this could signal its end.
                            // But usually, AI text comes *before* or *with* the audio.
                        } else {
                            // This is likely the AI's text response
                            aiResponseDiv.textContent = `<strong>AI:</strong> ${event.data}`; // Set for a new response
                            currentAudioChunks = []; // Clear chunks for a new AI response
                            expectingAudio = true; // Start collecting audio chunks
                            console.log("Started expecting audio for new AI response.");
                        }
                    } else if (event.data instanceof Blob || event.data instanceof ArrayBuffer) {
                        // It's binary audio data (MP3 chunks from TTS)
                        if (expectingAudio) {
                            // Ensure data is a Blob for consistent handling
                            const blobData = event.data instanceof ArrayBuffer ? new Blob([event.data], { type: 'audio/mpeg' }) : event.data;
                            currentAudioChunks.push(blobData);
                            console.log(`Received audio chunk. Current chunks total size: ${currentAudioChunks.reduce((sum, chunk) => sum + chunk.size, 0)} bytes`);

                            // Reset the timeout. If no new chunks arrive within this time,
                            // we assume the full audio has been received.
                            if (audioChunkTimeout) clearTimeout(audioChunkTimeout);
                            audioChunkTimeout = setTimeout(async () => {
                                if (expectingAudio && currentAudioChunks.length > 0) {
                                    console.log("Audio chunk timeout. Assuming end of audio stream. Playing collected audio.");
                                    const audioBlob = new Blob(currentAudioChunks, { type: 'audio/mpeg' });
                                    await playNextAudio(audioBlob);
                                    expectingAudio = false; // Finished with this audio turn
                                    currentAudioChunks = []; // Clear for next turn
                                } else {
                                    console.log("Audio chunk timeout, but no audio collected or not expecting.");
                                }
                            }, 200); // 200ms delay. Adjust if needed based on network conditions.
                        } else {
                            console.warn("Received unexpected audio chunk. Not currently expecting audio or already played.");
                        }
                    }
                };

                ws.onclose = () => {
                    console.log("WebSocket closed");
                    startButton.disabled = false;
                    stopButton.disabled = true;
                    startButton.classList.remove('recording');
                    if (audioChunkTimeout) clearTimeout(audioChunkTimeout); // Clear any pending audio timeouts
                    currentAudioChunks = []; // Clear any residual chunks
                    expectingAudio = false;
                };

                ws.onerror = (error) => {
                    console.error("WebSocket Error:", error);
                    startButton.disabled = false;
                    stopButton.disabled = true;
                    startButton.classList.remove('recording');
                    if (audioChunkTimeout) clearTimeout(audioChunkTimeout); // Clear any pending audio timeouts
                    currentAudioChunks = []; // Clear any residual chunks
                    expectingAudio = false;
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

                // Stop media stream and disconnect audio nodes
                if (mediaStream) {
                    mediaStream.getTracks().forEach(track => track.stop());
                }
                if (processor) {
                    processor.disconnect();
                }
                if (audioInput) {
                    audioInput.disconnect();
                }
                if (audioContext) {
                    audioContext.close();
                }
                
                // Clear any pending audio processing on stop
                if (audioChunkTimeout) clearTimeout(audioChunkTimeout);
                currentAudioChunks = [];
                expectingAudio = false;
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
    
    # Asynchronous queue for STT results to be processed in the main event loop.
    stt_results_queue = asyncio.Queue()

    # Flag to control streaming and task lifecycle
    is_recording_active = True

    async def consume_audio_stream():
        """
        Consumes audio chunks from the WebSocket and puts them into a synchronous queue.
        Also handles receiving the initial sample rate from the frontend.
        """
        nonlocal is_recording_active, stt_sample_rate_hertz
        try:
            # Loop while recording is active AND the websocket is connected
            while is_recording_active and websocket.client_state == WebSocketState.CONNECTED:
                try:
                    message = await websocket.receive()
                except RuntimeError as e:
                    logger.warning(f"WebSocket receive error (likely disconnect): {e}")
                    is_recording_active = False
                    break # Exit loop if receive fails due to disconnect
                
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
        nonlocal is_recording_active # Need to be able to set this to False on error

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

            # Define a synchronous function to run in a separate thread
            def _run_stt_synchronously() -> List[str]:
                """
                Synchronous function to run STT in a separate thread.
                Collects all final transcripts and returns them.
                """
                logger.info("Starting synchronous STT recognition in a new thread.")
                collected_transcripts = []
                try:
                    # This call returns a synchronous iterator
                    responses_sync_iterator = speech_client.streaming_recognize(
                        streaming_config, synchronous_request_generator()
                    )
                    
                    for response in responses_sync_iterator:
                        for result in response.results:
                            if result.alternatives:
                                transcript_text = result.alternatives[0].transcript
                                logger.info(f"STT Final Transcript (from thread): {transcript_text}")
                                collected_transcripts.append(transcript_text)
                                if result.is_final:
                                    # For single_utterance=True, we expect one final result per user turn.
                                    return collected_transcripts # Exit synchronous loop and return
                except Exception as e:
                    logger.error(f"Error within synchronous STT thread: {e}")
                finally:
                    logger.info("Synchronous STT thread finished.")
                    return collected_transcripts # Ensure transcripts are returned even on error

            # Run the synchronous STT function in a separate thread
            final_transcripts = await asyncio.to_thread(_run_stt_synchronously)
            logger.info(f"STT processing task completed. Final transcripts: {final_transcripts}")

            # Put collected final transcripts into the async queue for main loop processing
            for transcript_text in final_transcripts:
                await stt_results_queue.put(transcript_text)
            await stt_results_queue.put(None) # Sentinel to signal end of this utterance's results

        except Exception as e:
            logger.error(f"Error during speech transcription task setup/execution: {e}")
            is_recording_active = False # Ensure recording stops if STT fails
            # Also signal error/end to the STT results queue
            await stt_results_queue.put(None)

    async def process_stt_results():
        """
        Consumes final STT transcripts from the queue and processes them.
        """
        nonlocal is_recording_active
        try:
            while is_recording_active: # Loop while recording is active
                transcript_text = await stt_results_queue.get()
                if transcript_text is None: # Sentinel for end of utterance
                    logger.info("Received STT results sentinel, stopping processing for this turn.")
                    # This break will exit the while loop for the current turn.
                    # The outer try/finally or task cancellation will handle full shutdown.
                    break 
                
                # Check if WebSocket is still connected before processing and sending response
                if websocket.client_state == WebSocketState.CONNECTED:
                    await process_final_transcript(transcript_text)
                else:
                    logger.info("WebSocket disconnected, skipping processing of STT results.")
                    is_recording_active = False # Ensure loop terminates if client disconnected
                    break
        except Exception as e:
            logger.error(f"Error processing STT results: {e}")
            is_recording_active = False # Stop if error occurs
        finally:
            logger.info("STT results processing task finished.")
            # Ensure the queue is marked as done for the current task, if it was consuming.
            # This is more for graceful shutdown of the queue itself if it were persistent.

    async def process_final_transcript(transcript_text: str):
        """
        Processes the final transcribed text by sending it to Dialogflow (or Gemini directly).
        Then triggers TTS to speak the AI's response.
        """
        logger.info(f"Processing final transcript: '{transcript_text}'")

        if not transcript_text.strip():
            logger.info("Empty transcript, skipping LLM/Dialogflow.")
            return

        # Send transcript back to frontend for display (this ensures it's shown after processing)
        # Only send if the websocket is still connected
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(f"You: {transcript_text}")
        else:
            logger.info("WebSocket disconnected, cannot send user transcript back to frontend.")
            return # Exit early if disconnected

        # --- Option 1: Direct LLM (Gemini) Call (uncomment to use) ---
        # response_text = await get_gemini_response(transcript_text)

        # --- Option 2: Dialogflow Integration (Recommended for structure) ---
        response_text = await get_dialogflow_response(transcript_text, dialogflow_session_path)

        if response_text:
            logger.info(f"AI Response Text: {response_text}")
            # Only stream TTS if the websocket is still connected
            if websocket.client_state == WebSocketState.CONNECTED:
                await stream_tts_and_send_to_client(response_text)
            else:
                logger.info("WebSocket disconnected, cannot stream TTS audio.")
        else:
            logger.info("No AI response text generated.")
            # Only send error message if the websocket is still connected
            if websocket.client_state == WebSocketState.CONNECTED:
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
            # Wrap the synchronous Dialogflow call in asyncio.to_thread
            response = await asyncio.to_thread(
                dialogflow_sessions_client.detect_intent, session=session_path, query_input=query_input
            )
            logger.info(f"Dialogflow Raw Response: {response}") # Log raw Dialogflow response
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
            ssml_gender=tts.SsmlVoiceGender.FEMALE, # Changed from NEUTRAL to FEMALE
            name="en-US-Wavenet-F" # Changed from Neural2-C to a common Wavenet voice
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
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(text)
            else:
                logger.info("WebSocket disconnected, cannot send AI text back to frontend.")
                return

            # Chunk the audio and send it over the WebSocket
            chunk_size = 4096 # Adjust as needed for smoother streaming vs. network overhead
            for i in range(0, len(audio_content), chunk_size):
                chunk = audio_content[i:i + chunk_size]
                # Check if the client is still connected before sending
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_bytes(chunk)
                else:
                    logger.info("WebSocket disconnected during TTS streaming, stopping audio.")
                    break # Stop streaming if client disconnects
                await asyncio.sleep(0.001) # Small delay to yield control and allow other tasks to run
            logger.info("TTS audio streamed to client.")

        except Exception as e:
            logger.error(f"Error during TTS generation/streaming: {e}")
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text("I'm sorry, I can't speak right now.")

    # --- Main WebSocket Task Orchestration ---
    try:
        # Create the audio consumer task (receives from frontend, puts into sync queue)
        audio_consumer_task = asyncio.create_task(consume_audio_stream())
        
        # Create the transcription task (pulls from sync queue, calls STT, processes response)
        transcribe_task = asyncio.create_task(transcribe_speech())

        # Create the STT results processing task (consumes from async queue, calls LLM/Dialogflow, TTS)
        stt_results_processor_task = asyncio.create_task(process_stt_results())

        # Wait for all tasks to complete, or for a WebSocketDisconnect
        # We explicitly don't use FIRST_COMPLETED here to allow tasks to finish their work.
        # The `is_recording_active` flag and `None` sentinels are key for graceful exits.
        await asyncio.gather(
            audio_consumer_task,
            transcribe_task,
            stt_results_processor_task,
            return_exceptions=True # Allow tasks to fail without stopping gather immediately
        )

        logger.info("All WebSocket tasks completed.")

    except WebSocketDisconnect:
        logger.info("Client disconnected gracefully.")
    except Exception as e:
        logger.error(f"WebSocket endpoint encountered an unexpected error: {e}")
    finally:
        # This `finally` block ensures cleanup when the `websocket_endpoint` coroutine exits.
        # It's crucial to set `is_recording_active = False` here to signal all tasks to stop.
        is_recording_active = False 
        # Also, ensure the synchronous queue is signaled to prevent its thread from hanging.
        # This is important if consume_audio_stream or transcribe_speech were still running.
        try:
            sync_recognize_requests_queue.put_nowait(None)
        except queue.Full:
            pass # Ignore if queue is full during shutdown
        
        # And the async queue for results
        try:
            stt_results_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass # Ignore if queue is full during shutdown
        
        # Finally, close the websocket. This should be the last step.
        # Check if the websocket is still connected before attempting to close,
        # to avoid the "Unexpected ASGI message 'websocket.close'" error.
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()
