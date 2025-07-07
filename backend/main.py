# backend/main.py
import asyncio
import os
import io
import json
import logging
import queue
from starlette.websockets import WebSocketState
from typing import AsyncGenerator, List, Optional, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel # Import BaseModel for webhook request parsing

from google.cloud import speech_v1p1beta1 as speech
from google.cloud import texttospeech_v1 as tts
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from google.cloud import dialogflow_v2 as dialogflow
from google.api_core.exceptions import GoogleAPIError

from config import settings

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
try:
    speech_client = speech.SpeechClient()
    tts_client = tts.TextToSpeechClient()
    dialogflow_sessions_client = dialogflow.SessionsClient()
    logger.info("Google Cloud clients (Speech, TTS, Dialogflow) initialized successfully using Service Account.")

    vertexai.init(project=settings.GOOGLE_CLOUD_PROJECT_ID, location="us-central1")
    logger.info(f"Vertex AI initialized for project: {settings.GOOGLE_CLOUD_PROJECT_ID}, location: us-central1")

    gemini_model = GenerativeModel('gemini-1.5-pro')
    logger.info("Gemini model ('gemini-1.5-pro') initialized via Vertex AI.")

except GoogleAPIError as e:
    logger.error(f"Failed to initialize Google Cloud clients: {e}")
    logger.warning("Ensure your GOOGLE_APPLICATION_CREDENTIALS environment variable is set correctly and the service account has the necessary permissions (Speech-to-Text Admin, Text-to-Speech Admin, Dialogflow API Client, Vertex AI User).")
except Exception as e:
    logger.error(f"An unexpected error occurred during client initialization: {e}")
    logger.warning("Please check your environment setup and API configurations.")

gemini_config = GenerationConfig(temperature=0.7, top_p=0.9, top_k=40)


# --- Dialogflow Webhook Request Model ---
# This Pydantic model helps parse the incoming JSON from Dialogflow webhook
class DialogflowWebhookRequest(BaseModel):
    responseId: Optional[str] = None
    queryResult: Dict[str, Any]
    session: Optional[str] = None
    originalDetectIntentRequest: Optional[Dict[str, Any]] = None

# --- Helper function for calculations ---
def _perform_calculation(num1: float, num2: float, operation: str) -> str:
    """Performs the specified mathematical operation."""
    result = None
    response_text = ""

    if operation == "add":
        result = num1 + num2
    elif operation == "subtract":
        result = num1 - num2
    elif operation == "multiply" or operation == "times":
        result = num1 * num2
    elif operation == "divide":
        if num2 != 0:
            result = num1 / num2
        else:
            response_text = "I cannot divide by zero."
    elif operation == "power":
        result = num1 ** num2
    elif operation == "remainder" or operation == "mod" or operation == "modulo":
        if num2 != 0:
            result = num1 % num2
        else:
            response_text = "I cannot calculate remainder with zero."
    else:
        response_text = f"I don't recognize the operation '{operation}'. Please try add, subtract, multiply, divide, power, or remainder."

    if result is not None:
        response_text = f"The answer is {result}."
    elif not response_text: # If no specific error message was set by division/remainder by zero
        response_text = "I couldn't perform that calculation. Please ensure you provided valid numbers and operation."
    
    return response_text

# --- Dialogflow Webhook Endpoint ---
@app.post("/webhook")
async def dialogflow_webhook(request_body: DialogflowWebhookRequest):
    """
    Handles incoming webhook requests from Dialogflow.
    This endpoint will perform the math calculation based on detected intent and parameters.
    """
    logger.info(f"Received Dialogflow webhook request: {request_body.dict()}")

    query_result = request_body.queryResult
    intent_display_name = query_result.get("intent", {}).get("displayName")
    parameters = query_result.get("parameters", {})

    fulfillment_text = "I'm sorry, I couldn't process your request." # Default response

    if intent_display_name == "Calculate Math":
        try:
            num1 = parameters.get("number1")
            num2 = parameters.get("number2")
            operation = parameters.get("operation")

            if num1 is not None and num2 is not None and operation:
                # Dialogflow parameters are often strings, convert to float
                num1_float = float(num1)
                num2_float = float(num2)
                fulfillment_text = _perform_calculation(num1_float, num2_float, operation)
            else:
                fulfillment_text = "I need two numbers and an operation to perform the calculation."
                logger.warning(f"Missing parameters for Calculate Math intent: num1={num1}, num2={num2}, operation={operation}")
        except Exception as e:
            logger.error(f"Error in webhook calculation logic: {e}")
            fulfillment_text = "I encountered an error while trying to calculate that."
    else:
        # For other intents, you might rely on Dialogflow's default fulfillment
        # or implement other webhook logic here.
        fulfillment_text = query_result.get("fulfillmentText", "I'm not sure how to respond to that.")
        logger.info(f"Webhook received intent '{intent_display_name}', using Dialogflow's fulfillment text.")

    # Construct the Dialogflow webhook response
    webhook_response = {
        "fulfillmentText": fulfillment_text,
        "payload": {
            "google": {
                "expectUserResponse": True,
                "richResponse": {
                    "items": [
                        {
                            "simpleResponse": {
                                "textToSpeech": fulfillment_text,
                                "displayText": fulfillment_text
                            }
                        }
                    ]
                }
            }
        }
    }
    return JSONResponse(content=webhook_response)


# --- HTML for simple testing (optional) ---
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
            let mediaStream;
            let audioContext;
            let audioInput;
            let processor;
            let currentSampleRate = 16000;

            const startButton = document.getElementById('startButton');
            const stopButton = document.getElementById('stopButton');
            const transcriptDiv = document.getElementById('transcript');
            const aiResponseDiv = document.getElementById('aiResponse');

            startButton.onclick = async () => {
                startButton.disabled = true;
                stopButton.disabled = false;
                transcriptDiv.innerHTML = '<strong>You:</strong> ';
                aiResponseDiv.innerHTML = '<strong>AI:</strong> ';
                startButton.classList.add('recording');

                ws = new WebSocket("ws://localhost:8000/ws");

                ws.onopen = () => {
                    console.log("WebSocket opened");
                    navigator.mediaDevices.getUserMedia({ audio: true })
                        .then(stream => {
                            mediaStream = stream;
                            audioContext = new (window.AudioContext || window.webkitAudioContext)();
                            currentSampleRate = audioContext.sampleRate;
                            console.log("AudioContext sample rate:", currentSampleRate);

                            if (ws.readyState === WebSocket.OPEN) {
                                ws.send(JSON.stringify({ type: 'sample_rate', value: currentSampleRate }));
                            }

                            audioInput = audioContext.createMediaStreamSource(mediaStream);
                            processor = audioContext.createScriptProcessor(4096, 1, 1);

                            processor.onaudioprocess = (event) => {
                                const inputData = event.inputBuffer.getChannelData(0);
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
                        })
                        .catch(err => {
                            console.error("Error accessing microphone:", err);
                            startButton.disabled = false;
                            stopButton.disabled = true;
                            startButton.classList.remove('recording');
                            alert("Microphone access denied or error: " + err.message);
                        });
                };

                ws.onmessage = (event) => {
                    if (typeof event.data === 'string') {
                        if (event.data.startsWith("You:")) {
                            transcriptDiv.textContent = event.data;
                        } else {
                            aiResponseDiv.textContent += event.data;
                        }
                    } else if (event.data instanceof Blob) {
                        const audioUrl = URL.createObjectURL(event.data);
                        const audio = new Audio(audioUrl);
                        audio.play();
                        audio.onended = () => URL.revokeObjectURL(audioUrl);
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
                startButton.classList.remove('recording');

                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send("END_OF_SPEECH");
                    ws.close();
                }

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

    stt_sample_rate_hertz = 0
    stt_config_set = asyncio.Event()

    session_id = str(websocket.client.host) + "-" + str(websocket.client.port)
    dialogflow_session_path = dialogflow_sessions_client.session_path(
        settings.GOOGLE_CLOUD_PROJECT_ID, session_id
    )
    logger.info(f"Dialogflow Session Path: {dialogflow_session_path}")

    sync_recognize_requests_queue = queue.Queue()
    stt_results_queue = asyncio.Queue()
    is_recording_active = True

    async def consume_audio_stream():
        nonlocal is_recording_active, stt_sample_rate_hertz
        try:
            while is_recording_active and websocket.client_state == WebSocketState.CONNECTED:
                try:
                    message = await websocket.receive()
                except RuntimeError as e:
                    logger.warning(f"WebSocket receive error (likely disconnect): {e}")
                    is_recording_active = False
                    break
                
                if message["type"] == "websocket.receive" and "bytes" in message:
                    audio_chunk = message["bytes"]
                    if audio_chunk == b'END_OF_SPEECH':
                        logger.info("Received END_OF_SPEECH signal from client.")
                        is_recording_active = False
                        break
                    
                    await stt_config_set.wait()
                    sync_recognize_requests_queue.put_nowait(speech.StreamingRecognizeRequest(audio_content=audio_chunk))
                
                elif message["type"] == "websocket.receive" and "text" in message:
                    try:
                        data = json.loads(message["text"])
                        if data.get("type") == "sample_rate":
                            stt_sample_rate_hertz = int(data["value"])
                            logger.info(f"Received sample rate from frontend: {stt_sample_rate_hertz} Hz")
                            stt_config_set.set()
                    except json.JSONDecodeError:
                        logger.debug(f"Received non-JSON text message: {message['text']}")
        except WebSocketDisconnect:
            logger.info("Client disconnected during audio consumption.")
            is_recording_active = False
        except Exception as e:
            logger.error(f"Error consuming audio stream: {e}")
            is_recording_active = False
        finally:
            sync_recognize_requests_queue.put_nowait(None)


    def synchronous_request_generator():
        while True:
            request = sync_recognize_requests_queue.get()
            if request is None:
                break
            yield request

    async def transcribe_speech():
        nonlocal is_recording_active

        try:
            await stt_config_set.wait()

            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=stt_sample_rate_hertz,
                language_code="en-US",
                enable_automatic_punctuation=True,
            )
            streaming_config = speech.StreamingRecognitionConfig(
                config=config,
                interim_results=False,
                single_utterance=True,
            )

            def _run_stt_synchronously() -> List[str]:
                logger.info("Starting synchronous STT recognition in a new thread.")
                collected_transcripts = []
                try:
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
                                    return collected_transcripts
                except Exception as e:
                    logger.error(f"Error within synchronous STT thread: {e}")
                finally:
                    logger.info("Synchronous STT thread finished.")
                    return collected_transcripts

            final_transcripts = await asyncio.to_thread(_run_stt_synchronously)
            logger.info(f"STT processing task completed. Final transcripts: {final_transcripts}")

            for transcript_text in final_transcripts:
                await stt_results_queue.put(transcript_text)
            await stt_results_queue.put(None)

        except Exception as e:
            logger.error(f"Error during speech transcription task setup/execution: {e}")
            is_recording_active = False
            await stt_results_queue.put(None)

    async def process_stt_results():
        nonlocal is_recording_active
        try:
            while is_recording_active:
                transcript_text = await stt_results_queue.get()
                if transcript_text is None:
                    logger.info("Received STT results sentinel, stopping processing for this turn.")
                    break 
                
                if websocket.client_state == WebSocketState.CONNECTED:
                    await process_final_transcript(transcript_text)
                else:
                    logger.info("WebSocket disconnected, skipping processing of STT results.")
                    is_recording_active = False
                    break
        except Exception as e:
            logger.error(f"Error processing STT results: {e}")
            is_recording_active = False
        finally:
            logger.info("STT results processing task finished.")

    async def process_final_transcript(transcript_text: str):
        """
        Processes the final transcribed text by sending it to Dialogflow.
        The response from Dialogflow (which might come from our webhook)
        is then used for TTS and sent back to the client.
        """
        logger.info(f"Processing final transcript: '{transcript_text}'")

        if not transcript_text.strip():
            logger.info("Empty transcript, skipping Dialogflow.")
            return

        # Send transcript back to frontend for display
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(f"You: {transcript_text}")
        else:
            logger.info("WebSocket disconnected, cannot send user transcript back to frontend.")
            return

        # Call Dialogflow. Dialogflow will handle routing to our webhook
        # if the 'Calculate Math' intent is matched and webhook fulfillment is enabled.
        # We now just expect the fulfillment_text to contain the answer or a default response.
        response_text = await get_dialogflow_response(transcript_text, dialogflow_session_path)

        if response_text:
            logger.info(f"AI Response Text: {response_text}")
            if websocket.client_state == WebSocketState.CONNECTED:
                await stream_tts_and_send_to_client(response_text)
            else:
                logger.info("WebSocket disconnected, cannot stream TTS audio.")
        else:
            logger.info("No AI response text generated from Dialogflow.")
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text("I'm sorry, I couldn't generate a response.")


    async def get_gemini_response(prompt: str) -> str:
        """
        Calls the Gemini LLM directly using the async client from Vertex AI.
        (This function is currently not used, as Dialogflow integration is active)
        """
        if not gemini_model:
            logger.error("Gemini model not initialized.")
            return "AI model not initialized."
        try:
            response = await gemini_model.generate_content_async(
                prompt,
                generation_config=gemini_config,
            )
            return response.candidates[0].content.parts[0].text
        except Exception as e:
            logger.error(f"Error calling Gemini LLM: {e}")
            return "I'm sorry, I'm having trouble connecting to the AI at the moment."

    async def get_dialogflow_response(text: str, session_path: str) -> str:
        """
        Sends text to Dialogflow and gets a response.
        This function now expects Dialogflow to handle webhook fulfillment
        and return the final fulfillment_text.
        """
        query_input = dialogflow.QueryInput(text=dialogflow.TextInput(text=text, language_code="en-US"))
        try:
            # Wrap the synchronous Dialogflow call in asyncio.to_thread
            response = await asyncio.to_thread(
                dialogflow_sessions_client.detect_intent, session=session_path, query_input=query_input
            )
            logger.info(f"Dialogflow Raw Response (from main loop): {response}")
            logger.info(f"Dialogflow Query Result (from main loop): {response.query_result.fulfillment_text}")
            return response.query_result.fulfillment_text
        except Exception as e:
            logger.error(f"Error calling Dialogflow: {e}")
            return "I'm sorry, I'm having trouble with my understanding right now."


    async def stream_tts_and_send_to_client(text: str):
        synthesis_input = tts.SynthesisInput(text=text)
        voice = tts.VoiceSelectionParams(
            language_code="en-US",
            ssml_gender=tts.SsmlVoiceGender.FEMALE,
            name="en-US-Wavenet-F"
        )
        audio_config = tts.AudioConfig(
            audio_encoding=tts.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000
        )

        try:
            response = await asyncio.to_thread(
                tts_client.synthesize_speech, input=synthesis_input, voice=voice, audio_config=audio_config
            )
            audio_content = response.audio_content

            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(text)
            else:
                logger.info("WebSocket disconnected, cannot send AI text back to frontend.")
                return

            chunk_size = 4096
            for i in range(0, len(audio_content), chunk_size):
                chunk = audio_content[i:i + chunk_size]
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_bytes(chunk)
                else:
                    logger.info("WebSocket disconnected during TTS streaming, stopping audio.")
                    break
                await asyncio.sleep(0.001)
            logger.info("TTS audio streamed to client.")

        except Exception as e:
            logger.error(f"Error during TTS generation/streaming: {e}")
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text("I'm sorry, I can't speak right now.")

    # --- Main WebSocket Task Orchestration ---
    try:
        audio_consumer_task = asyncio.create_task(consume_audio_stream())
        transcribe_task = asyncio.create_task(transcribe_speech())
        stt_results_processor_task = asyncio.create_task(process_stt_results())

        await asyncio.gather(
            audio_consumer_task,
            transcribe_task,
            stt_results_processor_task,
            return_exceptions=True
        )

        logger.info("All WebSocket tasks completed.")

    except WebSocketDisconnect:
        logger.info("Client disconnected gracefully.")
    except Exception as e:
        logger.error(f"WebSocket endpoint encountered an unexpected error: {e}")
    finally:
        is_recording_active = False 
        try:
            sync_recognize_requests_queue.put_nowait(None)
        except queue.Full:
            pass
        
        try:
            stt_results_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass
        
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()
