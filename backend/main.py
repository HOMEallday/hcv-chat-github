# backend/main.py
import asyncio
import os
import io
import json
import logging
import queue
from starlette.websockets import WebSocketState
from typing import AsyncGenerator, List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
# Import Google Cloud clients
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import texttospeech_v1 as tts
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
# --- MODIFICATION 1: Import the Async Client ---
from google.cloud import dialogflow_v2 as dialogflow
from google.api_core.exceptions import GoogleAPIError

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma

from config import settings  # Import your settings module

from google.cloud.dialogflow_v2.types import WebhookRequest, WebhookResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize global variables for clients and models
speech_client = None
tts_client = None
dialogflow_sessions_client = None # This will now be an Async client
gemini_model = None

# Global variables for RAG components
vectorstore: Optional[Chroma] = None
retrieval_chain = None
gemini_llm: Optional[ChatVertexAI] = None

#Define lifespan context manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the FastAPI application.
    All initialization logic is moved here.
    """
    global speech_client, tts_client, dialogflow_sessions_client, gemini_model
    global vectorstore, retrieval_chain, gemini_llm # Declare RAG globals here

    logger.info("Application startup initiated.")
    try:
        # --- EXISTING: Google Cloud Clients Initialization (MOVED HERE) ---
        speech_client = speech.SpeechClient()
        tts_client = tts.TextToSpeechClient()
        
        # --- MODIFICATION 2: Use the SessionsAsyncClient ---
        # This prevents the server from deadlocking when a webhook is called.
        dialogflow_sessions_client = dialogflow.SessionsAsyncClient()
        logger.info("Google Cloud clients (Speech, TTS, Dialogflow Async) initialized successfully.")

        # Initialize Vertex AI for Gemini models 
        vertexai.init(project=settings.GOOGLE_CLOUD_PROJECT_ID, location="us-central1")
        logger.info(f"Vertex AI initialized for project: {settings.GOOGLE_CLOUD_PROJECT_ID}, location: us-central1")

        # Your original Vertex AI GenerativeModel instance
        MODEL_NAME = "gemini-2.5-flash"
        gemini_model = GenerativeModel(MODEL_NAME, generation_config=GenerationConfig(temperature=0.7, top_p=0.9, top_k=40))
        logger.info(f"Gemini model ('{MODEL_NAME}') initialized via Vertex AI.")

        # --- LangChain-compatible Gemini model for RAG ---
        gemini_llm = ChatVertexAI(model_name=MODEL_NAME, temperature=0.0) # Lower temp for RAG
        logger.info("LangChain ChatVertexAI (gemini_llm) initialized for RAG.")

        # --- RAG Component Initialization ---
        embeddings = VertexAIEmbeddings(model_name="text-embedding-005") 
        logger.info("VertexAIEmbeddings initialized for RAG.")


        # Adjust this path based on your *exact* directory structure:
        chroma_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'chroma_db')
        logger.info(f"Attempting to load Chroma DB from: {chroma_db_path}")

        if not os.path.exists(chroma_db_path):
            logger.error(f"Chroma DB directory not found at: {chroma_db_path}. RAG will not work.")
            vectorstore = None
            retrieval_chain = None
        else:
            try:
                # Load the existing Chroma vector store
                vectorstore = Chroma(
                    persist_directory=chroma_db_path,
                    embedding_function=embeddings
                )
                logger.info(f"Chroma vector store loaded from {chroma_db_path}.")

                # Create a retriever from the vector store
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 similar documents

                # Define the RAG prompt
                rag_prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are an AI assistant. Use the following context to answer the user's question. If the question cannot be answered from the context, state that you don't have enough information.\n\nContext:\n{context}"),
                    ("user", "{input}")
                ])

                # Create the document stuffing chain (combines retrieved docs into prompt)
                document_chain = create_stuff_documents_chain(gemini_llm, rag_prompt)

                # Create the full retrieval chain (retriever + document stuffing chain)
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                logger.info("LangChain RAG chain (with semantic search) created successfully.")

            except Exception as e:
                logger.error(f"Error loading Chroma DB or creating RAG chain: {e}")
                logger.warning("RAG (semantic search) functionality may be unavailable. Ensure your Chroma DB is correctly set up with compatible embeddings.")
                vectorstore = None
                retrieval_chain = None

    except GoogleAPIError as e:
        logger.error(f"Failed to initialize Google Cloud clients or Vertex AI (RAG): {e}")
        logger.warning("Ensure your GOOGLE_APPLICATION_CREDENTIALS environment variable is set correctly and the service account has the necessary permissions.")
        vectorstore = None
        retrieval_chain = None
        gemini_llm = None
    except Exception as e:
        logger.error(f"An unexpected error occurred during client or RAG initialization: {e}")
        logger.warning("Core functionality or RAG functionality may be unavailable.")
        vectorstore = None
        retrieval_chain = None
        gemini_llm = None

    logger.info("Application startup completed. Yielding control to FastAPI.")
    yield # This indicates that the application is ready to receive requests.

    # --- Shutdown logic (runs when app is shutting down) ---
    logger.info("Application shutdown initiated.")
    logger.info("Application shutdown completed.")

# Pass the lifespan manager to FastAPI ---
app = FastAPI(lifespan=lifespan)

# Allow CORS for your frontend
origins = [
    "http://localhost:5173", # Default Vite dev server port
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Gemini Generation Configuration
gemini_config = GenerationConfig(temperature=0.7, top_p=0.9, top_k=40)


# --- Dialogflow Webhook Request Model ---
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
    elif not response_text:
        response_text = "I couldn't perform that calculation. Please ensure you provided valid numbers and operation."
    return response_text

# --- Dialogflow Webhook Endpoint ---
@app.post("/webhook")
async def dialogflow_webhook(request_body: DialogflowWebhookRequest):
    """
    Handles incoming webhook requests from Dialogflow.
    This endpoint will perform the math calculation based on detected intent and parameters.
    """
    logger.info(f"\nReceived Dialogflow webhook request: {request_body.dict()}")

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
        fulfillment_text = query_result.get("fulfillmentText", "I'm not sure how to respond to that.")
        logger.info(f"Webhook received intent '{intent_display_name}', using Dialogflow's fulfillment text.")

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
    logger.info(f"Processed Dialogflow webhook request: {webhook_response}")
    return JSONResponse(content=webhook_response)

# --- HTML for simple testing ---
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
            const startButton = document.getElementById('startButton');
            const stopButton = document.getElementById('stopButton');
            const transcriptDiv = document.getElementById('transcript');
            const aiResponseDiv = document.getElementById('aiResponse');

            let ws;
            let mediaStream;
            let audioContext;
            let audioInput;
            let processor;
            let currentSampleRate;

            async function playNextAudio(audioBlob) {
                if (!audioBlob || audioBlob.size === 0) {
                    console.warn("Attempted to play empty audio blob.");
                    return;
                }
                const audioUrl = URL.createObjectURL(audioBlob);
                const audio = new Audio(audioUrl);
                audio.onended = () => URL.revokeObjectURL(audioUrl);
                audio.onerror = (e) => console.error('Audio playback error:', e);
                try {
                    await audio.play();
                } catch (error) {
                    console.error("Error attempting to play audio:", error);
                }
            }

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

                ws.onmessage = async (event) => {
                    if (typeof event.data === 'string') {
                        if (event.data.startsWith("You:")) {
                            transcriptDiv.textContent = event.data;
                        } else {
                            aiResponseDiv.innerHTML = `<strong>AI:</strong> ${event.data}`;
                        }
                    } else if (event.data instanceof Blob) {
                        await playNextAudio(event.data);
                    }
                };

                ws.onclose = () => {
                    console.log("WebSocket closed by server.");
                    startButton.disabled = false;
                    stopButton.disabled = true;
                    startButton.classList.remove('recording');
                    if (audioContext && audioContext.state !== 'closed') {
                        audioContext.close();
                    }
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
                }
                
                if (mediaStream) {
                    mediaStream.getTracks().forEach(track => track.stop());
                }
                if (processor) processor.disconnect();
                if (audioInput) audioInput.disconnect();
            };
        </script>
    </body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html)

async def process_final_transcript(transcript_text: str, websocket: WebSocket, dialogflow_session_path: str):
    logger.info(f"Processing final transcript: '{transcript_text}'")

    if not transcript_text.strip():
        logger.info("Empty transcript, skipping processing.")
        return

    if websocket.client_state == WebSocketState.CONNECTED:
        await websocket.send_text(f"You: {transcript_text}")
    else:
        logger.info("WebSocket disconnected, cannot send user transcript back to frontend.")
        return

    response_text = ""
    try:
        # 1. First, try Dialogflow for specific intents
        dialogflow_response = await get_dialogflow_response(transcript_text, dialogflow_session_path)
        
        dialogflow_handled = False
        # Improved check for Dialogflow handling the query
        error_phrases = [
            "Can you say that again?",
            "I didn't get that. Can you say it again?",
            "Sorry, what was that?",
            "Sorry, can you say that again?",
            "I missed what you said. What was that?",
            "One more time?",
            "What was that?",
            "Say that one more time?",
            "I'm sorry, I didn't get that.",
            "I'm sorry, can you repeat that?",
            "I'm sorry, I couldn't process your request.", # Your original phrases
            "I didn't understand that.",
            "I'm not sure how to respond to that.",
            "I'm sorry, I'm having trouble with my understanding right now."
        ]
        # Check if the response is not a generic error and not the initial static text
        if dialogflow_response and not any(phrase in dialogflow_response for phrase in error_phrases) and "Performing that calculation" not in dialogflow_response:
            response_text = dialogflow_response
            dialogflow_handled = True
            logger.info(f"Dialogflow handled the query with final webhook response: {response_text}")
        else:
            logger.info(f"Dialogflow gave initial or error response: '{dialogflow_response}'. Falling back...")

        
        # 2. Fallback to RAG if Dialogflow didn't provide a specific answer
        if not dialogflow_handled and retrieval_chain:
            logger.info(f"Dialogflow did not provide a specific answer. Attempting RAG (semantic search)...\n.\n.\n.\n.\n")
            try:
                rag_result = await retrieval_chain.ainvoke({"input": transcript_text})
                response_text = rag_result.get("answer", "I couldn't find a relevant answer in my documents.")
                logger.info(f"RAG Chain (Semantic) Response: {response_text}")
            except Exception as rag_e:
                logger.error(f"Error during RAG chain (semantic) invocation: {rag_e}")
                response_text = "I couldn't find information about that in my documents using semantic search."
        elif not dialogflow_handled:
            # If RAG is also not available, use the initial (potentially error) response from Dialogflow
            response_text = dialogflow_response if dialogflow_response else "I'm not sure how to respond to that."


    except Exception as e:
        logger.error(f"Error during response generation (Dialogflow/RAG): {e}")
        response_text = "I encountered an error trying to process your request."

    if response_text:
        logger.info(f"Final AI Response Text to be sent: {response_text}")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(response_text)
            await stream_tts_and_send_to_client(response_text, websocket)
    else:
        logger.info("No AI response text generated.")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text("I'm sorry, I couldn't generate a response.")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established.")

    stt_sample_rate_hertz = 0
    stt_config_set = asyncio.Event()

    session_id = str(websocket.client.host) + "-" + str(websocket.client.port)
    # The async client uses the same session_path method
    dialogflow_session_path = dialogflow_sessions_client.session_path(
        settings.GOOGLE_CLOUD_PROJECT_ID, session_id
    )
    logger.info(f"Dialogflow Session Path: {dialogflow_session_path}")

    sync_recognize_requests_queue = queue.Queue()
    stt_results_queue = asyncio.Queue()

    async def consume_audio_stream():
        nonlocal stt_sample_rate_hertz
        try:
            while websocket.client_state == WebSocketState.CONNECTED:
                message = await websocket.receive()
                
                if message["type"] == "websocket.receive" and "bytes" in message:
                    await stt_config_set.wait()
                    sync_recognize_requests_queue.put_nowait(speech.StreamingRecognizeRequest(audio_content=message["bytes"]))
                
                elif message["type"] == "websocket.receive" and "text" in message:
                    if message["text"] == "END_OF_SPEECH":
                        logger.info("Received END_OF_SPEECH signal from client.")
                        break
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
        except Exception as e:
            logger.error(f"Error consuming audio stream: {e}")
        finally:
            sync_recognize_requests_queue.put_nowait(None)

    def synchronous_request_generator():
        while True:
            request = sync_recognize_requests_queue.get()
            if request is None:
                break
            yield request

    async def transcribe_speech():
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
                                collected_transcripts.append(result.alternatives[0].transcript)
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
            await stt_results_queue.put(None)

    async def process_stt_results_consumer():
        try:
            while True:
                transcript_text = await stt_results_queue.get()
                if transcript_text is None:
                    logger.info("Received STT results sentinel, stopping processing.")
                    break 
                
                if websocket.client_state == WebSocketState.CONNECTED:
                    await process_final_transcript(transcript_text, websocket, dialogflow_session_path)
                else:
                    logger.warning("WebSocket disconnected while trying to process STT results.")
                    break
        except Exception as e:
            logger.error(f"Error processing STT results: {e}")
        finally:
            logger.info("STT results processing task finished.")

    consumer_task = asyncio.create_task(consume_audio_stream())
    transcribe_task = asyncio.create_task(transcribe_speech())
    processor_task = asyncio.create_task(process_stt_results_consumer())

    try:
        await asyncio.gather(consumer_task, transcribe_task, processor_task)
    except Exception as e:
        logger.error(f"An error occurred in websocket_endpoint: {e}")
    finally:
        for task in [consumer_task, transcribe_task, processor_task]:
            if not task.done():
                task.cancel()
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()
        logger.info("WebSocket connection closed and tasks cleaned up.")

# --- MODIFICATION 3: Convert helper function to async ---
async def get_dialogflow_response(transcript_text: str, session_path: str) -> str:
    """Asynchronous function to get a response from Dialogflow."""
    if dialogflow_sessions_client is None:
        logger.error("Dialogflow client not initialized.")
        return "I'm sorry, my Dialogflow service is not ready."

    text_input = dialogflow.TextInput(text=transcript_text, language_code="en-US")
    query_input = dialogflow.QueryInput(text=text_input)

    try:
        # Use await with the async client. This call now waits for the webhook to complete.
        response = await dialogflow_sessions_client.detect_intent(
            session=session_path, query_input=query_input
        )
        # The fulfillment_text will now contain the final response from the webhook.
        return response.query_result.fulfillment_text
    except GoogleAPIError as e:
        logger.error(f"Dialogflow API error: {e}")
        return "I'm sorry, I'm having trouble connecting to my understanding service right now."
    except Exception as e:
        logger.error(f"Error detecting intent with Dialogflow: {e}")
        return "I'm sorry, I'm having trouble with my understanding right now."

async def stream_tts_and_send_to_client(text_to_synthesize: str, websocket: WebSocket):
    """Synthesizes text to speech and sends it over the WebSocket."""
    if tts_client is None:
        logger.error("TTS client not initialized.")
        return

    synthesis_input = tts.SynthesisInput(text=text_to_synthesize)
    voice = tts.VoiceSelectionParams(language_code="en-US", name="en-US-Standard-C")
    audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.MP3)

    try:
        # This is a blocking (synchronous) call, so we run it in a thread
        # to avoid blocking the asyncio event loop.
        response = await asyncio.to_thread(
            tts_client.synthesize_speech,
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        audio_bytes = response.audio_content

        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_bytes(audio_bytes)
            logger.info(f"Sent {len(audio_bytes)} bytes of TTS audio.")
        else:
            logger.warning("WebSocket disconnected, unable to send TTS audio.")

    except Exception as e:
        logger.error(f"Error during TTS synthesis or sending audio: {e}")
