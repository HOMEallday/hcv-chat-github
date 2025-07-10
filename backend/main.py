# backend/main.py
import asyncio
import os
import io
import json
import logging
import queue
from starlette.websockets import WebSocketState
from enum import Enum
from typing import AsyncGenerator, List, Optional, Dict, Any
from contextlib import asynccontextmanager
import re
import time

from fastapi.staticfiles import StaticFiles 
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
from langchain_core.callbacks.base import BaseCallbackHandler
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
vectorstore: Optional[Chroma] = None
retrieval_chain = None
gemini_llm: Optional[ChatVertexAI] = None

# Define application states for managing conversation flow
class AppState(Enum):
    IDLE = "IDLE"
    INTRODUCTION = "INTRODUCTION"
    LESSON_DELIVERY = "LESSON_DELIVERY"
    LESSON_PAUSED = "LESSON_PAUSED"
    QNA = "QNA"
    QUIZ = "QUIZ"

# In main.py, replace the UsageCallback class with this final version.

class UsageCallback(BaseCallbackHandler):
    """
    A custom callback handler to capture token usage from Vertex AI.
    This version uses the correct path to the metadata inside the 'generations' object.
    """
    def __init__(self):
        super().__init__()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def on_llm_end(self, response, **kwargs):
        """
        This method is called when the LLM finishes its call.
        """
        # The token usage data is nested within the 'generations' object.
        try:
            # Check if the 'generations' list exists and is not empty
            if response.generations and response.generations[0]:
                
                # Get the 'generation_info' from the first generation result
                generation_info = response.generations[0][0].generation_info
                
                if generation_info:
                    usage_metadata = generation_info.get("usage_metadata", {})
                    self.prompt_tokens = usage_metadata.get("prompt_token_count", 0)
                    self.completion_tokens = usage_metadata.get("candidates_token_count", 0)

        except (KeyError, IndexError, AttributeError) as e:
            logger.warning(f"Could not extract token usage from response: {e}")

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
                retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # Retrieve top 3 similar documents

                # Define the RAG prompt
                rag_prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are an AI assistant for housing policies. Use the following context to answer the user's question.
                     
**CRITICAL INSTRUCTIONS:**
- Provide short responses. One or two sentences is ideal.
- **You must not use any markdown formatting.** This includes never using asterisks (*) for lists.
- Present lists naturally within a sentence. For example, instead of saying "* Item one * Item two", you should say "The policy includes item one, item two, and item three."
- Prioritize your answer to be brief. Only provide a detailed, multi-point summary if the user explicitly asks for "details" or a "detailed summary".
- If the question cannot be answered from the context, state that you don't have enough information.

**Context:**
{context}"""),
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

app.mount("/static", StaticFiles(directory="static"), name="static")

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
    <title>HCV Trainer</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; background-color: #f0f2f5; }

        /* --- FIX 1: Create a flex container for side-by-side layout --- */
        #main-content {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 20px; /* Adds space between the character and chat box */
            padding: 20px;
            max-width: 1200px; /* Widen the max-width to fit both elements */
            margin: auto;
        }

        /* --- FIX 2: Increase the size of the character's container --- */
        #character-container {
            height: 250px; /* Increased from 50px, adjust as needed */
        }
        #character-container img, #character-container video {
            height: 100%;
            width: auto;
        }

        /* --- Chat box styling (mostly unchanged) --- */
        #chat-container {
            flex: 1; /* Allows the chat box to fill the remaining space */
            max-width: 800px;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            min-height: 400px;
        }

        /* --- Other styles (unchanged) --- */
        .message { margin: 10px 0; padding: 10px 15px; border-radius: 18px; line-height: 1.5; max-width: 70%; }
        .user-message { background-color: #0084ff; color: white; text-align: left; margin-left: auto; }
        .ai-message { background-color: #e4e6eb; color: #050505; text-align: left; margin-right: auto; }
        #controls, #status-bar { text-align: center; margin-top: 20px; max-width: 800px; margin: 20px auto; }
        button { padding: 10px 20px; font-size: 1em; margin: 5px; cursor: pointer; border-radius: 20px; border: none; background-color: #0084ff; color: white; }
        button:disabled { background-color: #a0a0a0; cursor: not-allowed; }
        #status { font-weight: bold; color: #333; }
        #mic-indicator { width: 20px; height: 20px; background-color: grey; border-radius: 50%; display: inline-block; vertical-align: middle; margin-left: 10px; }
        #mic-indicator.listening { background-color: #e74c3c; animation: pulse 1.5s infinite; }
        @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(231, 76, 60, 0.7); } 70% { box-shadow: 0 0 0 10px rgba(231, 76, 60, 0); } 100% { box-shadow: 0 0 0 0 rgba(231, 76, 60, 0); } }
    </style>
</head>
<body>
    <div id="main-content">
        <div id="character-container">
            <img id="character-still" src="/static/talkinghouse.jpg" alt="AI Character" style="display: block;">
            <video id="character-talking" src="/static/realTalkHouse.mp4" style="display: none;" loop muted playsinline></video>
        </div>
        <div id="chat-container"><div id="conversation"></div></div>
    </div>

    <div id="status-bar">Current State: <span id="status">Connecting...</span> Mic: <span id="mic-indicator"></span></div>
    <div id="controls">
        <button id="pauseButton" style="display: none;">Pause Lesson</button>
        <button id="resumeButton" style="display: none;">Resume Lesson</button>
        <button id="talkButton">Push to Talk</button>
    </div>

    <script>
    // --- UI Elements ---
    const conversationDiv = document.getElementById('conversation');
    const statusSpan = document.getElementById('status');
    const micIndicator = document.getElementById('mic-indicator');
    const pauseButton = document.getElementById('pauseButton');
    const resumeButton = document.getElementById('resumeButton');
    const talkButton = document.getElementById('talkButton');
    // --- Re-added character elements ---
    const stillImage = document.getElementById('character-still');
    const talkingVideo = document.getElementById('character-talking');

    // --- State Variables ---
    let ws, mediaStream, processor, audioContext;
    let isListening = false, isPlaying = false;
    let audioQueue = [];
    
    // --- WebSocket Connection ---
    function connect() {
        ws = new WebSocket(`ws://${window.location.host}/ws`);
        ws.onopen = () => console.log("WebSocket connected.");
        ws.onclose = () => {
            console.log("WebSocket disconnected.");
            statusSpan.textContent = "Disconnected";
            if (isListening) stopListening();
        };
        ws.onerror = (error) => console.error("WebSocket Error:", error);

        ws.onmessage = (event) => {
            if (typeof event.data === 'string') {
                handleTextMessage(JSON.parse(event.data));
            } else if (event.data instanceof Blob) {
                audioQueue.push(event.data);
                if (!isPlaying) playNextAudio();
            }
        };
    }

    // --- Re-introducing animation logic ---
    async function playNextAudio() {
        if (audioQueue.length === 0) {
            isPlaying = false;
            talkingVideo.pause();
            talkingVideo.style.display = 'none';
            stillImage.style.display = 'block';
            updateUIForState(statusSpan.textContent);
            return;
        }

        isPlaying = true;
        updateUIForState(statusSpan.textContent);

        stillImage.style.display = 'none';
        talkingVideo.style.display = 'block';
        
        const audioBlob = audioQueue.shift();
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        
        audio.onended = () => {
            URL.revokeObjectURL(audioUrl);
            playNextAudio();
        };

        audio.onerror = (e) => {
            console.error('Audio playback error:', e);
            isPlaying = false;
            talkingVideo.pause();
            talkingVideo.style.display = 'none';
            stillImage.style.display = 'block';
            updateUIForState(statusSpan.textContent);
        };
        
        try {
            // This robust try/catch block prevents the app from getting stuck
            await Promise.all([audio.play(), talkingVideo.play()]);
        } catch (error) {
            console.error("Error playing media, possibly due to autoplay restrictions:", error);
            isPlaying = false;
            updateUIForState(statusSpan.textContent); // This is key to re-enabling the button
        }
    }

    // --- Other UI and Logic functions (unchanged) ---
    function handleTextMessage(message) {
        if (message.type === 'state_update') updateUIForState(message.state);
        else if (message.type === 'ai_response') addMessage(message.text, 'ai');
        else if (message.type === 'user_transcript') addMessage(message.text, 'user');
    }

    function updateUIForState(state) {
        statusSpan.textContent = state;
        pauseButton.style.display = (state === 'LESSON_DELIVERY') ? 'inline-block' : 'none';
        resumeButton.style.display = (state === 'LESSON_PAUSED') ? 'inline-block' : 'none';
        const isConversational = ['INTRODUCTION', 'QNA', 'QUIZ', 'LESSON_PAUSED'].includes(state);
        talkButton.disabled = isPlaying || !isConversational;
    }

    function addMessage(text, sender) {
        const messageElem = document.createElement('div');
        messageElem.classList.add('message', `${sender}-message`);
        messageElem.textContent = text;
        conversationDiv.appendChild(messageElem);
        conversationDiv.scrollTop = conversationDiv.scrollHeight;
    }

    async function startListening() {
        if (isListening || !ws || ws.readyState !== WebSocket.OPEN) return;
        
        if (!audioContext || audioContext.state === 'suspended') {
            audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            await audioContext.resume();
        }
        
        isListening = true;
        micIndicator.classList.add('listening');
        ws.send(JSON.stringify({ type: 'control', command: 'start_speech' }));

        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000 } });
        const audioInput = audioContext.createMediaStreamSource(mediaStream);
        processor = audioContext.createScriptProcessor(4096, 1, 1);
        processor.onaudioprocess = (event) => {
            const inputData = event.inputBuffer.getChannelData(0);
            const int16Array = new Int16Array(inputData.length);
            for (let i = 0; i < inputData.length; i++) {
                int16Array[i] = Math.max(-1, Math.min(1, inputData[i])) * 0x7FFF;
            }
            if (ws.readyState === WebSocket.OPEN) ws.send(int16Array.buffer);
        };
        audioInput.connect(processor);
        processor.connect(audioContext.destination);
    }

    function stopListening() {
        if (!isListening) return;
        isListening = false;
        micIndicator.classList.remove('listening');
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'control', command: 'end_speech' }));
        }
        if (mediaStream) mediaStream.getTracks().forEach(track => track.stop());
        if (processor) processor.disconnect();
    }

    // --- Event Listeners ---
    talkButton.onmousedown = startListening;
    talkButton.onmouseup = stopListening;
    pauseButton.onclick = () => ws.send(JSON.stringify({ type: 'control', command: 'pause_lesson' }));
    resumeButton.onclick = () => ws.send(JSON.stringify({ type: 'control', command: 'resume_lesson' }));

    // --- Start Connection ---
    connect();
    </script>
</body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html)
# In main.py

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established.")

    current_state = AppState.IDLE
    session_id = f"{websocket.client.host}-{websocket.client.port}"
    dialogflow_session_path = dialogflow_sessions_client.session_path(
        settings.GOOGLE_CLOUD_PROJECT_ID, session_id
    )

    # --- FIX: The audio queue is now the primary state 'switch' ---
    audio_input_queue = None 

    async def transition_to_state(new_state: AppState):
        nonlocal current_state
        current_state = new_state
        logger.info(f"Transitioning to state: {current_state.value}")
        await websocket.send_json({"type": "state_update", "state": current_state.value})

    try:
        await transition_to_state(AppState.INTRODUCTION)
        intro_text = "Hello! I'm your HCV trainer. Before we begin, what's your name?"
        await websocket.send_json({"type": "ai_response", "text": intro_text})
        await stream_tts_and_send_to_client(intro_text, websocket)

        while websocket.client_state == WebSocketState.CONNECTED:
            message = await websocket.receive()

            # --- FIX: Simplified audio handling logic ---
            if "bytes" in message:
                # If we have an active queue, put audio in it. No other checks needed.
                if audio_input_queue:
                    await audio_input_queue.put(message["bytes"])
            
            elif "text" in message:
                data = json.loads(message["text"])
                command = data.get("type") == "control" and data.get("command")

                if command == "start_speech":
                    # Only start a new session if one isn't already active.
                    if audio_input_queue is None:
                        logger.info("Client signaled start of speech. Creating new transcription task.")
                        audio_input_queue = asyncio.Queue() # Create the queue immediately
                        stt_results_queue = asyncio.Queue()
                        
                        transcription_task = asyncio.create_task(
                            transcribe_speech(audio_input_queue, stt_results_queue)
                        )
                        
                        async def result_handler():
                            transcript = await stt_results_queue.get()
                            await handle_response_by_state(transcript, websocket, dialogflow_session_path, transition_to_state, current_state)
                        
                        asyncio.create_task(result_handler())
                    else:
                        logger.warning("Received 'start_speech' while already listening. Ignoring.")


                elif command == "end_speech":
                    # Only end the session if one is active.
                    if audio_input_queue:
                        logger.info("Client signaled end of speech.")
                        await audio_input_queue.put(None) # Signal the end of the audio stream
                        audio_input_queue = None # Set the 'switch' to off
                    else:
                        logger.warning("Received 'end_speech' but was not listening.")
                
                elif command == "resume_lesson":
                    logger.info("Received resume command from button.")
                    if current_state == AppState.LESSON_PAUSED:
                        await transition_to_state(AppState.LESSON_DELIVERY)
                        resume_text = "Of course. Resuming the lesson."
                        await websocket.send_json({"type": "ai_response", "text": resume_text})
                        await stream_tts_and_send_to_client(resume_text, websocket)
                
                elif command == "pause_lesson":
                    logger.info("Received pause command.")
                    await transition_to_state(AppState.LESSON_PAUSED)

    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"Error in websocket endpoint: {e}", exc_info=True)
    finally:
        logger.info("WebSocket connection closed.")

async def handle_response_by_state(transcript: str, websocket: WebSocket, session_path: str, transition_func, current_state: AppState):
    """Orchestrates the AI's response and state transitions."""
    if transcript:
        await websocket.send_json({"type": "user_transcript", "text": transcript})

    response_text = ""
    if current_state == AppState.INTRODUCTION:
        user_name = extract_name(transcript)
        response_text = f"It's nice to meet you, {user_name}! Let's get started with our lesson."
        await websocket.send_json({"type": "ai_response", "text": response_text})
        await stream_tts_and_send_to_client(response_text, websocket)
        
        await transition_func(AppState.LESSON_DELIVERY)
        # --- MODIFICATION: Pass the transition function to start_lesson ---
        await start_lesson(websocket, transition_func)

    elif current_state == AppState.QNA:
        response_text = await get_rag_response(transcript, session_path)
        await websocket.send_json({"type": "ai_response", "text": response_text})
        await stream_tts_and_send_to_client(response_text, websocket)

    elif current_state == AppState.LESSON_PAUSED:
        if "resume" in transcript.lower() or "continue" in transcript.lower():
            await transition_func(AppState.LESSON_DELIVERY)
            response_text = "Of course. Resuming the lesson."
            await websocket.send_json({"type": "ai_response", "text": response_text})
            await stream_tts_and_send_to_client(response_text, websocket)
            # Add logic here to continue the lesson from where it left off
            
async def start_lesson(websocket: WebSocket, transition_func):
    """
    Delivers a lesson segment and then transitions to the Q&A state.
    """
    lesson_text = "Today, we will be covering the fundamentals of the HCV Program. [INSERT LESSON CONTENT HERE]."

    # 1. Deliver the lesson content
    await websocket.send_json({"type": "ai_response", "text": lesson_text})
    await stream_tts_and_send_to_client(lesson_text, websocket)
    logger.info("Lesson segment has been delivered.")
    
    # --- NEW: Transition to Q&A state after the lesson ---
    
    # A small delay for better conversational pacing
    await asyncio.sleep(1.0) 
    
    logger.info("Transitioning to Q&A state.")
    await transition_func(AppState.QNA)
    
    # 2. Prompt the user for questions
    qna_prompt = "Do you have any questions about what we just covered?"
    await websocket.send_json({"type": "ai_response", "text": qna_prompt})
    await stream_tts_and_send_to_client(qna_prompt, websocket)


async def get_rag_response(transcript_text: str, dialogflow_session_path: str) -> str:
    """
    Gets a response by first checking Dialogflow for specific intents, 
    then falling back to the RAG chain for general questions.
    Now includes performance and cost logging.
    """
    logger.info(f"Handling Q&A for: '{transcript_text}'")
    total_start_time = time.perf_counter()

    if not transcript_text.strip():
        return "I didn't catch that. Could you please repeat?"

    # --- 1. Check Dialogflow First ---
    df_start_time = time.perf_counter()
    dialogflow_response = await get_dialogflow_response(transcript_text, dialogflow_session_path)
    df_latency = time.perf_counter() - df_start_time

    if dialogflow_response and dialogflow_response.intent.display_name != "Default Fallback Intent":
        logger.info(f"Dialogflow handled the query with intent: '{dialogflow_response.intent.display_name}'.")
        response_text = dialogflow_response.fulfillment_text
        
        # Log Dialogflow performance (cost is per-request, not token-based)
        total_latency = time.perf_counter() - total_start_time
        logger.info(f"--- Dialogflow Performance ---")
        logger.info(f"Dialogflow Latency: {df_latency:.4f} seconds")
        logger.info(f"Total Latency: {total_latency:.4f} seconds")
        logger.info(f"Dialogflow Cost: ~$0.007 per request")
        logger.info(f"----------------------------")
        
        return response_text

    # --- 2. Fallback to RAG ---
    logger.info("Falling back to RAG system.")
    usage_callback = UsageCallback()
    rag_start_time = time.perf_counter()
    rag_result = await retrieval_chain.ainvoke(
        {"input": transcript_text},
        config={"callbacks": [usage_callback]}
    )
    rag_latency = time.perf_counter() - rag_start_time
    
    response_text = rag_result.get("answer", "I couldn't find an answer in my documents.")
    
    # --- 3. Log Performance & Cost ---
    log_and_calculate_cost(
        prompt_tokens=usage_callback.prompt_tokens,
        completion_tokens=usage_callback.completion_tokens
    )
    total_latency = time.perf_counter() - total_start_time
    logger.info(f"--- Latency Breakdown ---")
    logger.info(f"Dialogflow Check Latency: {df_latency:.4f} seconds")
    logger.info(f"RAG Chain Latency: {rag_latency:.4f} seconds")
    logger.info(f"Total Response Latency: {total_latency:.4f} seconds")
    logger.info(f"-------------------------")

    return response_text

# In main.py, replace the entire transcribe_speech function with this one.

async def transcribe_speech(audio_input_queue: asyncio.Queue, stt_results_queue: asyncio.Queue):
    """
    Transcribes a single utterance using the sync/async bridge pattern.
    This version is more robust and processes the result immediately.
    """
    sync_bridge_queue = queue.Queue()

    def sync_stt_call():
        def audio_generator():
            while True:
                chunk = sync_bridge_queue.get()
                if chunk is None:
                    break
                yield speech.StreamingRecognizeRequest(audio_content=chunk)

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
            enable_automatic_punctuation=True,
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=False # We only want the final result
        )

        try:
            responses = speech_client.streaming_recognize(
                config=streaming_config,
                requests=audio_generator()
            )

            # --- FIX: Process the response stream and return on the first final result ---
            for response in responses:
                if response.results and response.results[0].is_final:
                    transcript = response.results[0].alternatives[0].transcript
                    logger.info(f"STT Final Transcript: {transcript}")
                    stt_results_queue.put_nowait(transcript)
                    return # Exit as soon as we have what we need

            # If the loop finishes without ever finding a final result,
            # send an empty string to unblock the handler.
            logger.warning("STT stream ended without a final transcript.")
            stt_results_queue.put_nowait("")

        except Exception as e:
            logger.error(f"Error in sync STT call: {e}")
            stt_results_queue.put_nowait("") # Ensure we don't block on error

    # This part remains the same: it bridges the queues and runs the sync call in a thread.
    stt_thread = asyncio.to_thread(sync_stt_call)
    while True:
        chunk = await audio_input_queue.get()
        sync_bridge_queue.put(chunk)
        if chunk is None:
            break
    await stt_thread

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
        return response.query_result
    except GoogleAPIError as e:
        logger.error(f"Dialogflow API error: {e}")
        return "I'm sorry, I'm having trouble connecting to my understanding service right now."
    except Exception as e:
        logger.error(f"Error detecting intent with Dialogflow: {e}")
        return "I'm sorry, I'm having trouble with my understanding right now."

async def stream_tts_and_send_to_client(text_to_synthesize: str, websocket: WebSocket):
    # ... (This function is unchanged and works well) ...
    if not text_to_synthesize: return
    synthesis_input = tts.SynthesisInput(text=text_to_synthesize)
    voice = tts.VoiceSelectionParams(language_code="en-US", name="en-US-Standard-C")
    audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.MP3)
    try:
        response = await asyncio.to_thread(
            tts_client.synthesize_speech,
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_bytes(response.audio_content)
            logger.info(f"Sent {len(response.audio_content)} bytes of TTS audio.")
    except Exception as e:
        logger.error(f"Error during TTS synthesis: {e}")
# --- Correct `transcribe_speech` to be used by the new endpoint ---
async def transcribe_speech_utterance(audio_input_queue: asyncio.Queue, stt_results_queue: asyncio.Queue):
    sync_bridge_queue = queue.Queue()

    def sync_stt_call():
        def audio_generator():
            while True:
                chunk = sync_bridge_queue.get()
                if chunk is None: break
                yield speech.StreamingRecognizeRequest(audio_content=chunk)
        
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz=16000, language_code="en-US", enable_automatic_punctuation=True
        )
        streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=False)

        try:
            responses = speech_client.streaming_recognize(config=streaming_config, requests=audio_generator())
            transcript = ""
            for r in responses:
                if r.results and r.results[0].is_final:
                    transcript = r.results[0].alternatives[0].transcript
            
            logger.info(f"STT Final Transcript: {transcript}")
            stt_results_queue.put_nowait(transcript)
        except Exception as e:
            logger.error(f"Error in sync STT call: {e}")
            stt_results_queue.put_nowait("") # Ensure the handler doesn't wait forever

    stt_thread = asyncio.to_thread(sync_stt_call)

    while True:
        chunk = await audio_input_queue.get()
        sync_bridge_queue.put(chunk)
        if chunk is None:
            break
    
    await stt_thread

def extract_name(transcript: str) -> str:
    """
    Extracts a name from an introductory sentence using pattern matching.
    """
    if not transcript:
        return "there"

    # Pattern 1: Look for phrases like "my name is [Name]", "I'm [Name]", etc.
    # This looks for a capitalized word after the introductory phrase.
    match = re.search(r"(?:my name is|I'm|I am)\s+([A-Z]\w+)", transcript, re.IGNORECASE)
    if match:
        return match.group(1)

    # Fallback Pattern 2: If no specific phrase is found, assume the last word is the name.
    # This handles cases like "It's Ethan" or just saying the name "Ethan".
    words = transcript.strip().split()
    if words:
        potential_name = words[-1].strip(".,!?") # Remove trailing punctuation
        # A simple check to avoid using a common greeting as a name
        if potential_name.lower() not in ["hello", "hi", "hey"]:
            return potential_name

    # If all else fails, use a friendly default.
    return "there"

def log_and_calculate_cost(prompt_tokens: int, completion_tokens: int, model_name: str = "gemini-2.5-flash"):
        """
        Logs token usage and calculates the estimated cost of a RAG query.
        """
        PRICING = {
            "gemini-2.5-flash": {
                "prompt": 0.0003,
                "completion": 0.0025
            },
            # You could add other models here
        }
        cost = 0
        if model_name in PRICING:
            prompt_cost = (prompt_tokens / 1000) * PRICING[model_name]["prompt"]
            completion_cost = (completion_tokens / 1000) * PRICING[model_name]["completion"]
            cost = prompt_cost + completion_cost

        logger.info("--- RAG Performance & Cost ---")
        logger.info(f"Prompt Tokens: {prompt_tokens}")
        logger.info(f"Completion Tokens: {completion_tokens}")
        logger.info(f"Total Tokens: {prompt_tokens + completion_tokens}")
        logger.info(f"Estimated Cost: ${cost:.6f}")
        logger.info("------------------------------")