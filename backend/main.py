import asyncio
import os
import json
import logging
import queue
import re
import time
from contextlib import asynccontextmanager
from enum import Enum
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from google.api_core.exceptions import GoogleAPIError
from google.cloud import dialogflow_v2 as dialogflow
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import texttospeech_v1 as tts
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from pydantic import BaseModel
import vertexai
from starlette.websockets import WebSocketState

from config import settings

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global Variables ---
speech_client = None
tts_client = None
dialogflow_sessions_client = None
retrieval_chain = None
gemini_llm: Optional[ChatVertexAI] = None

# --- Application State Enum ---
class AppState(Enum):
    IDLE = "IDLE"
    INTRODUCTION = "INTRODUCTION"
    LESSON_DELIVERY = "LESSON_DELIVERY"
    LESSON_PAUSED = "LESSON_PAUSED"
    QNA = "QNA"
    QUIZ = "QUIZ"

# --- LangChain Callback for Token Usage ---
class UsageCallback(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def on_llm_end(self, response, **kwargs):
        try:
            if response.generations and response.generations[0]:
                generation_info = response.generations[0][0].generation_info
                if generation_info:
                    usage_metadata = generation_info.get("usage_metadata", {})
                    self.prompt_tokens = usage_metadata.get("prompt_token_count", 0)
                    self.completion_tokens = usage_metadata.get("candidates_token_count", 0)
        except (KeyError, IndexError, AttributeError) as e:
            logger.warning(f"Could not extract token usage from response: {e}")

# --- FastAPI Lifespan Manager (for startup and shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global speech_client, tts_client, dialogflow_sessions_client, retrieval_chain, gemini_llm
    logger.info("--- Application Startup Initiated ---")
    try:
        # Steps 1-4: Initialize clients (These are all working correctly)
        speech_client = speech.SpeechClient()
        tts_client = tts.TextToSpeechClient()
        dialogflow_sessions_client = dialogflow.SessionsAsyncClient()
        vertexai.init(project=settings.GOOGLE_CLOUD_PROJECT_ID, location="us-central1")
        gemini_llm = ChatVertexAI(
            model_name="gemini-2.5-flash", 
            temperature=0.2,
        )
        embeddings = VertexAIEmbeddings(model_name="text-embedding-005")
        
        # Step 5: Load ChromaDB (Working correctly)
        chroma_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'chroma_db')
        if os.path.exists(chroma_db_path):
            vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings)
            
            # --- Step 6: Build RAG chain ---
            retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

            # --- FIX: Restored the full prompt template with the required variables ---
            rag_prompt = ChatPromptTemplate.from_template("""
You are an AI assistant for housing policies. Provide EXTREMELY concise, one-paragraph summaries.
CRITICAL INSTRUCTIONS:
- Your response MUST be a single, short paragraph.
- You MUST NOT use any markdown formatting.
- If the context does not contain the answer, ONLY say "I don't have enough information to answer that."
- Get straight to the point.

CONTEXT:
{context}

QUESTION:
{input}
""")
            
            document_chain = create_stuff_documents_chain(gemini_llm, rag_prompt)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            logger.info(">>> RAG chain built successfully.")
        else:
            logger.error(f"Chroma DB not found at {chroma_db_path}. RAG will be unavailable.")

    except Exception as e:
        logger.error(f"!!! STARTUP FAILED at one of the steps above !!!")
        logger.error(f"Error details: {e}", exc_info=True)

    yield
    logger.info("--- Application Shutdown ---")

# --- FastAPI App Initialization ---
app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DialogflowWebhookRequest(BaseModel):
    queryResult: Dict[str, Any]
    session: Optional[str] = None

def _perform_calculation(num1: float, num2: float, operation: str) -> str:
    """Performs the specified mathematical operation."""
    result = None
    if operation == "add" or operation == "plus":
        result = num1 + num2
    elif operation == "subtract" or operation == "minus":
        result = num1 - num2
    elif operation == "multiply" or operation == "times":
        result = num1 * num2
    elif operation == "divide":
        if num2 != 0:
            result = num1 / num2
        else:
            return "I cannot divide by zero."
    else:
        return f"I don't recognize the operation '{operation}'."

    if result is not None:
        # Format to remove unnecessary .0 for whole numbers
        if result == int(result):
            return f"The answer is {int(result)}."
        else:
            return f"The answer is {result}."
    return "I couldn't perform that calculation."

@app.post("/webhook")
async def dialogflow_webhook(request_body: DialogflowWebhookRequest):
    """
    Handles incoming webhook requests from Dialogflow for calculations.
    """
    query_result = request_body.queryResult
    intent_name = query_result.get("intent", {}).get("displayName")
    parameters = query_result.get("parameters", {})
    
    fulfillment_text = "I'm sorry, I couldn't process your request."

    if intent_name == "Calculate Math": # Make sure this matches your intent name in Dialogflow
        try:
            num1 = parameters.get("number1")
            num2 = parameters.get("number2") # Dialogflow often names them number and number1
            operation = parameters.get("operation")

            if num1 is not None and num2 is not None and operation:
                fulfillment_text = _perform_calculation(float(num1), float(num2), operation)
            else:
                fulfillment_text = "I need two numbers and an operation to do math."
        except Exception as e:
            logger.error(f"Error in webhook calculation: {e}")
            fulfillment_text = "I ran into an error trying to calculate that."
    
    return JSONResponse(content={"fulfillmentText": fulfillment_text})

# --- HTML Frontend ---
html = """
<!DOCTYPE html>
<html>
<head>
    <title>HCV Trainer</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; background-color: #f0f2f5; }
        #main-content { display: flex; justify-content: center; align-items: center; gap: 20px; padding: 20px; max-width: 1200px; margin: auto; }
        #character-container { height: 250px; width: 250px; }
        #character-container img, #character-container video { height: 100%; width: 100%; object-fit: cover; }
        #chat-container { flex: 1; max-width: 800px; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); min-height: 400px; }
        .message { margin: 10px 0; padding: 10px 15px; border-radius: 18px; line-height: 1.5; max-width: 70%; }
        .user-message { background-color: #0084ff; color: white; margin-left: auto; }
        .ai-message { background-color: #e4e6eb; color: #050505; }
        #controls, #status-bar { text-align: center; max-width: 800px; margin: 20px auto; }
        button { padding: 10px 20px; font-size: 1em; margin: 5px; cursor: pointer; border-radius: 20px; border: none; background-color: #0084ff; color: white; }
        button:disabled { background-color: #a0a0a0; }
        #mic-indicator { width: 20px; height: 20px; background-color: grey; border-radius: 50%; display: inline-block; vertical-align: middle; margin-left: 10px; }
        #mic-indicator.listening { background-color: #e74c3c; }
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
            ws.onclose = () => { statusSpan.textContent = "Disconnected"; if (isListening) stopListening(); };
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

        // --- Audio Playback & Animation (This part is correct) ---
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
            
            audio.onended = () => { URL.revokeObjectURL(audioUrl); playNextAudio(); };
            audio.onerror = (e) => { isPlaying = false; updateUIForState(statusSpan.textContent); };
            
            try {
                await Promise.all([audio.play(), talkingVideo.play()]);
            } catch (error) {
                console.error("Error playing media:", error);
                isPlaying = false;
                updateUIForState(statusSpan.textContent);
            }
        }

        // --- FIX: Reverting to createScriptProcessor for debugging ---
        async function startListening() {
            if (isListening || !ws || ws.readyState !== WebSocket.OPEN) return;
            
            try {
                if (!audioContext || audioContext.state === 'suspended') {
                    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                    await audioContext.resume();
                }
                
                isListening = true;
                micIndicator.classList.add('listening');
                ws.send(JSON.stringify({ type: 'control', command: 'start_speech' }));

                mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000 } });
                const audioInput = audioContext.createMediaStreamSource(mediaStream);
                
                // Using the older, simpler method
                processor = audioContext.createScriptProcessor(4096, 1, 1);
                
                processor.onaudioprocess = (event) => {
                    if (!isListening) return;
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

            } catch (err) {
                console.error("Mic start error:", err);
                isListening = false;
                micIndicator.classList.remove('listening');
            }
        }

        function stopListening() {
            if (!isListening) return;
            isListening = false;
            micIndicator.classList.remove('listening');
            
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'control', command: 'end_speech' }));
            }
            if (mediaStream) {
                mediaStream.getTracks().forEach(track => track.stop());
            }
            if (processor) {
                processor.disconnect();
                processor = null;
            }
        }

        // --- Other UI and Logic functions ---
        function handleTextMessage(message) {
            if (message.type === 'state_update') updateUIForState(message.state);
            else if (message.type === 'ai_response') addMessage(message.text, 'ai');
            else if (message.type === 'user_transcript') addMessage(message.text, 'user');
        }

        function updateUIForState(state) {
            statusSpan.textContent = state;
            document.getElementById('pauseButton').style.display = (state === 'LESSON_DELIVERY') ? 'inline-block' : 'none';
            document.getElementById('resumeButton').style.display = (state === 'LESSON_PAUSED') ? 'inline-block' : 'none';
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

# --- Main WebSocket Endpoint ---
@app.get("/")
async def get():
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established.")
    current_state = AppState.IDLE
    dialogflow_session_path = ""
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
            if "bytes" in message:
                if audio_input_queue: await audio_input_queue.put(message["bytes"])
            elif "text" in message:
                data = json.loads(message["text"])
                command = data.get("type") == "control" and data.get("command")

                if command == "start_speech":
                    if audio_input_queue is None:
                        logger.info("Client signaled start of speech.")
                        audio_input_queue = asyncio.Queue()
                        stt_results_queue = asyncio.Queue()
                        
                        session_id = f"{websocket.client.host}-{websocket.client.port}-{time.time()}"
                        dialogflow_session_path = dialogflow_sessions_client.session_path(
                            settings.GOOGLE_CLOUD_PROJECT_ID, session_id
                        )
                        asyncio.create_task(transcribe_speech(audio_input_queue, stt_results_queue))
                        
                        async def result_handler():
                            transcript = await stt_results_queue.get()
                            await handle_response_by_state(transcript, websocket, dialogflow_session_path, transition_to_state, current_state)
                        
                        asyncio.create_task(result_handler())

                elif command == "end_speech":
                    if audio_input_queue:
                        logger.info("Client signaled end of speech.")
                        await audio_input_queue.put(None)
                        audio_input_queue = None

    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"Error in websocket endpoint: {e}", exc_info=True)
    finally:
        logger.info("WebSocket connection closed.")

# --- Helper Functions ---
def extract_name(transcript: str) -> str:
    if not transcript: return "there"
    match = re.search(r"(?:my name is|I'm|I am)\s+([A-Z]\w+)", transcript, re.IGNORECASE)
    if match: return match.group(1)
    words = transcript.strip().split()
    if words:
        potential_name = words[-1].strip(".,!?")
        if potential_name.lower() not in ["hello", "hi", "hey"]: return potential_name
    return "there"

def log_and_calculate_cost(prompt_tokens: int, completion_tokens: int, model_name: str = "gemini-2.5-flash"):
    PRICING = {"gemini-2.5-flash": {"prompt": 0.00013, "completion": 0.00038}}
    cost = 0
    if model_name in PRICING:
        cost = ((prompt_tokens / 1000) * PRICING[model_name]["prompt"]) + ((completion_tokens / 1000) * PRICING[model_name]["completion"])
    logger.info(f"--- RAG Cost --- Tokens: {prompt_tokens}p + {completion_tokens}c = {prompt_tokens + completion_tokens}t | Est. Cost: ${cost:.6f}")

async def get_dialogflow_response(transcript_text: str, session_path: str):
    text_input = dialogflow.TextInput(text=transcript_text, language_code="en-US")
    query_input = dialogflow.QueryInput(text=text_input)
    try:
        response = await dialogflow_sessions_client.detect_intent(session=session_path, query_input=query_input)
        return response.query_result
    except GoogleAPIError as e:
        logger.error(f"Dialogflow API error: {e}")
        return f"Dialogflow Error: {e}"

async def handle_response_by_state(transcript: str, websocket: WebSocket, session_path: str, transition_func, current_state: AppState):
    if transcript:
        await websocket.send_json({"type": "user_transcript", "text": transcript})

    if current_state == AppState.INTRODUCTION:
        user_name = extract_name(transcript)
        response_text = f"It's nice to meet you, {user_name}! Let's get started with our lesson."
        await websocket.send_json({"type": "ai_response", "text": response_text})
        await stream_tts_and_send_to_client(response_text, websocket)
        await transition_func(AppState.LESSON_DELIVERY)
        await start_lesson(websocket, transition_func)
    elif current_state == AppState.QNA:
        response_text = await get_rag_response(transcript, session_path)
        await websocket.send_json({"type": "ai_response", "text": response_text})
        await stream_tts_and_send_to_client(response_text, websocket)

async def start_lesson(websocket: WebSocket, transition_func):
    lesson_text = "Today, we will cover the fundamentals of the HCV Program."
    await websocket.send_json({"type": "ai_response", "text": lesson_text})
    await stream_tts_and_send_to_client(lesson_text, websocket)
    await asyncio.sleep(1.0) 
    await transition_func(AppState.QNA)
    qna_prompt = "Do you have any questions about what we just covered?"
    await websocket.send_json({"type": "ai_response", "text": qna_prompt})
    await stream_tts_and_send_to_client(qna_prompt, websocket)

async def get_rag_response(transcript_text: str, dialogflow_session_path: str) -> str:
    logger.info(f"Handling Q&A for: '{transcript_text}'")
    if not transcript_text.strip(): return "I didn't catch that. Please repeat."
    
    dialogflow_response = await get_dialogflow_response(transcript_text, dialogflow_session_path)
    if not isinstance(dialogflow_response, str) and dialogflow_response.intent.display_name != "Default Fallback Intent":
        return dialogflow_response.fulfillment_text

    if not retrieval_chain: return "My document search system isn't configured."
    logger.info("Falling back to RAG system.")
    usage_callback = UsageCallback()
    rag_result = await retrieval_chain.ainvoke(
        {"input": transcript_text},
        config={"callbacks": [usage_callback]}
    )
    log_and_calculate_cost(usage_callback.prompt_tokens, usage_callback.completion_tokens)
    return rag_result.get("answer", "I couldn't find an answer for that in my documents.")

async def transcribe_speech(audio_input_queue: asyncio.Queue, stt_results_queue: asyncio.Queue):
    sync_bridge_queue = queue.Queue()
    def sync_stt_call():
        def audio_generator():
            while True:
                chunk = sync_bridge_queue.get()
                if chunk is None: break
                yield speech.StreamingRecognizeRequest(audio_content=chunk)
        
        config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz=16000, language_code="en-US", enable_automatic_punctuation=True)
        streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=False)
        try:
            responses = speech_client.streaming_recognize(config=streaming_config, requests=audio_generator())
            for response in responses:
                if response.results and response.results[0].is_final:
                    transcript = response.results[0].alternatives[0].transcript
                    logger.info(f"STT Final Transcript: {transcript}")
                    stt_results_queue.put_nowait(transcript)
                    return
            logger.warning("STT stream ended without a final transcript.")
            stt_results_queue.put_nowait("")
        except Exception as e:
            logger.error(f"Error in sync STT call: {e}")
            stt_results_queue.put_nowait("")

    stt_thread = asyncio.to_thread(sync_stt_call)
    while True:
        chunk = await audio_input_queue.get()
        sync_bridge_queue.put(chunk)
        if chunk is None: break
    await stt_thread

async def stream_tts_and_send_to_client(text_to_synthesize: str, websocket: WebSocket):
    if not text_to_synthesize: return
    synthesis_input = tts.SynthesisInput(text=text_to_synthesize)
    voice = tts.VoiceSelectionParams(language_code="en-US", name="en-US-Standard-C")
    audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.MP3)
    try:
        response = await asyncio.to_thread(
            tts_client.synthesize_speech, input=synthesis_input, voice=voice, audio_config=audio_config
        )
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_bytes(response.audio_content)
    except Exception as e:
        logger.error(f"Error during TTS synthesis: {e}")