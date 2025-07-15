import asyncio
import os
import json
import logging
import queue
import re
import time
from contextlib import asynccontextmanager
from enum import Enum, auto
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
    LESSON_PROLOGUE = "LESSON_PROLOGUE"
    LESSON_DELIVERY = "LESSON_DELIVERY"
    LESSON_QUESTION = "LESSON_QUESTION"
    LESSON_FEEDBACK = "LESSON_FEEDBACK"
    LESSON_QNA = "LESSON_QNA"
    QNA = "QNA"
    QUIZ_START = "QUIZ_START"
    QUIZ_QUESTION = "QUIZ_QUESTION"
    QUIZ_FEEDBACK = "QUIZ_FEEDBACK"
    QUIZ_COMPLETE = "QUIZ_COMPLETE"


# --- Lesson & Quiz Content ---
lesson_flow = [
    {
        "type": "lecture",
        "text": "Today, we're going to dive into one of the most fundamental aspects of our work: determining a family's eligibility for the Housing Choice Voucher, or HCV, program."
    },
    {
        "type": "lecture",
        "text": "This session is based on the official Housing Choice Voucher Program Guidebook for Eligibility Determination and Denial of Assistance. Our goal today is to understand the core requirements set by HUD and how as a Public Housing Authority, or PHA, we apply them."
    },
    {
        "type": "question",
        "text": "What does PHA stand for?",
        "correct_answer": "Public Housing Authority",
        "feedback_correct": "That's right! Nicely done.",
        "feedback_incorrect": "Not quite. PHA stands for Public Housing Authority. It's a key term we'll be using a lot."
    },
    {
        "type": "lecture",
        "text": "It's absolutely critical that we strive for objectivity and consistency in every single case. We must always give families the chance to explain their circumstances and understand the basis for our decisions."
    },
    {
        "type": "lecture",
        "text": "And above all, every action we take must comply with all federal, state, and local fair housing and non-discrimination laws."
    },
    {
        "type": "qna_prompt",
        "text": "That's the end of this section. Do you have any questions before we continue?"
    }
]

quiz_questions = [
    {
        "type": "multiple_choice",
        "text": "First question: What is the primary goal of the Housing Choice Voucher program?",
        "options": ["A. To provide luxury housing", "B. To give families access to decent, safe, and sanitary housing", "C. To build new apartment complexes"],
        "correct_answer": "B"
    },
    {
        "type": "multiple_choice",
        "text": "Second question: What must a PHA comply with during eligibility determination?",
        "options": ["A. Only federal housing laws", "B. The tenant's personal preferences", "C. All federal, state, and local fair housing and non-discrimination laws"],
        "correct_answer": "C"
    }
]


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
        # Initialize clients
        speech_client = speech.SpeechClient()
        tts_client = tts.TextToSpeechClient()
        dialogflow_sessions_client = dialogflow.SessionsAsyncClient()
        vertexai.init(project=settings.GOOGLE_CLOUD_PROJECT_ID, location="us-central1")
        gemini_llm = ChatVertexAI(
            model_name="gemini-2.5-flash",
            temperature=0.2,
        )
        embeddings = VertexAIEmbeddings(model_name="text-embedding-005")

        # Load ChromaDB
        chroma_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'chroma_db')
        if os.path.exists(chroma_db_path):
            vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings)

            # Build RAG chain
            retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
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
        logger.error(f"!!! STARTUP FAILED !!! Error details: {e}", exc_info=True)

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
        if result == int(result):
            return f"The answer is {int(result)}."
        else:
            return f"The answer is {result}."
    return "I couldn't perform that calculation."

@app.post("/webhook")
async def dialogflow_webhook(request_body: DialogflowWebhookRequest):
    query_result = request_body.queryResult
    intent_name = query_result.get("intent", {}).get("displayName")
    parameters = query_result.get("parameters", {})
    fulfillment_text = "I'm sorry, I couldn't process your request."

    if intent_name == "Calculate Math":
        try:
            num1, num2 = parameters.get("number1"), parameters.get("number2")
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
        #startup-overlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(240, 242, 245, 0.95); display: flex; flex-direction: column; justify-content: center; align-items: center; z-index: 100; }
        #lesson-menu { background-color: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); text-align: center; }
        .lesson-button { display: block; width: 100%; padding: 15px 20px; font-size: 1.1em; margin-top: 10px; cursor: pointer; border: 1px solid #ccc; background-color: #fff; }
        #main-content { display: flex; justify-content: center; align-items: center; gap: 20px; padding: 20px; max-width: 1200px; margin: auto; }
        #character-container { height: 250px; width: 250px; }
        #character-container img, #character-container video { height: 100%; width: 100%; object-fit: cover; }
        #chat-container { flex: 1; max-width: 800px; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); min-height: 400px; display: flex; flex-direction: column; }
        #conversation { flex-grow: 1; overflow-y: auto; }
        .message { margin: 10px 0; padding: 10px 15px; border-radius: 18px; line-height: 1.5; max-width: 70%; word-wrap: break-word; }
        .user-message { background-color: #0084ff; color: white; margin-left: auto; }
        .ai-message { background-color: #e4e6eb; color: #050505; }
        #quiz-options { margin-top: 15px; display: flex; flex-direction: column; gap: 10px; }
        .quiz-button { padding: 12px; font-size: 1em; text-align: left; }
        #status-bar { text-align: center; max-width: 800px; margin: 20px auto; }
        #mic-indicator { width: 20px; height: 20px; background-color: grey; border-radius: 50%; display: inline-block; vertical-align: middle; margin-left: 10px; }
        #mic-indicator.listening { background-color: #e74c3c; }
    </style>
</head>
<body>
    <div id="startup-overlay">
        <div id="lesson-menu">
            <h2>HCV Training Program</h2>
            <button id="lesson-1-btn" class="lesson-button">Lesson 1: Program Fundamentals</button>
            <button class="lesson-button" disabled>Lesson 2: Coming Soon</button>
        </div>
    </div>
    <div id="main-content">
        <div id="character-container">
            <img id="character-still" src="/static/talkinghouse.jpg" alt="AI Character" style="display: block;">
            <video id="character-talking" src="/static/realTalkHouse.mp4" style="display: none;" loop muted playsinline></video>
        </div>
        <div id="chat-container">
            <div id="conversation"></div>
            <div id="quiz-options"></div>
        </div>
    </div>
    <div id="status-bar">Current State: <span id="status">Waiting to Start...</span> Mic: <span id="mic-indicator"></span></div>

    <script>
        const conversationDiv = document.getElementById('conversation');
        const quizOptionsDiv = document.getElementById('quiz-options');
        const statusSpan = document.getElementById('status');
        const micIndicator = document.getElementById('mic-indicator');
        const stillImage = document.getElementById('character-still');
        const talkingVideo = document.getElementById('character-talking');

        const workletCode = `
            class AudioProcessor extends AudioWorkletProcessor {
                process(inputs) {
                    const pcmData = inputs[0][0];
                    if (pcmData) this.port.postMessage(new Int16Array(pcmData.map(val => Math.max(-1, Math.min(1, val)) * 0x7FFF)));
                    return true;
                }
            }
            registerProcessor('audio-processor', AudioProcessor);
        `;

        let ws, mediaStream, audioContext, workletNode, audioInput, vad;
        let isPlaying = false;
        let audioQueue = [];

        class VoiceActivityDetector {
            constructor(onSpeechStart, onSpeechEnd, silenceThreshold = 1.0) {
                this.onSpeechStart = onSpeechStart;
                this.onSpeechEnd = onSpeechEnd;
                this.silenceThreshold = silenceThreshold;
                this.analyser = null;
                this.isSpeaking = false;
                this.silenceStartTime = 0;
                this.animationFrameId = null;
            }
            start(context, sourceNode) {
                this.analyser = context.createAnalyser();
                this.analyser.fftSize = 512;
                sourceNode.connect(this.analyser);
                this.isSpeaking = false;
                this.monitor();
            }
            stop() { if (this.animationFrameId) cancelAnimationFrame(this.animationFrameId); this.isSpeaking = false; }
            monitor = () => {
                const dataArray = new Uint8Array(this.analyser.frequencyBinCount);
                this.analyser.getByteFrequencyData(dataArray);
                const averageVolume = dataArray.reduce((sum, val) => sum + val, 0) / dataArray.length;
                if (averageVolume > 5) {
                    this.silenceStartTime = 0;
                    if (!this.isSpeaking) { this.isSpeaking = true; this.onSpeechStart(); }
                } else if (this.isSpeaking) {
                    if (this.silenceStartTime === 0) this.silenceStartTime = Date.now();
                    else if ((Date.now() - this.silenceStartTime) > this.silenceThreshold * 1000) { this.onSpeechEnd(); this.isSpeaking = false; }
                }
                this.animationFrameId = requestAnimationFrame(this.monitor);
            }
        }

        async function initializeApp() {
            document.getElementById('startup-overlay').style.display = 'none';
            if (!audioContext) {
                try {
                    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                    const blob = new Blob([workletCode], { type: 'application/javascript' });
                    await audioContext.audioWorklet.addModule(URL.createObjectURL(blob));
                    workletNode = new AudioWorkletNode(audioContext, 'audio-processor');
                } catch (e) { console.error("Error initializing audio context:", e); return; }
            }
            await audioContext.resume();
            connect();
        }

        function connect() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);
            ws.onopen = () => console.log("WebSocket connected.");
            ws.onclose = () => { statusSpan.textContent = "Disconnected"; if (vad) vad.stop(); };
            ws.onerror = (error) => console.error("WebSocket Error:", error);
            ws.onmessage = (event) => {
                if (typeof event.data === 'string') handleTextMessage(JSON.parse(event.data));
                else if (event.data instanceof Blob) { audioQueue.push(event.data); if (!isPlaying) playNextAudio(); }
            };
        }

        async function playNextAudio() {
            if (vad) vad.stop();
            if (audioQueue.length === 0) {
                isPlaying = false;
                talkingVideo.pause();
                talkingVideo.style.display = 'none';
                stillImage.style.display = 'block';
                updateUIForState(statusSpan.textContent);
                if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'control', command: 'tts_finished' }));
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
            audio.onerror = () => isPlaying = false;
            try { await Promise.all([audio.play(), talkingVideo.play()]); }
            catch (error) { isPlaying = false; }
        }

        async function startContinuousListening() {
            if (isPlaying || !audioContext) return;
            const state = statusSpan.textContent;
            const isConversational = ['INTRODUCTION', 'QNA', 'LESSON_QUESTION', 'LESSON_QNA'].includes(state);
            if (!isConversational) return; // Don't listen during quiz questions

            try {
                await audioContext.resume();
                mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000, echoCancellation: true, noiseSuppression: true } });
                audioInput = audioContext.createMediaStreamSource(mediaStream);
                audioInput.connect(workletNode);
                workletNode.port.onmessage = (event) => { if (ws.readyState === WebSocket.OPEN) ws.send(event.data.buffer); };
                vad = new VoiceActivityDetector(
                    () => { micIndicator.classList.add('listening'); ws.send(JSON.stringify({ type: 'control', command: 'start_speech' })); },
                    () => { stopContinuousListening(); ws.send(JSON.stringify({ type: 'control', command: 'end_speech' })); }
                );
                vad.start(audioContext, audioInput);
            } catch (err) { console.error("Mic start error:", err); }
        }

        function stopContinuousListening() {
            if (vad) vad.stop();
            if (mediaStream) mediaStream.getTracks().forEach(track => track.stop());
            if (audioInput) audioInput.disconnect();
            micIndicator.classList.remove('listening');
        }
        
        function handleTextMessage(message) {
            clearQuizOptions();
            if (message.type === 'state_update') {
                updateUIForState(message.state);
            } else if (message.type === 'ai_response') {
                addMessage(message.text, 'ai');
            } else if (message.type === 'user_transcript') {
                addMessage(message.text, 'user');
            } else if (message.type === 'quiz_question') {
                addMessage(message.text, 'ai');
                displayQuizOptions(message.options);
            }
        }
        
        function displayQuizOptions(options) {
            options.forEach(optionText => {
                const button = document.createElement('button');
                button.className = 'lesson-button quiz-button';
                button.innerHTML = optionText; // Use innerHTML to render formatted text
                button.onclick = () => {
                    const selectedOption = optionText.charAt(0); // e.g., "A"
                    addMessage(optionText, 'user');
                    if (ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({ type: 'quiz_answer', answer: selectedOption }));
                    }
                    clearQuizOptions();
                };
                quizOptionsDiv.appendChild(button);
            });
        }
        
        function clearQuizOptions() {
            quizOptionsDiv.innerHTML = '';
        }

        function updateUIForState(state) {
            statusSpan.textContent = state;
            if (state === 'QUIZ_QUESTION') {
                stopContinuousListening(); // Disable mic for quiz buttons
            } else {
                startContinuousListening(); // Re-enable for conversational states
            }
        }

        function addMessage(text, sender) {
            const messageElem = document.createElement('div');
            messageElem.classList.add('message', sender + '-message');
            messageElem.innerHTML = text.replace(/\\n/g, '<br>');
            conversationDiv.appendChild(messageElem);
            conversationDiv.scrollTop = conversationDiv.scrollHeight;
        }
        document.getElementById('lesson-1-btn').onclick = initializeApp;
    </script>
</body>
</html>
"""

# --- Main Endpoints ---
@app.get("/")
async def get():
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established.")

    # --- STATE MANAGEMENT ---
    current_state = AppState.IDLE
    dialogflow_session_path = ""
    audio_input_queue = None
    lesson_step = 0
    quiz_step = 0

    async def transition_to_state(new_state: AppState):
        nonlocal current_state
        current_state = new_state
        logger.info(f"Transitioning to state: {current_state.value}")
        await websocket.send_json({"type": "state_update", "state": current_state.value})

    async def advance_lesson():
        nonlocal lesson_step
        if lesson_step < len(lesson_flow):
            step_data = lesson_flow[lesson_step]
            lesson_step += 1

            if step_data["type"] == "lecture":
                await transition_to_state(AppState.LESSON_DELIVERY)
                await websocket.send_json({"type": "ai_response", "text": step_data["text"]})
                await stream_tts_and_send_to_client(step_data["text"], websocket)
            elif step_data["type"] == "question":
                await transition_to_state(AppState.LESSON_QUESTION)
                await websocket.send_json({"type": "ai_response", "text": step_data["text"]})
                await stream_tts_and_send_to_client(step_data["text"], websocket)
            elif step_data["type"] == "qna_prompt":
                await transition_to_state(AppState.LESSON_QNA)
                await websocket.send_json({"type": "ai_response", "text": step_data["text"]})
                await stream_tts_and_send_to_client(step_data["text"], websocket)
        else:
            await transition_to_state(AppState.QUIZ_START)
            end_text = "You've completed the lesson! Now for a final quiz."
            await websocket.send_json({"type": "ai_response", "text": end_text})
            await stream_tts_and_send_to_client(end_text, websocket)

    async def advance_quiz():
        nonlocal quiz_step
        if quiz_step < len(quiz_questions):
            question_data = quiz_questions[quiz_step]
            quiz_step += 1
            await transition_to_state(AppState.QUIZ_QUESTION)
            # Send question text and options separately for button creation
            await websocket.send_json({
                "type": "quiz_question", 
                "text": question_data['text'],
                "options": question_data['options']
            })
            tts_text = f"{question_data['text']}\n" + "\n".join(question_data['options'])
            await stream_tts_and_send_to_client(tts_text, websocket)
        else:
            await transition_to_state(AppState.QUIZ_COMPLETE)
            end_text = "You've completed the quiz! Great job."
            await websocket.send_json({"type": "ai_response", "text": end_text})
            await stream_tts_and_send_to_client(end_text, websocket)

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
                msg_type = data.get("type")

                if msg_type == "control":
                    command = data.get("command")
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
                                if transcript:
                                    await handle_response_by_state(transcript, websocket, dialogflow_session_path, transition_to_state, current_state, advance_lesson, lesson_step)
                                else:
                                    logger.info("Received empty transcript. Re-prompting user.")
                                    reprompt_text = "I didn't catch that. Could you please say it again?"
                                    await websocket.send_json({"type": "ai_response", "text": reprompt_text})
                                    await stream_tts_and_send_to_client(reprompt_text, websocket)

                            asyncio.create_task(result_handler())

                    elif command == "end_speech":
                        if audio_input_queue:
                            logger.info("Client signaled end of speech.")
                            await audio_input_queue.put(None)
                            audio_input_queue = None
                    elif command == "tts_finished":
                        logger.info(f"Client finished TTS in state: {current_state.value}")
                        if current_state in [AppState.LESSON_PROLOGUE, AppState.LESSON_DELIVERY, AppState.LESSON_FEEDBACK]:
                            await advance_lesson()
                        elif current_state == AppState.QUIZ_START:
                            await advance_quiz()
                        elif current_state == AppState.QUIZ_FEEDBACK:
                             await advance_quiz()
                
                elif msg_type == "quiz_answer":
                    answer = data.get("answer")
                    question_data = quiz_questions[quiz_step - 1]
                    correct_option_letter = question_data['correct_answer']
                    
                    is_correct = (answer == correct_option_letter)
                    feedback = "Correct!" if is_correct else f"Not quite. The correct answer was {correct_option_letter}."
                    
                    await transition_to_state(AppState.QUIZ_FEEDBACK)
                    await websocket.send_json({"type": "ai_response", "text": feedback})
                    await stream_tts_and_send_to_client(feedback, websocket)


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
    cost = ((prompt_tokens / 1000) * PRICING[model_name]["prompt"]) + ((completion_tokens / 1000) * PRICING[model_name]["completion"]) if model_name in PRICING else 0
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

async def handle_response_by_state(transcript: str, websocket: WebSocket, session_path: str, transition_func, current_state: AppState, advance_lesson_func, lesson_step: int):
    await websocket.send_json({"type": "user_transcript", "text": transcript})

    if current_state == AppState.INTRODUCTION:
        user_name = extract_name(transcript)
        response_text = f"It's nice to meet you, {user_name}! Let's get started."
        await websocket.send_json({"type": "ai_response", "text": response_text})
        await stream_tts_and_send_to_client(response_text, websocket)
        await transition_func(AppState.LESSON_PROLOGUE)

    elif current_state == AppState.LESSON_QUESTION:
        step_data = lesson_flow[lesson_step - 1]
        feedback = step_data['feedback_correct'] if step_data['correct_answer'].lower() in transcript.lower() else step_data['feedback_incorrect']
        await transition_func(AppState.LESSON_FEEDBACK)
        await websocket.send_json({"type": "ai_response", "text": feedback})
        await stream_tts_and_send_to_client(feedback, websocket)

    elif current_state == AppState.LESSON_QNA:
        transcript_lower = transcript.lower().strip()
        negative_responses = ["no", "nope", "nah", "i don't", "no thanks"]
        is_negative = any(word in transcript_lower for word in negative_responses)

        if is_negative:
            await advance_lesson_func()
        else:
            await transition_func(AppState.QNA)
            if transcript_lower in ["yes", "yeah", "yep", "sure"]:
                qna_prompt = "Great, what's your question?"
                await websocket.send_json({"type": "ai_response", "text": qna_prompt})
                await stream_tts_and_send_to_client(qna_prompt, websocket)
            else:
                # The user asked the question directly, so answer it immediately.
                response_text = await get_rag_response(transcript, session_path)
                await websocket.send_json({"type": "ai_response", "text": response_text})
                await stream_tts_and_send_to_client(response_text, websocket)
                # Ask if they have more questions and wait for a new response.
                qna_follow_up = "Do you have any other questions?"
                await websocket.send_json({"type": "ai_response", "text": qna_follow_up})
                await stream_tts_and_send_to_client(qna_follow_up, websocket)
                await transition_func(AppState.LESSON_QNA)

    elif current_state == AppState.QNA:
        response_text = await get_rag_response(transcript, session_path)
        await websocket.send_json({"type": "ai_response", "text": response_text})
        await stream_tts_and_send_to_client(response_text, websocket)
        qna_follow_up = "Do you have any other questions?"
        await websocket.send_json({"type": "ai_response", "text": qna_follow_up})
        await stream_tts_and_send_to_client(qna_follow_up, websocket)
        await transition_func(AppState.LESSON_QNA)

async def get_rag_response(transcript_text: str, dialogflow_session_path: str) -> str:
    logger.info(f"Handling Q&A for: '{transcript_text}'")
    if not transcript_text.strip(): return "I didn't catch that. Please repeat."

    dialogflow_response = await get_dialogflow_response(transcript_text, dialogflow_session_path)
    if not isinstance(dialogflow_response, str) and dialogflow_response.intent.display_name != "Default Fallback Intent":
        return dialogflow_response.fulfillment_text

    if not retrieval_chain: return "My document search system isn't configured."
    logger.info("Falling back to RAG system.")
    usage_callback = UsageCallback()
    rag_result = await retrieval_chain.ainvoke({"input": transcript_text}, config={"callbacks": [usage_callback]})
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
                    stt_results_queue.put_nowait(transcript); return
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