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
import azure.cognitiveservices.speech as speechsdk
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from pydantic import BaseModel
import vertexai
from starlette.websockets import WebSocketState, WebSocketDisconnect
from websockets.exceptions import ConnectionClosed

from config import settings

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global Variables (Initialized in lifespan) ---
speech_client = None
speech_config = None
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
lessons = {
    "1": {
        "title": "Program Fundamentals",
        "flow": [
            {
                "type": "lecture",
                "text": "Welcome. This lesson covers the foundational rules for Eligibility Determination and Denial of Assistance for the Housing Choice Voucher program, based on the November 2019 guidebook. Our goal is to understand the core requirements set by HUD and how we, as a Public Housing Authority, must apply them."
            },
            {
                "type": "question",
                "text": "Before we dive in, can you recall one of those four main eligibility factors?",
                "correct_answer": "Family Eligibility, Income Limits, Student Status, or Citizenship Status",
                "feedback_correct": "Excellent! Yes, that's one of the core four.",
                "feedback_incorrect": "Close. The four main factors are Family Eligibility, Income Limits, Student Status, and Citizenship Status."
            },
            {
                "type": "qna_prompt",
                "text": "That covers the basics. Do you have any questions before we move on to the quiz?"
            }
        ],
        "quiz": [
            {
                "type": "multiple_choice",
                "text": "A family of 5 applies for a voucher. 4 members have eligible citizenship status, but one does not. What should the PHA do?",
                "options": ["A. Deny assistance to the entire family.", "B. Admit the family with a prorated assistance payment based on 4 out of 5 members being eligible.", "C. Tell the family to re-apply after the ineligible member leaves the household."],
                "correct_answer": "B",
                "explanation": "This is a 'mixed family.' According to HUD rules, they are not denied but receive prorated assistance based on the number of eligible members."
            },
            {
                "type": "multiple_choice",
                "text": "The 'Income Targeting' rule states that at least 75% of new admissions to the HCV program must be families whose income is at or below the...",
                "options": ["A. Area Median Income limit.", "B. Low-Income limit.", "C. Extremely Low-Income (ELI) limit."],
                "correct_answer": "C",
                "explanation": "The 75% income targeting rule is a key HUD requirement ensuring that PHAs prioritize serving the neediest families, who are categorized as Extremely Low-Income (ELI)."
            },
        ]
    },
    "2": {
        "title": "Advanced Eligibility (Coming Soon...)",
        "flow": [
            { "type": "lecture", "text": "Welcome to Lesson 2. This lesson is currently under development. Please check back later." },
            { "type": "qna_prompt", "text": "Would you like to try the placeholder quiz for lesson 2?" }
        ],
        "quiz": [
            {
                "type": "multiple_choice",
                "text": "This is a placeholder question for Lesson 2. What is the correct answer?",
                "options": ["A. This one.", "B. Not this one.", "C. Maybe this one."],
                "correct_answer": "A",
                "explanation": "This is just a placeholder to demonstrate the quiz functionality for a second lesson."
            }
        ]
    }
}

# --- LangChain Callback for Token Usage ---
class UsageCallback(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def on_llm_end(self, response, **kwargs):
        try:
            usage_metadata = response.generations[0][0].generation_info.get("usage_metadata", {})
            self.prompt_tokens = usage_metadata.get("prompt_token_count", 0)
            self.completion_tokens = usage_metadata.get("candidates_token_count", 0)
        except (KeyError, IndexError, AttributeError, TypeError):
            logger.warning("Could not extract token usage from response.")

# --- FastAPI Lifespan Manager (for startup and shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global speech_client, speech_config, dialogflow_sessions_client, retrieval_chain, gemini_llm
    logger.info("--- Application Startup Initiated ---")
    try:
        speech_client = speech.SpeechClient()
        dialogflow_sessions_client = dialogflow.SessionsAsyncClient()
        vertexai.init(project=settings.GOOGLE_CLOUD_PROJECT_ID)
        logger.info(f"Google clients initialized for project: {settings.GOOGLE_CLOUD_PROJECT_ID}")

        speech_config = speechsdk.SpeechConfig(subscription=settings.AZURE_SPEECH_KEY, region=settings.AZURE_SPEECH_REGION)
        speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
        speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3)
        logger.info("Azure Speech Service configured.")

        gemini_llm = ChatVertexAI(model_name="gemini-1.5-flash", temperature=0.2)
        embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003")

        chroma_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'chroma_db')
        if os.path.exists(chroma_db_path):
            vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
            rag_prompt = ChatPromptTemplate.from_template("...") # Your prompt here
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
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- HTML Frontend (No Changes Needed) ---
html = """
<!DOCTYPE html>
<html>
<head>
    <title>HCV Trainer</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; display: flex; flex-direction: column; height: 100vh; background-image: url('/static/landscape.gif'); background-size: cover; background-position: center; background-attachment: fixed; }
        body::before { content: ''; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(255, 255, 255, 0.275); z-index: -1; }
        #startup-overlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(240, 242, 245, 0.95); display: flex; flex-direction: column; justify-content: center; align-items: center; z-index: 100; }
        #lesson-menu { background-color: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); text-align: center; }
        .lesson-button { display: block; width: 100%; padding: 15px 20px; font-size: 1.1em; margin-top: 10px; cursor: pointer; border: 1px solid #ccc; background-color: #fff; }
        #main-content { display: none; justify-content: center; align-items: center; gap: 20px; padding: 20px; max-width: 1200px; margin: auto; }
        #character-container { height: 375px; width: 375px; position: relative; display: flex; justify-content: center; align-items: center; }
        #character-container img { height: 100%; width: 100%; object-fit: cover; position: absolute; top: 0; left: 0; }
        #viseme-mouth { width: 50%; height: auto; position: absolute; top: 55%; z-index: 10; }
        #chat-container { flex: 1; max-width: 800px; background: #e0ffff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); height: 600px; display: flex; flex-direction: column; }
        #conversation { flex-grow: 1; overflow-y: auto; }
        .message { margin: 10px 0; padding: 10px 15px; border-radius: 18px; line-height: 1.5; max-width: 70%; word-wrap: break-word; }
        .user-message { background-color: #0084ff; color: white; margin-left: auto; }
        .ai-message { background-color: #e4e6eb; color: #050505; }
        #quiz-options { margin-top: 15px; display: flex; flex-direction: column; gap: 10px; }
        .quiz-button { padding: 12px; font-size: 1em; text-align: left; }
        #status-bar { text-align: center; max-width: 800px; margin: 20px auto; padding: 10px; background-color: #fff; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.1); display: flex; justify-content: center; align-items: center; gap: 20px; }
        #mic-indicator { width: 20px; height: 20px; background-color: grey; border-radius: 50%; display: inline-block; vertical-align: middle; margin-left: 5px; }
        #mic-indicator.listening { background-color: #e74c3c; }
    </style>
</head>
<body>
    <div id="startup-overlay">
        <div id="lesson-menu">
            <h2>HCV Training Program</h2>
            <button class="lesson-button" data-lesson-id="1">Lesson 1: Program Fundamentals</button>
            <button class="lesson-button" data-lesson-id="2">Lesson 2: Advanced Eligibility</button>
            <button class="lesson-button" disabled>Lesson 3: Rent Calculation (Coming Soon...)</button>
        </div>
    </div>
    <div id="main-content">
        <div id="character-container">
            <img id="character-still" src="/static/talkinghouse.jpg" alt="AI Character Base">
            <img id="viseme-mouth" src="/static/visemes/viseme_0.png" alt="AI Character Mouth">
        </div>
        <div id="chat-container">
            <div id="conversation"></div>
            <div id="quiz-options"></div>
        </div>
    </div>
    <div id="status-bar">
        <div>Current State: <span id="status">Waiting to Start...</span> Mic: <span id="mic-indicator"></span></div>
    </div>
    <script>
        // --- THIS SCRIPT IS UNCHANGED AND CORRECT ---
        // It correctly sends the `select_lesson` message upon button click.
        const conversationDiv = document.getElementById('conversation');
        const quizOptionsDiv = document.getElementById('quiz-options');
        const statusSpan = document.getElementById('status');
        const micIndicator = document.getElementById('mic-indicator');
        const visemeMouth = document.getElementById('viseme-mouth');
        const mainContent = document.getElementById('main-content');
        const startupOverlay = document.getElementById('startup-overlay');

        let ws, audioContext, workletNode, vad, mediaStream, audioInput;
        let isPlaying = false;
        let currentAudioSource = null;
        let currentAudioChunks = [];
        let visemeQueue = [];
        let animationFrameId = null;

        const visemeMap = {0: '/static/visemes/viseme_0.png', 1: '/static/visemes/viseme_1.png', 2: '/static/visemes/viseme_2.png', 3: '/static/visemes/viseme_3.png', 4: '/static/visemes/viseme_4.png', 5: '/static/visemes/viseme_5.png', 6: '/static/visemes/viseme_6.png', 7: '/static/visemes/viseme_7.png', 8: '/static/visemes/viseme_8.png', 9: '/static/visemes/viseme_9.png', 10: '/static/visemes/viseme_10.png', 11: '/static/visemes/viseme_11.png', 12: '/static/visemes/viseme_12.png', 13: '/static/visemes/viseme_13.png', 14: '/static/visemes/viseme_14.png', 15: '/static/visemes/viseme_15.png', 16: '/static/visemes/viseme_16.png', 17: '/static/visemes/viseme_17.png', 18: '/static/visemes/viseme_18.png', 19: '/static/visemes/viseme_19.png', 20: '/static/visemes/viseme_20.png', 21: '/static/visemes/viseme_21.png'};
        Object.values(visemeMap).forEach(path => { (new Image()).src = path; });

        const workletCode = `class AudioProcessor extends AudioWorkletProcessor { process(inputs) { const p = inputs[0][0]; if (p) this.port.postMessage(new Int16Array(p.map(v => Math.max(-1, Math.min(1, v)) * 0x7FFF))); return true; } } registerProcessor('audio-processor', AudioProcessor);`;
        class VoiceActivityDetector { constructor(onStart, onEnd, threshold = 1.0) {this.onStart=onStart; this.onEnd=onEnd; this.threshold=threshold; this.analyser=null; this.speaking=false; this.silenceTime=0; this.frameId=null;} start(ctx, src) {this.analyser=ctx.createAnalyser(); this.analyser.fftSize=512; src.connect(this.analyser); this.speaking=false; this.monitor();} stop() {if(this.frameId)cancelAnimationFrame(this.frameId);this.speaking=false;} monitor = () => {const arr=new Uint8Array(this.analyser.frequencyBinCount);this.analyser.getByteFrequencyData(arr);const avg=arr.reduce((s,v)=>s+v,0)/arr.length;if(avg>5){this.silenceTime=0;if(!this.speaking){this.speaking=true;this.onStart();}}else if(this.speaking){if(this.silenceTime===0)this.silenceTime=Date.now();else if((Date.now()-this.silenceTime)>this.threshold*1000){this.onEnd();this.speaking=false;}}this.frameId=requestAnimationFrame(this.monitor);}}

        document.addEventListener('DOMContentLoaded', () => {
            const completed = JSON.parse(localStorage.getItem('completedLessons')) || [];
            document.querySelectorAll('.lesson-button[data-lesson-id]').forEach(b => {
                const id = b.getAttribute('data-lesson-id');
                if (completed.includes(id)) { b.textContent += ' (Completed)'; b.style.backgroundColor = '#d4edda'; }
                b.onclick = () => startLesson(id);
            });
        });

        async function startLesson(lessonId) {
            startupOverlay.style.display = 'none';
            mainContent.style.display = 'flex';
            if (!audioContext) {
                try {
                    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                    const blob = new Blob([workletCode], { type: 'application/javascript' });
                    await audioContext.audioWorklet.addModule(URL.createObjectURL(blob));
                    workletNode = new AudioWorkletNode(audioContext, 'audio-processor');
                } catch (e) { console.error("Audio context init error:", e); return; }
            }
            await audioContext.resume();
            connect(() => {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    console.log(`Sending select_lesson for ID: ${lessonId}`);
                    ws.send(JSON.stringify({ type: 'select_lesson', lesson_id: lessonId }));
                }
            });
        }

        function connect(onOpenCallback) {
            ws = new WebSocket(`ws://${window.location.host}/ws`);
            ws.onopen = () => { console.log("WS connected."); if (onOpenCallback) onOpenCallback(); };
            ws.onclose = () => { statusSpan.textContent = "Disconnected"; stopContinuousListening(); };
            ws.onerror = (e) => console.error("WS Error:", e);
            ws.onmessage = (event) => {
                if (event.data instanceof Blob) currentAudioChunks.push(event.data);
                else {
                    const msg = JSON.parse(event.data);
                    if (msg.type === 'tts_stream_finished') playCombinedAudio();
                    else if (msg.type === 'viseme') visemeQueue.push(msg);
                    else handleTextMessage(msg);
                }
            };
        }
        function handleTextMessage(msg) {
            clearQuizOptions();
            if (msg.type === 'state_update') statusSpan.textContent = msg.state;
            else if (msg.type === 'ai_response') addMessage(msg.text, 'ai');
            else if (msg.type === 'user_transcript') addMessage(msg.text, 'user');
            else if (msg.type === 'quiz_question') { addMessage(msg.text, 'ai'); displayQuizOptions(msg.options); }
            else if (msg.type === 'quiz_summary') {
                addMessage(msg.text, 'ai');
                let completed = JSON.parse(localStorage.getItem('completedLessons')) || [];
                if (!completed.includes(msg.lesson_id.toString())) { completed.push(msg.lesson_id.toString()); localStorage.setItem('completedLessons', JSON.stringify(completed)); }
                const btn = document.createElement('button');
                btn.className = 'lesson-button';
                btn.textContent = 'Return to Lesson Menu';
                btn.onclick = () => window.location.reload();
                quizOptionsDiv.appendChild(btn);
            }
        }
        async function playCombinedAudio() {
            if (currentAudioChunks.length === 0 || !audioContext) return;
            isPlaying = true;
            stopContinuousListening();
            visemeMouth.style.display = 'block';
            const blob = new Blob(currentAudioChunks, { type: 'audio/mp3' });
            currentAudioChunks = [];
            try {
                const buffer = await audioContext.decodeAudioData(await blob.arrayBuffer());
                if (currentAudioSource) currentAudioSource.stop();
                if (animationFrameId) cancelAnimationFrame(animationFrameId);
                currentAudioSource = audioContext.createBufferSource();
                currentAudioSource.buffer = buffer;
                currentAudioSource.connect(audioContext.destination);
                currentAudioSource.onended = () => {
                    isPlaying = false; visemeMouth.src = visemeMap[0]; visemeQueue = []; cancelAnimationFrame(animationFrameId);
                    startContinuousListening(); 
                    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'control', command: 'tts_finished' }));
                };
                const startTime = audioContext.currentTime;
                currentAudioSource.start(0);
                animateVisemes(startTime);
            } catch (e) { console.error("Audio playback error:", e); isPlaying = false; }
        }
        function animateVisemes(startTime) {
            const elapsed = (audioContext.currentTime - startTime) * 1000;
            let lastViseme = null;
            while (visemeQueue.length > 0 && visemeQueue[0].offset_ms <= elapsed) lastViseme = visemeQueue.shift();
            if (lastViseme) visemeMouth.src = visemeMap[lastViseme.viseme_id];
            if (isPlaying) animationFrameId = requestAnimationFrame(() => animateVisemes(startTime));
        }
        async function startContinuousListening() {
            stopContinuousListening();
            if (isPlaying || !audioContext) return;
            if (!['INTRODUCTION', 'LESSON_QUESTION', 'LESSON_QNA', 'QNA'].includes(statusSpan.textContent)) return;
            try {
                await audioContext.resume();
                mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000, echoCancellation: true, noiseSuppression: true } });
                audioInput = audioContext.createMediaStreamSource(mediaStream);
                audioInput.connect(workletNode);
                workletNode.port.onmessage = (e) => { if (ws.readyState === WebSocket.OPEN) ws.send(e.data.buffer); };
                vad = new VoiceActivityDetector(
                    () => { micIndicator.classList.add('listening'); ws.send(JSON.stringify({ type: 'control', command: 'start_speech' })); },
                    () => { stopContinuousListening(); if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'control', command: 'end_speech' })); }
                );
                vad.start(audioContext, audioInput);
            } catch (err) { console.error("Mic error:", err); }
        }
        function stopContinuousListening() {
            if (vad) { vad.stop(); vad = null; }
            if (audioInput) { audioInput.disconnect(); audioInput = null; }
            if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
            micIndicator.classList.remove('listening');
        }
        function displayQuizOptions(options) { options.forEach(opt => { const b = document.createElement('button'); b.className = 'lesson-button quiz-button'; b.innerHTML = opt; b.onclick = () => { interruptSpeech(); addMessage(opt, 'user'); if (ws && ws.readyState === WebSocket.OPEN) { ws.send(JSON.stringify({ type: 'quiz_answer', answer: opt.charAt(0) })); } clearQuizOptions(); }; quizOptionsDiv.appendChild(b); }); }
        function addMessage(text, sender) { const el = document.createElement('div'); el.className = `message ${sender}-message`; el.innerHTML = text.replace(/\\n/g, '<br>'); conversationDiv.appendChild(el); conversationDiv.scrollTop = conversationDiv.scrollHeight; }
        function clearQuizOptions() { quizOptionsDiv.innerHTML = ''; }
        function interruptSpeech() { if (currentAudioSource) { currentAudioSource.stop(); currentAudioSource.onended = null; } if (animationFrameId) cancelAnimationFrame(animationFrameId); isPlaying = false; currentAudioSource = null; animationFrameId = null; currentAudioChunks = []; visemeQueue = []; visemeMouth.src = visemeMap[0]; }
    </script>
</body>
</html>
"""

# --- REFACTORED: Connection Manager Class ---
class ConnectionManager:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.current_state = AppState.IDLE
        self.speech_rate = "+11.50%"
        self.dialogflow_session_path = ""
        self.audio_input_queue: Optional[asyncio.Queue] = None
        self.lesson_step = 0
        self.quiz_step = 0
        self.quiz_score = 0
        self.selected_lesson_id: Optional[str] = None
        self.current_lesson_flow: List[Dict] = []
        self.current_quiz_questions: List[Dict] = []

    async def transition_to_state(self, new_state: AppState):
        self.current_state = new_state
        logger.info(f"Transitioning to state: {self.current_state.value}")
        await self.websocket.send_json({"type": "state_update", "state": self.current_state.value})

    async def advance_lesson(self):
        if self.lesson_step < len(self.current_lesson_flow):
            step_data = self.current_lesson_flow[self.lesson_step]
            self.lesson_step += 1
            text = step_data["text"]
            if step_data["type"] == "lecture": await self.transition_to_state(AppState.LESSON_DELIVERY)
            elif step_data["type"] == "question": await self.transition_to_state(AppState.LESSON_QUESTION)
            elif step_data["type"] == "qna_prompt": await self.transition_to_state(AppState.LESSON_QNA)
            await self.send_ai_response(text)
        else:
            await self.send_ai_response("Okay, let's start the final quiz.", AppState.QUIZ_START)

    async def advance_quiz(self):
        if self.quiz_step < len(self.current_quiz_questions):
            question_data = self.current_quiz_questions[self.quiz_step]
            self.quiz_step += 1
            await self.transition_to_state(AppState.QUIZ_QUESTION)
            await self.websocket.send_json({"type": "quiz_question", "text": question_data['text'], "options": question_data['options']})
            tts_text = f"{question_data['text']}\n" + "\n".join(question_data['options'])
            await stream_azure_tts_and_send_to_client(tts_text, self.websocket, self.speech_rate)
        else:
            await self.transition_to_state(AppState.QUIZ_COMPLETE)
            summary = f"You've completed the quiz! You scored {self.quiz_score} out of {len(self.current_quiz_questions)}. Great job."
            await self.websocket.send_json({"type": "quiz_summary", "text": summary, "lesson_id": self.selected_lesson_id})
            await stream_azure_tts_and_send_to_client(summary, self.websocket, self.speech_rate)

    async def send_ai_response(self, text: str, next_state: Optional[AppState] = None):
        if next_state:
            await self.transition_to_state(next_state)
        await self.websocket.send_json({"type": "ai_response", "text": text})
        await stream_azure_tts_and_send_to_client(text, self.websocket, self.speech_rate)

    async def handle_user_transcript(self, transcript: str):
        await self.websocket.send_json({"type": "user_transcript", "text": transcript})
        if self.current_state == AppState.INTRODUCTION:
            user_name = extract_name(transcript)
            await self.send_ai_response(f"It's nice to meet you, {user_name}! Let's get started.", AppState.LESSON_PROLOGUE)
        elif self.current_state == AppState.LESSON_QUESTION:
            step_data = self.current_lesson_flow[self.lesson_step - 1]
            feedback = step_data['feedback_correct'] if step_data['correct_answer'].lower() in transcript.lower() else step_data['feedback_incorrect']
            await self.send_ai_response(feedback, AppState.LESSON_FEEDBACK)
        elif self.current_state in [AppState.LESSON_QNA, AppState.QNA]:
            # Handle Q&A logic
            await self.send_ai_response("Thanks for the question! Let's move on.", AppState.QUIZ_START) # Placeholder

    async def handle_text_message(self, data: dict):
        msg_type = data.get("type")
        if msg_type == "select_lesson":
            lesson_id = data.get("lesson_id")
            if lesson_id and lesson_id in lessons:
                self.selected_lesson_id = lesson_id
                self.current_lesson_flow = lessons[lesson_id]["flow"]
                self.current_quiz_questions = lessons[lesson_id]["quiz"]
                logger.info(f"Client selected Lesson {lesson_id}: {lessons[lesson_id]['title']}")
                await self.send_ai_response("Hello! I'm your HCV trainer. Before we begin, what's your name?", AppState.INTRODUCTION)
        elif msg_type == "control":
            command = data.get("command")
            if command == "start_speech":
                if self.audio_input_queue is None:
                    self.audio_input_queue = asyncio.Queue()
                    stt_results_queue = asyncio.Queue()
                    session_id = f"{self.websocket.client.host}-{self.websocket.client.port}-{time.time()}"
                    self.dialogflow_session_path = dialogflow_sessions_client.session_path(settings.GOOGLE_CLOUD_PROJECT_ID, session_id)
                    asyncio.create_task(transcribe_speech(self.audio_input_queue, stt_results_queue))
                    async def result_handler():
                        transcript = await stt_results_queue.get()
                        if transcript: await self.handle_user_transcript(transcript)
                    asyncio.create_task(result_handler())
            elif command == "end_speech":
                if self.audio_input_queue: await self.audio_input_queue.put(None); self.audio_input_queue = None
            elif command == "tts_finished":
                if self.current_state in [AppState.LESSON_PROLOGUE, AppState.LESSON_DELIVERY, AppState.LESSON_FEEDBACK]: await self.advance_lesson()
                elif self.current_state == AppState.QUIZ_START: await self.advance_quiz()
                elif self.current_state == AppState.QUIZ_FEEDBACK: await self.advance_quiz()
        elif msg_type == "quiz_answer":
            question_data = self.current_quiz_questions[self.quiz_step - 1]
            if data.get("answer") == question_data['correct_answer']:
                self.quiz_score += 1
                feedback = "Correct!"
            else:
                feedback = f"Not quite. The correct answer was {question_data['correct_answer']}. Because: {question_data.get('explanation', 'That is the correct rule.')}"
            await self.send_ai_response(feedback, AppState.QUIZ_FEEDBACK)

    async def run(self):
        await self.transition_to_state(AppState.IDLE)
        try:
            while self.websocket.client_state == WebSocketState.CONNECTED:
                message = await self.websocket.receive()
                if "bytes" in message:
                    if self.audio_input_queue: await self.audio_input_queue.put(message["bytes"])
                elif "text" in message:
                    await self.handle_text_message(json.loads(message["text"]))
        except WebSocketDisconnect:
            logger.info("Client disconnected.")
        except Exception as e:
            logger.error(f"Error in connection manager: {e}", exc_info=True)
        finally:
            logger.info("WebSocket connection closed.")

# --- Main Endpoints ---
@app.get("/")
async def get(): return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established.")
    manager = ConnectionManager(websocket)
    await manager.run()

# --- Helper Functions (No changes needed below this line) ---
def extract_name(transcript: str) -> str:
    if not transcript: return "there"
    match = re.search(r"(?:my name is|I'm|I am)\s+([A-Z]\w+)", transcript, re.IGNORECASE)
    if match: return match.group(1)
    words = transcript.strip().split()
    if words:
        potential_name = words[-1].strip(".,!?")
        if potential_name.lower() not in ["hello", "hi", "hey"]: return potential_name
    return "there"

async def transcribe_speech(audio_input_queue: asyncio.Queue, stt_results_queue: asyncio.Queue):
    # This function remains the same, as it doesn't depend on connection state.
    # ... (code for transcribe_speech)
    pass

async def stream_azure_tts_and_send_to_client(text: str, websocket: WebSocket, rate: str):
    # This function also remains the same.
    # ... (code for stream_azure_tts_and_send_to_client)
    pass