// src/App.jsx (FINAL TWO-CONTEXT SOLUTION)
import { useState, useEffect, useRef, useCallback } from 'react';
import './App.css';

// --- Viseme Image Map & Child Components (Unchanged) ---
const visemeMap = { 0: '/visemes/viseme_0.png', 1: '/visemes/viseme_1.png', 2: '/visemes/viseme_2.png', 3: '/visemes/viseme_3.png', 4: '/visemes/viseme_4.png', 5: '/visemes/viseme_5.png', 6: '/visemes/viseme_6.png', 7: '/visemes/viseme_7.png', 8: '/visemes/viseme_8.png', 9: '/visemes/viseme_9.png', 10: '/visemes/viseme_10.png', 11: '/visemes/viseme_11.png', 12: '/visemes/viseme_12.png', 13: '/visemes/viseme_13.png', 14: '/visemes/viseme_14.png', 15: '/visemes/viseme_15.png', 16: '/visemes/viseme_16.png', 17: '/visemes/viseme_17.png', 18: '/visemes/viseme_18.png', 19: '/visemes/viseme_19.png', 20: '/visemes/viseme_20.png', 21: '/visemes/viseme_21.png' };
const StartupOverlay = ({ onStart }) => ( <div id="startup-overlay"> <div id="lesson-menu"> <h2>HCV Training Program</h2> <button id="lesson-1-btn" className="lesson-button" onClick={onStart}> Lesson 1: Program Fundamentals </button> </div> </div> );
const Character = ({ visemeSrc }) => ( <div id="character-container"> <img id="character-still" src="/talkinghouse.jpg" alt="AI Character Base" /> <img id="viseme-mouth" src={visemeSrc || visemeMap[0]} alt="AI Character Mouth" /> </div> );
const Conversation = ({ messages }) => { const endRef = useRef(null); useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]); return ( <div id="conversation"> {messages.map((msg, index) => ( <div key={index} className={`message ${msg.sender}-message`}> <span dangerouslySetInnerHTML={{ __html: msg.text.replace(/\\n/g, '<br>') }}></span> </div> ))} <div ref={endRef} /> </div> ); };
const QuizOptions = ({ options, onSelect }) => ( <div id="quiz-options"> {options.map((option, index) => ( <button key={index} className="lesson-button quiz-button" onClick={() => onSelect(option)}> {option} </button> ))} </div> );
const StatusBar = ({ status, isListening, speed, onSpeedChange }) => ( <div id="status-bar"> <div> Current State: <span id="status">{status}</span> Mic: <span id="mic-indicator" className={isListening ? 'listening' : ''}></span> </div> <div id="speed-controls"> <span>Speed:</span> <button className={`speed-button ${speed === '+0.00%' ? 'active' : ''}`} onClick={() => onSpeedChange('+0.00%')}>Slow</button> <button className={`speed-button ${speed === '+11.50%' ? 'active' : ''}`} onClick={() => onSpeedChange('+11.50%')}>Normal</button> <button className={`speed-button ${speed === '+50.00%' ? 'active' : ''}`} onClick={() => onSpeedChange('+50.00%')}>Fast</button> </div> </div> );

class VoiceActivityDetector { // Unchanged
    constructor(onSpeechStart, onSpeechEnd, silenceThreshold = 1.0) { this.onSpeechStart = onSpeechStart; this.onSpeechEnd = onSpeechEnd; this.silenceThreshold = silenceThreshold; this.analyser = null; this.isSpeaking = false; this.silenceStartTime = 0; this.animationFrameId = null; }
    start(analyserNode) { this.analyser = analyserNode; this.isSpeaking = false; this.monitor(); }
    stop() { if (this.animationFrameId) { cancelAnimationFrame(this.animationFrameId); this.animationFrameId = null; } if (this.isSpeaking) { this.onSpeechEnd(); } this.isSpeaking = false; }
    monitor = () => { if (!this.analyser) return; const dataArray = new Uint8Array(this.analyser.frequencyBinCount); this.analyser.getByteFrequencyData(dataArray); const averageVolume = dataArray.reduce((sum, val) => sum + val, 0) / dataArray.length;
        if (averageVolume > 5) { this.silenceStartTime = 0; if (!this.isSpeaking) { this.isSpeaking = true; this.onSpeechStart(); }
        } else if (this.isSpeaking) { if (this.silenceStartTime === 0) { this.silenceStartTime = Date.now(); } else if ((Date.now() - this.silenceStartTime) > this.silenceThreshold * 1000) { this.isSpeaking = false; this.onSpeechEnd(); } }
        this.animationFrameId = requestAnimationFrame(this.monitor);
    }
}

function App() {
    const [isStarted, setIsStarted] = useState(false);
    const [status, setStatus] = useState('Waiting to Start...');
    const [isListening, setIsListening] = useState(false);
    const [messages, setMessages] = useState([]);
    const [quizOptions, setQuizOptions] = useState([]);
    const [visemeSrc, setVisemeSrc] = useState(null);
    const [speed, setSpeed] = useState('+11.50%');

    const ws = useRef(null);
    const playbackContext = useRef(null); // The permanent "sound card" for bot playback
    const micContext = useRef(null); // The temporary "sound card" for the mic
    const vad = useRef(null);
    const mediaStream = useRef(null);
    const isPlaying = useRef(false);
    const currentAudioChunks = useRef([]);
    const visemeQueue = useRef([]);
    const animationFrameId = useRef(null);
    const currentAudioSource = useRef(null);
    const isListeningRef = useRef(false);
    const statusRef = useRef(status);

    useEffect(() => { statusRef.current = status; }, [status]);

    const stopContinuousListening = useCallback(() => {
        if (vad.current) { vad.current.stop(); vad.current = null; }
        if (mediaStream.current) { mediaStream.current.getTracks().forEach(track => track.stop()); mediaStream.current = null; }
        if (micContext.current && micContext.current.state !== 'closed') {
            micContext.current.close();
            micContext.current = null;
        }
        setIsListening(false);
        isListeningRef.current = false;
    }, []);

    const startContinuousListening = useCallback(async () => {
        if (isPlaying.current || isListeningRef.current) return;
        stopContinuousListening();
        try {
            // Create a brand new, temporary context just for the microphone
            micContext.current = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            await micContext.current.resume();
            
            const workletCode = `class AudioProcessor extends AudioWorkletProcessor { process(inputs) { const pcmData = inputs[0][0]; if (pcmData) this.port.postMessage(new Int16Array(pcmData.map(val => Math.max(-1, Math.min(1, val)) * 0x7FFF))); return true; } } registerProcessor('audio-processor', AudioProcessor);`;
            const blob = new Blob([workletCode], { type: 'application/javascript' });
            await micContext.current.audioWorklet.addModule(URL.createObjectURL(blob));
            const workletNode = new AudioWorkletNode(micContext.current, 'audio-processor');

            mediaStream.current = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000, echoCancellation: false, noiseSuppression: true } });
            const audioInput = micContext.current.createMediaStreamSource(mediaStream.current);
            const splitter = micContext.current.createGain();
            const analyserForVAD = micContext.current.createAnalyser();
            analyserForVAD.fftSize = 512;
            audioInput.connect(splitter);
            splitter.connect(analyserForVAD);
            splitter.connect(workletNode);
            
            workletNode.port.onmessage = (event) => { if (ws.current?.readyState === WebSocket.OPEN) { ws.current.send(event.data.buffer); } };
            const onSpeechStart = () => { setIsListening(true); isListeningRef.current = true; if (ws.current?.readyState === WebSocket.OPEN) { ws.current.send(JSON.stringify({ type: 'control', command: 'start_speech' })); } };
            const onSpeechEnd = () => { if (isListeningRef.current) { if (ws.current?.readyState === WebSocket.OPEN) { ws.current.send(JSON.stringify({ type: 'control', command: 'end_speech' })); } stopContinuousListening(); } };
            vad.current = new VoiceActivityDetector(onSpeechStart, onSpeechEnd);
            vad.current.start(analyserForVAD);
        } catch (err) { console.error("Mic start FAILED:", err); }
    }, [stopContinuousListening]);

    const animateVisemes = useCallback((audioStartTime) => {
        if (!playbackContext.current) return;
        const elapsedTime = (playbackContext.current.currentTime - audioStartTime) * 1000;
        let latestViseme = null;
        while (visemeQueue.current.length > 0 && visemeQueue.current[0].offset_ms <= elapsedTime) { latestViseme = visemeQueue.current.shift(); }
        if (latestViseme) { setVisemeSrc(visemeMap[latestViseme.viseme_id]); }
        if (isPlaying.current) { animationFrameId.current = requestAnimationFrame(() => animateVisemes(audioStartTime)); }
    }, []);

    const playCombinedAudio = useCallback(async () => {
        if (currentAudioChunks.current.length === 0 || !playbackContext.current) return;
        isPlaying.current = true;
        stopContinuousListening();
        const audioBlob = new Blob(currentAudioChunks.current, { type: 'audio/mp3' });
        currentAudioChunks.current = [];
        try {
            const arrayBuffer = await audioBlob.arrayBuffer();
            const audioBuffer = await playbackContext.current.decodeAudioData(arrayBuffer);
            if (currentAudioSource.current) { currentAudioSource.current.stop(); }
            if (animationFrameId.current) { cancelAnimationFrame(animationFrameId.current); }
            currentAudioSource.current = playbackContext.current.createBufferSource();
            currentAudioSource.current.buffer = audioBuffer;
            currentAudioSource.current.connect(playbackContext.current.destination);
            currentAudioSource.current.onended = () => {
                isPlaying.current = false;
                setVisemeSrc(visemeMap[0]);
                visemeQueue.current = [];
                cancelAnimationFrame(animationFrameId.current);
                if (ws.current?.readyState === WebSocket.OPEN) { ws.current.send(JSON.stringify({ type: 'control', command: 'tts_finished' })); }
                const isConversational = ['INTRODUCTION', 'LESSON_QUESTION', 'LESSON_QNA', 'QNA'].includes(statusRef.current);
                if (isConversational) { startContinuousListening(); }
            };
            const audioStartTime = playbackContext.current.currentTime;
            currentAudioSource.current.start(0);
            animateVisemes(audioStartTime);
        } catch (e) { console.error("Error playing audio:", e); isPlaying.current = false; }
    }, [startContinuousListening, stopContinuousListening, animateVisemes]);

    const connect = useCallback(() => {
        ws.current = new WebSocket('ws://localhost:8000/ws');
        ws.current.onopen = () => { console.log("WebSocket connection OPENED successfully!"); };
        ws.current.onclose = () => { console.log("WebSocket connection CLOSED."); setStatus("Disconnected"); stopContinuousListening(); };
        ws.current.onerror = (error) => { console.error("WebSocket ERRORED!", error); };
        ws.current.onmessage = (event) => {
            if (event.data instanceof Blob) { currentAudioChunks.current.push(event.data); }
            else {
                const message = JSON.parse(event.data);
                switch (message.type) {
                    case 'state_update': setStatus(message.state); setQuizOptions([]); break;
                    case 'ai_response': setMessages(prev => [...prev, { sender: 'ai', text: message.text }]); break;
                    case 'user_transcript': setMessages(prev => [...prev, { sender: 'user', text: message.text }]); break;
                    case 'quiz_question': setMessages(prev => [...prev, { sender: 'ai', text: message.text }]); setQuizOptions(message.options); break;
                    case 'quiz_summary': setMessages(prev => [...prev, { sender: 'ai', text: message.text }]); break;
                    case 'viseme': visemeQueue.current.push(message); break;
                    case 'tts_stream_finished': playCombinedAudio(); break;
                    default: console.log("Unknown message type:", message.type);
                }
            }
        };
    }, [stopContinuousListening, playCombinedAudio]);

    const initializeApp = useCallback(async () => {
        setIsStarted(true);
        if (!playbackContext.current) {
            try {
                // Only create the playback context here. The mic context is created on demand.
                playbackContext.current = new (window.AudioContext || window.webkitAudioContext)();
                await playbackContext.current.resume();
            } catch (e) { console.error("Error initializing audio context:", e); return; }
        }
        connect();
    }, [connect]);
    
    // Remaining handlers are fine
    const handleQuizSelection = (option) => { const selectedOption = option.charAt(0); setMessages(prev => [...prev, { sender: 'user', text: option }]); if (ws.current?.readyState === WebSocket.OPEN) { ws.current.send(JSON.stringify({ type: 'quiz_answer', answer: selectedOption })); } setQuizOptions([]); };
    const handleSpeedChange = (newRate) => { setSpeed(newRate); if (ws.current?.readyState === WebSocket.OPEN) { ws.current.send(JSON.stringify({ type: 'set_speed', rate: newRate })); } };
    useEffect(() => { Object.values(visemeMap).forEach(path => { const img = new Image(); img.src = path; }); }, []);
    if (!isStarted) return <StartupOverlay onStart={initializeApp} />;
    return ( <div className="App"> <div id="main-content"> <Character visemeSrc={visemeSrc} /> <div id="chat-container"> <Conversation messages={messages} /> <QuizOptions options={quizOptions} onSelect={handleQuizSelection} /> </div> </div> <StatusBar status={status} isListening={isListening} speed={speed} onSpeedChange={handleSpeedChange} /> </div> );
}

export default App;