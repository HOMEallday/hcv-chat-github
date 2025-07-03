// frontend/src/App.jsx
import React, { useState, useRef, useEffect } from 'react';
import './App.css'; // Basic styling

const App = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [aiResponse, setAiResponse] = useState('');
  const mediaRecorderRef = useRef(null);
  const audioContextRef = useRef(null);
  const websocketRef = useRef(null);

  // WebSocket URL - important! Use ws:// for HTTP, wss:// for HTTPS
  // For local development, use ws://localhost:8000/ws
  const WS_URL = 'ws://localhost:8000/ws';

  useEffect(() => {
    // Setup WebSocket connection
    websocketRef.current = new WebSocket(WS_URL);

    websocketRef.current.onopen = () => {
      console.log('WebSocket connection established.');
    };

    websocketRef.current.onmessage = (event) => {
      // Assuming backend sends text response first, then audio chunks
      // This is a simplified handler. In a real app, you'd parse JSON messages
      // to distinguish text from audio. For now, we'll assume text or audio.

      if (typeof event.data === 'string') {
        // This is likely the AI's text response
        setAiResponse(prev => prev + event.data); // Append for streaming text
      } else if (event.data instanceof Blob) {
        // This is an audio chunk
        const audioBlob = event.data;
        const url = URL.createObjectURL(audioBlob);
        const audio = new Audio(url);
        audio.play();
        audio.onended = () => URL.revokeObjectURL(url); // Clean up
      }
    };

    websocketRef.current.onclose = (event) => {
      console.log('WebSocket connection closed:', event.code, event.reason);
    };

    websocketRef.current.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    return () => {
      // Clean up WebSocket on component unmount
      if (websocketRef.current) {
        websocketRef.current.close();
      }
    };
  }, []);

  const startRecording = async () => {
    setTranscript(''); // Clear previous transcript
    setAiResponse(''); // Clear previous AI response
    setIsRecording(true);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      const scriptProcessor = audioContextRef.current.createScriptProcessor(4096, 1, 1); // Buffer size, input channels, output channels

      scriptProcessor.onaudioprocess = (event) => {
        const inputBuffer = event.inputBuffer.getChannelData(0);
        // Convert Float32Array to Int16Array for Google STT (LINEAR16 encoding)
        const int16Array = new Int16Array(inputBuffer.length);
        for (let i = 0; i < inputBuffer.length; i++) {
          int16Array[i] = Math.max(-1, Math.min(1, inputBuffer[i])) * 0x7FFF; // Scale to Int16 range
        }
        if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN) {
          websocketRef.current.send(int16Array.buffer);
        }
      };

      source.connect(scriptProcessor);
      scriptProcessor.connect(audioContextRef.current.destination); // Connect to speakers to hear yourself (optional)
      mediaRecorderRef.current = { stream, source, scriptProcessor }; // Store references to stop
    } catch (error) {
      console.error('Error accessing microphone:', error);
      setIsRecording(false);
    }
  };

  const stopRecording = () => {
    setIsRecording(false);
    if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN) {
      // Send a signal to the backend that recording has stopped
      websocketRef.current.send('END_OF_SPEECH');
    }

    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.scriptProcessor.disconnect();
      mediaRecorderRef.current.source.disconnect();
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      audioContextRef.current.close(); // Close audio context
      mediaRecorderRef.current = null;
    }
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h1>AI Learning Companion</h1>
      <button
        onMouseDown={startRecording}
        onMouseUp={stopRecording}
        onTouchStart={startRecording}
        onTouchEnd={stopRecording}
        disabled={isRecording}
        style={{
          padding: '15px 30px',
          fontSize: '1.2em',
          backgroundColor: isRecording ? '#e74c3c' : '#2ecc71',
          color: 'white',
          border: 'none',
          borderRadius: '5px',
          cursor: 'pointer',
          transition: 'background-color 0.3s ease',
        }}
      >
        {isRecording ? 'Listening...' : 'Push to Talk'}
      </button>

      {transcript && (
        <div style={{ marginTop: '20px', border: '1px solid #ccc', padding: '15px', borderRadius: '5px' }}>
          <strong>You:</strong> {transcript}
        </div>
      )}

      {aiResponse && (
        <div style={{ marginTop: '20px', border: '1px solid #2980b9', padding: '15px', borderRadius: '5px', backgroundColor: '#e8f6ff' }}>
          <strong>AI:</strong> {aiResponse}
        </div>
      )}
    </div>
  );
};

export default App;