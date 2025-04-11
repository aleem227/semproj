import React, { useRef, useState, useEffect } from 'react';
// Remove the TensorFlow.js import since we're not using it anymore
// import * as tf from '@tensorflow/tfjs';
import './App.css';

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  // Remove the model state since we're using the backend API
  // const [model, setModel] = useState(null);
  const [predictions, setPredictions] = useState({ age: null, gender: null });
  const [isLoading, setIsLoading] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [backendReady, setBackendReady] = useState(false);

  // Initialize camera with better error handling
  const startCamera = async () => {
    try {
      // First check if the browser supports getUserMedia
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Your browser does not support webcam access');
      }
      
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: 224, height: 224 }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setCameraActive(true);
      }
    } catch (err) {
      console.error("Error accessing webcam:", err);
      
      // Provide more specific error messages
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        alert("Camera access denied. Please allow camera access in your browser settings.");
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        alert("No camera found. Please connect a camera and try again.");
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        alert("Camera is already in use by another application. Please close other apps using the camera.");
      } else {
        alert(`Error accessing webcam: ${err.message}. Please check permissions.`);
      }
    }
  };

  // Stop camera
  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setCameraActive(false);
    }
  };

  // Check if backend is ready
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const response = await fetch('http://localhost:5000/health');
        if (response.ok) {
          setBackendReady(true);
          console.log("Backend is ready");
        } else {
          console.error("Backend health check failed");
        }
      } catch (err) {
        console.error("Error connecting to backend:", err);
        alert("Could not connect to the backend server. Please make sure it's running.");
      }
    };

    checkBackend();
    
    // Cleanup on component unmount
    return () => {
      stopCamera();
    };
  }, []);

  // Capture image from webcam
  const captureImage = () => {
    if (!videoRef.current || !canvasRef.current) return;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    
    // Set canvas dimensions to match video
    canvas.width = 224;
    canvas.height = 224;
    
    // Draw the current video frame to the canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Process the image and make prediction
    processImage(canvas);
  };

  // Process image and make prediction
  const processImage = async (canvas) => {
    setIsLoading(true);
    
    try {
      // Get image as base64 data URL
      const imageDataUrl = canvas.toDataURL('image/jpeg');
      
      // Send to backend API
      const response = await fetch('http://localhost:5000/predict/resnet34', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: imageDataUrl
        }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      const result = await response.json();
      
      setPredictions({
        age: Math.round(result.age),
        gender: result.gender > 0.5 ? 'Female' : 'Male'
      });
      
    } catch (err) {
      console.error("Error processing image:", err);
      alert("Error processing image. See console for details.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>UTK Face Predictor</h1>
      
      <div className="controls">
        {!cameraActive ? (
          <button onClick={startCamera} disabled={isLoading}>
            Start Camera
          </button>
        ) : (
          <button onClick={stopCamera} disabled={isLoading}>
            Stop Camera
          </button>
        )}
        
        <button 
          onClick={captureImage} 
          disabled={!cameraActive || isLoading || !backendReady}
        >
          Capture and Predict
        </button>
      </div>
      
      <div className="content">
        <div className="video-container">
          <video 
            ref={videoRef}
            autoPlay
            playsInline
            muted
            style={{ width: '224px', height: '224px' }}
          />
          <canvas 
            ref={canvasRef} 
            style={{ display: 'none' }}
          />
        </div>
        
        <div className="results">
          {!backendReady && (
            <div className="error">Backend server not connected</div>
          )}
          {isLoading ? (
            <div className="loading">Processing...</div>
          ) : (
            predictions.age !== null && (
              <div>
                <h2>Predictions:</h2>
                <p>Age: {predictions.age} years</p>
                <p>Gender: {predictions.gender}</p>
              </div>
            )
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
