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
  const [capturedImage, setCapturedImage] = useState(null);

  // Initialize camera
  const startCamera = async () => {
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Your browser does not support webcam access');
      }
      
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: 640, height: 480 }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setCameraActive(true);
      }
    } catch (err) {
      console.error("Error accessing webcam:", err);
      
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
      }
    };

    checkBackend();
    
    // Cleanup on component unmount
    return () => {
      stopCamera();
    };
  }, []);

  // Start camera automatically when component mounts
  useEffect(() => {
    startCamera();
  }, []);

  // Capture image from webcam
  const captureImage = () => {
    if (!videoRef.current || !canvasRef.current) return;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    
    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw the current video frame to the canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Save captured image
    const imageUrl = canvas.toDataURL('image/jpeg');
    setCapturedImage(imageUrl);
    
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
      const response = await fetch('http://localhost:5000/predict', {
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

  // Reset the process
  const resetCapture = () => {
    setCapturedImage(null);
    setPredictions({ age: null, gender: null });
  };

  return (
    <div className="app-container">
      <div className="project-introduction">
        <div className="project-title">Machine Learning - Semester Project</div>
        <div className="authors">
          <ul>
            <li>Muhammad Aleem Shakeel</li>
            {/* <li>Sheza Naqvi</li>
            <li>Zainab</li> */}
          </ul>
        </div>
      </div>
      <div className="header">
        <h1>Age & Gender Predictor</h1>
        {!backendReady && <div className="backend-status">Backend disconnected</div>}
      </div>
      
      <div className="main-content">
        <div className="video-section">
          <div className="video-container">
            {!capturedImage ? (
              <video 
                ref={videoRef}
                autoPlay
                playsInline
                muted
              />
            ) : (
              <img 
                src={capturedImage} 
                alt="Captured" 
                className="captured-image"
              />
            )}
            
            <div className="video-overlay">
              {!capturedImage && cameraActive && (
                <svg className="frame-lines" viewBox="0 0 100 100" preserveAspectRatio="none">
                  <line x1="30" y1="0" x2="30" y2="10" />
                  <line x1="0" y1="30" x2="10" y2="30" />
                  
                  <line x1="70" y1="0" x2="70" y2="10" />
                  <line x1="90" y1="30" x2="100" y2="30" />
                  
                  <line x1="30" y1="90" x2="30" y2="100" />
                  <line x1="0" y1="70" x2="10" y2="70" />
                  
                  <line x1="70" y1="90" x2="70" y2="100" />
                  <line x1="90" y1="70" x2="100" y2="70" />
                </svg>
              )}
            </div>
            
            <canvas ref={canvasRef} style={{ display: 'none' }} />
          </div>
          
          <div className="camera-controls">
            {!capturedImage ? (
              <>
                <div className="camera-toggle-container">
                  <input 
                    type="checkbox"
                    id="camera-toggle"
                    className="camera-toggle-checkbox"
                    checked={cameraActive}
                    onChange={cameraActive ? stopCamera : startCamera}
                    disabled={isLoading}
                  />
                  <label 
                    htmlFor="camera-toggle" 
                    className="camera-toggle-label"
                    title={cameraActive ? "Turn camera off" : "Turn camera on"}
                  >
                    <div className="camera-toggle-inner">
                      <div className="camera-toggle-switch">
                        <svg className="camera-icon" viewBox="0 0 24 24">
                          <path d="M18 10.48V6c0-1.1-.9-2-2-2H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2v-4.48l4 3.98v-11l-4 3.98zm-2-.79V18H4V6h12v3.69z" />
                        </svg>
                        <svg className="camera-off-icon" viewBox="0 0 24 24">
                          <path d="M18 10.48V6c0-1.1-.9-2-2-2H8.83l2 2H16v3.17l1.83 1.83L18 10.48zm2.5-.48L15.17 4H12c-1.1 0-2 .9-2 2h-4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h12c.34 0 .65-.09.93-.24L20.5 10zM2.1 4.93L4 6.83V14c0 1.1.9 2 2 2h8.17l1.73 1.73c-.28.15-.59.24-.93.24H6c-1.1 0-2-.9-2-2V8.83L1.9 6.73 2.1 4.93z" />
                          <path d="M3.55 2.44L2.1 3.89l18 18 1.45-1.45z" />
                        </svg>
                      </div>
                    </div>
                    <span className="toggle-status">
                      {cameraActive ? "Camera On" : "Camera Off"}
                    </span>
                  </label>
                </div>
                
                <button 
                  className="primary-button"
                  onClick={captureImage} 
                  disabled={!cameraActive || isLoading || !backendReady}
                >
                  Capture Image
                </button>
              </>
            ) : (
              <button 
                className="secondary-button"
                onClick={resetCapture}
                disabled={isLoading}
              >
                Take New Photo
              </button>
            )}
          </div>
        </div>
        
        <div className="results-section">
          <div className="results-card">
            <h2>Analysis Results</h2>
            
            {isLoading ? (
              <div className="loading-container">
                <div className="loading-spinner"></div>
                <p>Analyzing image...</p>
              </div>
            ) : predictions.age !== null ? (
              <div className="prediction-results">
                <div className="result-item">
                  <div className="result-label">Age</div>
                  <div className="result-value">{predictions.age} years</div>
                </div>
                <div className="result-item">
                  <div className="result-label">Gender</div>
                  <div className="result-value">{predictions.gender}</div>
                </div>
              </div>
            ) : (
              <div className="empty-results">
                <p>Capture an image to see predictions</p>
              </div>
            )}
          </div>
          
          <div className="info-card">
            <h3>How it works</h3>
            <p>This app uses a deep learning model trained on the UTK Face dataset to predict age and gender from facial images.</p>
            <p>For best results:</p>
            <ul>
              <li>Ensure good lighting</li>
              <li>Face the camera directly</li>
              <li>Remove hats or sunglasses</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
