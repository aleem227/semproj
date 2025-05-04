import React, { useRef, useState, useEffect } from 'react';
// Re-enable TensorFlow.js import
import * as tf from '@tensorflow/tfjs';
import './App.css';

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  // Re-add model state
  const [model, setModel] = useState(null);
  const [predictions, setPredictions] = useState({ age: null, gender: null });
  const [isLoading, setIsLoading] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  // Remove backend ready state since we'll use client-side model
  const [modelLoading, setModelLoading] = useState(false);
  const [capturedImage, setCapturedImage] = useState(null);
  // Add latency state
  const [latency, setLatency] = useState(null);
  // Add selected model state
  const [selectedModel, setSelectedModel] = useState(null);
  // Add model load time state
  const [modelLoadTime, setModelLoadTime] = useState(null);
  // Add prediction start time state
  const [predictionStartTime, setPredictionStartTime] = useState(null);

  // Load model when component mounts or when selected model changes
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!selectedModel) return; // Don't load if no model selected

    let isMounted = true;
    let disposed = false;
    setModelLoading(true);

    const loadModel = async () => {
      const startLoadTime = performance.now();
      const modelPath = `/model/${selectedModel}/model.json`;
      let loadedModel = null;
      let error = null;

      try {
        loadedModel = await tf.loadLayersModel(modelPath);
      } catch (err) {
        error = err;
      }

      if (loadedModel) {
        if (model) {
          try {
            model.dispose();
          } catch (err) {
            console.error('Error disposing previous model:', err);
          }
        }
        if (isMounted && !disposed) {
          setModel(loadedModel);
          setModelLoadTime((performance.now() - startLoadTime).toFixed(2));
          setModelLoading(false);
        } else {
          loadedModel.dispose();
        }
      } else {
        setModelLoading(false);
        console.error(`Error loading ${selectedModel} model:`, error);
        alert(`Failed to load model: ${error.message}`);
      }
    };

    loadModel();

    return () => {
      isMounted = false;
      disposed = true;
    };
  }, [selectedModel]);

  // Handle model selection change
  const handleModelChange = (e) => {
    setSelectedModel(e.target.value);
    setPredictions({ age: null, gender: null });
    setLatency(null);
  };

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
        videoRef.current.play(); // Explicitly play the video
        setCameraActive(true); // Set active immediately
        
        // Additional check after metadata loads
        videoRef.current.onloadedmetadata = () => {
          if (videoRef.current.readyState >= 2) {
            setCameraActive(true);
          }
        };
      }
    } catch (err) {
      console.error("Error accessing webcam:", err);
      setCameraActive(false);
      alert(`Camera error: ${err.message}`);
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

  // Start camera automatically when component mounts
  useEffect(() => {
    startCamera();
    
    // Cleanup on component unmount
    return () => {
      stopCamera();
    };
  }, []);

  // Capture image from webcam
  const captureImage = () => {
    if (!videoRef.current || !canvasRef.current) return;
    if (!model) {
      alert("Model is not loaded yet. Please wait.");
      return;
    }
    
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

  // Process image and make prediction using TensorFlow.js
  const processImage = async (canvas) => {
    setIsLoading(true);

    // Start timing here
    const startTime = performance.now();

    try {
      if (!model) {
        throw new Error('Model is not loaded yet');
      }

      // Preprocess the image for the model
      const imageData = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height);
      
      // Convert to tensor and normalize
      let tensor = tf.browser.fromPixels(imageData)
        .resizeBilinear([224, 224]) // Resize to model input size
        .toFloat()
        .div(tf.scalar(255.0)); // Normalize to [0,1]

      // Apply mean/std normalization (per channel)
      const mean = tf.tensor1d([0.485, 0.456, 0.406]);
      const std = tf.tensor1d([0.229, 0.224, 0.225]);
      tensor = tensor.sub(mean).div(std);

      tensor = tensor.expandDims(0); // Add batch dimension
      
      // Make prediction
      const result = await model.predict(tensor);
      
      // Process the prediction results
      // Assuming the model outputs [age, gender]
      const predictions = await result[0].dataSync(); // Get age
      const gender = await result[1].dataSync(); // Get gender
      
      // Calculate latency
      const endTime = performance.now();
      setLatency((endTime - startTime).toFixed(2));
      
      // Clean up tensor to prevent memory leaks
      tensor.dispose();
      result.forEach(t => t.dispose());
      

      // setPredictions({
      //   age: Math.round(predictions[0]) - 9,
      //   gender: gender[0] > 0.5 ? 'Female' : 'Male'
      // });


      setPredictions({
        age: Math.round(predictions[0]) - (selectedModel === 'resnet34' ? 5 : 9),
        gender: gender[0] > 0.5 ? 'Female' : 'Male'
      });
    } catch (err) {
      console.error("Error processing image:", err);
      alert(`Error processing image: ${err.message}`);
      resetCapture();
    } finally {
      setIsLoading(false);
    }
  };

  // Reset the process
  const resetCapture = async () => {
    setCapturedImage(null);
    setPredictions({ age: null, gender: null });
    setLatency(null);
    await startCamera();
  };

  return (
    <>
      {modelLoading && (
        <div className="model-loading-overlay">
          <div className="model-loading-spinner"></div>
          <div className="model-loading-text">Loading model...</div>
        </div>
      )}
      <div className="app-container">
        <div className="project-introduction">
          <div className="project-title">Machine Learning - Semester Project</div>
          <div className="authors">
            <ul>
              <li>Muhammad Aleem Shakeel</li>
              <li>Sheza Naqvi</li>
              <li>Zainab Shahzad</li>
            </ul>
          </div>
        </div>
        <div className="header">
          <h1>Predicting Age & Gender</h1>
        </div>
        
        <div className="model-selection">
          <h3>Select Model</h3>
          <div className="model-options">
            <label className={`model-option ${selectedModel === 'resnet34' ? 'selected' : ''}`}>
              <input
                type="radio"
                name="model"
                value="resnet34"
                checked={selectedModel === 'resnet34'}
                onChange={handleModelChange}
                disabled={isLoading || modelLoading}
              />
              <div className="model-details">
                <h4>ResNet-34 (84MB)</h4>
                <p className="model-description">Lightweight model with good latency and accuracy tradeoff</p>
                <div className="model-specs">
                  <span className="spec-item">
                    <span role="img" aria-label="Fast">⚡</span> Fast inference
                  </span>
                  <span className="spec-item">
                    <span role="img" aria-label="Balanced">✓</span> Balanced accuracy
                  </span>
                  <span className="spec-item">
                    <span role="img" aria-label="Mobile">📱</span> Mobile-friendly
                  </span>
                </div>
              </div>
            </label>
            
            <label className={`model-option ${selectedModel === 'resnet152' ? 'selected' : ''}`}>
              <input
                type="radio"
                name="model"
                value="resnet152"
                checked={selectedModel === 'resnet152'}
                onChange={handleModelChange}
                disabled={isLoading || modelLoading}
              />
              <div className="model-details">
                <h4>ResNet-152 (228MB)</h4>
                <p className="model-description">State-of-the-art model with high accuracy</p>
                <div className="model-specs">
                  <span className="spec-item">
                    <span role="img" aria-label="Accurate">🎯</span> Superior accuracy
                  </span>
                  <span className="spec-item">
                    <span role="img" aria-label="Slower">⏱️</span> Higher latency
                  </span>
                  <span className="spec-item">
                    <span role="img" aria-label="Desktop">💻</span> Better for desktop
                  </span>
                </div>
              </div>
            </label>
          </div>
        </div>
        
        <div className="main-content">
          <div className="video-section">
            <div className={`video-container ${cameraActive ? 'camera-active' : ''}`}>
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
                    disabled={!cameraActive || isLoading || modelLoading || !model || !selectedModel}
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
                  <p>Analyzing image with {selectedModel}...</p>
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
                  <div className="performance-metrics">
                    <div className="result-item">
                      <div className="result-label">Model Load Time</div>
                      <div className="result-value">{modelLoadTime} ms</div>
                    </div>
                    {latency && (
                      <div className="result-item">
                        <div className="result-label">Processing Time</div>
                        <div className="result-value">{latency} ms</div>
                      </div>
                    )}
                  </div>
                  <div className="model-info">Using {selectedModel === 'resnet34' ? 'ResNet-34' : 'ResNet-152'} model</div>
                </div>
              ) : (
                <div className="empty-results">
                  <p>Please select a model to begin.</p>
                </div>
              )}
            </div>
            
            <div className="info-card">
              <h3>How it works</h3>
              <p>This app uses a TensorFlow.js model running directly in your browser to predict age and gender from facial images.</p>
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
    </>
  );
}

export default App;