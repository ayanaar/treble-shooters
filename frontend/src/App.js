import React, { useState, useEffect, useRef } from 'react';
import * as Tone from 'tone';
import './App.css';

// Mapping configuration for row to note
const rowToNote = ["A5", "G5", "E5", "D5", "C5", "A4", "G4", "E4", "D4", "C4", "A3", "G3", "E3", "D3", "C3"];
const colorToInstrument = { 1: "violin", 2: "piano", 3: "flute" };

const testMatrix = [
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 1, 2, 0, 2, 0, 2, 0, 2, 0, 2],
  [2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 1, 2, 0, 2, 0, 2, 0, 2, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
];


//Scale matrix to be 15 by 20 to allow for 3 octaves and 5 bars for 4/4
function scaleMatrix(originalMatrix, targetRows, targetCols) {
  if (!originalMatrix.length) return [];
  
  const originalRows = originalMatrix.length;
  const originalCols = originalMatrix[0].length;
  
  const scaledMatrix = Array(targetRows).fill().map(() => Array(targetCols).fill(0));
  
  const rowRatio = originalRows / targetRows;
  const colRatio = originalCols / targetCols;
  
  for (let y = 0; y < targetRows; y++) {
    for (let x = 0; x < targetCols; x++) {
      const origY = Math.floor(y * rowRatio);
      const origX = Math.floor(x * colRatio);
      
      if (origY < originalRows && origX < originalCols) {
        scaledMatrix[y][x] = originalMatrix[origY][origX];
      }
    }
  }
  
  return scaledMatrix;
}


function App() {
  const [matrix, setMatrix] = useState(scaleMatrix(testMatrix, 15, 20));
  const [currentColumn, setCurrentColumn] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [volume, setVolume] = useState(0.5);
  const [tempo, setTempo] = useState(110);
  const [isAudioReady, setIsAudioReady] = useState(false);
 
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  const previousColumnRef = useRef(null);
  const activeNotes = useRef({});
  const synths = useRef({});
  const volumeNode = useRef(null);
  const lastStartTime = useRef(0);
  // const matrixRef = useRef([]);

  // Initialize Tone.js instruments 
  // Uses polySynth right now but can change to use samplers later for more realistic sound
  useEffect(() => {
    volumeNode.current = new Tone.Volume().toDestination();
    volumeNode.current.volume.value = Tone.gainToDb(volume);

    synths.current = {
      violin: new Tone.PolySynth(Tone.Synth, {
        envelope: {
          attack: 0.1,
          decay: 0.1,
          sustain: 0.3,
          release: 0.5
        },
        volume: -8
      }).connect(volumeNode.current),
      
      piano: new Tone.PolySynth(Tone.Synth, {
        envelope: {
          attack: 0.01,
          decay: 0.1,
          sustain: 0.3,
          release: 0.4
        },
        volume: -10
      }).connect(volumeNode.current),
      
      flute: new Tone.PolySynth(Tone.FMSynth, {
        envelope: {
          attack: 0.1,
          decay: 0.1,
          sustain: 0.4,
          release: 0.5
        },
        modulationIndex: 5,
        volume: -9
      }).connect(volumeNode.current)
    };

    setIsAudioReady(true);

    return () => {
      Object.values(synths.current).forEach(synth => synth.dispose());
      if (volumeNode.current) volumeNode.current.dispose();
    };
  }, []);

  // Fetch matrix from backend
  // useEffect(() => {
  //   const fetchMatrix = async () => {
  //     try {
  //       const ping = await fetch('http://localhost:5000', { 
  //         method: 'HEAD',
  //         mode: 'cors'
  //       });
        
  //       if (!ping.ok) throw new Error("Backend not responding");
  
  //       // Fetch the matrix data from backend
  //       const response = await fetch('http://localhost:5000/api/matrix', {
  //         headers: {
  //           'Accept': 'application/json',
  //         },
  //       });
  
  //       if (!response.ok) {
  //         throw new Error(`HTTP ${response.status} - ${await response.text()}`);
  //       }
  
  //       const data = await response.json();
        
  //       if (!data?.matrix) {
  //         throw new Error("Invalid matrix data format");
  //       }
  
  //       const scaledMatrix = scaleMatrix(data.matrix, 15, 20);
  //       setMatrix(scaledMatrix);
  //       matrixRef.current = scaledMatrix;
  //       setError(null);
  //     } catch (err) {
  //       setError(`Connection failed: ${err.message}`);
  //       fetchMatrix();
  //     } finally {
  //       setIsLoading(false);
  //     }
  //   };
  
  //   fetchMatrix();
  //   const intervalId = setInterval(fetchMatrix, 100);
  //   return () => clearInterval(intervalId);
  // }, []);
  
  // Play notes in the current column given an index
  const playColumn = (columnIndex) => {
    const now = Tone.now();
    
    const startTime = Math.max(now, lastStartTime.current + 0.01);
    lastStartTime.current = startTime;

    //Check if notes are sustained (i.e. were played in previous row) and release if not continuing
    if (previousColumnRef.current !== null) {
      matrix.forEach((row, rowIndex) => {
        const prevColor = row[previousColumnRef.current];
        const currColor = row[columnIndex];
        
        if (prevColor > 0 && currColor === 0) { 
          const noteKey = `${rowIndex}-${prevColor}`;
          if (activeNotes.current[noteKey]) {
            synths.current[colorToInstrument[prevColor]]
              .triggerRelease(rowToNote[rowIndex], startTime);
            delete activeNotes.current[noteKey];
          }
        }
      });
    }
  
    //Play new notes for this column
    matrix.forEach((row, rowIndex) => {
      const colorCode = row[columnIndex];
      if (colorCode > 0) {
        const noteKey = `${rowIndex}-${colorCode}`;
        if (!activeNotes.current[noteKey]) {
          synths.current[colorToInstrument[colorCode]]
            .triggerAttack(rowToNote[rowIndex], startTime);
          activeNotes.current[noteKey] = true;
        }
      }
    });
  
    previousColumnRef.current = columnIndex;
  };


  // Update volume when/if it changes
  useEffect(() => {
    if (volumeNode.current) {
      volumeNode.current.volume.value = Tone.gainToDb(volume);
    }
  }, [volume]);

  // Start/stop the playback
  const togglePlayback = async () => {
    if (isPlaying) {
      setIsPlaying(false);
      clearTimeout(animationRef.current);
      
        Object.keys(synths.current).forEach(instrument => {
      synths.current[instrument].releaseAll();
    });
      
      activeNotes.current = {};
      previousColumnRef.current = null;
      lastStartTime.current = 0;
    } else {
      await Tone.start();
      setIsPlaying(true);
      setCurrentColumn(0);
      previousColumnRef.current = null;
      lastStartTime.current = Tone.now();
      playColumn(0);
      animatePlayback();
    }
  };

  // Control the playback timing
  const animatePlayback = () => {
    const interval = (60 / tempo) * 1000;
  
    animationRef.current = setTimeout(() => {
      setCurrentColumn(prevCol => {
        const nextColumn = (prevCol + 1) % matrix[0].length;
        playColumn(nextColumn);
        return nextColumn;
      });
      animatePlayback();
    }, interval);
  };

  
  // Draw the matrix visualization to see the colours on the grid
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !matrix.length) return;
    
    const ctx = canvas.getContext('2d');
    const cellSize = 20;
    const width = matrix[0].length * cellSize;
    const height = matrix.length * cellSize;
    
    canvas.width = width;
    canvas.height = height;
    
    // Draw the matrix and fill in the cells
    for (let y = 0; y < matrix.length; y++) {
      for (let x = 0; x < matrix[0].length; x++) {
        switch (matrix[y][x]) {
          case 1: ctx.fillStyle = 'red'; break;
          case 2: ctx.fillStyle = 'green'; break;
          case 3: ctx.fillStyle = 'blue'; break;
          default: ctx.fillStyle = 'black'; break;
        }
        
        ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
        
        // Draw the grid lines so there's more clarity
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.strokeRect(x * cellSize, y * cellSize, cellSize, cellSize);
      }
    }

    // Have a white line/block to highlight current column to show that it is the one currently being playeds
    if (currentColumn !== null) {
      ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
      ctx.fillRect(
        currentColumn * cellSize, 
        0, 
        cellSize, 
        matrix.length * cellSize
      );
    }
  }, [matrix, currentColumn]);


  return (
    <div className="App">
      <h1>Melody Wall</h1>
      
      <div className="controls">
        <button onClick={togglePlayback} disabled={!isAudioReady}>
          {isPlaying ? 'Stop' : 'Play'}
        </button>
        
        <div className="slider-control">
          <label>Volume:</label>
          <input 
            type="range" 
            min="0" 
            max="1" 
            step="0.01" 
            value={volume} 
            onChange={(e) => setVolume(parseFloat(e.target.value))} 
          />
        </div>
        
        <div className="slider-control">
          <label>Tempo (BPM):</label>
          <input 
            type="range" 
            min="40" 
            max="200" 
            step="1" 
            value={tempo} 
            onChange={(e) => setTempo(parseInt(e.target.value))} 
          />
          <span>{tempo}</span>
        </div>
      </div>
      
      <div className="visualization">
        <canvas ref={canvasRef}></canvas>
      </div>
      
      <div className="legend">
        <h3>Legend!!!</h3>
        <p><span className="color-box red"></span> Red = Violin</p>
        <p><span className="color-box green"></span> Green = Piano</p>
        <p><span className="color-box blue"></span> Blue = Flute</p>
      </div>
    </div>
  );
}

export default App;