import React, { useState, useEffect } from 'react';
import { 
  UploadCloud, FileText, Database, AlertCircle, BarChart2, 
  Table as TableIcon, Sparkles, Settings, PlayCircle, Activity,
  LayoutDashboard, BrainCircuit, Grid, LineChart as LineChartIcon, Download, ChevronRight, Eye, Gamepad2, ImageIcon, Binary,
  Rocket, Zap, ShieldCheck
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';

// --- Mini-Game Component for the Loading Screen ---
const TicTacToe = () => {
  const [board, setBoard] = useState(Array(9).fill(null));
  const [xIsNext, setXIsNext] = useState(true);

  const calculateWinner = (squares) => {
    const lines = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]];
    for (let i = 0; i < lines.length; i++) {
      const [a, b, c] = lines[i];
      if (squares[a] && squares[a] === squares[b] && squares[a] === squares[c]) return squares[a];
    }
    return null;
  };

  const winner = calculateWinner(board);
  const isDraw = !winner && board.every(square => square !== null);

  const handleClick = (i) => {
    if (board[i] || winner) return;
    const newBoard = [...board];
    newBoard[i] = xIsNext ? 'X' : 'O';
    setBoard(newBoard);
    setXIsNext(!xIsNext);
  };

  return (
    <div className="flex flex-col items-center justify-center w-full">
      <div className="mb-3 text-sm font-bold text-slate-600">
        {winner ? `Winner: ${winner} 🎉` : isDraw ? "It's a Draw! 🤝" : `Next player: ${xIsNext ? 'X' : 'O'}`}
      </div>
      <div className="grid grid-cols-3 gap-2 mb-4 bg-slate-200 p-2 rounded-xl">
        {board.map((square, i) => (
          <button
            key={i}
            onClick={() => handleClick(i)}
            className={`w-14 h-14 sm:w-16 sm:h-16 flex items-center justify-center text-2xl font-bold rounded-lg shadow-sm transition-all bg-white hover:bg-slate-50 ${square === 'X' ? 'text-teal-500' : 'text-rose-500'}`}
          >
            {square}
          </button>
        ))}
      </div>
      <button onClick={() => { setBoard(Array(9).fill(null)); setXIsNext(true); }} className="px-4 py-2 bg-slate-800 text-white text-xs font-bold rounded-lg hover:bg-slate-700 transition-colors">
        Restart Game
      </button>
    </div>
  );
};

export default function App() {
  // --- NEW: Landing Page State ---
  const [isStarted, setIsStarted] = useState(false);

  // Application State
  const [modality, setModality] = useState(null); // 'tabular' or 'vision'
  const [file, setFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [edaData, setEdaData] = useState(null);
  const [error, setError] = useState(null);
  
  // Navigation State
  const [activeTab, setActiveTab] = useState('eda');

  // Model Training State
  const [targetColumn, setTargetColumn] = useState("");
  const [taskType, setTaskType] = useState("");
  const [aiReasoning, setAiReasoning] = useState("");
  const [isTraining, setIsTraining] = useState(false);
  const [trainResults, setTrainResults] = useState(null);

  useEffect(() => {
    if (targetColumn && edaData && edaData.column_types) {
      const colType = edaData.column_types[targetColumn];
      if (colType === 'categorical') {
        setTaskType('classification');
        setAiReasoning(`AI Recommendation: Classification is recommended because '${targetColumn}' contains categorical/text data.`);
      } else if (colType === 'numeric') {
        setTaskType('regression');
        setAiReasoning(`AI Recommendation: Regression is recommended because '${targetColumn}' contains continuous numeric data.`);
      }
    } else {
      setTaskType("");
      setAiReasoning("");
    }
  }, [targetColumn, edaData]);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setIsUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const endpoint = modality === 'tabular' 
        ? 'http://127.0.0.1:8000/api/upload-dataset/' 
        : 'http://127.0.0.1:8000/api/upload-vision-dataset/';

      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Upload failed');
      }

      const data = await response.json();
      setEdaData(data);
      setTargetColumn("");
      setTaskType("");
      setTrainResults(null);
      setActiveTab('eda'); 
    } catch (err) {
      setError(err.message);
    } finally {
      setIsUploading(false);
    }
  };

  const handleTrain = async () => {
    if (modality === 'tabular' && (!file || !targetColumn || !taskType)) return;
    if (modality === 'vision' && !file) return;
    
    setIsTraining(true);
    setError(null);
    setTrainResults(null);

    const formData = new FormData();
    formData.append('file', file);
    
    if (modality === 'tabular') {
      formData.append('target_column', targetColumn);
      formData.append('task_type', taskType);
    }

    try {
      const endpoint = modality === 'tabular' 
        ? 'http://127.0.0.1:8000/api/train-model/' 
        : 'http://127.0.0.1:8000/api/train-vision-model/';

      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Training failed');
      }

      const data = await response.json();
      setTrainResults(data);
      setActiveTab('training'); 
    } catch (err) {
      setError(err.message);
    } finally {
      setIsTraining(false);
    }
  };

  const navTabs = [
    { id: 'eda', label: 'Data Exploration', icon: LayoutDashboard, disabled: false },
    { id: 'training', label: 'Model Training', icon: BrainCircuit, disabled: false },
    { id: 'explainability', label: 'Feature Importance', icon: Eye, disabled: !trainResults || modality === 'vision' }, // Hidden for vision currently
    { id: 'matrices', label: 'Confusion Matrices', icon: Grid, disabled: !trainResults || trainResults.task_type !== 'classification' },
    { id: 'graphs', label: 'Performance Graphs', icon: LineChartIcon, disabled: !trainResults },
    { id: 'download', label: 'Download Models', icon: Download, disabled: !trainResults },
  ];

  // ==========================================
  // VIEW 0: SAAS LANDING PAGE
  // ==========================================
  if (!isStarted) {
    return (
      <div className="min-h-screen bg-slate-50 font-sans selection:bg-teal-200">
        {/* Navigation Bar */}
        <nav className="flex items-center justify-between px-8 py-6 max-w-7xl mx-auto">
          <div className="flex items-center gap-2 text-teal-600 font-extrabold text-2xl tracking-tight">
            <BrainCircuit size={32} />
            <span>Easy<span className="text-slate-800">ML</span></span>
          </div>
          <button 
            onClick={() => setIsStarted(true)}
            className="hidden md:block px-6 py-2.5 bg-slate-900 text-white rounded-full font-semibold hover:bg-slate-800 transition-colors shadow-sm"
          >
            Go to App
          </button>
        </nav>

        {/* Hero Section */}
        <main className="max-w-7xl mx-auto px-8 pt-16 pb-24 flex flex-col items-center text-center animate-in fade-in slide-in-from-bottom-8 duration-700">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-teal-50 border border-teal-100 text-teal-700 text-sm font-bold mb-8 shadow-sm">
            <Sparkles size={16} className="text-teal-500" />
            <span>Now supporting Computer Vision & Deep Learning</span>
          </div>
          
          <h1 className="text-5xl md:text-7xl font-extrabold text-slate-900 tracking-tight leading-tight max-w-4xl mb-6">
            Train Enterprise AI Models <br className="hidden md:block"/>
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-teal-500 to-indigo-600">Without Writing Code.</span>
          </h1>
          
          <p className="text-lg md:text-xl text-slate-500 max-w-2xl mb-10 leading-relaxed">
            Upload your Tabular Data or Images. Our autonomous pipeline handles the EDA, preprocessing, algorithm selection, and hyperparameter tuning in seconds.
          </p>
          
          <button 
            onClick={() => setIsStarted(true)}
            className="group relative inline-flex items-center justify-center px-8 py-4 font-bold text-white bg-slate-900 rounded-full overflow-hidden transition-all hover:scale-105 shadow-xl hover:shadow-2xl"
          >
            <span className="absolute inset-0 w-full h-full -mt-1 rounded-lg opacity-30 bg-gradient-to-b from-transparent via-transparent to-black"></span>
            <span className="relative flex items-center gap-2 text-lg">
              Launch Workspace <ChevronRight size={20} className="group-hover:translate-x-1 transition-transform" />
            </span>
          </button>
        </main>

        {/* Feature Grid */}
        <section className="bg-white border-t border-slate-200 py-24">
          <div className="max-w-7xl mx-auto px-8 grid grid-cols-1 md:grid-cols-3 gap-12">
            <div className="flex flex-col items-center text-center space-y-4">
              <div className="w-16 h-16 bg-blue-50 text-blue-600 rounded-2xl flex items-center justify-center mb-2 shadow-sm">
                <LayoutDashboard size={32} />
              </div>
              <h3 className="text-xl font-bold text-slate-800">Automated EDA</h3>
              <p className="text-slate-500 leading-relaxed">Instantly visualize missing values, data distributions, and correlations the moment you upload your dataset.</p>
            </div>
            <div className="flex flex-col items-center text-center space-y-4">
              <div className="w-16 h-16 bg-teal-50 text-teal-600 rounded-2xl flex items-center justify-center mb-2 shadow-sm">
                <Zap size={32} />
              </div>
              <h3 className="text-xl font-bold text-slate-800">Multi-Algorithm Racing</h3>
              <p className="text-slate-500 leading-relaxed">Simultaneously train Random Forests, Logistic Regression, and MobileNetV2 CNNs to find the perfect fit.</p>
            </div>
            <div className="flex flex-col items-center text-center space-y-4">
              <div className="w-16 h-16 bg-purple-50 text-purple-600 rounded-2xl flex items-center justify-center mb-2 shadow-sm">
                <ShieldCheck size={32} />
              </div>
              <h3 className="text-xl font-bold text-slate-800">Explainable AI (XAI)</h3>
              <p className="text-slate-500 leading-relaxed">Break open the black box. Generate feature importance graphs and confusion matrices to audit your models.</p>
            </div>
          </div>
        </section>
      </div>
    );
  }

  // ==========================================
  // VIEW 1: UPLOAD SCREEN (After clicking Get Started)
  // ==========================================
  if (!edaData) {
    return (
      <div className="min-h-screen bg-gray-50 text-gray-800 font-sans p-8 flex flex-col items-center justify-center">
        <header className="mb-10 text-center">
          <h1 className="text-4xl font-extrabold text-slate-900 mb-2 flex justify-center items-center gap-3">
            <BrainCircuit size={36} className="text-teal-600" />
            EasyML
          </h1>
          <p className="text-gray-500 text-lg">Select your project type and upload your dataset to begin.</p>
        </header>

        <div className="bg-white p-10 rounded-2xl shadow-sm border border-gray-200 w-full max-w-3xl text-center">
          
          {/* STEP 1: Choose Modality */}
          {!modality ? (
            <div className="animate-in fade-in zoom-in-95 duration-300">
              <h3 className="text-2xl font-bold text-slate-800 mb-6">What are you building today?</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                
                <button 
                  onClick={() => setModality('tabular')}
                  className="flex flex-col items-center justify-center p-8 border-2 border-slate-200 rounded-2xl hover:border-teal-500 hover:bg-teal-50 transition-all group"
                >
                  <Binary size={48} className="text-slate-400 group-hover:text-teal-500 mb-4 transition-colors" />
                  <h4 className="text-xl font-bold text-slate-700 group-hover:text-teal-700 mb-2">Tabular Data</h4>
                  <p className="text-sm text-slate-500">Predict values or categories using CSV/Excel files.</p>
                </button>

                <button 
                  onClick={() => setModality('vision')}
                  className="flex flex-col items-center justify-center p-8 border-2 border-slate-200 rounded-2xl hover:border-rose-500 hover:bg-rose-50 transition-all group"
                >
                  <ImageIcon size={48} className="text-slate-400 group-hover:text-rose-500 mb-4 transition-colors" />
                  <h4 className="text-xl font-bold text-slate-700 group-hover:text-rose-700 mb-2">Computer Vision</h4>
                  <p className="text-sm text-slate-500">Classify images by uploading a ZIP file of folders.</p>
                </button>

              </div>
            </div>
          ) : (
            
            /* STEP 2: Upload File */
            <div className="animate-in slide-in-from-right-8 duration-300 relative">
              <button 
                onClick={() => { setModality(null); setFile(null); setError(null); }}
                className="absolute -top-4 -left-4 text-slate-400 hover:text-slate-600 text-sm font-bold flex items-center"
              >
                ← Back
              </button>

              <div className={`border-2 border-dashed rounded-xl p-12 mb-6 flex flex-col items-center justify-center transition-colors ${modality === 'tabular' ? 'border-teal-200 bg-teal-50/50 hover:bg-teal-50' : 'border-rose-200 bg-rose-50/50 hover:bg-rose-50'}`}>
                <UploadCloud size={64} className={`mb-4 ${modality === 'tabular' ? 'text-teal-400' : 'text-rose-400'}`} />
                <h3 className="text-xl font-semibold text-gray-700 mb-2">
                  Upload your {modality === 'tabular' ? 'CSV' : 'ZIP'} Dataset
                </h3>
                <p className="text-gray-500 mb-6">
                  {modality === 'tabular' ? 'Must be a valid .csv file' : 'ZIP file containing folders of images (e.g., cats/, dogs/)'}
                </p>
                
                <input 
                  type="file" 
                  accept={modality === 'tabular' ? ".csv" : ".zip"} 
                  onChange={handleFileChange} 
                  className="hidden" 
                  id="file-upload" 
                />
                <label 
                  htmlFor="file-upload" 
                  className={`cursor-pointer text-white px-6 py-2.5 rounded-lg font-medium transition-colors shadow-sm ${modality === 'tabular' ? 'bg-teal-600 hover:bg-teal-700' : 'bg-rose-600 hover:bg-rose-700'}`}
                >
                  Select Dataset
                </label>
              </div>
              
              {file && (
                <div className="flex items-center justify-between bg-gray-50 p-4 rounded-lg border border-gray-200 mb-6">
                  <div className="flex items-center gap-3 overflow-hidden">
                    <FileText className="text-gray-400 shrink-0" />
                    <span className="font-medium text-gray-700 truncate">{file.name}</span>
                  </div>
                  <span className="text-sm text-gray-500 bg-gray-200 px-2 py-1 rounded">{(file.size / 1024).toFixed(1)} KB</span>
                </div>
              )}

              {error && (
                <div className="bg-red-50 text-red-600 p-4 rounded-lg flex items-center gap-2 mb-6">
                  <AlertCircle size={20} />
                  <p>{error}</p>
                </div>
              )}

              <button 
                onClick={handleUpload} 
                disabled={!file || isUploading} 
                className={`w-full py-3 rounded-xl font-bold text-lg shadow-sm transition-all ${
                  !file || isUploading 
                    ? 'bg-gray-200 text-gray-400 cursor-not-allowed' 
                    : modality === 'tabular' ? 'bg-teal-600 text-white hover:bg-teal-700' : 'bg-rose-600 text-white hover:bg-rose-700'
                }`}
              >
                {isUploading ? 'Processing Dataset...' : 'Enter Workspace'}
              </button>
            </div>
          )}
        </div>
      </div>
    );
  }

  // ==========================================
  // VIEW 2: WORKSPACE (After data is loaded)
  // ==========================================
  return (
    <div className="h-screen flex bg-gray-50 text-gray-800 font-sans overflow-hidden">
      
      {/* SIDEBAR NAVIGATION */}
      <div className="w-72 bg-slate-900 text-slate-300 flex flex-col shrink-0">
        <div className="p-6 flex items-center gap-3 text-white border-b border-slate-800">
          <BrainCircuit className="text-teal-400" size={28} />
          <h2 className="text-xl font-bold tracking-wide">EasyML</h2>
        </div>
        
        <div className="p-4 flex flex-col gap-2 flex-1 overflow-y-auto">
          <div className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2 mt-2 px-2">Pipeline</div>
          {navTabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                disabled={tab.disabled}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center justify-between px-4 py-3 rounded-xl transition-all ${
                  tab.disabled ? 'opacity-50 cursor-not-allowed hidden' : 
                  isActive ? 'bg-teal-500/10 text-teal-400' : 'hover:bg-slate-800 hover:text-white'
                }`}
              >
                <div className="flex items-center gap-3">
                  <Icon size={20} />
                  <span className="font-medium">{tab.label}</span>
                </div>
                {isActive && <ChevronRight size={16} />}
              </button>
            );
          })}
        </div>

        <div className="p-4 border-t border-slate-800">
          <button onClick={() => { setEdaData(null); setFile(null); setTrainResults(null); setModality(null); setIsStarted(false); }} className="w-full py-2.5 bg-slate-800 hover:bg-slate-700 text-white rounded-lg text-sm font-medium transition-colors">
            Exit Workspace
          </button>
        </div>
      </div>

      {/* MAIN CONTENT AREA */}
      <div className="flex-1 overflow-y-auto p-8 relative">
        <div className="max-w-5xl mx-auto space-y-8 pb-20">
          
          <div className="flex items-center gap-3 text-slate-500 mb-6">
            <FileText size={20} />
            <span className="font-medium">Active Dataset: <strong className="text-slate-800">{edaData.filename}</strong></span>
          </div>

          {/* TAB 1: EDA */}
          {activeTab === 'eda' && (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 space-y-8">
              <h2 className="text-3xl font-extrabold text-slate-800">Data Exploration</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-200 flex items-center gap-4">
                  <div className="bg-emerald-100 p-4 rounded-full text-emerald-600"><Database size={28} /></div>
                  <div><p className="text-gray-500 font-medium text-sm">Total Rows</p><p className="text-3xl font-bold text-gray-800">{edaData.total_rows?.toLocaleString()}</p></div>
                </div>
                <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-200 flex items-center gap-4">
                  <div className="bg-violet-100 p-4 rounded-full text-violet-600"><TableIcon size={28} /></div>
                  <div><p className="text-gray-500 font-medium text-sm">Total Features</p><p className="text-3xl font-bold text-gray-800">{edaData.total_columns}</p></div>
                </div>
                <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-200 flex items-center gap-4">
                  <div className="bg-rose-100 p-4 rounded-full text-rose-600"><AlertCircle size={28} /></div>
                  <div><p className="text-gray-500 font-medium text-sm">Missing Values</p><p className="text-3xl font-bold text-gray-800">{edaData.missing_values ? Object.values(edaData.missing_values).reduce((a, b) => a + b, 0).toLocaleString() : 0}</p></div>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                <div className="lg:col-span-1 space-y-6">
                  <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-200">
                    <h3 className="text-lg font-bold text-gray-800 mb-4 border-b pb-2">Data Types</h3>
                    <div className="space-y-3 max-h-64 overflow-y-auto pr-2">
                      {edaData.column_types && Object.entries(edaData.column_types).map(([col, type]) => (
                        <div key={col} className="flex justify-between items-center text-sm">
                          <span className="font-medium text-gray-700 truncate max-w-[150px]" title={col}>{col}</span>
                          <span className={`px-2 py-1 rounded text-xs font-semibold ${
                            type === 'numeric' ? 'bg-blue-50 text-blue-700' : 'bg-green-50 text-green-700'
                          }`}>
                            {type}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="lg:col-span-2 bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden flex flex-col">
                  <div className="p-6 border-b border-gray-200 bg-gray-50/50">
                    <h3 className="text-lg font-bold text-gray-800">Dataset Preview (First 5 Rows)</h3>
                  </div>
                  <div className="overflow-x-auto p-4">
                    <table className="w-full text-sm text-left whitespace-nowrap">
                      <thead className="text-xs text-gray-500 uppercase bg-gray-50">
                        <tr>
                          {edaData.preview && edaData.preview[0] && Object.keys(edaData.preview[0]).map((col) => (
                            <th key={col} className="px-4 py-3 font-semibold">{col}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {edaData.preview && edaData.preview.map((row, idx) => (
                          <tr key={idx} className="border-b last:border-0 border-gray-100 hover:bg-gray-50 transition-colors">
                            {Object.values(row).map((val, vIdx) => (
                              <td key={vIdx} className="px-4 py-3 text-gray-600">
                                {val === null ? <span className="text-red-400 italic">NaN</span> : String(val)}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 2: MODEL TRAINING & LEADERBOARD */}
          {activeTab === 'training' && (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 space-y-8">
              <h2 className="text-3xl font-extrabold text-slate-800">Model Training Setup</h2>
              
              {isTraining ? (
                // --- NEW LOADING SCREEN W/ MINI-GAME ---
                <div className="bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden p-8 md:p-12 flex flex-col items-center justify-center min-h-[500px] animate-in zoom-in-95 duration-500">
                  <div className="relative mb-6">
                    <div className="absolute inset-0 bg-teal-400 rounded-full blur-xl opacity-20 animate-pulse"></div>
                    <BrainCircuit size={64} className="text-teal-500 animate-pulse relative z-10" />
                  </div>
                  <h3 className="text-2xl font-bold text-slate-800 mb-2 text-center">Training your AI Models...</h3>
                  <p className="text-gray-500 text-center max-w-md mb-8 text-sm">
                    Depending on the size of your dataset and the algorithms being used, this process can take anywhere from a few seconds to a few minutes. <strong>Please don't close this tab!</strong>
                  </p>

                  <div className="w-full max-w-md bg-slate-50 border border-slate-200 rounded-2xl p-6 shadow-inner">
                    <h4 className="font-bold text-slate-700 flex items-center justify-center gap-2 mb-4">
                      <Gamepad2 size={18} className="text-indigo-500"/> While you wait, play a game!
                    </h4>
                    <TicTacToe />
                  </div>
                </div>
              ) : modality === 'tabular' ? (
                // --- TABULAR SETUP SCREEN ---
                <div className="bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden">
                  <div className="p-8 grid grid-cols-1 md:grid-cols-2 gap-8">
                    <div className="space-y-6">
                      <div>
                        <label className="block text-sm font-bold text-gray-700 mb-2">1. Target Column (To Predict)</label>
                        <select value={targetColumn} onChange={(e) => setTargetColumn(e.target.value)} className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-teal-500 bg-gray-50">
                          <option value="">-- Choose a column --</option>
                          {edaData.column_types && Object.keys(edaData.column_types).map(col => (<option key={col} value={col}>{col}</option>))}
                        </select>
                      </div>
                      <div>
                        <label className="block text-sm font-bold text-gray-700 mb-2">2. Machine Learning Task</label>
                        <select value={taskType} onChange={(e) => setTaskType(e.target.value)} className="w-full p-3 border border-gray-300 rounded-lg bg-gray-50" disabled={!targetColumn}>
                          <option value="">-- Auto-selected by AI --</option>
                          <option value="classification">Classification (Categories)</option>
                          <option value="regression">Regression (Numbers)</option>
                        </select>
                      </div>
                    </div>

                    <div className="flex flex-col justify-between">
                      {aiReasoning ? (
                        <div className="bg-violet-50 border border-violet-200 p-5 rounded-xl flex items-start gap-3">
                          <Sparkles className="text-violet-600 shrink-0 mt-0.5" />
                          <div>
                            <h4 className="font-bold text-violet-900 mb-1">AI Assistant</h4>
                            <p className="text-violet-700 text-sm leading-relaxed">{aiReasoning}</p>
                          </div>
                        </div>
                      ) : (
                        <div className="bg-gray-50 border border-gray-200 p-5 rounded-xl flex items-start gap-3 text-gray-400 h-full">
                          <Sparkles className="shrink-0 mt-0.5" />
                          <p className="text-sm">Select a target column to get AI recommendations.</p>
                        </div>
                      )}
                      <button onClick={handleTrain} disabled={!targetColumn || !taskType || isTraining} className={`mt-6 w-full py-4 rounded-xl font-bold text-lg flex items-center justify-center gap-2 transition-all ${!targetColumn || !taskType || isTraining ? 'bg-gray-200 text-gray-400 cursor-not-allowed' : 'bg-slate-900 text-white hover:bg-black'}`}>
                        <PlayCircle size={24} className={isTraining ? 'animate-spin' : ''} />
                        {isTraining ? 'Training Models...' : 'Start Pipeline'}
                      </button>
                    </div>
                  </div>
                </div>
              ) : (
                // --- VISION SETUP SCREEN ---
                <div className="bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden p-8 grid gap-8">
                  <div className="bg-rose-50 border border-rose-200 p-6 rounded-xl flex flex-col justify-center">
                      <h3 className="text-xl font-bold text-rose-800 mb-2">Computer Vision Pipeline</h3>
                      <p className="text-rose-700">We will automatically convert your images into tensor arrays and train image classification algorithms based on your folder names.</p>
                  </div>
                  <div className="flex flex-col justify-end">
                    <button onClick={handleTrain} disabled={isTraining} className={`w-full py-4 rounded-xl font-bold text-lg flex items-center justify-center gap-2 transition-all ${isTraining ? 'bg-gray-200 text-gray-400 cursor-not-allowed' : 'bg-rose-600 text-white hover:bg-rose-700 shadow-md'}`}>
                      <PlayCircle size={24} className={isTraining ? 'animate-spin' : ''} />
                      {isTraining ? 'Processing Images...' : 'Start Vision Training'}
                    </button>
                  </div>
                </div>
              )}

              {trainResults && (
                <div className="bg-white border border-gray-200 rounded-2xl shadow-sm overflow-hidden animate-in fade-in">
                  <div className="bg-slate-50 p-6 border-b border-gray-200 flex justify-between items-center">
                    <h4 className="font-bold text-slate-800 text-lg flex items-center gap-2"><Activity /> Algorithm Leaderboard</h4>
                    <span className="bg-emerald-100 text-emerald-800 px-4 py-1.5 rounded-full text-sm font-bold flex items-center gap-1">
                      Winner: {trainResults.best_model} <Sparkles size={14}/>
                    </span>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm text-left">
                      <thead className="text-xs text-gray-500 uppercase bg-gray-50 border-b border-gray-200">
                        <tr>
                          <th className="px-6 py-4 font-semibold">Algorithm</th>
                          {trainResults.task_type === 'classification' ? (
                            <><th className="px-6 py-4 font-semibold">Accuracy</th><th className="px-6 py-4 font-semibold">F1 Score</th><th className="px-6 py-4 font-semibold">Recall</th><th className="px-6 py-4 font-semibold">CV Score</th></>
                          ) : (
                            <><th className="px-6 py-4 font-semibold">RMSE</th><th className="px-6 py-4 font-semibold">R² Score</th></>
                          )}
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(trainResults.models).map(([modelName, metrics]) => (
                          <tr key={modelName} className={`border-b border-gray-100 last:border-0 transition-colors ${trainResults.best_model === modelName ? 'bg-emerald-50/30' : 'hover:bg-gray-50'}`}>
                            <td className="px-6 py-4 font-bold text-slate-700 flex items-center gap-2">
                                {trainResults.best_model === modelName && <Sparkles size={16} className="text-emerald-500" />}
                                {modelName}
                            </td>
                            {trainResults.task_type === 'classification' ? (
                              <>
                                <td className="px-6 py-4 text-gray-600 font-medium">{(metrics.accuracy * 100).toFixed(2)}%</td>
                                <td className="px-6 py-4 text-gray-600">{(metrics.macro_avg_f1 * 100).toFixed(2)}%</td>
                                <td className="px-6 py-4 text-gray-600">{(metrics.macro_avg_recall * 100).toFixed(2)}%</td>
                                <td className="px-6 py-4 text-gray-600">{(metrics.cv_score_mean * 100).toFixed(2)}%</td>
                              </>
                            ) : (
                              <>
                                <td className="px-6 py-4 text-gray-600 font-medium">{metrics.rmse?.toFixed(4)}</td>
                                <td className="px-6 py-4 text-gray-600">{(metrics.r2_score * 100).toFixed(2)}%</td>
                              </>
                            )}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="p-4 bg-slate-50 border-t border-gray-200 flex gap-4 justify-end">
                    {modality === 'tabular' && (
                      <button onClick={() => setActiveTab('explainability')} className="text-indigo-600 font-bold text-sm hover:text-indigo-800 flex items-center gap-1">
                        View Feature Importance <ChevronRight size={16} />
                      </button>
                    )}
                    <button onClick={() => setActiveTab('matrices')} className="text-teal-600 font-bold text-sm hover:text-teal-800 flex items-center gap-1">
                      View Detailed Metrics <ChevronRight size={16} />
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* TAB 2.5: FEATURE IMPORTANCE (Explainable AI) */}
          {activeTab === 'explainability' && trainResults && (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 space-y-8">
              <div className="flex items-center gap-4 mb-2">
                <div className="bg-indigo-100 p-4 rounded-xl text-indigo-600">
                  <Eye size={32} />
                </div>
                <div>
                  <h2 className="text-3xl font-extrabold text-slate-800">Feature Importance (XAI)</h2>
                  <p className="text-gray-500 mt-1">Discover which columns in your dataset had the biggest impact on the model's predictions.</p>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {Object.entries(trainResults.models).map(([modelName, metrics]) => {
                  if (!metrics.feature_importance) return null;
                  
                  return (
                    <div key={modelName} className="bg-white border border-gray-200 p-6 rounded-xl shadow-sm relative h-[450px] flex flex-col">
                      {trainResults.best_model === modelName && (
                        <div className="absolute top-4 right-4 bg-emerald-100 text-emerald-800 text-xs font-bold px-3 py-1 rounded-full flex items-center gap-1">
                          <Sparkles size={12} /> Winner
                        </div>
                      )}
                      <h4 className="font-bold text-gray-800 mb-2 flex items-center gap-2">
                        <Activity size={18} className="text-indigo-500"/> {modelName}
                      </h4>
                      <p className="text-xs text-gray-500 mb-6">Top driving factors for predictions</p>
                      
                      <div className="flex-1 w-full h-full min-h-[300px]">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart 
                            data={metrics.feature_importance} 
                            layout="vertical" 
                            margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="#f0f0f0" />
                            <XAxis type="number" hide />
                            <YAxis 
                              dataKey="feature" 
                              type="category" 
                              axisLine={false} 
                              tickLine={false} 
                              tick={{fontSize: 12, fill: '#4b5563'}}
                              width={100}
                            />
                            <Tooltip 
                              formatter={(value) => [(value * 100).toFixed(2) + '%', 'Importance']}
                              contentStyle={{borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'}}
                            />
                            <Bar 
                              dataKey="importance" 
                              fill={trainResults.best_model === modelName ? "#10b981" : "#6366f1"} 
                              radius={[0, 4, 4, 0]}
                              barSize={20}
                            />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* TAB 3: CONFUSION MATRICES */}
          {activeTab === 'matrices' && trainResults?.task_type === 'classification' && (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 space-y-8">
              <h2 className="text-3xl font-extrabold text-slate-800">Confusion Matrices</h2>
              <p className="text-gray-500">Compare where each algorithm made correct predictions (Green) vs. mistakes (Red).</p>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {Object.entries(trainResults.models).map(([modelName, metrics]) => (
                  <div key={modelName} className="bg-white border border-gray-200 p-6 rounded-xl shadow-sm relative">
                    {trainResults.best_model === modelName && (
                      <div className="absolute top-4 right-4 bg-emerald-100 text-emerald-800 text-xs font-bold px-3 py-1 rounded-full flex items-center gap-1">
                        <Sparkles size={12} /> Winner
                      </div>
                    )}
                    <h4 className="font-bold text-gray-800 mb-6 flex items-center gap-2">
                      <TableIcon size={18} className="text-teal-500"/> {modelName}
                    </h4>
                    
                    <div className="flex flex-col items-center">
                      <div className="grid grid-cols-3 gap-1 mb-2 w-full max-w-sm">
                        <div className="col-span-1"></div>
                        <div className="col-span-2 text-center text-xs font-bold text-gray-500 mb-1">Predicted</div>
                        
                        <div className="col-span-1 flex items-center justify-end pr-2 text-xs font-bold text-gray-500" style={{writingMode: 'vertical-rl', transform: 'rotate(180deg)'}}>Actual</div>
                        
                        <div className="col-span-2 grid grid-cols-2 gap-2">
                          {metrics.confusion_matrix && metrics.confusion_matrix.map((row, i) => 
                            row.map((cell, j) => {
                              const maxVal = Math.max(...metrics.confusion_matrix.flat());
                              const opacity = Math.max(0.1, cell / maxVal);
                              const isCorrect = i === j;
                              return (
                                <div key={`${i}-${j}`} 
                                     className="relative flex flex-col items-center justify-center p-4 rounded-lg border border-gray-100 overflow-hidden min-h-[80px]"
                                     style={{ backgroundColor: isCorrect ? `rgba(16, 185, 129, ${opacity})` : `rgba(239, 68, 68, ${opacity})` }}
                                >
                                  <span className={`relative z-10 text-xl font-bold ${opacity > 0.5 ? 'text-white' : 'text-gray-800'}`}>{cell}</span>
                                </div>
                              )
                            })
                          )}
                        </div>
                      </div>
                      <div className="flex justify-between w-full max-w-[150px] mt-2 ml-10 text-xs text-gray-500">
                        <span>{trainResults.classes ? trainResults.classes["0"] : '0'}</span>
                        <span>{trainResults.classes ? (trainResults.classes["1"] || 'N/A') : '1'}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* TAB 4: PERFORMANCE GRAPHS */}
          {activeTab === 'graphs' && (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 space-y-8">
              <h2 className="text-3xl font-extrabold text-slate-800">Performance Graphs</h2>
              
              {trainResults?.task_type === 'classification' && Object.values(trainResults.models).some(m => m.roc_data) ? (
                <>
                  <p className="text-gray-500">Compare the ROC Curves for each algorithm. A curve closer to the top-left corner (AUC closer to 1.0) is better.</p>
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    {Object.entries(trainResults.models).map(([modelName, metrics]) => {
                      if (!metrics.roc_data) return null;
                      return (
                        <div key={modelName} className="bg-white border border-gray-200 p-6 rounded-xl shadow-sm h-full min-h-[350px] flex flex-col relative">
                          {trainResults.best_model === modelName && (
                            <div className="absolute top-4 right-4 bg-emerald-100 text-emerald-800 text-xs font-bold px-3 py-1 rounded-full flex items-center gap-1">
                              <Sparkles size={12} /> Winner
                            </div>
                          )}
                          <h4 className="font-bold text-gray-800 mb-6 flex items-center justify-between">
                            <span className="flex items-center gap-2"><Activity size={18} className="text-blue-500"/> {modelName}</span>
                            <span className="text-sm font-medium bg-blue-50 text-blue-700 px-2 py-1 rounded">AUC: {metrics.roc_auc?.toFixed(3)}</span>
                          </h4>
                          <div className="flex-1 w-full h-full min-h-[250px]">
                            <ResponsiveContainer width="100%" height="100%">
                              <LineChart data={metrics.roc_data} margin={{ top: 5, right: 5, bottom: 20, left: -20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                                <XAxis dataKey="fpr" type="number" domain={[0, 1]} label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -15, style: {fontSize: 12, fill: '#6b7280'} }} tick={{fontSize: 12}} />
                                <YAxis type="number" domain={[0, 1]} label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft', style: {fontSize: 12, fill: '#6b7280'} }} tick={{fontSize: 12}} />
                                <Tooltip 
                                  formatter={(value) => value.toFixed(3)} 
                                  labelFormatter={(label) => `FPR: ${Number(label).toFixed(3)}`}
                                  contentStyle={{borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'}}
                                />
                                <Line type="monotone" dataKey="tpr" stroke="#3b82f6" strokeWidth={3} dot={false} activeDot={{ r: 6 }} name="TPR" />
                                <Line type="linear" dataKey="fpr" stroke="#94a3b8" strokeWidth={2} strokeDasharray="5 5" dot={false} name="Random" />
                              </LineChart>
                            </ResponsiveContainer>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </>
              ) : (
                <div className="bg-white border border-gray-200 p-10 rounded-xl text-center shadow-sm max-w-2xl mx-auto">
                  <Activity size={48} className="mx-auto text-gray-300 mb-4" />
                  <h3 className="text-xl font-bold text-gray-700">No Graphs Available</h3>
                  <p className="text-gray-500 mt-2">ROC curves and detailed performance graphs are primarily generated for Classification tasks with specific probability outputs.</p>
                </div>
              )}
            </div>
          )}

          {/* TAB 5: DOWNLOAD CENTER */}
          {activeTab === 'download' && trainResults && (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 space-y-8">
              <div className="flex items-center gap-4 mb-6">
                <div className="bg-teal-100 p-4 rounded-xl text-teal-600">
                  <Download size={32} />
                </div>
                <div>
                  <h2 className="text-3xl font-extrabold text-slate-800">Download Center</h2>
                  <p className="text-gray-500 mt-1">Export your trained models as .joblib files for deployment in your own applications.</p>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {Object.entries(trainResults.models).map(([modelName, metrics]) => (
                  <div key={modelName} className="bg-white border border-gray-200 p-6 rounded-2xl shadow-sm relative flex flex-col h-full hover:shadow-md transition-shadow">
                    {trainResults.best_model === modelName && (
                      <div className="absolute top-4 right-4 bg-emerald-100 text-emerald-800 text-xs font-bold px-3 py-1 rounded-full flex items-center gap-1 shadow-sm">
                        <Sparkles size={12} /> Winner
                      </div>
                    )}
                    
                    <div className="mb-4 pr-16">
                      <h4 className="font-bold text-gray-800 text-lg mb-2">{modelName}</h4>
                      <div className="text-sm font-medium text-gray-500 flex items-center gap-2">
                        {trainResults.task_type === 'classification' ? (
                          <span className="text-blue-700 bg-blue-50 border border-blue-200 px-2 py-0.5 rounded">Accuracy: {(metrics.accuracy * 100).toFixed(1)}%</span>
                        ) : (
                          <span className="text-blue-700 bg-blue-50 border border-blue-200 px-2 py-0.5 rounded">RMSE: {metrics.rmse?.toFixed(3)}</span>
                        )}
                      </div>
                    </div>

                    <div className="flex-1 text-sm text-gray-500 mb-6 bg-slate-50 p-3 rounded-lg border border-gray-100">
                      This serialized pipeline includes your configured imputer, scaler/encoder, and the {modelName} algorithm.
                    </div>

                    <a 
                      href={`http://127.0.0.1:8000/api/download-model/${metrics.download_file}`}
                      download={metrics.download_file}
                      className={`w-full py-3 rounded-xl font-bold text-sm flex items-center justify-center gap-2 transition-all shadow-sm ${
                        trainResults.best_model === modelName 
                          ? 'bg-teal-600 text-white hover:bg-teal-700 hover:shadow-md' 
                          : 'bg-slate-100 text-slate-700 hover:bg-slate-200 hover:text-slate-900'
                      }`}
                    >
                      <Download size={18} />
                      Download Model
                    </a>
                  </div>
                ))}
              </div>
              
              <div className="bg-slate-900 border border-slate-800 p-8 rounded-2xl mt-8 shadow-lg">
                <h4 className="font-bold text-white mb-4 flex items-center gap-2 text-lg">
                  <BrainCircuit size={20} className="text-teal-400" /> How to use these files in Python
                </h4>
                <div className="bg-slate-950 p-5 rounded-xl overflow-x-auto border border-slate-800">
                  <pre className="text-teal-400 text-sm font-mono leading-relaxed">
{`import joblib
import pandas as pd

# 1. Load the model pipeline
model = joblib.load('downloaded_model.joblib')

# 2. Prepare your new data (columns must match original dataset)
new_data = pd.DataFrame({
    'Feature_1': [value],
    'Feature_2': [value]
})

# 3. Make predictions!
predictions = model.predict(new_data)
print(predictions)`}
                  </pre>
                </div>
              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  );
}