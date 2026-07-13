import { useState, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './App.css'

interface PipelineResponse {
  success: boolean;
  text_outputs?: any[];
  "semantic novelties"?: any[];
  token_similarities?: any;
  clustering_results?: {
    MDS_results: number[][];
    Agglomerative_coloring: number[];
  };
  accuracy_scores?: any;
  "OpenRouter Errors"?: string;
  "error message"?: string;
}

function App() {
  const [question, setquestion] = useState("");
  const [baseline, setbaseline] = useState("");
  const [available_models, setAvailableModels] = useState([]);
  const [selected_models, setselected_models] = useState<string[]>([])
  const [apiKey, setApiKey] = useState<string>("");
  const [loading, setloading] = useState(false)
  const [pipelineData, setPipelineData] = useState<PipelineResponse | null>(null);
  const [error_msg, setErrorMsg] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<string>("work");
  const [numClusters, setnumClusters] = useState(2)
  const [Plot, setPlot] = useState<any>(null);
  const [theme, setTheme] = useState<'light' | 'dark'>('dark')

  useEffect(() => {
    let cancelled = false;

    const loadModels = async () => {
      const api = (window as any).pywebview?.api;
      if (!api) {
        setTimeout(loadModels, 100);
        return;
      }

      const models = await api.get_models();
      if (!cancelled) {
        setAvailableModels(models);
      }
    };

    loadModels();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    Promise.all([
      import('react-plotly.js/factory'),
      import('plotly.js-dist-min')
    ]).then(([factory, plotly]) => {
      const createPlotlyComponent = factory.default;
      const Plotly = plotly.default;
      setPlot(() => createPlotlyComponent(Plotly));
    });
  }, []);

  useEffect(() => {
    const loadSavedApiKey = async () => {
      if ((window as any).pywebview?.api) {
        const result = await (window as any).pywebview.api.get_api_key();
        if (result?.success && result.API_key) {
          setApiKey(result.API_key);
        }
      }
    };

    loadSavedApiKey();
  }, []);

  const handleModelToggle = async(model_name: string, checked: boolean) => {
    const updatedModels = checked 
      ? [...selected_models, model_name] 
      : selected_models.filter(m => m !== model_name);
    
    setselected_models(updatedModels);
    
    if (numClusters > updatedModels.length && updatedModels.length >= 2) {
      setnumClusters(updatedModels.length);
    }
  }
  
  const setAPIKey = async() => {
    if(!apiKey){
      setErrorMsg("API Key cannot be empty.")
    }
    const saved = (await (window as any).pywebview.api.save_api_key(apiKey))
    if(!saved) {
      setErrorMsg("Your API Key pattern is off. It must begin with a sk-or-v1. Go get it from OpenRouter if you don't have one yet!")
    }else{
      setErrorMsg(null)
    }
  }

  const runPipeline = async() => {
    setloading(true)
    var response = (await (window as any).pywebview.api.return_outputs(question, baseline, selected_models, numClusters))
    if(response["success"]){
      setPipelineData(response)
      if("OpenRouter Errors" in response){
        setErrorMsg(`Everything executed fine but OpenRouter returned errors, this is possibly because some models are too busy to respond:\n ${response["OpenRouter Errors"]}`)
      }else{
        setErrorMsg(null)
      }
    }else {
      setErrorMsg(response["error message"])
    }
    setloading(false)
  }
  const renderTextOutputs = () => {
    const outputs = pipelineData?.text_outputs;
    if (!outputs || !Array.isArray(outputs) || outputs.length === 0) return null;

    return (
      <div className={`w-full ${theme === 'dark' ? 'bg-slate-950 border-slate-800' : 'bg-white border-slate-200'} border rounded-lg p-4 flex flex-col gap-2`}>
        <h3 className={`text-sm font-semibold tracking-wider ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'} uppercase`}>Text Outputs</h3>
        <div className="flex-1 min-h-0 overflow-y-auto space-y-3 pr-1">
          {outputs.map((item: any, idx: number) => {
            const markdownText =
              typeof item === "string"
                ? item
                : item.text ?? JSON.stringify(item, null, 2);

            return (
              <div key={`text-output-${idx}`} className={`rounded border ${theme === 'dark' ? 'border-slate-800 bg-slate-900/50' : 'border-slate-200 bg-slate-50'} p-3`}>
                {typeof item !== "string" && item.model && (
                  <div className="text-xs font-mono text-slate-400 mb-2">{item.model}</div>
                )}
                <div className={`text-sm ${theme === 'dark' ? 'text-slate-200' : 'text-slate-800'} whitespace-pre-wrap`}>
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {markdownText}
                  </ReactMarkdown>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  const renderMDSPlot = () => {
    if (!pipelineData?.clustering_results) return null;
    const { MDS_results, Agglomerative_coloring } = pipelineData.clustering_results;
    const x = MDS_results.map(coords => coords[0]);
    const y = MDS_results.map(coords => coords[1]);

    return (
      <Plot
        data={[{
          x,
          y,
          mode: 'markers',
          type: 'scatter',
          marker: {
            size: 12,
            color: Agglomerative_coloring,
            colorscale: 'Viridis',
            line: { width: 1, color: theme === 'dark' ? '#475569' : '#cbd5e1' }
          },
          text: selected_models
        }]}
        layout={{
          autosize: true,
          margin: { l: 40, r: 20, t: 20, b: 40 },
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          xaxis: { gridcolor: theme === 'dark' ? '#1e293b' : '#e2e8f0', zerolinecolor: theme === 'dark' ? '#334155' : '#cbd5e1', tickfont: { color: theme === 'dark' ? '#94a3b8' : '#64748b' } },
          yaxis: { gridcolor: theme === 'dark' ? '#1e293b' : '#e2e8f0', zerolinecolor: theme === 'dark' ? '#334155' : '#cbd5e1', tickfont: { color: theme === 'dark' ? '#94a3b8' : '#64748b' } }
        }}
        useResizeHandler={true}
        className="w-full h-full"
      />
    );
  };

  const renderHeatmaps = () => {
    const drifts = pipelineData?.token_similarities;
    if (!drifts || !Array.isArray(drifts)) return null;

    return drifts.map((driftItem: any, idx: number) => {
      const modelName = driftItem[0];
      const zData = driftItem[1];

      return (
        <div key={`heatmap-${idx}`} className={`w-full h-[350px] ${theme === 'dark' ? 'bg-slate-900/50 border-slate-800' : 'bg-slate-50 border-slate-200'} rounded border p-2 flex flex-col gap-1`}>
          <span className="text-[10px] font-mono text-slate-400 block px-1">{modelName}</span>
          <div className="flex-1 min-h-0">
            <Plot
              data={[{
                z: zData,
                type: 'heatmap',
                colorscale: 'Viridis',
                zmin: 0.7,
                zmax: 1.0
              }]}
              layout={{
                autosize: true,
                margin: { l: 40, r: 20, t: 20, b: 40 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                xaxis: { tickfont: { color: theme === 'dark' ? '#94a3b8' : '#64748b' }, title: { text: 'Baseline', font: { size: 10, color: theme === 'dark' ? '#64748b' : '#94a3b8' } } },
                yaxis: { tickfont: { color: theme === 'dark' ? '#94a3b8' : '#64748b' }, title: { text: 'Response', font: { size: 10, color: theme === 'dark' ? '#64748b' : '#94a3b8' } } }
              }}
              useResizeHandler={true}
              className="w-full h-full"
            />
          </div>
        </div>
      );
    });
  };

  const renderNoveltyPlots = () => {
    const novelties = pipelineData?.["semantic novelties"];
    if (!novelties || novelties.length === 0) return null;

    return novelties.map((item, idx) => {
      const modelName = Object.keys(item)[0];
      const scores = item[modelName] as number[];

      return (
        <div key={`novelty-${idx}`} className={`w-full h-[350px] ${theme === 'dark' ? 'bg-slate-900/50 border-slate-800' : 'bg-slate-50 border-slate-200'} rounded border p-2 flex flex-col gap-1`}>
          <span className="text-[10px] font-mono text-slate-400 block px-1">{modelName}</span>
          <div className="flex-1 min-h-0">
            <Plot
              data={[{
                x: Array.from({ length: scores.length }, (_, i) => i),
                y: scores,
                type: 'scatter',
                mode: 'lines+markers',
                marker: { color: '#3b82f6' },
                line: { color: '#3b82f6' }
              }]}
              layout={{
                autosize: true,
                margin: { l: 40, r: 20, t: 20, b: 40 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                xaxis: { gridcolor: theme === 'dark' ? '#1e293b' : '#e2e8f0', tickfont: { color: theme === 'dark' ? '#94a3b8' : '#64748b' } },
                yaxis: { gridcolor: theme === 'dark' ? '#1e293b' : '#e2e8f0', tickfont: { color: theme === 'dark' ? '#94a3b8' : '#64748b' } }
              }}
              useResizeHandler={true}
              className="w-full h-full"
            />
          </div>
        </div>
      );
    });
  };

  return (
    <div className={`flex flex-col h-screen w-screen ${theme === 'dark' ? 'bg-slate-900 text-slate-100' : 'bg-white text-slate-900'} font-sans overflow-hidden`}>
      <div className={`flex justify-end items-center px-6 py-2 border-b ${theme === 'dark' ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-200'} h-10`}>
        <button 
          onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          className={`px-2 py-1 text-xs font-medium border rounded transition-colors ${theme === 'dark' ? 'bg-slate-900 border-slate-700 text-slate-300 hover:bg-slate-800 hover:text-white' : 'bg-white border-slate-300 text-slate-700 hover:bg-slate-100 hover:text-black'}`}
        >
          {theme === 'dark' ? 'Light Mode' : 'Dark Mode'}
        </button>
      </div>

      <div className="flex flex-1 overflow-hidden">
        <div className={`w-100 border-r ${theme === 'dark' ? 'border-slate-800 bg-slate-950' : 'border-slate-200 bg-slate-50'} p-6 flex flex-col gap-6 overflow-y-auto`}>
          <div>
            <h1 className={`text-xl font-bold tracking-tight ${theme === 'dark' ? 'text-white' : 'text-slate-950'}`}>Welcome to Model Assistant Inference Network (MAIN)</h1>
            <p className="text-xs text-slate-500 mt-1">Find out which model is the most effective! </p>
            <p className='text-xs text-yellow-600 mt-1'>This service sends your request to openrouter.ai, keep in mind that some free models will use your requests to train themselves.</p>
          </div>
          
          <div className="flex flex-col gap-1.5">
            <button 
              onClick={() => setActiveTab("work")}
              className={`text-left px-3 py-2 rounded text-sm font-medium transition-colors ${activeTab === 'work' ? 'bg-blue-600 text-white' : theme === 'dark' ? 'text-slate-400 hover:bg-slate-900 hover:text-slate-200' : 'text-slate-600 hover:bg-slate-200'}`}
            >
              Execution Workspace
            </button>
            <button 
              onClick={() => setActiveTab("settings")}
              className={`text-left px-3 py-2 rounded text-sm font-medium transition-colors ${activeTab === 'settings' ? 'bg-blue-600 text-white' : theme === 'dark' ? 'text-slate-400 hover:bg-slate-900 hover:text-slate-200' : 'text-slate-600 hover:bg-slate-200'}`}
            >
              App Settings
            </button>
            <button 
              onClick={() => setActiveTab("docs")}
              className={`text-left px-3 py-2 rounded text-sm font-medium transition-colors ${activeTab === 'docs' ? 'bg-blue-600 text-white' : theme === 'dark' ? 'text-slate-400 hover:bg-slate-900 hover:text-slate-200' : 'text-slate-600 hover:bg-slate-200'}`}
            >
              Documentation
            </button>
          </div>

          <hr className={`${theme === 'dark' ? 'border-slate-800' : 'border-slate-200'} my-1`} />

          {activeTab === "work" && (
            <div className="flex flex-col gap-5">
              <div className="flex flex-col gap-1.5">
                <label className="text-xs font-semibold tracking-wider text-slate-400 uppercase">Evaluation Question</label>
                <textarea
                  value={question}
                  onChange={(e) => setquestion(e.target.value)}
                  className={`w-full h-24 p-2.5 rounded ${theme === 'dark' ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-300'} border text-sm focus:outline-none focus:border-blue-500 resize-none`}
                  placeholder="Enter pipeline evaluation query..."
                />
              </div>

              <div className="flex flex-col gap-1.5">
                <label className="text-xs font-semibold tracking-wider text-slate-400 uppercase">Baseline Anchor</label>
                <textarea
                  value={baseline}
                  onChange={(e) => setbaseline(e.target.value)}
                  className={`w-full h-24 p-2.5 rounded ${theme === 'dark' ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-300'} border text-sm focus:outline-none focus:border-blue-500 resize-none`}
                  placeholder="Enter gold standard baseline..."
                />
              </div>

              <div className="flex flex-col gap-1.5">
                <div className="flex justify-between items-center">
                  <label className="text-xs font-semibold tracking-wider text-slate-400 uppercase">Available Models</label>
                </div>
                <div className={`w-full h-40 overflow-y-auto ${theme === 'dark' ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-300'} border rounded p-2 flex flex-col gap-2`}>
                  {available_models.length === 0 ? (
                    <span className="text-xs text-slate-500 p-1">No models fetched. Try again.</span>
                  ) : (
                    available_models.map((model_name) => (
                      <label key={model_name} className={`flex items-center gap-2.5 text-sm ${theme === 'dark' ? 'text-slate-300 hover:text-white' : 'text-slate-700 hover:text-black'} cursor-pointer`}>
                        <input
                          type="checkbox"
                          checked={selected_models.includes(model_name)}
                          onChange={(e) => handleModelToggle(model_name, e.target.checked)}
                          className="rounded border-slate-700 bg-slate-800 text-blue-600 focus:ring-0 focus:ring-offset-0"
                        />
                        {model_name}
                      </label>
                    ))
                  )}
                </div>
              </div>

              <div className="flex flex-col gap-1.5">
                <div className="flex justify-between items-center">
                  <label className="text-xs font-semibold tracking-wider text-slate-400 uppercase">Clusters (K)</label>
                  <span className="text-xs font-mono text-blue-400">{numClusters}</span>
                </div>
                <input
                  type="range"
                  min="2"
                  max={Math.max(2, selected_models.length)}
                  value={numClusters}
                  disabled={selected_models.length < 2}
                  onChange={(e) => setnumClusters(parseInt(e.target.value))}
                  className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
                />
              </div>

              <button
                onClick={runPipeline}
                disabled={loading || selected_models.length < 2 || numClusters > selected_models.length}
                className="w-full py-2.5 px-4 rounded bg-blue-600 hover:bg-blue-700 disabled:bg-slate-800 disabled:text-slate-500 disabled:cursor-not-allowed text-white text-sm font-medium tracking-wide transition-colors"
              >
                {loading ? "Running Matrix Operations..." : "Execute Pipeline"}
              </button>
            </div>
          )}
        </div>

        <div className={`flex-1 ${theme === 'dark' ? 'bg-slate-900' : 'bg-slate-100'} p-6 flex flex-col gap-6 overflow-y-auto`}>
          {activeTab === "work" && (
            <div className="flex flex-col gap-6 h-full min-h-0">
              {error_msg && (
                <div className={`p-4 rounded border text-sm ${error_msg.startsWith('Everything executed fine') ? 'bg-amber-950/40 border-amber-800 text-amber-200' : 'bg-rose-950/40 border-rose-800 text-rose-200'}`}>
                  {error_msg}
                </div>
              )}

              {!pipelineData ? (
                <div className={`flex-1 flex flex-col items-center justify-center border border-dashed ${theme === 'dark' ? 'border-slate-800' : 'border-slate-300'} rounded-lg p-12 text-center`}>
                  <p className="text-slate-400 text-sm">
                    {loading ? "Awaiting hardware output tensors from python bridge..." : "Setup your benchmark and run it! For how to test your models and what to look out for, check the Documentation page."}
                  </p>
                </div>
              ) : (
                <div className="flex flex-col gap-6 flex-1 min-h-0">
                  <div className={`w-full ${theme === 'dark' ? 'bg-slate-950 border-slate-800' : 'bg-white border-slate-200'} border rounded-lg p-4 flex flex-col gap-2`}>
                    <h3 className={`text-sm font-semibold tracking-wider ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'} uppercase`}>MDS Latent Clustering Map</h3>
                    <div className={`min-h-[350px] ${theme === 'dark' ? 'bg-slate-900/50 border-slate-800' : 'bg-white border-slate-200'} rounded border flex items-center justify-center text-xs text-slate-500 overflow-hidden`}>
                      {renderMDSPlot()}
                    </div>
                  </div>

                  <div className={`w-full ${theme === 'dark' ? 'bg-slate-950 border-slate-800' : 'bg-white border-slate-200'} border rounded-lg p-4 flex flex-col gap-2`}>
                    <h3 className={`text-sm font-semibold tracking-wider ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'} uppercase`}>Token Distance Similarity Grid</h3>
                    <div className="flex flex-col gap-3 overflow-y-auto pr-1">
                      {renderHeatmaps()}
                    </div>
                  </div>

                  <div className={`w-full ${theme === 'dark' ? 'bg-slate-950 border-slate-800' : 'bg-white border-slate-200'} border rounded-lg p-4 flex flex-col gap-2`}>
                    <h3 className={`text-sm font-semibold tracking-wider ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'} uppercase`}>Semantic Novelties Orthogonalization Plot</h3>
                    <div className="flex flex-col gap-3 overflow-y-auto pr-1">
                      {renderNoveltyPlots()}
                    </div>
                  </div>

                  {renderTextOutputs()}
                </div>
              )}
            </div>
          )}

          {activeTab === "settings" && (
            <div className="flex flex-col gap-6 max-w-xl">
              <div>
                <h2 className={`text-lg font-bold ${theme === 'dark' ? 'text-white' : 'text-slate-950'}`}>Application Settings</h2>
                <p className="text-sm text-slate-400 mt-1">Configure environment variables and endpoint authentication hooks.</p>
              </div>

              <div className={`p-5 ${theme === 'dark' ? 'bg-slate-950 border-slate-800' : 'bg-white border-slate-200'} border rounded-lg flex flex-col gap-4`}>
                <div className="flex flex-col gap-1.5">
                  <label className="text-xs font-semibold tracking-wider text-slate-400 uppercase">OpenRouter API Key</label>
                  <input
                    type="password"
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    className={`w-full p-2.5 rounded ${theme === 'dark' ? 'bg-slate-900 border-slate-800 text-white' : 'bg-slate-50 border-slate-300 text-slate-900'} text-sm font-mono focus:outline-none focus:border-blue-500`}
                    placeholder="sk-or-v1-..."
                  />
                  <p className="text-xs text-green-500 mt-1">
                    Keys are stored locally in application runtime memory and passed securely to the local python execution stack. No developers or malicious bots from the internet can access your key. Get your key from openrouter.ai if you don't have one yet!
                  </p>
                  <button className='w-full py-2 px-4 rounded bg-blue-600 hover:bg-blue-700 text-sm font-medium text-white mt-3 transition-colors'
                    onClick={setAPIKey}
                  >
                    Save
                  </button>
                </div>
              </div>
            </div>
          )}

          {activeTab === "docs" && (
            <div className="flex flex-col gap-4 max-w-3xl">
              <div>
                <h2 className={`text-lg font-bold ${theme === 'dark' ? 'text-white' : 'text-slate-950'}`}>Documentation</h2>
                <p className="text-sm text-slate-400 mt-1">This software is still in Beta, what do you expect? Come back soon, the docs will be ready in a bit. </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App