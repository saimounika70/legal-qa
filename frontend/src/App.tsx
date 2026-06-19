import { useState, useRef } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";

const API = "http://localhost:8000";

interface Source {
  content: string;
  similarity: number;
  rank: number;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
}

interface DocInfo {
  doc_id: string;
  filename: string;
  chunks: number;
}

export default function App() {
  const [docInfo, setDocInfo] = useState<DocInfo | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [summary, setSummary] = useState("");
  const [showSources, setShowSources] = useState<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const uploadFile = async (file: File) => {
    setUploading(true);
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await axios.post(`${API}/upload`, form);
      setDocInfo({
        doc_id: res.data.doc_id,
        filename: file.name,
        chunks: res.data.chunks || 0
      });
      setMessages([{
        role: "assistant",
        content: `Document **${file.name}** uploaded successfully. ${res.data.chunks || ''} chunks indexed. Ask me anything about it.`
      }]);
      // auto-fetch summary
      fetchSummary(res.data.doc_id);
    } catch (e: any) {
      alert(e.response?.data?.detail || "Upload failed");
    }
    setUploading(false);
  };

  const fetchSummary = async (doc_id: string) => {
    try {
      const res = await axios.get(`${API}/summary/${doc_id}`);
      setSummary(res.data.summary);
    } catch {}
  };

  const askQuestion = async () => {
    if (!question.trim() || !docInfo) return;
    const q = question;
    setQuestion("");
    setMessages(prev => [...prev, { role: "user", content: q }]);
    setLoading(true);
    try {
      const res = await axios.post(`${API}/question`, {
        doc_id: docInfo.doc_id,
        question: q,
        include_history: true
      });
      setMessages(prev => [...prev, {
        role: "assistant",
        content: res.data.answer,
        sources: res.data.sources
      }]);
    } catch (e: any) {
      setMessages(prev => [...prev, {
        role: "assistant",
        content: "Error: " + (e.response?.data?.detail || "Something went wrong")
      }]);
    }
    setLoading(false);
  };

  const s = styles;

  return (
    <div style={s.page}>
      {/* Header */}
      <div style={s.header}>
        <div>
          <h1 style={s.title}>⚖️ Indian Legal Q&A</h1>
          <p style={s.subtitle}>
            Upload Indian court judgments or legal documents.
            Ask questions in plain English. Get cited answers.
          </p>
        </div>
        {docInfo && (
          <div style={s.docBadge}>
            <span style={{ color: "#3fb950" }}>●</span> {docInfo.filename}
            <span style={s.chunkCount}>{docInfo.chunks} chunks</span>
          </div>
        )}
      </div>

      <div style={s.body}>
        {/* Left: Upload + Summary */}
        <div style={s.sidebar}>
          {/* Upload area */}
          <div
            style={s.uploadBox}
            onClick={() => fileInputRef.current?.click()}
            onDragOver={e => e.preventDefault()}
            onDrop={e => {
              e.preventDefault();
              const file = e.dataTransfer.files[0];
              if (file?.name.endsWith('.pdf')) uploadFile(file);
            }}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              style={{ display: "none" }}
              onChange={e => {
                const file = e.target.files?.[0];
                if (file) uploadFile(file);
              }}
            />
            {uploading ? (
              <p style={{ color: "#8b949e" }}>Processing PDF...</p>
            ) : (
              <>
                <div style={{ fontSize: "2rem", marginBottom: "0.5rem" }}>📄</div>
                <p style={{ color: "#8b949e", fontSize: "0.9rem" }}>
                  Drop a PDF here or click to upload
                </p>
                <p style={{ color: "#484f58", fontSize: "0.75rem", marginTop: "0.3rem" }}>
                  Indian court judgments, RTI docs, legal notices
                </p>
              </>
            )}
          </div>

          {/* Sample questions */}
          {docInfo && (
            <div style={s.sampleBox}>
              <p style={s.sampleTitle}>Try asking:</p>
              {[
                "What was the court's final decision?",
                "What sections of IPC were cited?",
                "Who are the parties involved?",
                "What was the main legal issue?",
                "What evidence was considered?"
              ].map(q => (
                <button
                  key={q}
                  style={s.sampleBtn}
                  onClick={() => { setQuestion(q); }}
                >
                  {q}
                </button>
              ))}
            </div>
          )}

          {/* Summary */}
          {summary && (
            <div style={s.summaryBox}>
              <p style={s.sampleTitle}>Document Summary</p>
              <div style={{ fontSize: "0.8rem", color: "#8b949e", lineHeight: 1.6 }}>
                <ReactMarkdown>{summary}</ReactMarkdown>
              </div>
            </div>
          )}
        </div>

        {/* Right: Chat */}
        <div style={s.chatArea}>
          {/* Messages */}
          <div style={s.messages}>
            {messages.length === 0 && (
              <div style={s.emptyState}>
                <p>Upload a legal document to get started.</p>
                <p style={{ fontSize: "0.8rem", color: "#484f58", marginTop: "0.5rem" }}>
                  Powered by RAG: ChromaDB + sentence-transformers + LLaMA 3
                </p>
              </div>
            )}
            {messages.map((msg, i) => (
              <div key={i} style={msg.role === "user" ? s.userMsg : s.assistantMsg}>
                <div style={s.msgRole}>
                  {msg.role === "user" ? "You" : "⚖️ Legal Assistant"}
                </div>
                <div style={s.msgContent}>
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                </div>
                {msg.sources && (
                  <div style={{ marginTop: "0.5rem" }}>
                    <button
                      style={s.sourceToggle}
                      onClick={() => setShowSources(showSources === i ? null : i)}
                    >
                      {showSources === i ? "Hide" : "Show"} {msg.sources.length} sources
                    </button>
                    {showSources === i && (
                      <div style={{ marginTop: "0.5rem" }}>
                        {msg.sources.map((src, j) => (
                          <div key={j} style={s.sourceCard}>
                            <div style={s.sourceHeader}>
                              <span>Excerpt {src.rank}</span>
                              <span style={{ color: "#3fb950" }}>
                                {Math.round(src.similarity * 100)}% match
                              </span>
                            </div>
                            <p style={s.sourceText}>{src.content}</p>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
            {loading && (
              <div style={s.assistantMsg}>
                <div style={s.msgRole}>⚖️ Legal Assistant</div>
                <div style={{ color: "#8b949e", fontSize: "0.9rem" }}>
                  Searching document and generating answer...
                </div>
              </div>
            )}
          </div>

          {/* Input */}
          <div style={s.inputArea}>
            <input
              style={s.input}
              value={question}
              onChange={e => setQuestion(e.target.value)}
              onKeyDown={e => e.key === "Enter" && !e.shiftKey && askQuestion()}
              placeholder={docInfo ? "Ask a question about the document..." : "Upload a document first"}
              disabled={!docInfo || loading}
            />
            <button
              style={{
                ...s.sendBtn,
                opacity: (!docInfo || loading || !question.trim()) ? 0.5 : 1
              }}
              onClick={askQuestion}
              disabled={!docInfo || loading || !question.trim()}
            >
              Ask
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  page: { minHeight: "100vh", background: "#0d1117", color: "#e6edf3", fontFamily: "system-ui, sans-serif", display: "flex", flexDirection: "column" },
  header: { padding: "1.5rem 2rem", borderBottom: "1px solid #21262d", display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "1rem" },
  title: { fontSize: "1.5rem", fontWeight: 700, marginBottom: "0.3rem" },
  subtitle: { color: "#8b949e", fontSize: "0.9rem" },
  docBadge: { background: "#161b22", border: "1px solid #30363d", borderRadius: "8px", padding: "0.5rem 1rem", fontSize: "0.85rem", display: "flex", alignItems: "center", gap: "0.5rem" },
  chunkCount: { background: "#21262d", borderRadius: "20px", padding: "0.1rem 0.6rem", fontSize: "0.75rem", color: "#8b949e" },
  body: { display: "flex", flex: 1, overflow: "hidden", height: "calc(100vh - 80px)" },
  sidebar: { width: "320px", borderRight: "1px solid #21262d", padding: "1rem", overflowY: "auto", display: "flex", flexDirection: "column", gap: "1rem" },
  uploadBox: { border: "2px dashed #30363d", borderRadius: "10px", padding: "2rem", textAlign: "center", cursor: "pointer", transition: "border-color 0.2s" },
  sampleBox: { background: "#161b22", border: "1px solid #21262d", borderRadius: "10px", padding: "1rem" },
  sampleTitle: { fontSize: "0.75rem", fontWeight: 600, color: "#8b949e", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "0.7rem" },
  sampleBtn: { display: "block", width: "100%", background: "transparent", border: "1px solid #21262d", borderRadius: "6px", padding: "0.5rem 0.8rem", color: "#58a6ff", fontSize: "0.8rem", cursor: "pointer", textAlign: "left", marginBottom: "0.4rem" },
  summaryBox: { background: "#161b22", border: "1px solid #21262d", borderRadius: "10px", padding: "1rem" },
  chatArea: { flex: 1, display: "flex", flexDirection: "column" },
  messages: { flex: 1, overflowY: "auto", padding: "1.5rem", display: "flex", flexDirection: "column", gap: "1rem" },
  emptyState: { textAlign: "center", color: "#8b949e", marginTop: "4rem" },
  userMsg: { background: "#161b22", border: "1px solid #21262d", borderRadius: "10px", padding: "1rem", alignSelf: "flex-end", maxWidth: "80%" },
  assistantMsg: { background: "#0d1117", border: "1px solid #21262d", borderRadius: "10px", padding: "1rem", maxWidth: "90%" },
  msgRole: { fontSize: "0.75rem", fontWeight: 600, color: "#8b949e", marginBottom: "0.4rem" },
  msgContent: { fontSize: "0.9rem", lineHeight: 1.6 },
  sourceToggle: { background: "transparent", border: "1px solid #30363d", borderRadius: "6px", padding: "0.3rem 0.8rem", color: "#8b949e", fontSize: "0.75rem", cursor: "pointer" },
  sourceCard: { background: "#161b22", border: "1px solid #21262d", borderRadius: "8px", padding: "0.8rem", marginBottom: "0.5rem" },
  sourceHeader: { display: "flex", justifyContent: "space-between", fontSize: "0.75rem", color: "#8b949e", marginBottom: "0.4rem" },
  sourceText: { fontSize: "0.8rem", color: "#8b949e", lineHeight: 1.5 },
  inputArea: { padding: "1rem 1.5rem", borderTop: "1px solid #21262d", display: "flex", gap: "0.8rem" },
  input: { flex: 1, background: "#161b22", border: "1px solid #30363d", borderRadius: "8px", padding: "0.7rem 1rem", color: "#e6edf3", fontSize: "0.9rem", outline: "none" },
  sendBtn: { background: "#238636", color: "#fff", border: "none", borderRadius: "8px", padding: "0.7rem 1.5rem", cursor: "pointer", fontWeight: 600, fontSize: "0.9rem" }
};