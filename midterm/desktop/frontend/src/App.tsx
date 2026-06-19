import { useEffect, useRef, useState } from "react";
import {
  Checkpoints,
  Health,
  LoadCheckpoint,
  Predict,
} from "../wailsjs/go/main/App";
import { main } from "../wailsjs/go/models";
import { formatProb, stripDataUrl } from "./lib";
import "./App.css";

type Phase = "loading" | "ready" | "predicting" | "switching" | "failed";

function App() {
  const [phase, setPhase] = useState<Phase>("loading");
  const [checkpoints, setCheckpoints] = useState<string[]>([]);
  const [current, setCurrent] = useState("");
  const [imageB64, setImageB64] = useState("");
  const [preview, setPreview] = useState("");
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<main.PredictResp | null>(null);
  const [error, setError] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  // Poll /health khi khởi động cho tới khi model sẵn sàng (tối đa ~90s)
  useEffect(() => {
    let cancelled = false;
    let attempts = 0;
    const MAX_ATTEMPTS = 90;
    const poll = async () => {
      try {
        const h = await Health();
        if (cancelled) return;
        if (h.ready) {
          const c = await Checkpoints();
          setCheckpoints(c.checkpoints);
          setCurrent(c.current);
          setPhase("ready");
          return;
        }
      } catch {
        /* sidecar chưa lên — thử lại */
      }
      if (cancelled) return;
      attempts += 1;
      if (attempts >= MAX_ATTEMPTS) {
        setError(
          "Không khởi động được model sau 90 giây. Kiểm tra .venv ở repo root và checkpoint trong midterm/checkpoints/, và xem log terminal."
        );
        setPhase("failed");
        return;
      }
      setTimeout(poll, 1000);
    };
    poll();
    return () => {
      cancelled = true;
    };
  }, []);

  const onFile = (f: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result as string;
      setPreview(dataUrl);
      setImageB64(stripDataUrl(dataUrl));
      setResult(null);
    };
    reader.readAsDataURL(f);
  };

  const onRun = async () => {
    if (!imageB64 || !question.trim()) return;
    setPhase("predicting");
    setError("");
    try {
      setResult(await Predict(imageB64, question));
    } catch (e) {
      setError(String(e));
    } finally {
      setPhase("ready");
    }
  };

  const onSwitch = async (name: string) => {
    if (name === current) return;
    setPhase("switching");
    setResult(null);
    setError("");
    try {
      const h = await LoadCheckpoint(name);
      setCurrent(h.checkpoint);
    } catch (e) {
      setError(String(e));
    } finally {
      setPhase("ready");
    }
  };

  const busy = phase === "loading" || phase === "predicting" || phase === "switching" || phase === "failed";
  const canRun = phase === "ready" && !!imageB64 && !!question.trim();
  const top1 = result?.answers?.[0];

  return (
    <div className="app">
      <header className="topbar">
        <h1>MedVQA Demo — VQA-RAD</h1>
        <div className="status">
          <label>
            Checkpoint:&nbsp;
            <select
              value={current}
              disabled={busy}
              onChange={(e) => onSwitch(e.target.value)}
            >
              {checkpoints.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </label>
          <span className={`dot ${phase === "ready" ? "ok" : phase === "failed" ? "fail" : "wait"}`} />
          <span>
            {phase === "loading" && "Đang tải model…"}
            {phase === "switching" && "Đang đổi checkpoint…"}
            {phase === "predicting" && "Đang suy luận…"}
            {phase === "ready" && "Sẵn sàng"}
            {phase === "failed" && "Không khởi động được"}
          </span>
        </div>
      </header>

      <main className="grid">
        <section className="left">
          <div
            className="dropzone"
            onClick={() => fileRef.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              const f = e.dataTransfer.files?.[0];
              if (f) onFile(f);
            }}
          >
            {preview ? (
              <img src={preview} alt="preview" />
            ) : (
              <span>Kéo-thả hoặc bấm để chọn ảnh</span>
            )}
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              hidden
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) onFile(f);
              }}
            />
          </div>

          <label className="field">
            Câu hỏi (tiếng Anh):
            <input
              type="text"
              value={question}
              placeholder="is there cardiomegaly?"
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && onRun()}
            />
          </label>

          <button className="run" disabled={!canRun} onClick={onRun}>
            {phase === "predicting" ? "Đang chạy…" : "Run"}
          </button>
          {error && <p className="error">{error}</p>}
        </section>

        <section className="right">
          {top1 && (
            <div className="answer">
              <span className="label">Đáp án</span>
              <span className="value">{top1.answer}</span>
            </div>
          )}
          {result && (
            <div className="top5">
              <span className="label">Top-5</span>
              {result.answers.map((a) => (
                <div className="bar-row" key={a.answer}>
                  <span className="name">{a.answer}</span>
                  <div className="bar">
                    <div className="fill" style={{ width: `${a.prob * 100}%` }} />
                  </div>
                  <span className="pct">{formatProb(a.prob)}</span>
                </div>
              ))}
            </div>
          )}
          {result &&
            (result.has_attention && result.heatmap ? (
              <div className="heatmap">
                <span className="label">Attention heatmap</span>
                <img src={`data:image/png;base64,${result.heatmap}`} alt="heatmap" />
              </div>
            ) : (
              <p className="note">
                Checkpoint “{current}” không có attention để vẽ heatmap.
              </p>
            ))}
        </section>
      </main>
    </div>
  );
}

export default App;
