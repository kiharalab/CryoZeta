import { useState, useRef, useEffect, useCallback } from "react";
import type { Entry } from "../types";
import { validateEntries, prepareForExport } from "../validation";
import { Rocket, X, Loader2, Terminal, Square } from "lucide-react";

interface LaunchDialogProps {
  entries: Entry[];
}

interface LaunchConfig {
  json_path: string;
  output_dir: string;
  gpu: string;
  seeds: string;
  n_sample: number;
  n_step: number;
  n_cycle: number;
  num_select: number;
  interpolation: boolean;
  skip_detection: boolean;
  skip_inference: boolean;
  skip_combine: boolean;
}

export function LaunchDialog({ entries }: LaunchDialogProps) {
  const [open, setOpen] = useState(false);
  const [launching, setLaunching] = useState(false);
  const [result, setResult] = useState<{ ok: boolean; message: string } | null>(null);
  const [config, setConfig] = useState<LaunchConfig>({
    json_path: "",
    output_dir: "",
    gpu: "0",
    seeds: "101",
    n_sample: 5,
    n_step: 20,
    n_cycle: 10,
    num_select: 5,
    interpolation: false,
    skip_detection: false,
    skip_inference: false,
    skip_combine: false,
  });

  // Log viewer state
  const [sessionName, setSessionName] = useState<string | null>(null);
  const [logLines, setLogLines] = useState<string[]>([]);
  const [jobRunning, setJobRunning] = useState(false);
  const logEndRef = useRef<HTMLDivElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  const errors = validateEntries(entries);
  const canLaunch = errors.length === 0 && config.json_path && config.output_dir && config.gpu;

  // Auto-scroll to bottom when new log lines arrive
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logLines]);

  // Clean up EventSource on unmount or close
  const closeEventSource = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
  }, []);

  useEffect(() => {
    return closeEventSource;
  }, [closeEventSource]);

  const connectToLogs = useCallback((session: string) => {
    closeEventSource();
    setLogLines([]);
    setJobRunning(true);

    const es = new EventSource(`/api/logs?session=${encodeURIComponent(session)}`);
    eventSourceRef.current = es;

    es.onmessage = (event) => {
      setLogLines((prev) => [...prev, event.data]);
    };

    es.addEventListener("done", () => {
      setJobRunning(false);
      es.close();
      eventSourceRef.current = null;
    });

    es.onerror = () => {
      setJobRunning(false);
      es.close();
      eventSourceRef.current = null;
    };
  }, [closeEventSource]);

  const handleLaunch = async () => {
    setLaunching(true);
    setResult(null);
    try {
      const data = prepareForExport(entries);
      const res = await fetch("/api/launch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ entries: data, config }),
      });
      const body = await res.json();
      if (res.ok) {
        setResult({ ok: true, message: body.message || "Job launched" });
        const session = body.session;
        if (session) {
          setSessionName(session);
          connectToLogs(session);
        }
      } else {
        setResult({ ok: false, message: body.error || "Launch failed" });
      }
    } catch (e: any) {
      setResult({ ok: false, message: e.message || "Network error" });
    } finally {
      setLaunching(false);
    }
  };

  const handleTerminate = async () => {
    if (!sessionName) return;
    try {
      await fetch("/api/terminate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session: sessionName }),
      });
    } catch {
      // SSE will detect the session ending regardless
    }
  };

  const handleClose = () => {
    closeEventSource();
    setSessionName(null);
    setLogLines([]);
    setJobRunning(false);
    setResult(null);
    setOpen(false);
  };

  const patch = (p: Partial<LaunchConfig>) => setConfig((c) => ({ ...c, ...p }));

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        className="btn-sm rounded-lg bg-emerald-600 text-white font-semibold hover:bg-emerald-500 shadow-sm active:scale-[0.97] transition-all"
      >
        <Rocket className="h-3.5 w-3.5" /> Launch
      </button>
    );
  }

  // Log viewer mode (after successful launch)
  if (sessionName) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center">
        <div className="absolute inset-0 bg-black/50" onClick={handleClose} />

        <div className="relative z-10 w-full max-w-2xl rounded-xl border bg-card shadow-2xl flex flex-col" style={{ maxHeight: "80vh" }}>
          {/* Header */}
          <div className="flex items-center justify-between border-b px-5 py-4">
            <div className="flex items-center gap-2.5">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-slate-800">
                <Terminal className="h-4 w-4 text-emerald-400" />
              </div>
              <div>
                <h3 className="text-sm font-semibold">Job Output</h3>
                <p className="text-[11px] text-muted-foreground font-mono">{sessionName}</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              {/* Status badge */}
              {jobRunning ? (
                <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-100 px-2.5 py-0.5 text-xs font-medium text-emerald-700">
                  <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
                  Running
                </span>
              ) : (
                <span className="inline-flex items-center gap-1.5 rounded-full bg-slate-100 px-2.5 py-0.5 text-xs font-medium text-slate-600">
                  <span className="h-1.5 w-1.5 rounded-full bg-slate-400" />
                  Finished
                </span>
              )}
              <button onClick={handleClose} className="btn-icon text-slate-400 hover:text-slate-600">
                <X className="h-4 w-4" />
              </button>
            </div>
          </div>

          {/* Terminal panel */}
          <div className="flex-1 overflow-y-auto bg-slate-900 px-4 py-3 font-mono text-xs leading-5 text-emerald-400 min-h-[300px]">
            {logLines.length === 0 && jobRunning && (
              <div className="text-slate-500">Waiting for output...</div>
            )}
            {logLines.map((line, i) => (
              <div key={i} className="whitespace-pre-wrap break-all">
                {line}
              </div>
            ))}
            <div ref={logEndRef} />
          </div>

          {/* Footer */}
          <div className="flex items-center justify-end gap-2 border-t px-5 py-3">
            {jobRunning && (
              <button
                onClick={handleTerminate}
                className="btn-sm rounded-lg bg-red-600 text-white font-semibold hover:bg-red-500
                           shadow-sm transition-all inline-flex items-center gap-1.5"
              >
                <Square className="h-3 w-3 fill-current" /> Terminate
              </button>
            )}
            <button onClick={handleClose} className="btn-outline btn-sm rounded-lg">
              Close
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/50" onClick={handleClose} />

      <div className="relative z-10 w-full max-w-lg rounded-xl border bg-card shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b px-5 py-4">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-100">
              <Terminal className="h-4 w-4 text-emerald-700" />
            </div>
            <div>
              <h3 className="text-sm font-semibold">Launch CryoZeta Job</h3>
              <p className="text-[11px] text-muted-foreground">Runs in a new tmux session</p>
            </div>
          </div>
          <button onClick={handleClose} className="btn-icon text-slate-400 hover:text-slate-600">
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="px-5 py-4 space-y-4 max-h-[70vh] overflow-y-auto">
          {/* Validation */}
          {errors.length > 0 && (
            <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-600">
              Fix {errors.length} validation {errors.length === 1 ? "error" : "errors"} before launching
            </div>
          )}

          {/* Required fields */}
          <div className="space-y-3">
            <h4 className="text-xs font-bold uppercase tracking-widest text-slate-400">Required</h4>
            <div>
              <label className="label-muted mb-1 block">Input JSON save path</label>
              <input type="text" value={config.json_path}
                onChange={(e) => patch({ json_path: e.target.value })}
                placeholder="/path/to/save/input.json"
                className="input" />
            </div>
            <div>
              <label className="label-muted mb-1 block">Output directory</label>
              <input type="text" value={config.output_dir}
                onChange={(e) => patch({ output_dir: e.target.value })}
                placeholder="/path/to/output"
                className="input" />
            </div>
            <div>
              <label className="label-muted mb-1 block">GPU ID</label>
              <input type="text" value={config.gpu}
                onChange={(e) => patch({ gpu: e.target.value })}
                placeholder="0"
                className="input w-24" />
            </div>
          </div>

          {/* Advanced */}
          <details>
            <summary className="cursor-pointer text-xs font-bold uppercase tracking-widest text-slate-400 hover:text-slate-600">
              Advanced Options
            </summary>
            <div className="mt-3 space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="label-muted mb-1 block">Seeds</label>
                  <input type="text" value={config.seeds}
                    onChange={(e) => patch({ seeds: e.target.value })}
                    className="input" />
                </div>
                <div>
                  <label className="label-muted mb-1 block">N_sample</label>
                  <input type="number" min={1} value={config.n_sample}
                    onChange={(e) => patch({ n_sample: parseInt(e.target.value) || 5 })}
                    className="input" />
                </div>
                <div>
                  <label className="label-muted mb-1 block">N_step</label>
                  <input type="number" min={1} value={config.n_step}
                    onChange={(e) => patch({ n_step: parseInt(e.target.value) || 20 })}
                    className="input" />
                </div>
                <div>
                  <label className="label-muted mb-1 block">N_cycle</label>
                  <input type="number" min={1} value={config.n_cycle}
                    onChange={(e) => patch({ n_cycle: parseInt(e.target.value) || 10 })}
                    className="input" />
                </div>
                <div>
                  <label className="label-muted mb-1 block">num_select</label>
                  <input type="number" min={1} value={config.num_select}
                    onChange={(e) => patch({ num_select: parseInt(e.target.value) || 5 })}
                    className="input" />
                </div>
              </div>

              {/* Toggles */}
              <div className="space-y-2">
                <label className="flex items-center gap-2 text-sm">
                  <input type="checkbox" checked={config.interpolation}
                    onChange={(e) => patch({ interpolation: e.target.checked })}
                    className="rounded" />
                  Use interpolation model
                </label>
                <label className="flex items-center gap-2 text-sm">
                  <input type="checkbox" checked={config.skip_detection}
                    onChange={(e) => patch({ skip_detection: e.target.checked })}
                    className="rounded" />
                  Skip detection (Stage 1)
                </label>
                <label className="flex items-center gap-2 text-sm">
                  <input type="checkbox" checked={config.skip_inference}
                    onChange={(e) => patch({ skip_inference: e.target.checked })}
                    className="rounded" />
                  Skip inference (Stage 2)
                </label>
                <label className="flex items-center gap-2 text-sm">
                  <input type="checkbox" checked={config.skip_combine}
                    onChange={(e) => patch({ skip_combine: e.target.checked })}
                    className="rounded" />
                  Skip combine (Stage 3)
                </label>
              </div>
            </div>
          </details>

          {/* Result */}
          {result && (
            <div className={`rounded-lg border p-3 text-sm ${
              result.ok
                ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                : "border-red-200 bg-red-50 text-red-700"
            }`}>
              {result.message}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-2 border-t px-5 py-3">
          <button onClick={handleClose} className="btn-outline btn-sm rounded-lg">
            Cancel
          </button>
          <button
            onClick={handleLaunch}
            disabled={!canLaunch || launching}
            className="btn-sm rounded-lg bg-emerald-600 text-white font-semibold hover:bg-emerald-500
                       disabled:opacity-50 disabled:cursor-not-allowed shadow-sm transition-all
                       inline-flex items-center gap-2"
          >
            {launching ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Rocket className="h-3.5 w-3.5" />}
            {launching ? "Launching..." : "Launch in tmux"}
          </button>
        </div>
      </div>
    </div>
  );
}
