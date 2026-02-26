import { useState, useCallback, useMemo } from "react";
import type { Entry } from "./types";
import { emptyEntry } from "./types";
import { validateEntries, prepareForExport } from "./validation";
import { EntryCard } from "./components/EntryCard";
// import { LaunchDialog } from "./components/LaunchDialog";
import {
  Plus, Download, Upload, Code2, AlertCircle, CheckCircle2,
  Dna, Ban,
} from "lucide-react";

type Toast = { id: number; type: "success" | "error"; message: string };
let toastId = 0;

export default function App() {
  const [entries, setEntries] = useState<Entry[]>([emptyEntry()]);
  const [showPreview, setShowPreview] = useState(false);
  const [toasts, setToasts] = useState<Toast[]>([]);

  // Live validation — re-runs whenever entries change
  const errors = useMemo(() => validateEntries(entries), [entries]);

  const toast = (type: Toast["type"], message: string) => {
    const id = ++toastId;
    setToasts((t) => [...t, { id, type, message }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 3500);
  };

  const updateEntry = useCallback((idx: number, patch: Partial<Entry>) => {
    setEntries((prev) => prev.map((e, i) => (i === idx ? { ...e, ...patch } : e)));
  }, []);

  const removeEntry = useCallback((idx: number) => {
    setEntries((prev) => prev.filter((_, i) => i !== idx));
  }, []);

  const isValid = errors.length === 0;

  const handleDownload = () => {
    const data = prepareForExport(entries);
    const blob = new Blob([JSON.stringify(data, null, 4)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "cryozeta_input.json";
    a.click();
    URL.revokeObjectURL(url);
    toast("success", "JSON downloaded");
  };

  const handleImport = () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".json";
    input.onchange = (ev) => {
      const file = (ev.target as HTMLInputElement).files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const data = JSON.parse(reader.result as string);
          if (Array.isArray(data)) {
            setEntries(data);
            toast("success", `Loaded ${data.length} entries from ${file.name}`);
          } else {
            toast("error", "JSON root must be an array of entries");
          }
        } catch {
          toast("error", "Failed to parse JSON file");
        }
      };
      reader.readAsText(file);
    };
    input.click();
  };

  const exportJson = prepareForExport(entries);
  const totalEntities = entries.reduce((n, e) => n + e.sequences.length, 0);

  return (
    <div className="min-h-screen bg-background">
      {/* ── Header ── */}
      <header className="sticky top-0 z-50 border-b bg-gradient-to-r from-slate-900 via-slate-800 to-slate-900 text-white shadow-lg">
        <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-6">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-white/10">
              <Dna className="h-4 w-4" />
            </div>
            <div>
              <h1 className="text-sm font-semibold tracking-tight">CryoZeta</h1>
              <p className="text-[10px] text-slate-400">Input JSON Builder</p>
            </div>
          </div>

          {/* Stats */}
          <div className="hidden sm:flex items-center gap-4 text-xs text-slate-400">
            <span>{entries.length} {entries.length === 1 ? "entry" : "entries"}</span>
            <span className="h-3 w-px bg-slate-700" />
            <span>{totalEntities} {totalEntities === 1 ? "entity" : "entities"}</span>
          </div>

          {/* Actions */}
          <div className="flex items-center gap-1.5">
            <button onClick={handleImport}
              className="btn-ghost btn-sm text-slate-300 hover:text-white hover:bg-white/10 rounded-lg">
              <Upload className="h-3.5 w-3.5" /> Import
            </button>
            <button onClick={() => setShowPreview((p) => !p)}
              className={`btn-ghost btn-sm rounded-lg ${showPreview ? "text-white bg-white/10" : "text-slate-300 hover:text-white hover:bg-white/10"}`}>
              <Code2 className="h-3.5 w-3.5" /> JSON
            </button>
            <div className="mx-1 h-6 w-px bg-slate-700" />
            <button onClick={handleDownload}
              disabled={!isValid}
              className="btn-sm rounded-lg bg-white text-slate-900 font-semibold hover:bg-slate-100 shadow-sm active:scale-[0.97] transition-all
                         disabled:opacity-40 disabled:cursor-not-allowed inline-flex items-center gap-2"
              title={isValid ? "Download JSON" : "Fix validation errors first"}>
              {isValid ? <Download className="h-3.5 w-3.5" /> : <Ban className="h-3.5 w-3.5" />} Download
            </button>
            {/* <LaunchDialog entries={entries} /> */}
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-8">
        {/* ── Validation errors ── */}
        {/* Live validation status */}
        {errors.length > 0 ? (
          <div className="mb-6 rounded-xl border border-red-200 bg-red-50 p-4">
            <div className="flex items-center gap-2 mb-2">
              <AlertCircle className="h-4 w-4 text-red-500" />
              <span className="text-sm font-semibold text-red-700">
                {errors.length} {errors.length === 1 ? "issue" : "issues"} found
              </span>
            </div>
            <ul className="space-y-0.5">
              {errors.map((e, i) => (
                <li key={i} className="ml-6 list-disc text-sm text-red-600">{e}</li>
              ))}
            </ul>
          </div>
        ) : entries.some((e) => e.name || e.sequences.length > 0) ? (
          <div className="mb-6 rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4 text-emerald-500" />
            <span className="text-sm font-medium text-emerald-700">
              All {entries.length} {entries.length === 1 ? "entry" : "entries"} valid — ready to download
            </span>
          </div>
        ) : null}

        <div className={`grid gap-8 ${showPreview ? "lg:grid-cols-[1fr_420px]" : ""}`}>
          {/* ── Entry cards ── */}
          <div className="space-y-5">
            {entries.map((entry, idx) => (
              <EntryCard
                key={idx}
                index={idx}
                total={entries.length}
                entry={entry}
                onUpdate={(patch) => updateEntry(idx, patch)}
                onRemove={entries.length > 1 ? () => removeEntry(idx) : undefined}
              />
            ))}

            {/* Add entry */}
            <button
              onClick={() => setEntries((prev) => [...prev, emptyEntry()])}
              className="group flex w-full items-center justify-center gap-2 rounded-xl border-2 border-dashed border-slate-200 py-6 text-sm font-medium text-muted-foreground hover:border-primary/40 hover:text-primary hover:bg-primary/[0.02] transition-all"
            >
              <Plus className="h-4 w-4 transition-transform group-hover:scale-110" />
              Add Entry
            </button>
          </div>

          {/* ── JSON preview ── */}
          {showPreview && (
            <div className="sticky top-20 h-[calc(100vh-7rem)]">
              <div className="card h-full flex flex-col">
                <div className="flex items-center justify-between border-b px-4 py-3">
                  <div className="flex items-center gap-2">
                    <Code2 className="h-4 w-4 text-muted-foreground" />
                    <span className="text-sm font-semibold">JSON Preview</span>
                  </div>
                  <span className="badge bg-slate-100 text-slate-600">
                    {JSON.stringify(exportJson, null, 2).split("\n").length} lines
                  </span>
                </div>
                <pre className="flex-1 overflow-auto p-4 text-[11px] leading-relaxed font-mono text-slate-600 selection:bg-blue-100">
                  {JSON.stringify(exportJson, null, 2)}
                </pre>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* ── Toasts ── */}
      <div className="fixed bottom-6 right-6 z-50 flex flex-col gap-2">
        {toasts.map((t) => (
          <div
            key={t.id}
            className={`animate-in flex items-center gap-2 rounded-lg px-4 py-2.5 text-sm font-medium shadow-lg ${
              t.type === "success"
                ? "bg-emerald-600 text-white"
                : "bg-red-600 text-white"
            }`}
          >
            {t.type === "success" ? <CheckCircle2 className="h-4 w-4" /> : <AlertCircle className="h-4 w-4" />}
            {t.message}
          </div>
        ))}
      </div>
    </div>
  );
}
