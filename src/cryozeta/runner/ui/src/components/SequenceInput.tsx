import { useState, useRef, useEffect } from "react";
import { Upload } from "lucide-react";

interface SequenceInputProps {
  value: string;
  onChange: (seq: string) => void;
  onFasta: () => void;
  placeholder?: string;
  label: string;
  status: React.ReactNode;
  count: number;
  onCountChange: (n: number) => void;
}

const BLOCK = 10;

function buildRuler(startPos: number, lineLen: number): string {
  const blocks: string[] = [];
  for (let b = 0; b < Math.ceil(lineLen / BLOCK); b++) {
    const blockStart = b * BLOCK;
    const blockEnd = Math.min(blockStart + BLOCK, lineLen);
    const blockLen = blockEnd - blockStart;
    const chars = new Array(blockLen).fill(" ");

    for (let i = 0; i < blockLen; i++) {
      const pos = startPos + blockStart + i;
      if (pos % 10 === 0) {
        const numStr = String(pos);
        for (let j = 0; j < numStr.length; j++) {
          const idx = i - (numStr.length - 1) + j;
          if (idx >= 0) chars[idx] = numStr[j];
        }
      }
    }
    blocks.push(chars.join(""));
  }
  return blocks.join(" ");
}

function blockify(line: string): string {
  const parts: string[] = [];
  for (let i = 0; i < line.length; i += BLOCK) {
    parts.push(line.slice(i, i + BLOCK));
  }
  return parts.join(" ");
}

function SequenceViewer({ sequence, onFocus }: { sequence: string; onFocus: () => void }) {
  const clean = sequence.replace(/\s+/g, "").toUpperCase();
  const containerRef = useRef<HTMLDivElement>(null);
  const [lineWidth, setLineWidth] = useState(60);

  // Measure how many characters fit per line based on container width
  useEffect(() => {
    if (!containerRef.current) return;

    const measure = () => {
      const el = containerRef.current;
      if (!el) return;
      // available width minus padding (p-3 = 12px each side)
      const available = el.clientWidth - 24;
      // measure 1ch in the monospace font
      const probe = document.createElement("span");
      probe.style.fontFamily = "'JetBrains Mono', 'Fira Code', monospace";
      probe.style.fontSize = "11px";
      probe.style.position = "absolute";
      probe.style.visibility = "hidden";
      probe.textContent = "X";
      document.body.appendChild(probe);
      const chWidth = probe.getBoundingClientRect().width;
      document.body.removeChild(probe);

      // each block = BLOCK chars + 1 space (except last), so N blocks = N*BLOCK chars + (N-1) spaces
      // total chars for N blocks = N*BLOCK + (N-1) = N*(BLOCK+1) - 1
      // N*(BLOCK+1) - 1 <= available/chWidth  =>  N <= (available/chWidth + 1) / (BLOCK+1)
      const maxBlocks = Math.floor((available / chWidth + 1) / (BLOCK + 1));
      const blocks = Math.max(1, maxBlocks);
      setLineWidth(blocks * BLOCK);
    };

    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  if (!clean) return null;

  const lines: string[] = [];
  for (let i = 0; i < clean.length; i += lineWidth) {
    lines.push(clean.slice(i, i + lineWidth));
  }

  return (
    <div
      ref={containerRef}
      className="rounded-lg border bg-slate-50/80 p-3 overflow-x-auto max-h-60 overflow-y-auto cursor-text"
      onClick={onFocus}
    >
      <div className="font-mono text-[11px] leading-[1.1] whitespace-pre">
        {lines.map((line, li) => {
          const startPos = li * lineWidth + 1;
          const ruler = buildRuler(startPos, line.length);
          return (
            <div key={li} className={li > 0 ? "mt-2.5" : ""}>
              <div className="text-slate-400 select-none">{ruler}</div>
              <div className="text-slate-700 select-all">{blockify(line)}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function SequenceInput({
  value, onChange, onFasta, placeholder, label, status,
  count, onCountChange,
}: SequenceInputProps) {
  const clean = value.replace(/\s+/g, "");
  const [focused, setFocused] = useState(false);

  return (
    <div>
      {/* Header row */}
      <div className="flex items-center justify-between mb-1.5">
        <label className="label-muted">{label}</label>
        <div className="flex items-center gap-2">
          {status}
          <span className="text-[10px] text-slate-400">copies:</span>
          <input
            type="number" min={1} value={count}
            onChange={(e) => onCountChange(parseInt(e.target.value) || 1)}
            className="w-14 rounded border border-input bg-background px-1.5 py-0.5 text-xs text-center
                       focus:outline-none focus:ring-1 focus:ring-ring/20"
          />
          <button onClick={onFasta}
            className="btn-ghost h-6 px-1.5 text-[10px] text-slate-400 hover:text-slate-700 rounded">
            <Upload className="h-3 w-3" /> FASTA
          </button>
        </div>
      </div>

      {/* Viewer or editor */}
      {!focused && clean ? (
        <SequenceViewer sequence={value} onFocus={() => setFocused(true)} />
      ) : (
        <textarea
          autoFocus={focused}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          rows={3}
          className="textarea"
          onBlur={() => setFocused(false)}
        />
      )}
    </div>
  );
}
