import { useState } from "react";
import type { Entry, SequenceObj, SeqType, CovalentBond } from "../types";
import { SEQ_TYPES, TYPE_LABELS, TYPE_BADGE, emptySequence } from "../types";
import { EntityCard } from "./EntityCard";
import {
  ChevronDown, Trash2, Plus, Link2, MapPin, Hash, Layers,
} from "lucide-react";

interface EntryCardProps {
  index: number;
  total: number;
  entry: Entry;
  onUpdate: (patch: Partial<Entry>) => void;
  onRemove?: () => void;
}

export function EntryCard({ index, total, entry, onUpdate, onRemove }: EntryCardProps) {
  const [expanded, setExpanded] = useState(true);

  const updateSequence = (seqIdx: number, s: SequenceObj) => {
    const seqs = [...entry.sequences];
    seqs[seqIdx] = s;
    onUpdate({ sequences: seqs });
  };

  const removeSequence = (seqIdx: number) => {
    onUpdate({ sequences: entry.sequences.filter((_, i) => i !== seqIdx) });
  };

  const addSequence = (t: SeqType) => {
    onUpdate({ sequences: [...entry.sequences, emptySequence(t)] });
  };

  const updateBond = (bi: number, patch: Partial<CovalentBond>) => {
    const bonds = [...(entry.covalent_bonds || [])];
    bonds[bi] = { ...bonds[bi], ...patch };
    onUpdate({ covalent_bonds: bonds });
  };

  const removeBond = (bi: number) => {
    onUpdate({ covalent_bonds: (entry.covalent_bonds || []).filter((_, i) => i !== bi) });
  };

  const addBond = () => {
    onUpdate({
      covalent_bonds: [
        ...(entry.covalent_bonds || []),
        { left_entity: 1, left_position: 1, left_atom: "", right_entity: 1, right_position: 1, right_atom: "" },
      ],
    });
  };

  return (
    <div className="card overflow-hidden">
      {/* ── Header ── */}
      <div
        className="flex items-center gap-3 px-5 py-3.5 cursor-pointer select-none hover:bg-slate-50/80 transition-colors"
        onClick={() => setExpanded((p) => !p)}
      >
        <ChevronDown className={`h-4 w-4 text-slate-400 transition-transform duration-200 ${expanded ? "" : "-rotate-90"}`} />

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-foreground">
              {entry.name || `Entry ${index + 1}`}
            </span>
            {total > 1 && (
              <span className="badge bg-slate-100 text-slate-500 text-[10px]">
                {index + 1}/{total}
              </span>
            )}
          </div>
          {!expanded && entry.sequences.length > 0 && (
            <div className="flex items-center gap-1.5 mt-1">
              {entry.sequences.map((_, si) => {
                const key = Object.keys(entry.sequences[si])[0] as SeqType;
                return (
                  <span key={si} className={`badge text-[9px] py-0 ${TYPE_BADGE[key]}`}>
                    {TYPE_LABELS[key]}
                  </span>
                );
              })}
            </div>
          )}
        </div>

        <span className="text-xs text-muted-foreground tabular-nums">
          {entry.sequences.length} {entry.sequences.length === 1 ? "entity" : "entities"}
        </span>

        {onRemove && (
          <button
            onClick={(e) => { e.stopPropagation(); onRemove(); }}
            className="btn-icon text-slate-300 hover:text-red-500 hover:bg-red-50"
          >
            <Trash2 className="h-4 w-4" />
          </button>
        )}
      </div>

      {expanded && (
        <div className="border-t">
          {/* ── Entry fields ── */}
          <div className="bg-slate-50/50 px-5 py-4">
            <div className="grid gap-4 sm:grid-cols-4">
              {/* Name */}
              <div className="sm:col-span-1">
                <label className="label-muted mb-1.5 flex items-center gap-1.5">
                  <Hash className="h-3 w-3" /> Name
                </label>
                <input
                  type="text"
                  value={entry.name}
                  onChange={(e) => onUpdate({ name: e.target.value })}
                  placeholder="e.g. 9b0l"
                  className="input"
                />
              </div>

              {/* Map path */}
              <div className="sm:col-span-1">
                <label className="label-muted mb-1.5 flex items-center gap-1.5">
                  <MapPin className="h-3 w-3" /> Map Path
                </label>
                <input
                  type="text"
                  value={entry.map_path}
                  onChange={(e) => onUpdate({ map_path: e.target.value })}
                  placeholder="/path/to/map.mrc"
                  className="input"
                />
              </div>

              {/* Resolution */}
              <div>
                <label className="label-muted mb-1.5 block">Resolution (&Aring;)</label>
                <input
                  type="number" step={0.1} min={0}
                  value={entry.resolution}
                  onChange={(e) => onUpdate({ resolution: parseFloat(e.target.value) || 0 })}
                  className="input"
                />
              </div>

              {/* Contour */}
              <div>
                <label className="label-muted mb-1.5 block">Contour Level</label>
                <input
                  type="number" step={0.01} min={0}
                  value={entry.contour_level}
                  onChange={(e) => onUpdate({ contour_level: parseFloat(e.target.value) || 0 })}
                  className="input"
                />
              </div>
            </div>

            {/* Model seeds */}
            <div className="mt-3">
              <label className="label-muted mb-1.5 flex items-center gap-1.5">
                <Layers className="h-3 w-3" /> Model Seeds
                <span className="text-[10px] text-slate-400 font-normal">(optional, comma-separated)</span>
              </label>
              <input
                type="text"
                value={entry.modelSeeds.join(", ")}
                onChange={(e) => {
                  const seeds = e.target.value
                    .split(",")
                    .map((s) => parseInt(s.trim()))
                    .filter((n) => !isNaN(n));
                  onUpdate({ modelSeeds: seeds });
                }}
                placeholder="e.g. 1, 2, 3"
                className="input"
              />
            </div>
          </div>

          {/* ── Entities ── */}
          <div className="px-5 py-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-xs font-bold uppercase tracking-widest text-slate-400">
                Entities
              </h3>
              {entry.sequences.length > 0 && (
                <span className="badge bg-slate-100 text-slate-500">{entry.sequences.length}</span>
              )}
            </div>

            {entry.sequences.length === 0 ? (
              <div className="rounded-xl border-2 border-dashed border-slate-200 bg-slate-50/50 py-8 text-center">
                <p className="text-sm text-muted-foreground mb-3">No entities yet</p>
                <div className="flex flex-wrap justify-center gap-2">
                  {SEQ_TYPES.map((t) => (
                    <button key={t} onClick={() => addSequence(t)}
                      className={`btn-sm rounded-lg border font-medium transition-all hover:shadow-sm ${TYPE_BADGE[t]}`}
                    >
                      <Plus className="h-3 w-3" /> {TYPE_LABELS[t]}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <>
                <div className="space-y-3">
                  {entry.sequences.map((seq, si) => (
                    <EntityCard
                      key={si}
                      index={si}
                      seq={seq}
                      onUpdate={(s) => updateSequence(si, s)}
                      onRemove={() => removeSequence(si)}
                    />
                  ))}
                </div>
                <div className="mt-4 flex flex-wrap gap-1.5">
                  {SEQ_TYPES.map((t) => (
                    <button key={t} onClick={() => addSequence(t)}
                      className="btn-ghost btn-sm rounded-lg text-muted-foreground hover:text-foreground"
                    >
                      <Plus className="h-3 w-3" /> {TYPE_LABELS[t]}
                    </button>
                  ))}
                </div>
              </>
            )}
          </div>

          {/* ── Covalent Bonds ── */}
          <div className="border-t px-5 py-4">
            <details>
              <summary className="flex cursor-pointer items-center gap-2 text-xs font-bold uppercase tracking-widest text-slate-400 hover:text-slate-600">
                <Link2 className="h-3.5 w-3.5" />
                Covalent Bonds
                {(entry.covalent_bonds?.length || 0) > 0 && (
                  <span className="badge bg-slate-100 text-slate-500 ml-1">
                    {entry.covalent_bonds!.length}
                  </span>
                )}
              </summary>

              <div className="mt-3 space-y-2">
                {(entry.covalent_bonds || []).map((bond, bi) => (
                  <div key={bi} className="rounded-lg border bg-slate-50/80 p-3">
                    <div className="flex items-center gap-1.5 mb-2">
                      <span className="badge bg-slate-200 text-slate-600 text-[10px]">Bond {bi + 1}</span>
                      <button onClick={() => removeBond(bi)}
                        className="ml-auto btn-icon h-6 w-6 text-slate-400 hover:text-red-500 hover:bg-red-50">
                        <Trash2 className="h-3 w-3" />
                      </button>
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      <div className="space-y-1.5">
                        <span className="text-[10px] font-semibold text-slate-400 uppercase">Left</span>
                        <div className="grid grid-cols-4 gap-1.5">
                          <input type="number" min={1} placeholder="Entity" value={bond.left_entity}
                            onChange={(e) => updateBond(bi, { left_entity: parseInt(e.target.value) || 1 })}
                            className="input-sm" />
                          <input type="number" min={1} placeholder="Pos" value={bond.left_position}
                            onChange={(e) => updateBond(bi, { left_position: parseInt(e.target.value) || 1 })}
                            className="input-sm" />
                          <input type="text" placeholder="Atom" value={bond.left_atom}
                            onChange={(e) => updateBond(bi, { left_atom: e.target.value })}
                            className="input-sm" />
                          <input type="number" min={1} placeholder="Copy" value={bond.left_copy || ""}
                            onChange={(e) => updateBond(bi, { left_copy: parseInt(e.target.value) || undefined })}
                            className="input-sm" />
                        </div>
                      </div>
                      <div className="space-y-1.5">
                        <span className="text-[10px] font-semibold text-slate-400 uppercase">Right</span>
                        <div className="grid grid-cols-4 gap-1.5">
                          <input type="number" min={1} placeholder="Entity" value={bond.right_entity}
                            onChange={(e) => updateBond(bi, { right_entity: parseInt(e.target.value) || 1 })}
                            className="input-sm" />
                          <input type="number" min={1} placeholder="Pos" value={bond.right_position}
                            onChange={(e) => updateBond(bi, { right_position: parseInt(e.target.value) || 1 })}
                            className="input-sm" />
                          <input type="text" placeholder="Atom" value={bond.right_atom}
                            onChange={(e) => updateBond(bi, { right_atom: e.target.value })}
                            className="input-sm" />
                          <input type="number" min={1} placeholder="Copy" value={bond.right_copy || ""}
                            onChange={(e) => updateBond(bi, { right_copy: parseInt(e.target.value) || undefined })}
                            className="input-sm" />
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
                <button onClick={addBond}
                  className="btn-ghost btn-sm text-muted-foreground hover:text-foreground rounded-lg">
                  <Plus className="h-3 w-3" /> Add covalent bond
                </button>
              </div>
            </details>
          </div>
        </div>
      )}
    </div>
  );
}
