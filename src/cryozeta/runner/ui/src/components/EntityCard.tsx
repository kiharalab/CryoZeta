import type { SequenceObj, SeqType, ProteinModification, NucleicModification } from "../types";
import { seqTypeOf, TYPE_LABELS, TYPE_BADGE, TYPE_BORDER, COMMON_IONS, SEQ_TYPES } from "../types";
import { cn } from "../lib/utils";
import { validateSequence } from "../validation";
import { Trash2, Plus, FlaskConical, Database, Settings2 } from "lucide-react";
import { SequenceInput } from "./SequenceInput";

interface EntityCardProps {
  index: number;
  seq: SequenceObj;
  onUpdate: (s: SequenceObj) => void;
  onRemove: () => void;
}

function parseFasta(text: string): string {
  return text
    .split("\n")
    .filter((l) => !l.startsWith(">"))
    .join("")
    .replace(/\s+/g, "");
}

function handleFastaUpload(callback: (seq: string) => void) {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = ".fasta,.fa,.fna,.faa,.txt";
  input.onchange = (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => callback(parseFasta(reader.result as string));
    reader.readAsText(file);
  };
  input.click();
}

/** Inline validation badge for a sequence */
function SeqStatus({ seq, type }: { seq: string; type: "protein" | "dna" | "rna" }) {
  if (!seq) return <span className="text-[10px] text-slate-400">No sequence</span>;
  const clean = seq.replace(/\s+/g, "").toUpperCase();
  const errs = validateSequence(clean, type);
  const unit = type === "protein" ? "residues" : "bases";
  if (errs.length > 0) {
    return <span className="text-[10px] text-amber-600 font-medium">{errs[0]}</span>;
  }
  return (
    <span className="text-[10px] text-emerald-600 font-medium">
      {clean.length.toLocaleString()} {unit}
    </span>
  );
}

export function EntityCard({ index, seq, onUpdate, onRemove }: EntityCardProps) {
  const seqType = seqTypeOf(seq);

  const changeType = (newType: SeqType) => {
    if (newType === seqType) return;
    switch (newType) {
      case "proteinChain": onUpdate({ proteinChain: { sequence: "", count: 1 } }); break;
      case "dnaSequence": onUpdate({ dnaSequence: { sequence: "", count: 1 } }); break;
      case "rnaSequence": onUpdate({ rnaSequence: { sequence: "", count: 1 } }); break;
      case "ligand": onUpdate({ ligand: { ligand: "", count: 1 } }); break;
      case "ion": onUpdate({ ion: { ion: "NA", count: 1 } }); break;
    }
  };

  return (
    <div className={cn("rounded-xl border-l-[3px] border bg-white p-4 shadow-sm transition-shadow hover:shadow-md", TYPE_BORDER[seqType])}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2.5">
          <span className="text-xs font-semibold text-slate-400 tabular-nums">#{index + 1}</span>
          <span className={cn("badge", TYPE_BADGE[seqType])}>
            {TYPE_LABELS[seqType]}
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <select
            value={seqType}
            onChange={(e) => changeType(e.target.value as SeqType)}
            className="select py-1 text-xs h-7"
          >
            {SEQ_TYPES.map((t) => (
              <option key={t} value={t}>{TYPE_LABELS[t]}</option>
            ))}
          </select>
          <button onClick={onRemove}
            className="btn-icon h-7 w-7 text-slate-300 hover:text-red-500 hover:bg-red-50 rounded-lg">
            <Trash2 className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* Protein */}
      {"proteinChain" in seq && (
        <ProteinFields
          chain={seq.proteinChain}
          onChange={(patch) => onUpdate({ proteinChain: { ...seq.proteinChain, ...patch } })}
        />
      )}

      {/* DNA */}
      {"dnaSequence" in seq && (
        <NucleicFields label="DNA" seqType="dna"
          data={seq.dnaSequence} hasMsa={false}
          onChange={(patch) => onUpdate({ dnaSequence: { ...seq.dnaSequence, ...patch } })}
        />
      )}

      {/* RNA */}
      {"rnaSequence" in seq && (
        <NucleicFields label="RNA" seqType="rna"
          data={seq.rnaSequence} hasMsa={true}
          onChange={(patch) => onUpdate({ rnaSequence: { ...seq.rnaSequence, ...patch } })}
        />
      )}

      {/* Ligand */}
      {"ligand" in seq && (
        <div className="space-y-2">
          <div className="grid grid-cols-[1fr_90px] gap-3">
            <div>
              <label className="label-muted mb-1.5 flex items-center gap-1.5">
                <FlaskConical className="h-3 w-3" /> Ligand
                <span className="text-[10px] text-slate-400 font-normal">CCD code, SMILES, or FILE_path</span>
              </label>
              <input type="text" value={seq.ligand.ligand}
                onChange={(e) => onUpdate({ ligand: { ...seq.ligand, ligand: e.target.value } })}
                placeholder="e.g. CCD_ATP" className="input" />
            </div>
            <div>
              <label className="label-muted mb-1.5 block">Count</label>
              <input type="number" min={1} value={seq.ligand.count}
                onChange={(e) => onUpdate({ ligand: { ...seq.ligand, count: parseInt(e.target.value) || 1 } })}
                className="input" />
            </div>
          </div>
        </div>
      )}

      {/* Ion */}
      {"ion" in seq && (
        <div className="grid grid-cols-[1fr_90px] gap-3">
          <div>
            <label className="label-muted mb-1.5 block">Ion</label>
            <select value={seq.ion.ion}
              onChange={(e) => onUpdate({ ion: { ...seq.ion, ion: e.target.value } })}
              className="select">
              {COMMON_IONS.map((ion) => (
                <option key={ion} value={ion}>{ion}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="label-muted mb-1.5 block">Count</label>
            <input type="number" min={1} value={seq.ion.count}
              onChange={(e) => onUpdate({ ion: { ...seq.ion, count: parseInt(e.target.value) || 1 } })}
              className="input" />
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Protein fields ── */

function ProteinFields({
  chain, onChange,
}: {
  chain: { sequence: string; count: number; msa?: any; modifications?: ProteinModification[] };
  onChange: (patch: any) => void;
}) {
  return (
    <div className="space-y-3">
      <SequenceInput
        value={chain.sequence}
        onChange={(seq) => onChange({ sequence: seq })}
        onFasta={() => handleFastaUpload((seq) => onChange({ sequence: seq }))}
        placeholder="Paste protein sequence (ARNDCQEGHILKMFPSTWYVX)..."
        label="Sequence"
        status={<SeqStatus seq={chain.sequence} type="protein" />}
        count={chain.count}
        onCountChange={(n) => onChange({ count: n })}
      />

      {/* Advanced options */}
      <div className="flex flex-wrap gap-1.5">
        <AdvancedMsa data={chain} onChange={onChange} />
        <AdvancedPtm
          modifications={chain.modifications || []}
          posKey="ptmPosition" typeKey="ptmType" label="PTMs"
          onChange={(mods) => onChange({ modifications: mods })} />
      </div>
    </div>
  );
}

/* ── Nucleic acid fields ── */

function NucleicFields({
  label, seqType, data, hasMsa, onChange,
}: {
  label: string;
  seqType: "dna" | "rna";
  data: { sequence: string; count: number; msa?: any; modifications?: NucleicModification[] };
  hasMsa: boolean;
  onChange: (patch: any) => void;
}) {
  return (
    <div className="space-y-3">
      <SequenceInput
        value={data.sequence}
        onChange={(seq) => onChange({ sequence: seq })}
        onFasta={() => handleFastaUpload((seq) => onChange({ sequence: seq }))}
        placeholder={`Paste ${label} sequence...`}
        label={`${label} Sequence`}
        status={<SeqStatus seq={data.sequence} type={seqType} />}
        count={data.count}
        onCountChange={(n) => onChange({ count: n })}
      />

      <div className="flex flex-wrap gap-1.5">
        {hasMsa && <AdvancedMsa data={data} onChange={onChange} />}
        <AdvancedPtm
          modifications={data.modifications || []}
          posKey="basePosition" typeKey="modificationType" label="Modifications"
          onChange={(mods) => onChange({ modifications: mods })} />
      </div>
    </div>
  );
}

/* ── MSA collapsible ── */

function AdvancedMsa({ data, onChange }: { data: any; onChange: (p: any) => void }) {
  return (
    <details className="w-full group">
      <summary className="btn-ghost h-7 px-2 text-[11px] text-slate-400 hover:text-slate-700 rounded-lg cursor-pointer list-none flex items-center gap-1.5">
        <Database className="h-3 w-3" />
        MSA
        {data.msa?.precomputed_msa_dir && (
          <span className="badge bg-emerald-100 text-emerald-700 text-[9px] py-0 ml-1">configured</span>
        )}
      </summary>
      <div className="mt-2 grid grid-cols-2 gap-3 rounded-lg border bg-slate-50/80 p-3">
        <div>
          <label className="label-muted mb-1 block text-[10px]">Precomputed MSA directory</label>
          <input type="text" value={data.msa?.precomputed_msa_dir || ""}
            onChange={(e) => onChange({
              msa: { precomputed_msa_dir: e.target.value, pairing_db: data.msa?.pairing_db || "" }
            })}
            placeholder="/path/to/msa" className="input-sm" />
        </div>
        <div>
          <label className="label-muted mb-1 block text-[10px]">Pairing DB</label>
          <input type="text" value={data.msa?.pairing_db || ""}
            onChange={(e) => onChange({
              msa: { precomputed_msa_dir: data.msa?.precomputed_msa_dir || "", pairing_db: e.target.value }
            })}
            placeholder="uniref100" className="input-sm" />
        </div>
      </div>
    </details>
  );
}

/* ── Modifications / PTMs collapsible ── */

function AdvancedPtm({
  modifications, posKey, typeKey, label, onChange,
}: {
  modifications: any[];
  posKey: string;
  typeKey: string;
  label: string;
  onChange: (mods: any[]) => void;
}) {
  return (
    <details className="w-full group">
      <summary className="btn-ghost h-7 px-2 text-[11px] text-slate-400 hover:text-slate-700 rounded-lg cursor-pointer list-none flex items-center gap-1.5">
        <Settings2 className="h-3 w-3" />
        {label}
        {modifications.length > 0 && (
          <span className="badge bg-violet-100 text-violet-700 text-[9px] py-0 ml-1">{modifications.length}</span>
        )}
      </summary>
      <div className="mt-2 space-y-2 rounded-lg border bg-slate-50/80 p-3">
        {modifications.map((mod, mi) => (
          <div key={mi} className="flex items-center gap-2">
            <input type="number" min={1} placeholder="Position" value={mod[posKey]}
              onChange={(e) => {
                const mods = [...modifications];
                mods[mi] = { ...mods[mi], [posKey]: parseInt(e.target.value) || 0 };
                onChange(mods);
              }}
              className="input-sm w-20" />
            <input type="text" placeholder={posKey === "ptmPosition" ? "CCD_SEP" : "Modification type"}
              value={mod[typeKey]}
              onChange={(e) => {
                const mods = [...modifications];
                mods[mi] = { ...mods[mi], [typeKey]: e.target.value };
                onChange(mods);
              }}
              className="input-sm flex-1" />
            <button onClick={() => onChange(modifications.filter((_, i) => i !== mi))}
              className="btn-icon h-6 w-6 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded">
              <Trash2 className="h-3 w-3" />
            </button>
          </div>
        ))}
        <button
          onClick={() => onChange([...modifications, { [posKey]: 1, [typeKey]: "" }])}
          className="btn-ghost h-6 px-2 text-[10px] text-slate-400 hover:text-slate-700 rounded">
          <Plus className="h-3 w-3" /> Add
        </button>
      </div>
    </details>
  );
}
