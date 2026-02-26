export interface MsaInfo {
  precomputed_msa_dir: string;
  pairing_db: string;
}

export interface ProteinModification {
  ptmPosition: number;
  ptmType: string;
}

export interface NucleicModification {
  basePosition: number;
  modificationType: string;
}

export interface ProteinChain {
  sequence: string;
  count: number;
  msa?: MsaInfo;
  modifications?: ProteinModification[];
}

export interface NucleicSequence {
  sequence: string;
  count: number;
  msa?: MsaInfo;
  modifications?: NucleicModification[];
}

export interface LigandInfo {
  ligand: string;
  count: number;
}

export interface IonInfo {
  ion: string;
  count: number;
}

export type SequenceObj =
  | { proteinChain: ProteinChain }
  | { dnaSequence: NucleicSequence }
  | { rnaSequence: NucleicSequence }
  | { ligand: LigandInfo }
  | { ion: IonInfo };

export type SeqType =
  | "proteinChain"
  | "dnaSequence"
  | "rnaSequence"
  | "ligand"
  | "ion";

export interface CovalentBond {
  left_entity: number;
  left_position: number;
  left_atom: string;
  left_copy?: number;
  right_entity: number;
  right_position: number;
  right_atom: string;
  right_copy?: number;
}

export interface Entry {
  name: string;
  modelSeeds: number[];
  map_path: string;
  resolution: number;
  contour_level: number;
  sequences: SequenceObj[];
  covalent_bonds?: CovalentBond[];
}

// --- Helpers ---

export const SEQ_TYPES: SeqType[] = [
  "proteinChain",
  "dnaSequence",
  "rnaSequence",
  "ligand",
  "ion",
];

export const TYPE_LABELS: Record<SeqType, string> = {
  proteinChain: "Protein",
  dnaSequence: "DNA",
  rnaSequence: "RNA",
  ligand: "Ligand",
  ion: "Ion",
};

export const TYPE_BADGE: Record<SeqType, string> = {
  proteinChain: "bg-blue-100 text-blue-700",
  dnaSequence: "bg-orange-100 text-orange-700",
  rnaSequence: "bg-emerald-100 text-emerald-700",
  ligand: "bg-violet-100 text-violet-700",
  ion: "bg-rose-100 text-rose-700",
};

export const TYPE_BORDER: Record<SeqType, string> = {
  proteinChain: "border-l-blue-500",
  dnaSequence: "border-l-orange-500",
  rnaSequence: "border-l-emerald-500",
  ligand: "border-l-violet-500",
  ion: "border-l-rose-500",
};

export const COMMON_IONS = [
  "NA", "MG", "ZN", "CA", "FE", "MN", "CL", "K", "CO", "CU", "NI",
];

export function emptyEntry(): Entry {
  return {
    name: "",
    modelSeeds: [],
    map_path: "",
    resolution: 3.0,
    contour_level: 0.3,
    sequences: [],
  };
}

export function emptySequence(t: SeqType): SequenceObj {
  switch (t) {
    case "proteinChain":
      return { proteinChain: { sequence: "", count: 1 } };
    case "dnaSequence":
      return { dnaSequence: { sequence: "", count: 1 } };
    case "rnaSequence":
      return { rnaSequence: { sequence: "", count: 1 } };
    case "ligand":
      return { ligand: { ligand: "", count: 1 } };
    case "ion":
      return { ion: { ion: "NA", count: 1 } };
  }
}

export function seqTypeOf(s: SequenceObj): SeqType {
  if ("proteinChain" in s) return "proteinChain";
  if ("dnaSequence" in s) return "dnaSequence";
  if ("rnaSequence" in s) return "rnaSequence";
  if ("ligand" in s) return "ligand";
  return "ion";
}
