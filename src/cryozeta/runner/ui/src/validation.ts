import type { Entry, SequenceObj } from "./types";

const VALID_PROTEIN = new Set("ARNDCQEGHILKMFPSTWYVX");
const VALID_DNA = new Set("AGCTNIXU");
const VALID_RNA = new Set("AGCUNIX");

export function validateSequence(seq: string, seqType: string): string[] {
  const errors: string[] = [];
  if (!seq) {
    errors.push("Sequence is empty.");
    return errors;
  }
  const upper = seq.toUpperCase();
  const charsets: Record<string, Set<string>> = {
    protein: VALID_PROTEIN,
    dna: VALID_DNA,
    rna: VALID_RNA,
  };
  const valid = charsets[seqType];
  if (valid) {
    const invalid = [...new Set(upper)].filter((c) => !valid.has(c)).sort();
    if (invalid.length > 0) {
      errors.push(`Invalid ${seqType} characters: ${invalid.join(", ")}`);
    }
  }
  return errors;
}

function validateSeq(s: SequenceObj, prefix: string): string[] {
  const errors: string[] = [];
  if ("proteinChain" in s) {
    for (const e of validateSequence(s.proteinChain.sequence, "protein"))
      errors.push(`${prefix}: ${e}`);
  } else if ("dnaSequence" in s) {
    for (const e of validateSequence(s.dnaSequence.sequence, "dna"))
      errors.push(`${prefix}: ${e}`);
  } else if ("rnaSequence" in s) {
    for (const e of validateSequence(s.rnaSequence.sequence, "rna"))
      errors.push(`${prefix}: ${e}`);
  } else if ("ligand" in s) {
    if (!s.ligand.ligand) errors.push(`${prefix}: ligand value is missing`);
  } else if ("ion" in s) {
    if (!s.ion.ion) errors.push(`${prefix}: ion code is missing`);
  }
  return errors;
}

export function validateEntries(entries: Entry[]): string[] {
  const errors: string[] = [];
  const seen = new Set<string>();
  for (let i = 0; i < entries.length; i++) {
    const p = `Entry ${i + 1}`;
    const entry = entries[i];
    if (!entry.name) errors.push(`${p}: name is missing`);
    else if (seen.has(entry.name))
      errors.push(`${p}: duplicate name '${entry.name}'`);
    seen.add(entry.name);
    if (!entry.map_path) errors.push(`${p}: map_path is missing`);
    if (!entry.resolution || entry.resolution <= 0)
      errors.push(`${p}: resolution must be positive`);
    if (!entry.contour_level || entry.contour_level <= 0)
      errors.push(`${p}: contour_level must be positive`);
    if (!entry.sequences.length) errors.push(`${p}: no entities defined`);
    for (let j = 0; j < entry.sequences.length; j++) {
      errors.push(...validateSeq(entry.sequences[j], `${p}, Entity ${j + 1}`));
    }
  }
  return errors;
}

export function prepareForExport(entries: Entry[]): Entry[] {
  return structuredClone(entries).map((entry) => {
    entry.sequences = entry.sequences.map((s) => {
      if ("proteinChain" in s) {
        s.proteinChain.sequence = s.proteinChain.sequence
          .replace(/\s+/g, "")
          .toUpperCase();
        if (!s.proteinChain.msa?.precomputed_msa_dir) delete s.proteinChain.msa;
        if (!s.proteinChain.modifications?.length)
          delete s.proteinChain.modifications;
      }
      if ("dnaSequence" in s) {
        s.dnaSequence.sequence = s.dnaSequence.sequence
          .replace(/\s+/g, "")
          .toUpperCase();
        if (!s.dnaSequence.modifications?.length)
          delete s.dnaSequence.modifications;
      }
      if ("rnaSequence" in s) {
        s.rnaSequence.sequence = s.rnaSequence.sequence
          .replace(/\s+/g, "")
          .toUpperCase();
        if (!s.rnaSequence.msa?.precomputed_msa_dir) delete s.rnaSequence.msa;
        if (!s.rnaSequence.modifications?.length)
          delete s.rnaSequence.modifications;
      }
      return s;
    });
    if (!entry.covalent_bonds?.length) delete entry.covalent_bonds;
    return entry;
  });
}
