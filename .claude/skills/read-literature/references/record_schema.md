# Candidate record schema (one JSON object per line in `reports/literature/candidates.jsonl`)

```json
{
  "id": "dealmeida_2022",                      // proposed DatasetSpec id: firstauthor_year
  "title": "...",
  "citation": "de Almeida et al., Nat Genet (2022)",
  "doi": "10.xxxx/...", "pmid": null,
  "organism": "Drosophila melanogaster", "cell_context": "S2 cells",
  "modality": "enhancer",                      // promoter|enhancer|regulatory|utr5|utr3|rbs|coding|terminator|splice|polya|ires
  "assay": "STARR-seq",                        // MPRA|lentiMPRA|STARR-seq|Sort-seq|FACS-seq|polysome-MPRA|GPRA|...
  "hts_readout": true,                         // per-oligo readout counted by sequencing
  "readout": "log2 enrichment (RNA/DNA)",      // exactly what was measured
  "continuous": true,
  "n_sequences": null,                         // from the deposit/paper only — never from memory
  "seq_len_nt": null, "fixed_length": null,
  "library_type": "genomic tiles",             // random|designed|genomic tiles|variant pairs|saturation mutagenesis
  "replicates": null, "provided_splits": null,
  "data_location": {"repository": "GEO", "accession": "GSE...", "url": "...",
                    "files": [{"name": "...", "size": "..."}]},
  "processed_table_verified": false,           // true only after you saw the per-sequence file listing
  "access": "open",                            // open|login|controlled|on-request
  "license": null,
  "fit_verdict": "FIT_WITH_CAVEATS",           // STRONG_FIT|FIT_WITH_CAVEATS|REJECT
  "caveats": ["..."], "reject_reason": null,
  "project_value": {"score": 0, "new_organism": true, "new_modality": false,
                    "transfer_pair_with": [], "notes": "..."},
  "evidence": ["https://..."],                 // pages you actually opened
  "accessed": "YYYY-MM-DD",
  "status": "UNVERIFIED"                       // VERIFIED only after Step 5
}
```

Scan report table columns (`reports/literature/<date>-dataset-scan.md`):
`rank | id | verdict | score | organism | modality | assay | n | length | readout | location | status`
