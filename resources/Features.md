# QQL — Medical Knowledge Base Reference

> **Collection**: `medical_records`  
> **Departments**: Neurology (12), Cardiology (10), Oncology (9), Orthopedics (5), Pulmonology (5)  
> **Total**: 41 records | All records are `peer_reviewed: true`

---

### Concept: Create Collection

```commandline
-- Dense-only (default model, 384 dims)
CREATE COLLECTION medical_records

-- Pinned to a specific model (768 dims)
CREATE COLLECTION medical_records USING MODEL 'BAAI/bge-base-en-v1.5'

-- Hybrid (dense + sparse BM25) — recommended for clinical keyword + semantic search
CREATE COLLECTION medical_records HYBRID

-- Hybrid with custom dense model
CREATE COLLECTION medical_records USING HYBRID DENSE MODEL 'BAAI/bge-base-en-v1.5'
```

---

### Concept: SHOW COLLECTIONS + DROP COLLECTION

```commandline
SHOW COLLECTIONS

DROP COLLECTION medical_records
```

---

### Concept: INSERT INTO COLLECTION

```commandline
-- Minimal (text only)
INSERT INTO COLLECTION medical_records VALUES {
  'text': 'Beta blockers are contraindicated in patients with reactive airway disease and bradycardia'
}

-- With rich metadata
INSERT INTO COLLECTION medical_records VALUES {
  'text': 'Ischemic stroke occurs when a cerebral artery is occluded by thrombus or embolus causing neuronal death. IV alteplase within 4.5 hours significantly reduces disability. Mechanical thrombectomy extends the treatment window up to 24 hours in selected patients.',
  'title': 'Acute Ischemic Stroke Thrombolysis Protocol',
  'department': 'neurology',
  'sub_specialty': 'stroke_neurology',
  'document_type': 'emergency_protocol',
  'icd10_code': 'I63.9',
  'severity': 'critical',
  'year': 2024,
  'peer_reviewed': true
}

-- With a specific embedding model
INSERT INTO COLLECTION medical_records VALUES {
  'text': 'STEMI requires emergent revascularization. Primary PCI with door-to-balloon under 90 minutes is preferred.',
  'title': 'STEMI Primary PCI Protocol',
  'department': 'cardiology',
  'sub_specialty': 'interventional_cardiology',
  'document_type': 'emergency_protocol',
  'icd10_code': 'I21.3',
  'severity': 'critical',
  'year': 2024,
  'peer_reviewed': true
} USING MODEL 'BAAI/bge-small-en-v1.5'
```

#### Bulk Insert

```commandline
-- NEUROLOGY (batch 1 of 2)
INSERT BULK INTO COLLECTION medical_records VALUES [
  {
    'text': 'Alzheimers disease is characterized by progressive memory loss, cognitive decline, and accumulation of amyloid plaques and neurofibrillary tangles. Early detection using CSF Abeta42 and tau biomarkers is critical for disease-modifying therapy trials.',
    'title': 'Alzheimers Disease Overview',
    'department': 'neurology',
    'sub_specialty': 'cognitive_neurology',
    'document_type': 'clinical_guideline',
    'icd10_code': 'G30.9',
    'severity': 'high',
    'year': 2023,
    'peer_reviewed': true
  },
  {
    'text': 'Parkinsons disease involves degeneration of dopaminergic neurons in the substantia nigra, causing resting tremor, bradykinesia, rigidity, and postural instability. Levodopa is gold standard but long-term use causes dyskinesias.',
    'title': 'Parkinsons Disease Motor Management',
    'department': 'neurology',
    'sub_specialty': 'movement_disorders',
    'document_type': 'treatment_protocol',
    'icd10_code': 'G20',
    'severity': 'high',
    'year': 2024,
    'peer_reviewed': true
  },
  {
    'text': 'Epilepsy is defined by recurrent unprovoked seizures. First-line medications include valproate, lamotrigine, and levetiracetam. Drug-resistant epilepsy may benefit from surgical resection, vagus nerve stimulation, or responsive neurostimulation.',
    'title': 'Epilepsy Seizure Management',
    'department': 'neurology',
    'sub_specialty': 'epileptology',
    'document_type': 'clinical_guideline',
    'icd10_code': 'G40.909',
    'severity': 'moderate',
    'year': 2023,
    'peer_reviewed': true
  },
  {
    'text': 'Multiple sclerosis is an autoimmune demyelinating CNS disease. Relapsing-remitting MS is managed with disease-modifying therapies including interferon-beta, natalizumab, and ocrelizumab. MRI T2 lesion burden guides treatment escalation.',
    'title': 'Multiple Sclerosis DMT Selection',
    'department': 'neurology',
    'sub_specialty': 'neuroimmunology',
    'document_type': 'treatment_protocol',
    'icd10_code': 'G35',
    'severity': 'high',
    'year': 2024,
    'peer_reviewed': true
  },
  {
    'text': 'Migraine is a primary headache disorder with recurrent unilateral throbbing pain, photophobia, and nausea. CGRP monoclonal antibodies erenumab and fremanezumab are effective preventive therapies with favorable safety profiles.',
    'title': 'Migraine Prevention with CGRP Inhibitors',
    'department': 'neurology',
    'sub_specialty': 'headache_medicine',
    'document_type': 'research_summary',
    'icd10_code': 'G43.909',
    'severity': 'moderate',
    'year': 2023,
    'peer_reviewed': true
  }
]

-- NEUROLOGY (batch 2 of 2)
INSERT BULK INTO COLLECTION medical_records VALUES [
  {
    'text': 'Guillain-Barre syndrome is an acute inflammatory demyelinating polyneuropathy following infection. Rapid ascending weakness can cause respiratory failure. IVIG and plasmapheresis are equally effective first-line treatments.',
    'title': 'Guillain-Barre Syndrome ICU Management',
    'department': 'neurology',
    'sub_specialty': 'neuromuscular',
    'document_type': 'emergency_protocol',
    'icd10_code': 'G61.0',
    'severity': 'critical',
    'year': 2022,
    'peer_reviewed': true
  },
  {
    'text': 'Myasthenia gravis is caused by antibodies against acetylcholine receptors at the neuromuscular junction. Ptosis and diplopia are common presenting features. Pyridostigmine, immunosuppression, and thymectomy form the therapeutic triad.',
    'title': 'Myasthenia Gravis Treatment Algorithm',
    'department': 'neurology',
    'sub_specialty': 'neuromuscular',
    'document_type': 'clinical_guideline',
    'icd10_code': 'G70.01',
    'severity': 'high',
    'year': 2023,
    'peer_reviewed': true
  },
  {
    'text': 'ALS causes progressive degeneration of upper and lower motor neurons leading to paralysis and respiratory failure. Riluzole modestly extends survival. Tofersen, an antisense oligonucleotide, reduces neurofilament levels in SOD1-ALS.',
    'title': 'ALS Multidisciplinary Care and Novel Therapeutics',
    'department': 'neurology',
    'sub_specialty': 'neuromuscular',
    'document_type': 'research_summary',
    'icd10_code': 'G12.21',
    'severity': 'critical',
    'year': 2024,
    'peer_reviewed': true
  }
]

-- CARDIOLOGY (batch 1 of 2)
INSERT BULK INTO COLLECTION medical_records VALUES [
  {
    'text': 'STEMI requires emergent revascularization. Primary PCI with door-to-balloon under 90 minutes is preferred. Aspirin plus P2Y12 inhibitor dual antiplatelet therapy reduces reinfarction risk significantly.',
    'title': 'STEMI Primary PCI Protocol',
    'department': 'cardiology',
    'sub_specialty': 'interventional_cardiology',
    'document_type': 'emergency_protocol',
    'icd10_code': 'I21.3',
    'severity': 'critical',
    'year': 2024,
    'peer_reviewed': true
  },
  {
    'text': 'Heart failure with reduced ejection fraction is treated with the foundational quadruple therapy: ACE inhibitor or ARNi, beta-blocker, MRA, and SGLT2 inhibitor. Each drug class independently reduces all-cause mortality.',
    'title': 'HFrEF Quadruple Therapy Optimisation',
    'department': 'cardiology',
    'sub_specialty': 'heart_failure',
    'document_type': 'clinical_guideline',
    'icd10_code': 'I50.20',
    'severity': 'high',
    'year': 2024,
    'peer_reviewed': true
  },
  {
    'text': 'Atrial fibrillation stroke prevention uses CHA2DS2-VASc score. NOACs are preferred over warfarin due to predictable pharmacokinetics and lower intracranial haemorrhage risk. Rate control targets resting heart rate below 110 bpm.',
    'title': 'Atrial Fibrillation Anticoagulation and Rate Control',
    'department': 'cardiology',
    'sub_specialty': 'electrophysiology',
    'document_type': 'clinical_guideline',
    'icd10_code': 'I48.91',
    'severity': 'high',
    'year': 2023,
    'peer_reviewed': true
  }
]

-- ONCOLOGY (batch 1 of 2)
INSERT BULK INTO COLLECTION medical_records VALUES [
  {
    'text': 'Non-small cell lung cancer with EGFR exon 19 deletion or L858R mutation responds to osimertinib as first-line targeted therapy with superior PFS over chemotherapy and CNS penetration.',
    'title': 'EGFR-Mutated NSCLC Osimertinib First-Line',
    'department': 'oncology',
    'sub_specialty': 'thoracic_oncology',
    'document_type': 'treatment_protocol',
    'icd10_code': 'C34.10',
    'severity': 'high',
    'year': 2024,
    'peer_reviewed': true
  },
  {
    'text': 'Immune checkpoint inhibitor toxicities including pneumonitis, colitis, hepatitis, and endocrinopathies require graded immunosuppression. Grade 3 to 4 toxicities mandate permanent discontinuation and high-dose methylprednisolone followed by a steroid taper.',
    'title': 'Immune Checkpoint Inhibitor Toxicity Management',
    'department': 'oncology',
    'sub_specialty': 'immuno_oncology',
    'document_type': 'clinical_guideline',
    'icd10_code': 'T45.1X5A',
    'severity': 'high',
    'year': 2023,
    'peer_reviewed': true
  }
]

-- PULMONOLOGY
INSERT BULK INTO COLLECTION medical_records VALUES [
  {
    'text': 'COPD exacerbations are managed with short-acting bronchodilators, systemic corticosteroids, and antibiotics. Non-invasive positive pressure ventilation reduces intubation rates in hypercapnic respiratory failure significantly.',
    'title': 'COPD Acute Exacerbation Management',
    'department': 'pulmonology',
    'sub_specialty': 'obstructive_lung_disease',
    'document_type': 'emergency_protocol',
    'icd10_code': 'J44.1',
    'severity': 'high',
    'year': 2024,
    'peer_reviewed': true
  },
  {
    'text': 'ARDS is defined by bilateral infiltrates and PaO2-FiO2 below 300 of non-cardiogenic origin. Lung-protective ventilation with 6 mL/kg tidal volumes and plateau pressure below 30 cmH2O reduces mortality. Prone positioning for 16 hours improves oxygenation in severe ARDS.',
    'title': 'ARDS Lung Protective Ventilation and Proning',
    'department': 'pulmonology',
    'sub_specialty': 'critical_care',
    'document_type': 'emergency_protocol',
    'icd10_code': 'J80',
    'severity': 'critical',
    'year': 2024,
    'peer_reviewed': true
  },
  {
    'text': 'Severe asthma uncontrolled on high-dose ICS-LABA benefits from biologic add-on therapies. Mepolizumab targets eosinophilic inflammation, omalizumab targets IgE in allergic asthma, and tezepelumab blocks TSLP across multiple phenotypes.',
    'title': 'Severe Asthma Biologic Therapy Selection',
    'department': 'pulmonology',
    'sub_specialty': 'asthma',
    'document_type': 'treatment_protocol',
    'icd10_code': 'J45.51',
    'severity': 'high',
    'year': 2024,
    'peer_reviewed': true
  }
]

-- ORTHOPEDICS
INSERT BULK INTO COLLECTION medical_records VALUES [
  {
    'text': 'Total knee arthroplasty for end-stage osteoarthritis uses cemented fixation as gold standard in older patients. Enhanced recovery after surgery protocols with multimodal analgesia reduce length of stay and opioid consumption.',
    'title': 'Total Knee Arthroplasty ERAS Protocol',
    'department': 'orthopedics',
    'sub_specialty': 'joint_replacement',
    'document_type': 'treatment_protocol',
    'icd10_code': 'M17.11',
    'severity': 'moderate',
    'year': 2023,
    'peer_reviewed': true
  },
  {
    'text': 'Osteoporotic vertebral compression fractures are treated with balloon kyphoplasty for height restoration and pain relief. Bisphosphonate or denosumab therapy is essential to prevent subsequent fractures in the same patient.',
    'title': 'Vertebral Compression Fracture Kyphoplasty',
    'department': 'orthopedics',
    'sub_specialty': 'spine',
    'document_type': 'treatment_protocol',
    'icd10_code': 'M80.08XA',
    'severity': 'moderate',
    'year': 2022,
    'peer_reviewed': true
  }
]
```

---

### Concept: DELETE FROM

```commandline
-- By UUID (returned from INSERT output)
DELETE FROM medical_records WHERE id = '3f2e1a4b-8c91-4d0e-b123-abc123def456'

-- By integer ID
DELETE FROM medical_records WHERE id = 42
```

---

### Concept: Basic Semantic Search

```commandline
-- Top 5 results — open semantic query
SEARCH medical_records SIMILAR TO 'motor neuron degeneration treatment' LIMIT 5

-- With a specific embedding model (must match INSERT model)
SEARCH medical_records SIMILAR TO 'stroke thrombolysis window' LIMIT 10 USING MODEL 'BAAI/bge-small-en-v1.5'

-- Equality / inequality
SEARCH medical_records SIMILAR TO 'seizure management' LIMIT 10 WHERE department = 'neurology'
SEARCH medical_records SIMILAR TO 'respiratory failure ventilation' LIMIT 10 WHERE document_type != 'research_summary'

-- Range — recent guidelines only
SEARCH medical_records SIMILAR TO 'cardiac revascularization' LIMIT 5 WHERE year > 2022

-- BETWEEN — span a period of guideline updates
SEARCH medical_records SIMILAR TO 'immunotherapy toxicity management' LIMIT 10 WHERE year BETWEEN 2022 AND 2024

-- IN — filter to actionable document types
SEARCH medical_records SIMILAR TO 'airway management hypercapnia' LIMIT 10
  WHERE document_type IN ('emergency_protocol', 'treatment_protocol')

-- NOT IN — exclude lower-acuity records
SEARCH medical_records SIMILAR TO 'biologic therapy autoimmune' LIMIT 10
  WHERE severity NOT IN ('low', 'moderate')

-- Null checks — only records with a confirmed ICD-10 code
SEARCH medical_records SIMILAR TO 'demyelinating neuropathy IVIG' LIMIT 5 WHERE icd10_code IS NOT NULL

-- Full-text MATCH — keyword precision inside semantic search
SEARCH medical_records SIMILAR TO 'lung disease fibrosis' LIMIT 10
  WHERE title MATCH PHRASE 'antifibrotic therapy'

-- Logical AND — scoped to a department and recent year
SEARCH medical_records SIMILAR TO 'dopamine pathway neurodegeneration' LIMIT 10
  WHERE department = 'neurology' AND year >= 2023

-- Logical OR + AND — multi-department critical protocols
SEARCH medical_records SIMILAR TO 'respiratory cardiac arrest resuscitation' LIMIT 10
  WHERE (department = 'cardiology' OR department = 'pulmonology') AND severity = 'critical'

-- Single level nesting — filter by sub-specialty
SEARCH medical_records SIMILAR TO 'upper motor neuron disease antisense' LIMIT 5
  WHERE sub_specialty = 'neuromuscular'

-- Array of nested objects — hospital site data with department volumes
INSERT BULK INTO COLLECTION hospital_sites VALUES [
  {
    'text': 'Apollo Hospitals Hyderabad is a tertiary care centre with high-volume cardiac and neuro-surgery programs',
    'name': 'Apollo Hospitals Hyderabad',
    'country': {
      'name': 'India',
      'sites': [
        { 'name': 'Apollo Hyderabad', 'beds': 710, 'icu_beds': 120 },
        { 'name': 'Apollo Chennai',   'beds': 650, 'icu_beds': 100 }
      ]
    }
  },
  {
    'text': 'AIIMS Delhi is India largest public tertiary hospital with national referral programs across specialties',
    'name': 'AIIMS Delhi',
    'country': {
      'name': 'India',
      'sites': [
        { 'name': 'AIIMS Delhi',   'beds': 2478, 'icu_beds': 340 },
        { 'name': 'AIIMS Bhopal', 'beds': 960,  'icu_beds': 80  }
      ]
    }
  }
]

SEARCH hospital_sites SIMILAR TO 'large ICU tertiary care' LIMIT 5
  WHERE country.sites[].icu_beds > 100

-- Simple Hybrid search — combines dense semantic + sparse BM25 keyword signals
SEARCH medical_records SIMILAR TO 'EGFR mutation osimertinib lung cancer' LIMIT 10 USING HYBRID

-- Hybrid with WHERE — restrict to oncology protocols only
SEARCH medical_records SIMILAR TO 'checkpoint inhibitor PD-L1 immunotherapy' LIMIT 10
  USING HYBRID WHERE department = 'oncology' AND year >= 2022

-- Custom dense + sparse models — precision clinical retrieval
SEARCH medical_records SIMILAR TO 'ACE inhibitor hyperkalemia renal failure' LIMIT 5
  USING HYBRID DENSE MODEL 'BAAI/bge-base-en-v1.5' SPARSE MODEL 'prithivida/Splade_PP_en_v1'

-- Sparse only — keyword-heavy ICD/drug queries benefit from SPLADE
SEARCH medical_records SIMILAR TO 'levodopa dyskinesia substantia nigra bradykinesia' LIMIT 5 USING SPARSE

-- Custom sparse model
SEARCH medical_records SIMILAR TO 'natalizumab ocrelizumab relapsing remitting MS DMT' LIMIT 5
  USING SPARSE MODEL 'prithivida/Splade_PP_en_v1'
```

---

### Concept: RERANK — second-pass precision scoring

```commandline
-- Dense search + rerank — improve top-k ordering for clinical queries
SEARCH medical_records SIMILAR TO 'stroke thrombolysis mechanical thrombectomy window' LIMIT 5 RERANK

-- Hybrid + rerank — maximum precision for high-stakes clinical retrieval
SEARCH medical_records SIMILAR TO 'heart failure ejection fraction quadruple therapy' LIMIT 10
  USING HYBRID RERANK

-- With WHERE + rerank — scoped to recent peer-reviewed emergency protocols
SEARCH medical_records SIMILAR TO 'respiratory failure ventilation proning ARDS' LIMIT 10
  WHERE year > 2022 AND document_type = 'emergency_protocol' AND peer_reviewed = true
  RERANK

-- Custom cross-encoder reranker
SEARCH medical_records SIMILAR TO 'BRCA mutation PARP inhibitor prostate cancer' LIMIT 5
  RERANK MODEL 'BAAI/bge-reranker-large'

-- Everything combined — production-grade medical RAG retrieval pipeline
SEARCH medical_records SIMILAR TO 'neuromuscular junction antibody acetylcholine receptor treatment' LIMIT 10
  USING HYBRID DENSE MODEL 'BAAI/bge-base-en-v1.5'
  WHERE department = 'neurology'
    AND sub_specialty IN ('neuromuscular', 'neuroimmunology')
    AND severity IN ('high', 'critical')
    AND year >= 2022
  RERANK MODEL 'cross-encoder/ms-marco-MiniLM-L-6-v2'
```

---

### Concept: Query-time search parameter overrides

```commandline
-- Exact KNN — brute force, useful for recall debugging on medical_records
SEARCH medical_records SIMILAR TO 'anticoagulation atrial fibrillation NOAC CHA2DS2' LIMIT 10 EXACT

-- Boost HNSW exploration — higher ef for better recall on dense clinical embeddings
SEARCH medical_records SIMILAR TO 'CGRP monoclonal antibody migraine prevention' LIMIT 10
  WITH { hnsw_ef: 256 }

-- ACORN for filtered queries — avoids recall loss when filtering by severity = 'critical'
SEARCH medical_records SIMILAR TO 'ICU ventilator respiratory arrest management' LIMIT 10
  WHERE severity = 'critical' AND department = 'pulmonology'
  WITH { acorn: true }

-- Hybrid + exact mode — recall audit on the full hybrid pipeline
SEARCH medical_records SIMILAR TO 'biologic mepolizumab omalizumab tezepelumab asthma' LIMIT 10
  USING HYBRID EXACT
```

---

### Concept: Playing with .qql script files

```commandline
-- Seed the full medical knowledge base from file
qql> execute seed_medical_records.qql

-- Stop on first error — safe for production ingestion
qql> execute seed_medical_records.qql --stop-on-error

-- Export all 41 records to a backup file
qql> dump medical_records medical_records_backup.qql

-- Inside the shell
qql> DUMP COLLECTION medical_records medical_records_backup.qql

-- Round-trip: backup → drop → restore (useful before schema migrations)
qql> dump medical_records medical_records_v1_backup.qql
qql> DROP COLLECTION medical_records
qql> execute medical_records_v1_backup.qql
```