#use PDF_benchmarking_py312 environment to run this code
from gliner import GLiNER
from jsonLLM import(
    read_text_file, 
    get_text_files_from_directory
)
import pprint
def merge_entities(entities, original_text, max_gap=10):
    """Merge entities with the same label that are close together."""
    if not entities:
        return []
    
    # Group by label
    groups = {}
    for entity in sorted(entities, key=lambda x: x['start']):
        label = entity['label']
        if label not in groups:
            groups[label] = []
        
        # Check if we can merge with the last entity in this label group
        if groups[label] and entity['start'] - groups[label][-1]['end'] <= max_gap:
            # Extend the last entity
            last = groups[label][-1]
            last['text'] = original_text[last['start']:entity['end']].strip()
            last['end'] = entity['end']
        else:
            groups[label].append(entity)
    
    # Flatten and sort by position
    return sorted([entity for group in groups.values() for entity in group], 
                  key=lambda x: x['start'])

def split_text(text):
    """
    Make text into sections of at most 834 characters
    """
    sections = []
    if len(text) <= 834:
        sections.append(text)
    else:
        words = text.split()
        current_section = ""
        for word in words:
            if len(current_section) + len(word) + 1 <= 834:
                if current_section:
                    current_section += " "
                current_section += word
            else:
                sections.append(current_section)
                current_section = word
        if current_section:
            sections.append(current_section)
    return sections

def main():
    model = GLiNER.from_pretrained("numind/NuNerZero")
    #model = GLiNER.from_pretrained('urchade/gliner_multi-v2.1')
    labels = [
    "date_collected",
    "date_received",
    "date_verified",
    "report_type",
    "testing_context",
    "ordering_clinic",
    "testing_laboratory",
    "sequencing_scope",
    "gene_symbol",
    "refseq_mrna",
    "num_tested_genes",
    "sample_type",
    "analysis_type",
    "chromosome",
    "hgvsg",
    "hgvsc",
    "hgvsp",
    "transcript_id",
    "exon",
    "zygosity",
    "interpretation",
    "mafac",
    "mafan",
    "mafaf",
    "num_variants",
    "reference_genome"
    ]
    text = """
        123 Main St, West , Toronto, ON, A4B5G9
Toronto Cancer Hospital Division of Tumor Sequencing and Diagnostics
info@och,on.ca fax: 416-456-7890 phone: 437-416-6470
Draft MOLECULAR GENETICS LABORATORY RESULTS Patient Info: Patient Information: Name: Physician Name: DOB: Health Card: Sex: MRN #; Clinic: Windsor Regional Hospital (Windsor)
Procedure Date 2024-06-15
Accession Date 2024-06-16
Report Date 2024-06-17
Report Details
Genome Reference GRCh37 Sequencing Range Targeted variant testing, Methylation analysis
Referral Reason Clinical
Analysis Location Sinai Health System (Toronto) Table of findings: Gene Information Interpretation OTOF c.54366>T p.Glu1812Asp heterozy- Zikely pathogenic  gous
Summary: One likely pathogenic variant detected,. The details regarding the specific mutations are included below;
MRN:
Date: 2024-06-15
Page 1

123 Main St, West , Toronto, ON, A4B5G9
Toronto Cancer Hospital Division of Tumor info@och,on.ca Sequencing and fax: 416-456-7890 Diagnostics phone: 437-416-6470 Variant Interpretation: The interpretation of these variants is as follows: One likely pathogenic variant was detected in the sample.
Variant 1 of 1
Gene OTOF
Variant c.54366>1
Amino
Zygosity heterozygous
p.Glu1812Asp
Likely pathogenic
The 9.264626386>Tvariant occurs in chromosome chr2 within the OTOF gene , and it causes C.54366>T change at position 1812 in exon 7 causing the mu- tation p.Glu1812Asp This mutation has been identified in 46 families. It has a population frequency of 0.OOe+00 (0 alleles in 375853 total alleles tested) , indi- cating it is a very rare variant in the general population. It causes an amino acid substitution, which replaces glutamate with aspartate ClinVar and other genomic databases report the OTOF c.54366>T variant as clinically relevant based on aggregated evidence. This variant is classified as likely pathogenic. It is believed to negatively impact protein function and may play a role in disease development in affected indi- viduals, The affected nucleotide lies within a region that is highly conserved across ver- tebrate species, which suggests functional importance and evolutionary con- straint, This variant is implicated in oncogenesis and other disease processes according to ClinVar records (VCV accession: VCVO0O920952). Supporting studies and case reports can be found in the scientific literature. Rel- evant PubMedreferences include: 492984350,246228801,563728082,942746831, 956367708
MRN:
Date: 2024-06-15
Page 2

123 Main St, West , Toronto, ON, A4B5G9
Toronto Cancer Hospital Division of Tumor info@och,on.ca Sequencing and fax: 416-456-7890 Diagnostics phone: 437-416-6470 In accordance with existing evidence, this variantis therefore classified as a likely pathogenic variant; Test Details:
Genes Analyzed and mRNA sequence (NM_): IL6ST and NM_002184,4, OTOF and NM_194248.3. IGFZR and NM_000876.4. TCF12 and NM_001322159.3. NCKIPSD and NM_016453.4, DABZIP and NM_001395010.1. MDMZ and NM_002392.6. RAPIGDSI and NM_001 100427.2. 8 total genes tested,
Recommendations A precision oncology approach is advised: These mutations are known onco- genic drivers, Iinked to constitutive pathway activation: Targeted therapies, including hormone-correcting agents, may be considered, guided by clinical judgment, PI3K inhibitors could be explored in trials for PIK3CA-mutated cases; Germline testing is not indicated, as all mutations are consistent with somatic events, A multidisciplinary tumour board review is recommended to integrate findings into care. Additional testing may be pursued at the physician's discre_ tion;
Methodology ctDNA was sequenced with Targeted variant testing and was analyzed using Methylation analysis covering all coding exons and adjacent intronic regions. Target enrichment waS performed with hybrid capture (Twist Bioscience) , fol- lowed by Illumina NextSeq sequencing: Reads were aligned to GRCh37 using BWA-MEM, and variants called with GATK. Annotation was performed in VarSeq using population databases, predictive algorithms, and ClinVar; CNVs were assessed with CNVkit and confirmed by MLPA when applicable, Regions with pseudogene interference , such as PMS2 , were validated using long-range PCR and Sanger sequencing; Mean read depth exceeded 30Ox, with a minimum threshold of 5Ox. Analytical sensitivity is >99% for SNVs/indels and >95% for exon- level CNVs. Only variants classified as pathogenic, likely pathogenic, or variants of uncertain significance (VUS) are reported, per ACMG/AMP guidelines (PMID: 25741868) . Limitations This test was developed and validated in a certified clinical laboratory: Limi- tations include reduced sensitivity in pseudogene regions (e.g., PMS2 , CHEK2) , andinability to detect certain structural variants (e.g., MSHZ inversions) , deepin- tronic changes, or low-level mosaicism. PMS2 exons 11-15 are not analyzed. In- terpretations reflect current knowledge and may be updated as new evidence emerges;
MRN:
Date: 2024-06-15
Page 3

123 Main St, West , Toronto, ON, A4B5G9
Toronto Cancer Hospital Sequencing and Division of Tumor fax: 416-456-7890 info@och,on.ca Report Electronically Verified and Signed Diagnostics phone: 437-416-6470 by: 1 Mock
MRN:
Date: 2024-06-15
Page 4
        """
    texts = split_text(text)
    results = {}
    for text in texts:
        print("Starting with text", {text})

        entities = model.predict_entities(text, labels, threshold = 0.25)
        entities = merge_entities(entities, text)
        for entity in entities:
            if entity['label'] in results.keys():
                results[entity['label']] += " " + entity['text']
            else:
                results[entity['label']] = entity['text']
    output_data = {
                    "model": "numind/NuNerZero",
                    "status": "success",
                    "data": results,
                    "timestamp": "2025-06-19",
                    "source_file": "hospital2"
                }
    import os
    os.makedirs("getJSON/outJSON/glinerJSON", exist_ok=True)
    with open("getJSON/outJSON/glinerJSON/hospital2_numind_NuNerZero__response.json", "w") as f:
        import json
        json.dump(output_data, f, indent=2)
    print("Output data saved to getJSON/outJSON/hospital2_numind_NuNerZero__response.json")
main()