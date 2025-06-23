from gliner import GLiNER
from jsonLLM import(
    extract_json_from_response, 
    save_model_response, 
    read_text_file, 
    get_text_files_from_directory
)

def merge_entities(entities):
    if not entities:
        return []
    merged = []
    current = entities[0]
    for next_entity in entities[1:]:
        if next_entity['label'] == current['label'] and (next_entity['start'] == current['end'] + 1 or next_entity['start'] == current['end']):
            current['text'] = next_entity[current['start']: next_entity['end']].strip()
            current['end'] = next_entity['end']
        else:
            merged.append(current)
            current = next_entity
    # Append the last entity
    merged.append(current)
    return merged




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
        OCH
MRN:
Date Collected: 2024-06-15
Note this is @ draft dataset
Ontario Cancer Hospital 123 Main St, West, Toronto, ON, A4B5G9 info@och.on.ca fax: 416-456-7890 phone: 437 - 416-6470
Division of Tumor Sequencing and Diagnostics
LABORATORY RESULTS
Report electronically signed by:
Patient Info. First Name: Redacted Last Name: DOB; Sex: Health Card: Medical Record #; Sample tested: ctDNA Type of analysis: Methylation analysis Referral Reason: Clinical Referring Physician: Dates Collected ~ 2024-06-15 Assessed 5 2024-06-16 Reported ~ 2024-06-17
Results: One Iikely pathogenic variant detected:
Summary of Results: Gene Exon Base " Amino Zygosity Interpretation   Acid OTOF C,54366>1 pClul812Asp]heterozygous Likely pathogenic
Genes Analyzed: 8 total IL6ST, OTOF, IGFZR, TCF12, NCKIPSD , DABZIP, MDM2, RAPIGDSI ,
Ontario Cancer Hospital
Note that this is @ draft dataset
Page 1

OCH
MRN:
Date Collected: 2024-06-15
Note this is @ draft dataset
Test Details:
Findings: The interpretation of these variants is as follows: One likely pathogenic variant was detected in the sample. Variant 1 of 1 OTOF ( c.5436G>T p.Glu1812Asp ) The C.54366>T variant occurs at position 1812 and is located in exon 7 of the OTOF gene, within chromosome chr2 It causes an amino acid substitution, which replaces glutamate with aspartate ClinVar and other genomic databases report the OTOF c.54366>T variant aS clinically relevant based on aggregated evidence. This variant is considered Iikely pathogenic: It has been associated with deleterious effects on protein function and may contribute to disease in affected individuals. The affected nucleotide lies within a region that is highly conserved across vertebrate species, which suggests functional importance and evolutionary constraint; This variant is implicated in oncogenesis and other disease processes according to Clin- Var records (VCV accession: VCVO00920952) . Supporting studies and case reports can be found in the scientific literature. Relevant PubMed references include: 685937294,930988152 , 961839690, 786564703,204885742, 602541323, 536372756, 965974397 In accordance with existing evidence , this variant is therefore classified as a likely pathogenic variant;
Recommendations We recommend a precision oncology approach: These variants are associated with con- stitutive pathway activation and are well-established drivers of tumourigenesis.  Targeted therapies should be evaluated based on these molecular findings. Specifically, pharma- ceutical treatment is also recommended to correct hormone imbalances that may be caused by these mutations; However, the clinician's advice takes precedence. Addi- tionally, PI3K inhibitors could be explored in clinical trials for the PIK3CA-mutated context. Further germline testing is not indicated at this time, as all three mutations are consistent with somatic oncogenic events. Multidisciplinary tumour board review is advised to inte grate molecular findings into the patient's treatment plan: Further genetic testing may be required and completed at a physician's discretion: Methodology Genomic DNA was extracted and analyzed using a custom-designed targeted sequenc- ing panel encompassing all coding exons and at least 20 base pairs of flanking intronic regions for the specified genes.  Target enrichment waS performed using hybrid capture technology (Twist Bioscience)  followed by paired-endsequencing on the Illumina NextSeq platform. Sequencing reads were aligned to the GRCh37 /hg19 human genome reference using BWA-MEM, and variant calling was performed using GATK (Broad Institute): Annota- tion and interpretation of variants were conducted using VarSeq (Golden Helix) , incorpo- rating population frequency databases, in silico prediction tools, and ClinVar Exon-level copy number variations were evaluated using CNVkit and confirmed by MLPA (MRC Hol-
Ontario Cancer Hospital
Note that this is @ draft dataset
Page 2

OCH
MRN:
Date Collected: 2024-06-15
Note this is @ draft dataset
land) when applicable. Regions with known pseudogeneinterference, such as PMS2 , were validated with long-range PCR and Sanger sequencing: The average read depth across all targeted regions exceeded 30Ox , with a minimum depth threshold of 5Ox. The analytical sensitivity for single nucleotide variants and small indels is >o9%, and for exon-level CNVs, >95%, Only variants classified as pathogenic, likely pathogenic, or variants of uncertain significance (VUS) are reported, according to ACMGIAMP guidelines (PMID: 25741868). Limitations This test was developed and validated by a certified clinical laboratory: Limitations include reduced sensitivity in regions with pseudogenes (e.g.= PMS2 , CHEK2) , and inability to de tect certain structural variants (e.g. MSH2 inversion) , deep intronic changes, or low-level mosaicism, PMS2 exons 11-15 are not assessed due to pseudogene interference. Variant interpretation reflects current scientific knowledge and may evolve over time.
Ontario Cancer Hospital
Note that this is @ draft dataset
Page 3
        """
    entities = model.predict_entities(text, labels)
    #entities = merge_entities(entities)
    for entity in entities:
        print(entity['text'], "==>", entity['label'])
            
main()