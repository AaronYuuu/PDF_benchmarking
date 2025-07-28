#!/usr/bin/env Rscript
long_blurb <- function(dataset) {
  variants <- dataset$variants

  #if there are no variants, we're done
  if (length(variants) == 0) {
    return("No variants were detected.")
  }

  #parse HGVS strings
  hgvsps <- sapply(variants, `[[`, "hgvsp")
  hgvsp_data <- hgvsParseR::parseHGVS(hgvsps)
  #fix missing stop codon type in var_data table
  if (any(grep("Ter$", hgvsps))) {
    hgvsp_data$type[grep("Ter$",hgvsps)] <- "stop"
  }
  hgvscs <- sapply(variants, `[[`, "hgvsc")
  hgvsc_data <- hgvsParseR::parseHGVS(hgvscs)

  sapply(seq_along(variants), \(i) {
    variant <- as.list(variants[[i]])
    p_detail <- hgvsp_data[i, ]
    c_detail <- hgvsc_data[i, ]

    paste(
      "A heterozygous variant,",
      paste0(
        tex_escape(variant$transcript_id), "(", variant$gene_symbol, "):",
        variant$hgvsc, ","
      ),
      variant$hgvsp,
      "was detected in exon", variant$exon, "of this gene.",
      "The allele frequency of this variant in the population databases is:",
      paste0(
        "gnomAD: (", format(variant$mafaf * 100, digits=3), "\\% overall)."
      ),
      "The classification of this variant in databases",
      "with clinically curated data is:",
      paste0("ClinVar: \"", variant$interpretation, "\"."),
      "In silico prediction of the effect of this amino acid",
      "change on protein structure and function is:",
      if (grepl("pathogenic", variant$interpretation, ignore.case = TRUE)) {
        paste(
          "SIFT: \"deleterious\"; PolyPhen-2: \"probably damaging\";",
          "Align-GVGD:\"Class C0\".",
          paste0(
            "This variant has been reported in a patient with",
            "colon cancer (PMID:", generate_pubmed(1), ")."
          ),
          "Pathogenic variants in the", variant$gene_symbol, 
          "gene are associated with autosomal dominant",
          paste0(variant$gene_symbol, "-associated polyposis,"),
          "including familial adenomatous polyposis (FAP),",
          "gastric adenocarcinoma and proximal polyposis of the",
          "stomach (GAPPS) (OMIM\\#611731, ClinGen, GeneReviews)."
        )
      } else {
        paste(
          "SIFT: \"tolerated\"; PolyPhen-2: \"neutral\";",
          "Align-GVGD:\"Class C5\".",
          "This variant has not been reported in the literature."
        )
      },
      "Based on the current evidence, we interpret this variant as",
      tolower(variant$interpretation), "(ACMG category)."

    )
  }) |> paste(collapse = "\n\n")

}
