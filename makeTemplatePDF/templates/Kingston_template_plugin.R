summary_blurb <- function(dataset) {
  variants <- dataset$variants
  #if there are no variants, we're done
  if (length(variants) == 0) {
    return(paste("No variants", suffix))
  } else if (length(variants) == 1) {
    return(paste(
      "A variant was detected in the",
      variants[[1]]$gene_symbol, 
      "gene."
    ))
  } else {
    paste(
      "Variants were detected in the",
      paste(
        sapply(variants, `[[`, "gene_symbol")[[-length(variants)]],
        collapse = ", "
      ),
      "and",
      variants[[length(variants)]]$gene_symbol, 
      "genes."
    )
  }
}

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

  sapply(seq_along(variants), function(i) {
    variant <- as.list(variants[[i]])
    p_detail <- hgvsp_data[i, ]
    c_detail <- hgvsc_data[i, ]

    paste(
      "This", variant$type, "variant in the",
      variant$gene_symbol, "gene results in a",
      switch(c_detail$type,
        substitution = {
          paste(
            "single base substitution at position",
            c_detail$start, "of the coding sequence,"
          )
        },
        deletion = {
          paste(
            "deletion of", 
            c_detail$end - c_detail$start + 1,
            "nucleotides,"
          )
        },
        insertion = {
          paste(
            "insertion of", 
            nchar(c_detail$variant),
            "nucleotides (", c_detail$variant, "),"
          )
        },
        delins = {
          paste(
            "substitution of", 
            c_detail$end - c_detail$start + 1,
            "nucleotides with",
            nchar(c_detail$variant),
            "different nucleotides,"
          )
        }
      ),
      "which",
      switch(p_detail$type,
        substitution = {
          paste(
            "replaces a", aaname(p_detail$ancestral), 
            "at position", p_detail$start,
            "with a", aaname(p_detail$variant), ".",
            "This modification may alter protein function."
          )
        },
        frameshift = {
          paste(
            "modifies the reading frame to produce an alternate stop codon,",
            "resulting in a prematurely truncated protein.",
            "This truncated protein is presumed to be non-functional."
          )
        },
        stop = {
          paste(
            "introduces a premature stop codon,",
            "leading to a truncated protein.",
            "This truncated protein is presumed to be non-functional."
          )
        }
      ),
      "The variant has been reported multiple times in ClinVar,",
      "with conflicting interpretations.",
      "The variant has been observed in gnomAD with a frequency of",
      variant$mafaf, ".",
      "This variant is classified as", variant$interpretation, "."
    )
  }) |> paste(collapse = "\n\n")
}
# This frameshift variant in the BRCA1 gene results in a 
# deletion of four nucleotides (TCAA), which modifies the 
# reading frame to produce an alternate stop codon, resulting 
# in a prematurely truncated protein. This truncated protein, 
# with legacy nomenclature BRCA1 4181del4, is presumed to be 
# non-functional. The variant has been reported multiple times 
# in ClinVar, with consensus for a pathogenic classification. 
# Loss of one BRCA1 allele is consistent with an increased 
# risk of developing breast/ovarian cancer (PMID: 31897316). 
# This variant has also been identified in multiple independent 
# families with hereditary breast and ovarian cancer 
# (PMID: 23683081; 20104584; 21559243; 11802209). This is 
# classified as a pathogenic variant.
#
# Interpretation of these findings must be made in light of 
# the clinical history and evaluation of this individual. 
# Genetic counselling and appropriate clinical follow up are 
# recommended for this individual.
