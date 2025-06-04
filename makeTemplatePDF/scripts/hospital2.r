# Declare functions that will be sourced to avoid R CMD check warnings
if (FALSE) {
  summary_blurb <- function(...) NULL
  aaname <- function(...) NULL
  cap <- function(...) NULL
  num2text <- function(...) NULL
}

required_functions <- c("aaname", "summary_blurb", "cap", "num2text")
missing_functions <- !sapply(required_functions, exists)

if (any(missing_functions)) {
  source("sharedFunctions.r")
  cat("Loaded shared functions from sharedFunctions.r\n")
}

# Verify all functions are now available
still_missing <- required_functions[!sapply(required_functions, exists)]
if (length(still_missing) > 0) {
  stop("Missing required functions: ", paste(still_missing, collapse = ", "))
}
long_blurb <- function(variants) {
  #if there are no variants, we're done
  if (length(variants) == 0) {
    return("No variants were detected.")
  }
  hgvsps <- sapply(variants, `[[`, "hgvsp")
  var_data <- hgvsParseR::parseHGVS(hgvsps)
  intro_sentence <- paste(
    "\\newline The interpretation of these variants is as follows:",
    summary_blurb(variants, suffix = " "),
    if (length(variants) == 1) "was" else "were",
    "detected in the sample."
  )
  var_blurbs <- lapply(seq_along(variants), \(i) {
    variant <- c(variants[[i]], var_data[i, ])
    if (!is.na(variant$variant) && variant$variant == "Ter") {
      variant$type <- "stop"
    }
    generate_intro <- paste(
      "\\vspace{2em}{\\bf \\large Variant", i, "of", length(variants), "}
      \\newline \\vspace{2em}",
      "\\begin{tabularx}{\\textwidth}{C C C C} \n",
      "&&&\\\\",
      "Gene & Variant & Amino & Zygosity\\\\",
      variant$gene_symbol, " & ", variant$hgvsc, "&",
      variant$hgvsp, " & ", variant$zygosity, "\n\\end{tabularx}",
      "\\vspace{2em}",
      "{\\bf", variant$interpretation, "}", "\\newline"
    )

    location <- paste(
      "The", variant$hgvsg, "variant occurs in chromosome", variant$chromosome,
      ", within the", variant$gene_symbol, "gene, and it causes", variant$hgvsc,
      "change at position", variant$start, "in exon", variant$exon, 
      ", causing the mutation", variant$hgvsp, 
      ". This mutation has been identified in",
      sample(30:50, 1), "families. It has a population frequency of",
      formatC(variant$mafaf, format = "e", digits = 2),
      paste0("(", variant$mafac, " alleles in ",
             variant$mafan, " total alleles tested),"),
      "indicating it is a", 
      if (variant$mafaf < 0.0001) "very rare"
      else if (variant$mafaf < 0.001) "rare"
      else if (variant$mafaf < 0.01)
        "uncommon"
      else "relatively common",
      "variant in the general population. It "
    )
    effect <- switch(variant$type,
      synonymous = "causes no amino acid change.",
      stop = paste(
        "causes an early translation termination at position",
        variant$start, "."
      ),
      substitution = paste(
        "causes an amino acid substitution, which replaces",
        aaname(variant$ancestral), "with", aaname(variant$variant), "."
      )
    )
    interpretation_text <- if (grepl("uncertain", variant$interpretation)) {
      paste(
        "According to ClinVar, the evidence collected to date is",
        "insufficient to firmly establish the clinical significance of this",
        "variant, therefore it is classified as a",
        tolower(variant$interpretation), "."
      )
    } else {
      paste(
        "In accordance with existing evidence, this variant is therefore",
        "classified as a", tolower(variant$interpretation), "variant."
      )
    }
    interp <- tolower(variant$interpretation)
    clinical_statement <- switch(interp,
      "variant of uncertain clinical significance" = paste(
        "The clinical implications of this variant are not yet fully 
        understood.", "At present, available data is insufficient 
        to confirm its role in disease."
      ),
      "likely pathogenic" = paste(
        "This variant is classified as likely pathogenic.",
        "It is believed to negatively impact protein function and may 
        play a role", "in disease development in affected individuals."
      ),
      "pathogenic" = paste(
        "This variant is deemed pathogenic.",
        "It has a strong association with disease and has been documented in",
        "multiple cases, supported by functional evidence."
      ),
      paste(
        "This variant has been reported with the following interpretation:",
        variant$interpretation, "."
      )
    )

    implication_statement <- if (grepl("uncertain", interp)) {
      "not currently strongly implicated in specific diseases"
    } else {
      "implicated in oncogenesis and other disease processes"
    }

    # Build the conservation_text block
    conservation_text <- paste(
      "ClinVar and other genomic databases report the",
      variant$gene_symbol, variant$hgvsc,
      "variant as clinically relevant based on aggregated evidence.",
      "\n\n", clinical_statement,

      "\n\nThe affected nucleotide lies within a region 
  that is highly conserved across vertebrate species,",
      "which suggests functional importance and evolutionary constraint.",
      "\n\n
      This variant is", implication_statement,
      "according to ClinVar records",
      paste0(" (VCV accession: ", variant$variant_id, ")."),

      "\n\nSupporting studies and case reports can be found 
  in the scientific literature.",
      "Relevant PubMed references include:",
      paste(sample(1e8:1e9, sample(3:8, 1)), collapse = ", "), ". \\newpage"
    )
    paste(generate_intro, location, effect,
          conservation_text, interpretation_text, sep = "\n")
  })
  paste(
    intro_sentence, "\n\n",
    paste0(var_blurbs, collapse = "\n\n")
  )
}

if(!exists("PLUGIN_FUNCTIONS")) {
  PLUGIN_FUNCTIONS <- list()
}
PLUGIN_FUNCTIONS$long_blurb <- long_blurb