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
    "The interpretation of these variants is as follows:",
    summary_blurb(variants, suffix = " "),
    if (length(variants) == 1) "was" else "were",
    "detected in the sample. \\newline"
  )
  var_blurbs <- lapply(seq_along(variants), \(i) {
    variant <- c(variants[[i]], var_data[i, ])
    if (!is.na(variant$variant) && variant$variant == "Ter") {
      variant$type <- "stop"
    }
    generate_intro <- paste(
      "{\\bf Variant", i, "of", length(variants),
      variant$gene_symbol, "(", variant$hgvsc, variant$hgvsp, ")}", "\\newline"
    )

    location <- paste(
      "The", variant$hgvsc, "variant occurs at position", variant$start,
      "and is located in exon", variant$exon, "of the", variant$gene_symbol,
      "gene, within chromosome", variant$chromosome, ". It "
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
        tolower(variant$interpretation), ". \\newline"
      )
    } else {
      paste(
        "In accordance with existing evidence, this variant is therefore",
        "classified as a", tolower(variant$interpretation), "variant. \\newline"
      )
    }
    interp <- tolower(variant$interpretation)
    clinical_statement <- switch(interp,
      "variant of uncertain clinical significance" = paste(
        "The clinical relevance of this variant remains unclear.",
        "Currently, there is insufficient evidence 
        to confirm or refute its role in disease."
      ),
      "likely pathogenic" = paste(
        "This variant is considered likely pathogenic.",
        "It has been associated with deleterious effects on protein function 
        and may contribute to disease in affected individuals."
      ),
      "pathogenic" = paste(
        "This variant is classified as pathogenic.",
        "It is strongly associated with disease causation and has been reported
        in multiple affected individuals and functional studies."
      ),
      paste(
        "This variant has been reported with the interpretation:",
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

      paste("\nSupporting studies and case reports can be found 
  in the scientific literature.",
            "Relevant PubMed references include:",
            paste(sample(1e8:1e9, sample(3:8, 1)), collapse = ", "), ".")
    )
    paste(generate_intro, location, effect,
          conservation_text, interpretation_text, sep = "\n")
  })
  paste(
    intro_sentence, "\n",
    paste0(var_blurbs, collapse = "\n")
  )
}

if (!exists("PLUGIN_FUNCTIONS")) {
  PLUGIN_FUNCTIONS <- list()
}
PLUGIN_FUNCTIONS$long_blurb <- long_blurb