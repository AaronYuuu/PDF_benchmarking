

summary_blurb <- function(dataset) {
  variants <- dataset$variants
  #if there are no variants, we're done
  if (length(variants) == 0) {
    return(paste("{\\bf No variants were detected}"))
  }
  #retrieve interpretations
  interpretations <- sapply(variants, `[[`, "interpretation")
  #retrieve genes
  genes <- sapply(variants, `[[`, "gene_symbol")
  #which ones are P/LP?
  is_patho <- grepl("pathogenic", interpretations, ignore.case = TRUE)
  #divide genes in to P/LP and VUS lists
  pgenes <- unique(genes[is_patho])
  vgenes <- unique(genes[!is_patho])
  #how many patho and vus variants each?
  npatho <- sum(is_patho)
  nvus <- sum(!is_patho)

  #helper function c(1,2,3) -> "1, 2 and 3"
  list2text <- function(l) {
    if (length(l) < 1) {
      ""
    } else if (length(l) == 1) {
      as.character(l)
    } else {
      paste(
        paste(l[-length(l)], collapse = ", "),
        "and",
        l[length(l)]
      )
    }
  }

  if (npatho == 0) {
    text <- "{\\bf No pathogenic germline variants were detected.} However, "
  } else if (npatho == 1) {
    text <- paste(
      "{\\bf A pathogenic germline variant was detected in the",
      pgenes[[1]],
      "gene.}"
    )
  } else {
    text <- paste(
      "{\\bf Pathogenic germline variants were detected in the",
      list2text(pgenes),
      "genes.}"
    )
  }
  if (npatho > 0 && nvus > 0) {
    text <- paste(text, "In addition, ")
  }
  if (nvus == 1) {
    text <- paste(
      text,
      "a variant of uncertain significance (VUS) was detected in the",
      vgenes[[1]],
      "gene."
    )
  } else if (nvus > 1) {
    text <- paste(
      text,
      "variants of uncertain significance (VUS) were detected in the",
      list2text(vgenes),
      "genes."
    )
  }

  return(text)
}

# {\bf No pathogenic germline variants were detected:} 
# However, a variant of uncertain significance (VUS) was 
# detected in the PALB2 gene (see below for explanation)

long_blurb <- function(dataset) {
  variants <- dataset$variants
  #if there are no variants, we're done
  if (length(variants) == 0) {
    return(paste("{\\bf No variants were detected}"))
  }
  #retrieve interpretations
  interpretations <- sapply(variants, `[[`, "interpretation")
  is_patho <- grepl("pathogenic", interpretations, ignore.case = TRUE)

  ptext <- sub_blurb(variants[is_patho], "pathogenic")
  vtext <- sub_blurb(variants[!is_patho], "VUS")

  return(paste0(ptext, "\n\n", vtext))

}

sub_blurb <- function(variants, btype) {
  if (length(variants) == 0) {
    if (btype == "pathogenic") {
      return(paste(
        "No pathogenic variants were detected in any of the genes targeted by",
        "massively parallel sequencing and deletion/duplication",
        "analylsis.Although this result decreases the likelihood",
        "of hereditary cancer, it does not exclude a diagnosis of",
        "a hereditary cancer syndrome."
      ))
    } else {
      return("") #absence of #VUS wouldn't be mentioned
    }
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
      "A", btype, "variant",
      variant$hgvsc, ",", variant$hgvsp, ",",
      "was detected in the", variant$gene_symbol,
      "gene.",
      switch(p_detail$type,
        substitution = paste(
          "This missense variant is predicted to result in",
          "amino acid substitution of", aaname(p_detail$ancestral),
          "at codon", p_detail$start, "with",
          aaname(p_detail$variant), "."
        ),
        stop = paste(
          "This nonsense variant is predicted to result in",
          "an early translation termination at codon", p_detail$start,
          "leading to a truncated protein."
        ),
        frameshift = paste(
          "This frameshift variant is predicted to result in",
          "an early translation termination, leading to a",
          "truncated protein."
        )
      ),
      "this variant is present at a",
      ifelse (variant$mafaf < 1e-3, "low", "moderately high"),
      "frequency in the general population in the Genome Aggregation",
      "Database (gnomAD ALL:", variant$mafaf * 100, "\\%, dbSNP:",
      paste0("rs", paste(sample.int(9, 9), collapse = ""), ","),
      "ACMGG PM2), as well as in unaffected individuals [PMID:",
      generate_pubmed(1), "]. ",
      if (btype == "pathogenic") {
        paste(
          "Although the majority of pathogenic variants in",
          variants$gene_symbol, 
          "are truncating, this variant has been reported in individuals",
          "with breast cancer or ovarian cancer [PMID: ",
          generate_pubmed(), "] (ACMGG PS4\\_Supporting)"
        )
      },
      "In silico analyses are concordant regarding the",
      ifelse(btype == "pathogenic", btype, "benign"),
      "effect this varinat may have on",
      "protein structure and function. (ACMGG BP4), however,",
      "functional analyses have not been performed to verify these",
      "predictions. Clinvar contains an entry for this variant",
      "(Variation ID: ", paste(sample.int(9, 6), collapse = ""), ")",
      "where it is classified as", tolower(variant$interpretation),
      "by the majority of submitting laboratories. Given the available",
      "evidence, our laboratory currently classifies this variant as",
      tolower(variant$interpretation), 
      "(ACMGG: PS4\\_Supporting, PMZ, BPI, BPA)."
    )

  }) |> paste(collapse = "\n\n")
}
