
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

    text <- paste(
      "This individual is", variant$zygosity,
      "for a rare sequence variant in the",
      variant$gene_symbol, "gene. This variant is a",
      switch(c_detail$type,
        "substitution" = {
          paste(
            "single base substitution at position",
            c_detail$position, "of the coding sequence"
          )
        },
        "deletion" = {
          paste(
            "deletion of", 
            c_detail$end - c_detail$start + 1,
            "nucleotides at position",
            c_detail$start, 
            "of the coding sequence"
          )
        },
        "insertion" = {
          paste(
            "insertion of", 
            nchar(c_detail$variant),
            "nucleotides at position",
            c_detail$start, 
            "of the coding sequence"
          )
        },
        "delins" = {
          paste(
            "subtitution of", 
            c_detail$end - c_detail$start + 1,
            "nucleotides with",
            nchar(c_detail$variant),
            "different nucleotides at position",
            c_detail$start,
            "of the coding sequence"
          )
        }
      ),
      "and is predicted to result in",
      switch(p_detail$type,
        "frameshift" = {
          paste(
            "the frameshift of the open reading frame creating",
            "a premature stop codon (",
            variant$hgvsp,
            "). This variant is predicted to cause",
            "loss or disruption of the normal protein function",
            "through nonsense-mediated RNA decay or protein",
            "truncation."
          )
        },
        "stop" = {
          paste(
            "the early termination of the open reading frame (",
            variant$hgvsp,
            "). This variant is predicted to cause",
            "loss or disruption of the normal protein function",
            "through nonsense-mediated RNA decay or protein",
            "truncation."
          )
        },
        "synonymous" = {
          paste(
            "no amino acid change",
            variant$hgvsp,
            "This variant is predicted to have no",
            "impact on the protein function."
          )
        },
        "missense" = {
          paste(
            "a single amino acid substitution of a",
            aaname(p_detail$ancestral), "with a",
            aaname(p_detail$variant), "at position",
            p_detail$start,
            "of the protein (",
            variant$hgvsp, ")",
            "This variant could potentially alter the",
            "protein function, but the exact impact is",
            "currently unknown."
          )
        }
      ),
      "Based on the currently available evidence,",
      "this variant is classified as",
      variant$interpretation, " (ACMG category)."
    )

    return(text)

  }) |> paste(collapse = "\n\n")

}

# This individual is heterozygous for a rare sequence variant 
# in the BRCA2 gene. This variant is a deletion of two nucleotides 
# c.5722\_5723delCT and is predicted to result in the frameshift 
# of the open reading frame creating a premature stop codon 
# p.(Leu1908ArgfsTer2) This variant is predicted to cause loss 
# or disruption of the normal protein function through 
# nonsense-mediated RNA decay or protein truncation. Based on 
# the currently available evidence this variant is classified 
# as pathogenic mutation (ACMG category 1)
