
# generates a summary text blurb for a set of variants
summary_blurb <- function(dataset, suffix = "detected.") {
  variants <- dataset$variants
  #if there are no variants, we're done
  if (length(variants) == 0) {
    return(paste("No variants", suffix))
  }
  interpretations <- sapply(variants, `[[`, "interpretation")
  #rename "pathogenic" to "pathogenic variant", etc
  interpretations <- sapply(interpretations, \(iname) {
    #also, convert to lower case
    iname <- tolower(iname)
    if (!grepl("variant", iname)) {
      paste(iname, "variant")
    } else {
      iname
    }
  })
  #count how many there are of each type
  inter_table <- table(interpretations)
  #generate strings for each type/number (e.g "two pathogenic variants")
  vstrings <- sapply(names(inter_table), \(iname) {
    #translate the number to a text string (2 -> "two")
    numstr <- inter_table[[iname]] |> num2text()
    #add plural when number is greater 1
    if (inter_table[[iname]] > 1) {
      iname <- sub("variant", "variants", iname)
    }
    paste(numstr, iname)
  })
  #if there's only one type, we're done
  if (length(vstrings) == 1) {
    return(cap(paste(vstrings[[1]], suffix)))
  }
  #concatenate with commas and "and"
  paste(
    paste(vstrings[-length(vstrings)], collapse = ", "),
    "and", vstrings[[length(vstrings)]], suffix
  )
}



long_blurb <- function(dataset) {
  variants <- dataset$variants
  #if there are no variants, we're done
  if (length(variants) == 0) {
    return("No variants were detected.")
  }

  intro_sentence <- paste(
    summary_blurb(dataset, suffix = ""),
    if (length(variants) == 1) "was" else "were",
    "detected in this individual. The interpretation of this result",
    "is summarized below."
  )

  hgvsps <- sapply(variants, `[[`, "hgvsp")
  var_data <- hgvsParseR::parseHGVS(hgvsps)

  var_blurbs <- lapply(seq_along(variants), \(i) {
    variant <- c(variants[[i]], var_data[i, ])
    if (!is.na(variant$variant) && variant$variant == "Ter") {
      variant$type <- "stop"
    }
    paste(
      "The", variant$hgvsc, "variant in", variant$gene_symbol,
      switch(variant$type, 
        synonymous = "causes no amino acid change.",
        stop = paste(
          "causes an early translation termination at position",
          variant$start, "."
        ),
        substitution = paste(
          "causes an amino acid substitution, which replaces",
          aaname(variant$ancestral), "with", aaname(variant$variant),
          "at position", variant$start , "."
        )
      ),
      "It was identified in 1/250010 (0.0004\\%) of alleles tested from",
      "control populations in the Genome Aggregation Database (gnomAD).",
      "To the best of our knowledge, it has not been previously reported",
      "in the literature. The", paste0(variant$ancestral, variant$start),
      "residue is weakly conserved in evolution.",
      "In silico analysis programs (SIFT, PolyPhen-2, Mutation Taster) predict",
      "this variant",
      if (grepl("uncertain", variant$interpretation)) {
        "to be tolerated"
      } else {
        "not to be tolerated"
      }, ". This variant is listed in ClinVar ",
      paste0("(",variant$variant_id,")."), 
      if (grepl("uncertain", variant$interpretation)) {
        paste(
          "In our opinion, the evidence collected to date is",
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
    )
  })

  paste(
    intro_sentence, "\n\n",
    paste0(var_blurbs, collapse = "\n\n")
  )
}
