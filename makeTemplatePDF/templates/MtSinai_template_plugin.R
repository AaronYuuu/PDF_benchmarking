
summary_blurb <- function(dataset) {
  variants <- dataset$variants
  prefix <- "Sequencing  identified"
  suffix <- "listed above."
  #if there are no variants, we're done
  if (length(variants) == 0) {
    return(paste(prefix, "no variants"))
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
    return(paste(prefix, vstrings[[1]], suffix))
  }
  #concatenate with commas and "and"
  paste(
    prefix,
    paste(vstrings[-length(vstrings)], collapse = ", "),
    "and", vstrings[[length(vstrings)]], suffix,
    "The result is consistent with clinical history."
  )
}


long_blurb <- function(dataset) {
  variants <- dataset$variants
  header <- paste0(
    "\\hdashrule[0.5ex]{\\textwidth}{1pt}{1mm}\n\n",
    "\\vspace{-1ex}\n"
  )
  midline <- paste0(
    "\n\n\\vspace{-1ex}\n",
    "\\hdashrule[0.5ex]{\\textwidth}{1pt}{1mm}\n\n"
  )
  if (length(variants) == 0) {
    return(paste0(header, "No variants found.", midline))
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
      header,
      paste0(
        variant$gene_symbol, ", ",
        "EXON", sprintf("%02d",as.integer(variant$exon)), ", ",
        variant$hgvsc, ", ", variant$hgvsp, ", ",
        cap(variant$zygosity), ", ", cap(variant$interpretation)
      ),
      midline,
      "The ", variant$gene_symbol, " ", variant$hgvsc, ", ",
      variant$hgvsp, " variant was identified in multiple individuals",
      "with cancer syndromes (", 
      paste(paste0("PMID:",generate_pubmed()), collapse = ", "),
      "). The variant was also identified in ClinVar (classified as ",
      variant$interpretation, " by multiple submitters). ",
      "The variant was identified in controls in ",
      variant$mafac, "of", format(variant$mafan, big.mark = ","),
      " chromosomes at a frequency of ",
      variant$mafaf, "(Genome Aggregation Database Nov 1 2023 v4.0.0). ",
      if (grepl("pathogenic", variant$interpretation, ignore.case = TRUE)) {
        paste(
          "The variant has been found in the literature to segregate",
          "with disease in multiple families (",
          paste(paste0("PMID:", generate_pubmed()), collapse = ", "),
          "). Furthermore, in functional studies, the variant is",
          "demonstrated to disrupt protein stability and function (",
          paste(paste0("PMID:", generate_pubmed(2)), collapse = ", "),
          "). The ", p_detail$ref, c_detail$ref, " residue is conserved",
          "across mammals and other organisms, and computational",
          "analyses (PolyPhen-2, SIFT, AlignGVGD, MutationTaster) suggest",
          "that the variant may impact the protein."
        )
      } else {
        paste(
          "Functional studies did not demonstrate a disruption of protein",
          "stability and function (",
          paste(paste0("PMID:", generate_pubmed(2)), collapse = ", "),
          "). The ", p_detail$ref, c_detail$ref, " residue is moderately",
          "conserved across mammals and other organisms, and computational",
          "analyses (PolyPhen-2, SIFT, AlignGVGD, MutationTaster) suggest",
          "that the variant may not impact the protein."
        )
      },
      "The variant occurs outside of the splicing consensus sequence",
      "and in silico or computational prediction software programs",
      "(SpliceSiteFinder, MaxEntScan, NNSPLICE, GeneSplicer) do not predict",
      "a difference in splicing.",
      "In summary, based on the above information, the clinical",
      "significance of this variant is classified as",
      paste0(variant$interpretation, ".\n\n"),
      "{\\bf References (PMIDs)}:", paste(generate_pubmed(), collapse = ", "),
      "\n\n"
    )
  }) |> paste(collapse = "\n\n")

}
