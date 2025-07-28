
# generates a summary text blurb for a set of variants
summary_blurb <- function(dataset, suffix = "are present in this patient.") {
  variants <- dataset$variants
  #if there are no variants, we're done
  if (length(variants) == 0) {
    return(paste("No variants", suffix))
  }
  var_strs <- sapply(variants, \(v) {
    paste(v$gene_symbol, v$hgvsc)
  })
  if (length(variants) == 1) {
    return(paste("The", var_strs[1], suffix))
  }
  #concatenate with commas and "and"
  paste(
    paste(var_strs[-length(var_strs)], collapse = ", "),
    "and", var_strs[[length(var_strs)]], suffix
  )
}

long_blurb <- function(dataset) {
  variants <- dataset$variants
  #if there are no variants, we're done
  if (length(variants) == 0) {
    return("No variants were detected.")
  }

  #parse HGVS strings
  hgvsps <- sapply(variants, `[[`, "hgvsp")
  var_data <- hgvsParseR::parseHGVS(hgvsps)
  #fix missing stop codon type in var_data table
  if (any(grep("Ter$", hgvsps))) {
    var_data$type[grep("Ter$",hgvsps)] <- "stop"
  }

  #index variants by interpretation
  inter_idx <- tapply(
    seq_along(variants),
    sapply(variants, `[[`, "interpretation"),
    c
  )

  sapply(names(inter_idx), \(interpretation) {

    opener <- paste(
      "\\underline{\\bf", interpretation,
      "variants identified:}\n\n"
    )

    body <- sapply(inter_idx[[interpretation]], \(i) {
      #get variant and its details
      variant <- as.list(variants[[i]])
      var_details <- var_data[i, ]

      #header shows variant descriptor
      var_header <- paste0(
        "{\\bf ",
        variant$gene_symbol, " (",
        tex_escape(variant$transcript), "):",
        variant$hgvsc, " (",
        variant$hgvsp, ")}\n\n"
      )
      #body lists interpretation and evidence
      var_body1 <- paste0(
        "Based on the following evidence, the ",
        variant$gene_symbol, variant$hgvsc, "(",variant$hgvsp, 
        ") variant is classified as ", variant$interpretation, ":"
      )

      if (var_details$type == "frameshift") {
        var_body2 <- paste(
          "1) this frameshift variant leads to a premature ",
          "termination codon, which is predicted to result",
          "in a truncated or absent protein;"
        )
      } else if (var_details$type == "stop") {
        var_body2 <- paste(
          "this early termination variant is predicted to result",
          "in a truncated or absent protein;"
        )
      } else if (var_details$type == "synonymous") {
        var_body2 <- paste(
          "this synonymous variant is predicted to cause no",
          "amino acid change;"
        )
      } else {
        var_body2 <- paste(
          "this missense variant is predicted to result in",
          "an amino acid substitution from a",
          aaname(var_details$ancestral), "at position",
          var_details$start, "to a",
          aaname(var_details$variant)
        )
      }

      if (grepl("pathogenic", interpretation, ignore.case = TRUE)) {
        var_body3 <- paste(
          "2) this variant is reported in association with an",
          "increased risk of developing breast cancer (PMID:",
          paste(generate_pubmed(), collapse = ", "), ");"
        )
      } else {
        var_body3 <- paste(
          "2) this variant has not been reported in the literature;"
        )
      }

      if (variant$mafac > 0) {
        var_body4 <- paste0(
          "3) this variant is present in gnomAD (MAF = ",
          format(variant$mafaf), ", ",
          variant$mafac, "/", variant$mafan, " alleles);"
        )
      } else {
        var_body4 <- "3) this variant is not present in gnomAD;"
      }

      num_labs <- sample.int(9,1)+1
      var_body5 <- paste0(
        "4) this variant has been reported in ClinVar (",
        variant$variant_id, ") by ", num_labs,
        " clinical laboratories."
      )

      var_closer <- paste(
        "Based on currently available information, this variant",
        "is classified as", tolower(variant$interpretation),
        "according to the ACMG 2015 variant classification guidelines",
        "(PMID: 25741868).",
        if (grepl("pathogenic", interpretation, ignore.case = TRUE)) {
          paste(
            "Although this variant is not expected to cause highly",
            "penetrant Mendelian disease, it is an established",
            "cancer risk factor."
          )
        } else {
          paste(
            "This variant is not expected to cause highly penetrant",
            "Mendelian disease."
          )
        }
      )

      paste(
        var_header, var_body1, var_body2, var_body3,
        var_body4, var_body5, var_closer
      )

    }) |> paste(collapse = "\n\n")

    paste(opener, body)

  }) |> paste(collapse = "\n\n")
}


# Based on the following evidence, the CHEK2 c. 1100de|C (p. Thr367Metfs*15) variant is classified as pathogenic: 1) this
# frameshift variant leads to a premature termination codon, which is predicted to result in a truncated or absent protein; 2)
# this variant is reported in association with an increased risk of developing breast cancer (PMID: 18172190, 24918820,
# 17428320; 3) this variant has been reported in families affected with Li-Fraumeni syndrome (PMID: 10617473,
# 11479205); 4) this variant is present in gnomAD (MAF = 0.21%, 591/280390 alleles, 1 homozogyote) with a higher
# frequency reported in the European (Finnish) subpopulation (MAF = 0.87%, 219/25124 alleles); 5) this variant has been
# reported in Clin Var by 22 clinical laboratories, with conflicting classifications (20 pathogenic, 2 uncertain). Based on
# currently available information, this variant is classified as pathogenic according to the ACMG 2015 variant classification
# guidelines (PMID: 25741868). Although this variant is not expected to cause highly penetrant Mendelian disease, it is an
# established cancer risk factor.