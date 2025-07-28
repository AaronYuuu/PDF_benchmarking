
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
      "A", variant$interpretation, "in exon", 
      variant$exon, "of the", variant$gene_symbol,
      "gene was detected in this individual.",
      "This variant, ",variant$hgvsc, ", ",
      variant$hgvsp, ", is predicted to result in",
      switch(p_detail$type,
        substitution = paste(
          "the substitution of", aaname(p_detail$variant),
          "for the", aaname(p_detail$ancestral),
          "at codon", p_detail$start, "."
        ),
        frameshift = paste(
          "a frameshift beginning at codon", p_detail$start,
          "resulting in an early translation termination."
        ),
        stop = paste(
          "an early translation termination at codon", p_detail$start, "."
        )
      ),
      "This variant has been reported in Clinvar by multiple",
      "submitters as", variant$interpretation, ". ",
      "It does not appear to have been discussed in the literature.",
      "The allele frequency is ",variant$mafac, "/", variant$mafan,
      "in the Genome Aggregation Database (v2.1.1, non-cancer).",
      "In silico anlyses (Align GVGD, MutationTaster; POLYPHEN-2 and SIFT)",
      "were consistent in predicting this variant to be",
      tolower(variant$interpretation), ".",
      "This nucleotide is",
      if (grepl("pathogenic", variant$interpretation, ignore.case = TRUE)) {
        "conserved (phyloP = 0.7)."
      } else {
        "not conserved (phyloP = 0.1)."
      },
      "This variant would be classified as", tolower(variant$interpretation), "."

    )
  }) |> paste(collapse = "\n\n")
}

# % A variant of uncertain significance in exon 13 of the PALB2 gene was detected in this individual. 
# This variant, c.3538A>G, p.(Ile1180Val), is predicted to result in the substitution of valine for 
# the isoleucine at codon 1180. This variant has been reported in ClinVar by multiple submissions as 
# a variant of uncertain significance. It does not appear to have been discussed in the literature: 
# The allele frequency is 1/236920 in the Genome Aggregation Database (v2.1.l, non-cancer). In silico 
# analyses (Align GVGD, MutationTaster; POLYPHEN-2 and SIFT) were consistent in predicting this variant 
# to be benign. This nucleotide is weakly conserved (phyloP = 0.69). This variant would be classified 
# as a variant of uncertain significance. 
#
# % In addition, a variant of uncertain significance in exon 9 of the RAD51D gene was detected. 
# This variant, c.868C>T, p.(Arg290Trp): is predicted to result in the substitution of tryptophan 
# for the arginine at codon 290. This variant has been reported in ClinVar by multiple submissions 
# as a variant of uncertain significance.  It does not appear to have been discussed in the literature; 
# The allele frequency is 8/268292 in the Genome Aggregation Database (v2.1.1, non-cancer). 
# In silico analyses (Align GVGD; MutationTaster POLYPHEN-2 and SIFT) were consistent in predicting 
# this variant to be benign.  This nucleotide is not conserved (phyloP = -0.04). This variant would 
# be classified as a variant of uncertain significance. 
#
# % No conclusions as to the pathogenicity of these variants can be made from this analysis and thus 
# it cannot be used to modify the clinically determined risk of breast or ovarian cancer. Family studies 
# may be helpful in further elucidating the significance of these variants with respect to familial 
# breast/ovarian cancer: Genetic counseling and clinical follow-up are recommended.
