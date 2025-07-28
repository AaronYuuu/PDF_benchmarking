#FIXME: need a reset function for the cached data
#to be called by subsequent document interpolations!

#storage for variant groups
var_groups <- list()

get_var_groups <- function(variants) {
  if (length(var_groups) > 0) {
    return(var_groups)
  }

  #randomly select variants as somatic and germline
  is_somatic <- as.logical(sample.int(2, length(variants), replace = TRUE) - 1)
  interpretations <- sapply(variants, `[[`, "interpretation")
  is_plp <- grepl("pathogenic", interpretations, ignore.case = TRUE)

  var_groups <<- list(
    somatic_actionable = variants[is_somatic & is_plp],
    somatic_vus = variants[is_somatic & !is_plp],
    germline_plp = variants[!is_somatic & is_plp],
    germline_vus = variants[!is_somatic & !is_plp]
  )

  return(var_groups)
}

#tumor mutation burden storage
tmb <- NULL

get_tmb <- function(...) {
  if (!is.null(tmb)) {
    return(tmb)
  }
  tmb <<- sprintf("%.3f", runif(1, 0.1, 1.5))
  return(tmb)
}

vafs <- NULL
get_vafs <- function(vars) {
  if (!is.null(vafs)) {
    return(vafs)
  }
  ids <- sapply(vars, `[[`, "mega_hgvs")
  vafs <<- sprintf("%.02f", runif(length(vars), 5, 85)) |> setNames(ids)
  return(vafs)
}

#store modes of inheritance
inhs <- NULL
get_inhs <- function(vars) {
  if (!is.null(inhs)) {
    return(inhs)
  }
  ids <- sapply(vars, `[[`, "mega_hgvs")
  inhs <<- sapply(vars, `[[`, "chromosome") |>
    sapply(\(chrom) {
      paste0(
        switch(chrom, chrX = "X", "A"),
        sample(c("D", "R"), 1)
      )
    }) |>
    setNames(ids)
  return(inhs)
}

#reset function to delete data from previous documents
reset <- function(...) {
  inhs <<- NULL
  vafs <<- NULL
  tmb <<- NULL
  var_groups <<- list()
  return("")
}

findings_tables <- function(dataset) {
  variants <- dataset$variants

  tmb <- get_tmb()
  var_groups <- get_var_groups(variants)

  paste0(
    "{\\bf Somatic Findings:}\n\n",
    "\\vspace{1em}\n",
    "\\begin{tabular}{p{8cm} p{2cm} p{7cm}}\n",
    "Tumour Mutation Burden: & Not actionable & ", tmb, " mutations/Mb\\\\\n",
    " & & \\\\\n",
    "{\\bf SNV/indels and copy number changes} & Gene Name & Variant \\\\\n",
    if (length(var_groups$somatic_actionable) == 0) {
      "Clinically actionable variants in patient's tumour type & - & -\\\\\n"
    } else {
      paste0(
        "Clinically actionable variants in patient's tumour type",
        paste0(sapply(var_groups$somatic_actionable, \(v) {
          paste0(
            " & ", v$gene_symbol, " & ", v$hgvsc, " (", v$hgvsp, ")\\\\\n"
          )
        }), collapse = "")
      )
    },
    "Clinically actionable variants in a different tumour type & - & -\\\\\n",
    "Clinically actionable variants detected at low VAF & - & -\\\\\n",
    if (length(var_groups$somatic_vus) == 0) {
      "Oncogenic variants with uncertain clinical actionability & - & - \\\\\n"
    } else {
      paste0(
        "Oncogenic variants with uncertain clinical actionability",
        paste(sapply(var_groups$somatic_vus, \(v) {
          paste0(
            " & ", v$gene_symbol, " & ", v$hgvsc, " (", v$hgvsp, ")\\\\\n"
          )
        }), collapse = "")
      )
    },
    "Variants of uncertain clinical significance & - & -\\\\\n",
    "\\end{tabular}\n\n",
    "\\vspace{1em}\n",
    "{\\bf Germline Findings:}\n\n",
    "\\vspace{1em}\n",
    "\\begin{tabular}{p{8cm} p{2cm} p{7cm}}\n",
    "{\\bf SNV/indels and copy number changes} & Gene Name & Variant \\\\\n",
    if (length(var_groups$germline_plp) == 0) {
      "Pathogenic/Likely pathogenic variants & - & - \\\\\n"
    } else {
      paste0(
        "Pathogenic/Likely pathogenic variants",
        paste(sapply(var_groups$germline_plp, \(v) {
          paste0(
            " & ", v$gene_symbol, " & ", v$hgvsc, " (", v$hgvsp, ")\\\\\n"
          )
        }), collapse = "")
      )
    },
    if (length(var_groups$germline_vus) == 0) {
      "Variants of uncertain significance & - & - \\\\\n"
    } else {
      paste0(
        "Variants of uncertain significance",
        paste(sapply(var_groups$germline_vus, \(v) {
          paste0(
            " & ", v$gene_symbol, " & ", v$hgvsc, " (", v$hgvsp, ")\\\\\n"
          )
        }), collapse = "")
      )
    },
    "\\end{tabular}\n\n",
    "\\vspace{1em}\n",
    "Tumour DNA: ",
    paste(sapply(var_groups$somatic_actionable, \(v) {
      paste(
        "A clinically actionable variant in the",
        v$gene_symbol, 
        "gene was identified, which is associated",
        "with poor prognosis in angiosarcomas."
      )
    }), collapse = ""),
    paste(sapply(var_groups$somatic_vus, \(v) {
      paste(
        "An oncogenic variant with uncertain clinical actionability in the",
        v$gene_symbol, 
        "gene was identified."
      )
    }), collapse = ""),
    "Tumour mutation burden is not actionable.\n\n",
    "\\vspace{1em}\n",
    "Germline DNA: ",
    paste(sapply(var_groups$germline_plp, \(v) {
      paste(
        "A", v$interpretation, "variant in the",
        v$gene_symbol, 
        "gene was found in the germline of this patient."
      )
    }), collapse = ""),
    paste(sapply(var_groups$germline_vus, \(v) {
      paste(
        "A", v$interpretation, "in the",
        v$gene_symbol, 
        "gene was found in the germline of this patient."
      )
    }), collapse = ""),
    "The results suggest that these variants are mosaic ",
    "(\\textasciitilde 39\\%) in the peripheral blood of this patient."
  )

}

somatic_table <- function(dataset) {

  variants <- dataset$variants
  var_groups <- get_var_groups(variants)
  soma_vars <- c(var_groups$somatic_plp, var_groups$somatic_vus)
  
  if (length(soma_vars) == 0) {
    return("No variants to report.")
  }


  #variant allele frequencies are indexed using mega_hgvs as IDs
  vafs <- get_vafs(soma_vars)

  sapply(soma_vars, \(v) {
    paste0(
      v$gene_symbol, " & ", v$hgvsc, " & ",
      cap(v$chromosome), "(", dataset$reference_genome, "): & ",
      vafs[[v$mega_hgvs]], " & ", v$interpretation, "\\\\\n",
      tex_escape(v$transcript_id), " & (", v$hgvsp, ") & ", v$hgvsg, " & & \\\\"
    )
  }) |> paste(collapse = "\n")
}


somatic_blurb <- function(dataset) {

  variants <- dataset$variants
  var_groups <- get_var_groups(variants)
  soma_vars <- c(var_groups$somatic_plp, var_groups$somatic_vus)
  #variant allele frequencies are indexed using mega_hgvs as IDs
  vafs <- get_vafs(soma_vars)

  paste(
    "{\\bf Clinically actionable:} Variants with known therapeutic;",
    "prognostic or diagnostic actionability in the patients tumour ",
    "type or in a different tumour type\n\n",
    somatic_sub_blurb(var_groups$somatic_plp, vafs),
    "\n\n",
    "{\\bf Variants of Uncertain Clinical Significance (VUS)}:",
    "Variants of uncertain association with therapeutic, prognostic",
    "or diagnostic actionability.\n\n",
    somatic_sub_blurb(var_groups$somatic_vus, vafs)
  )

}

somatic_sub_blurb <- function(vars, vafs) {

  if (length(vars) == 0) {
    return("No variants found.")
  }

  #parse HGVS strings
  hgvsps <- sapply(vars, `[[`, "hgvsp")
  hgvsp_data <- hgvsParseR::parseHGVS(hgvsps)
  #fix missing stop codon type in var_data table
  if (any(grep("Ter$", hgvsps))) {
    hgvsp_data$type[grep("Ter$",hgvsps)] <- "stop"
  }
  # hgvscs <- sapply(vars, `[[`, "hgvsc")
  # hgvsc_data <- hgvsParseR::parseHGVS(hgvscs)

  sapply(seq_along(vars), \(i) {
    v <- vars[[i]]
    h <- hgvsp_data[i,]
    vaf <- vafs[[i]]
    paste0(
      "Variant: ", v$hgvsc, " (", v$hgvsp, ") in the ",
      v$gene_symbol, " gene\\\\",
      "Variant type: ",
      switch(h$type,
        substitution = "single amino acid substitution",
        stop = "stop gain SNV",
        frameshift = "frameshift variant"
      ),
      "\\\\\n",
      "Variant Allele Fraction: ", vaf, "\\% \\\\\n",
      "The ", v$hgvsc, " variant located in exon ", v$exon,
      "of the ", v$gene_symbol, " gene has been reported ",
      "twice in other tumors (Cosmic 1x, cBioPortal 1x). ",
      "Loss of ", v$gene_symbol, " expression is associated ",
      "with increased tumor growth, including in angiosarcomas ",
      "( PMIDs: ", paste(generate_pubmed(), collapse = ", "), "). ",
      if (grepl("pathogenic", v$interpretation, ignore.case = TRUE)) {
        paste0(
          v$gene_symbol, " loss may be elible for clinical trials ",
          "in other tumor types (mycancergenome.org). In addition, ",
          "preclinical evidence has shown efficacy of a number of ",
          "kinase inhibitors (PMID:", generate_pubmed(1), ")"
        )
      } else {
        paste0(
          "However; clinical actionability of this variant remains ",
          "uncertain at this time."
        )
      }
    )
  }) |> paste(collapse = "\n\n")
}

germline_table <- function(dataset) {
  variants <- dataset$variants
  var_groups <- get_var_groups(variants)
  gl_vars <- c(var_groups$germline_plp, var_groups$germline_vus)

  if (length(gl_vars) == 0) {
    return("No variants to report.")
  }

  inhs <- get_inhs(gl_vars)

  sapply(gl_vars, \(v) {
    paste0(
      v$gene_symbol, " & ", v$hgvsc, " & ",
      cap(v$chromosome), "(", dataset$reference_genome, "): & ",
      v$mafaf, " & ", v$zygosity, " & ", inhs[[v$mega_hgvs]], " & ",
      v$interpretation, "\\\\\n",
      tex_escape(v$transcript_id), " & (", v$hgvsp, ") & ", v$hgvsg, " & & \\\\"
    )
  }) |> paste(collapse = "\n")
}

germline_blurb <- function(dataset) {
  variants <- dataset$variants
  var_groups <- get_var_groups(variants)
  gl_vars <- c(var_groups$germaline_plp, var_groups$germaline_vus)

  inhs <- get_inhs(gl_vars)

  paste0(
    "{\\bf Pathogenic/Likely Pathogenic}: Variants associated with ",
    "cancer predisposition or therapeutic actionability in the patients ",
    "tumour type or in a different tumour type.\n\n",
    germline_sub_blurb(var_groups$germline_plp, inhs),
    "\n\n",
    germline_sub_blurb(var_groups$germline_vus, inhs),
    "\n\n",
    "Recommendations: This patient and/or guardian should receive ",
    "genetic counselling to discuss the implications of this result."
  )
}

germline_sub_blurb <- function(vars, inhs) {

  if (length(vars) == 0) {
    return("No variants found.")
  }

  #parse HGVS strings
  hgvsps <- sapply(vars, `[[`, "hgvsp")
  hgvsp_data <- hgvsParseR::parseHGVS(hgvsps)
  #fix missing stop codon type in var_data table
  if (any(grep("Ter$", hgvsps))) {
    hgvsp_data$type[grep("Ter$",hgvsps)] <- "stop"
  }

  sapply(seq_along(vars), \(i) {
    v <- vars[[i]]
    h <- hgvsp_data[i,]
    paste0(
      "Variant: ", v$hgvsc, " (", v$hgvsp, ") in the ",
      v$gene_symbol, "gene\\\\\n",
      "Mode of Inheritance: ", inhs[[v$mega_hgvs]], "\\\\\n",
      "Avg frequency data (gnomAD): ", v$mafaf, "\\\\n",
      "In silico Programs (Sift, PolyPhen, Mutation Taster): ",
      "Not assessed\n\n",
      "Comment: The ", v$interpretation, ", ",
      v$hgvsc, " ", v$hgvsp, " variant ",
      switch(
        h$type,
        substitution = paste(
          "replaces a", h$ancestral, "amino acid at position",
          h$start, "with a", h$variant, ". "
        ),
        stop = paste(
          "introduces an early translation termination at codon",
          h$start, ", thus producing a truncated protein. "
        ),
        frameshift = paste(
          "introduces in a frameshift starting at codon",
          h$start, "resulting in an early tranlation termination",
          "and thus produces a truncated protein. "
        )
      ),
      "To the best of our knowledge, this variant has not been ",
      "reported in the literature.",
      "It has been observed in the general population at a ",
      "minor allele frequency of ", v$mafaf, " (gnomAD), which is",
      "consistent with a ", v$interpretation, " classification. ",
      "Pathogenic variants in the ", v$gene_symbol, " gene are ",
      "associated with a number of cancer syndromes (PMIDs: ",
      generate_pubmed(), "). This variant should be interpreted in",
      "the context of clinical findings, family history, and other ",
      "experimental data. "
    )
  }) |> paste(collapse = "\n\n")
}
