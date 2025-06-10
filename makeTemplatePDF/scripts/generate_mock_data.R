#!/usr/bin/Rscript
# mockups.R generates a mock dataset in JSON format

library(yaml)
library(hgvsParseR)
library(RJSONIO)
library(argparser)
#parse command line arguments
ap <- arg_parser("generate a mock dataset in JSON format", name = "mockups.R")
ap <- add_argument(ap,
  "--amount",
  help = "number of reports to generate",
  default = 1L
)
ap <- add_argument(ap,
  "--outfile",
  help = "the output file",
  default = "mock_data.json"
)
args <- parse_args(ap)
num_reports <- args$amount
out_file <- args$outfile

#Load source data
field_values <- yaml.load_file("../data/field_values.yml")
gene_info <- read.csv("../data/gene_info.csv", row.names = 1)
exons_df <- read.csv("../data/exon_info.csv", stringsAsFactors = FALSE)

# Helper function to sample from fields
sample_field <- function(name, num = 1) sample(field_values[[name]], num)

#generate UUIDv4
gen_uuid <- function() {
  alphanum <- function(n) {
    paste(sample(c(letters[1:6], 0:9), n, replace = TRUE), collapse = "")
  }
  paste(
    alphanum(8), "-", alphanum(4), "-4", alphanum(3), "-", 
    sample(c("8", "9", "a", "b"), 1), alphanum(3), "-", alphanum(12),
    collapse = "", sep = ""
  )
}

# sample from a power-law distribution using rejection sampling
rpow <- function(n, alpha = 1000, range = 5e5) {
  # n: Number of samples to generate
  # alpha: Power-law exponent (must be > 1)
  # range: Range / tail-length of the distribution (default is 5e5)
  if (alpha <= 1) {
    stop("Alpha must be greater than 1 for a valid power-law distribution.")
  }
  # Define the target power-law probability density function
  power_law_pdf <- function(x) {
    ifelse(x >= 1, (alpha - 1) * x^(-alpha), 0)
  }
  # Define the proposal distribution (uniform in this case)
  proposal_pdf <- function(x) {
    ifelse(x >= 1, 1, 0)
  }
  # Find the maximum ratio of target PDF to proposal PDF
  M <- (alpha - 1) * 1^(-alpha)
  # Rejection sampling
  samples <- numeric(0)
  while (length(samples) < n) {
    # Sample from the proposal distribution
    x_proposal <- runif(1, min = 1, max = 1 * 10) # Adjust range as needed
    # Compute acceptance probability
    accprob <- power_law_pdf(x_proposal) / (M * proposal_pdf(x_proposal))
    # Accept or reject the sample
    if (runif(1) < accprob) {
      samples <- c(samples, x_proposal)
    }
  }
  # Scale the samples to the desired range
  samples_out <- samples * range - range
  return(samples_out)
}

# generate allele frequency, count and number
gen_maf <- function(amount = 1, max_pop = 5e5) {
  # sample allele number from normal distribution
  an <- round(rnorm(amount, mean = max_pop, sd = max_pop / 5))
  # sample allele count from power-law distribution
  ac <- round(rpow(amount, alpha = 1000, range = max_pop))
  # restrict allele count to be ranged between 0 and an, enrich zeroes
  ac <- mapply(\(a, n) {
    min(n, max(0, a))
    # ifelse(a > n, n, ifelse(a < 0, 0, a)
  }, ac - 100, an)
  #calculate allele frequency
  af <- ac / an
  return(data.frame(ac = ac, an = an, af = af))
}

# Generate a set of threee dates (collected, received, verified)
gen_dates <- function() {
  d1 <- paste0(
    "20", sample(20:25, 1), "-", sprintf("%02d", sample(12, 1)), "-", sprintf("%02d", sample(28, 1))
  ) |> as.Date()  # Use as.Date instead of as.POSIXct

  d2 <- d1 + round(rnorm(1, mean = 1, sd = 0.5))  # Adds days
  d3 <- d2 + round(rnorm(1, mean = 1, sd = 0.5))

  list(
    date_collected = format(d1, "%Y-%m-%d"),
    date_received = format(d2, "%Y-%m-%d"),
    date_verified = format(d3, "%Y-%m-%d")
  )
}

# Generate mock variants
# TODO: Future iterations could sample from Clinvar instead
gen_var <- function(gene, amount = 1) {
  cds <- gene_info[gene, "coding"]
  pos <- sample(nchar(cds), amount)
  from <- sapply(pos, \(p) substr(cds, p, p))
  to <- sapply(from, \(fr) sample(setdiff(c("A", "C", "G", "T"), fr), 1))
  data.frame(pos = pos, from = from, to = as.vector(to))
}

# Generate HGVS identifiers for variants
gen_hgvs <- function(var_data, gene) {
  b_c <- new.hgvs.builder.c()  # coding builder for hgvsc
  b_g <- new.hgvs.builder.g()  # genomic builder for hgvsg
  
  hgvsc <- sapply(seq_len(nrow(var_data)), \(i) {
    with(var_data[i, ], {
      b_c$substitution(pos, from, to)
    })
  })
  # Use genomic builder for genomic coordinates with g. prefix
  hgvsg <- sapply(seq_len(nrow(var_data)), \(i) {
    with(var_data[i, ], {
      b_g$substitution(gene_info[gene, "start_position"] + pos - 1, from, to)
    })
  })
  cds_seq <- gene_info[gene, "coding"]
  hgvs <- do.call(rbind,lapply(hgvsc, \(.hgvsc) translateHGVS(.hgvsc,cds_seq)))
  hgvs <- cbind(hgvsg=hgvsg,hgvs)
  hgvs
}

# Generate VCV accessions
gen_vcv <- function(amount=1) {
  #[SRV]CV[A-Z0-9]{9}
  replicate(amount, {
    paste0("VCV00", paste0(sample(10, 7, replace = TRUE) - 1, collapse = ""))
  })
}

# Declare external_gene_name as a global variable to avoid binding warnings
globalVariables(c("external_gene_name"))

find_exon_number <- function(chromosome, hgvsg, gene_symbol, exons_df) {
  # Normalize chromosome name
  variant_chr <- gsub("^chr", "", chromosome)

  # Extract numeric position from HGVSg (e.g., "g.77510022T>C" → 77510022)
  variant_pos <- as.numeric(sub("^g\\.(\\d+).*", "\\1", hgvsg))

  # Filter exons for matching gene and chromosome
  exon_match <- exons_df[
    exons_df$external_gene_name == gene_symbol &
      exons_df$chromosome_name == variant_chr &
      exons_df$exon_chrom_start <= variant_pos &
      exons_df$exon_chrom_end >= variant_pos,
  ]

  # Return the exon number (rank), or a random number if not found
  if (nrow(exon_match) > 0) {
    return(exon_match$rank[1])  # return first match
  } else {
    return(sample(1:20, 1))  # return a random exon number if not found
  }
}





#Sample random variants
sample_variants <- function(genes) {
  num_variants <- max(1, rpois(1, 1))
  replicate(num_variants, {
    data <- list()
    data$gene_symbol <- sample(genes, 1L)
    data$variant_id <- gen_vcv()
    data$chromosome <- paste0("chr", 
      gene_info[data$gene_symbol, "chromosome_name"]
    )
    var_data <- gen_var(data$gene_symbol)
    hgvs <- gen_hgvs(var_data, data$gene_symbol)
    data <- c(data, hgvs[, 1:3])
    # TODO: Add aapos, fromAA, toAA
    data$transcript_id <- gene_info[data$gene_symbol, "refseq_mrna"]

    # TODO: data$exon
    data$exon <- find_exon_number(
      data$chromosome,
      data$hgvsg,
      data$gene_symbol,
      exons_df
    )
    if (is.null(data$exon) || is.na(data$exon) || data$exon < 1) {
      data$exon <- sample(1:20, 1)  # fallback to a random exon number
    }

    #make function to query biomart for exon derived from variant position
    data$zygosity <- sample(
      field_values$zygosity, 1,
      prob = c(.8, .2)
    )
    data$interpretation <- sample(
      field_values$interpretation, 1,
      prob = c(.6, .3, .1)
    )
    maf <- gen_maf(1)[1, , drop = TRUE]
    data$mafac <- maf$ac
    data$mafan <- maf$an
    data$mafaf <- formatC(maf$af, format = "e", digits = 2)
    data
  }, simplify = FALSE)
}

gen_genes <- function() {
  gene_symbols <- sample_field("genes", round(runif(1, 5, 20)))
  refseq_accs <- gene_info[gene_symbols, "refseq_mrna"]
  mapply(
    \(s, a) list(gene_symbol = s, refseq_mrna = a),
    gene_symbols,
    refseq_accs,
    SIMPLIFY = FALSE
  )
}

# report_date"
# report_type": Pathology, Molecular Genetics
# testing_context": "Clinical",
# ordering_clinic"
# testing_laboratory"
# sequencing scope: "Gene panel","Targeted variant testing"
# tested_genes
# sample_type: "Amplified DNA"
# "analysis_type":"Variant analysis", "Fusion analysis"
# Per variant:
#   "variant_ids": "VCVO00483847"
#   "chromosome": chrX
#   hgvsc
#   hgvsg
#   hgvp
#   gene_symbol
#   transcript_id":  "NM_000179.2"
#   "exon"
#   "reference_genome"

# Generate mock-up data using sampling functions
generate_mockup <- function() {
  data <- list()
  data <- c(data, gen_dates())
  data$report_type <- sample_field("report_types")
  data$testing_context <- sample_field("testing_contexts")
  data$ordering_clinic <- sample_field("clinics")
  data$testing_laboratory <- sample_field("labs")
  data$sequencing_scope <- sample_field("scopes")
  data$tested_genes <- gen_genes()
  data$num_tested_genes <- length(data$tested_genes)
  data$sample_type <- sample_field("sample_types")
  data$analysis_type <- sample_field("analysis_types")
  data$variants <- sample_variants(names(data$tested_genes))
  data$num_variants <- length(data$variants)
  data$reference_genome <- sample_field("reference_genomes")
  data
}

uuids <- replicate(num_reports, {
  gen_uuid()
})

out <- replicate(num_reports, {
  generate_mockup()
}, simplify = FALSE)
names(out) <- uuids

#write JSON output to file
json_out <- toJSON(out)
cat(json_out, file = out_file)

cat("Done!")