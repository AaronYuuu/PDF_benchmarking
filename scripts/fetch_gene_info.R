
library(yaml)
library(biomaRt)
library(httr)
library(RJSONIO)
options(timeout=180)
#get gene list from field values
gene_table <- yaml.load_file("data/field_values.yml")

#query biomart for genes


get_connections <- function(){
  mirrors <- c(
    "https://useast.ensembl.org",
    "https://www.ensembl.org",
    "https://uswest.ensembl.org",
    "https://asia.ensembl.org"
  )
  connections <- lapply(mirrors, function(mirror) {
    tryCatch({
      useMart("ensembl", dataset = "hsapiens_gene_ensembl", host = mirror)
    }, error = function(e) {
      message(paste("Failed to connect to", mirror, ":", e$message))
      NULL
    })
  })
  connections <- Filter(Negate(is.null), connections)
  if (length(connections) == 0) {
    stop("No working connections to Ensembl found.")
  }
  return(connections)
}

Sys.sleep(5)  # wait a bit before trying again
ensembl_connections <- tryCatch(get_connections(), error = function(e) NULL)

if (is.null(ensembl_connections) || length(ensembl_connections) == 0) {
  stop("All Ensembl mirror connections failed. Try again later.")
}
ensembl <- ensembl_connections[[1]]

results <- getBM(
  attributes = c(
    "external_gene_name",
    "refseq_mrna",
    "transcript_is_canonical",
    "coding",
    "chromosome_name",
    "start_position"
  ),
  filters = "external_gene_name",
  values = gene_table$genes,
  mart = ensembl
)
exons <- getBM(
  attributes = c(
    "ensembl_gene_id",
    "external_gene_name",
    "exon_chrom_start",
    "exon_chrom_end",
    "rank", # rank or exon number within the transcript
    "ensembl_transcript_id", 
    "chromosome_name"
  ),
  filters = "external_gene_name",
  values = gene_table$genes,
  mart = ensembl
)

#filter out non-canonical transcripts and empty values
results_filtered <- results[which(
  results$transcript_is_canonical == 1 &
    results$coding != "Sequence unavailable" &
    !is.na(results$chromosome_name)
), ]
#filter out duplicates
results_filtered <- results_filtered[
  !duplicated(results_filtered$external_gene_name),
]
exons_filtered <- exons[which(
  !is.na(exons$exon_chrom_start) &
    !is.na(exons$exon_chrom_end) &
    !is.na(exons$rank)
), ]
#re-order table columns
results_filtered <- results_filtered[, c(
  "external_gene_name", "refseq_mrna",
  "chromosome_name", "start_position",
  "coding"
)]

# Check transcript IDs against entrez e-utils to get version code 
# and resolve cases with multiple IDs
counter <- 1
base_url <- "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
strsplit(results_filtered$refseq_mrna, ";") |> lapply(\(nms) {
  nm_str <- paste0(nms, collapse = ",")
  url <- paste0(base_url, "?db=nuccore&id=", nm_str, "&retmode=json")
  cat("Processing ", results_filtered[counter, "external_gene_name"], "\n")
  #sleep 300ms per iteration, to not stress out the eutils API
  Sys.sleep(0.3)
  response <- GET(url)
  counter <<- counter + 1
  if (status_code(response) == 200) {
    data <- fromJSON(content(response, "text"))
    uids <- data$result$uids
    if (length(uids) > 0) {
      accessions <- sapply(data$result[uids], \(e) e$accessionversion)
      #pick the one with the most revisions
      versions <- as.integer(sub("^.+\\.", "", accessions))
      #if there's a tie, pick the last one.
      winner_idx <- max(which(versions == max(versions)))
      return(accessions[winner_idx])
    }
  }
  #fallback: make something up; pick the last one and use version 1
  return(paste0(nms[length(nms)], ".1"))
}) -> refseq_accessions

#fix entries with contig names as chromosomes and update transcripts
results_final <- results_filtered
results_final$chromosome_name <- 
  gsub("^HSCHR|_.+$", "", results_filtered$chromosome_name)
results_final$refseq_mrna <- do.call(c, refseq_accessions)
# write result to file
write.csv(results_final, "data/gene_info.csv", row.names = FALSE)
write.csv(exons_filtered, "data/exon_info.csv", row.names = FALSE)
paste("Gene info, exon info, saved to 
data/gene_info.csv, data/exon_info.csv",
      sep = "\n") |> cat()
