
library(biomaRt)
ensembl <- useEnsembl(biomart = 'genes', dataset = 'hsapiens_gene_ensembl')
# Get the gene IDs for the genes of interest
genes_of_interest <- c("BRCA1", "TP53", "EGFR")
gene_ids <- getBM(attributes = c('ensembl_gene_id', 'external_gene_name'),
                  filters = 'external_gene_name',
                  values = genes_of_interest,
                  mart = ensembl)
# Print the gene IDs
print(gene_ids)