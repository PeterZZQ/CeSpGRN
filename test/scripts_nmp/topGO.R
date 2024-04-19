rm(list = ls())
gc()

# https://jdblischak.github.io/nw/analysis/mouse/go.html
library(topGO)
library(org.Mm.eg.db)

setwd("/localscratch/ziqi/CeSpGRN/")

# First create the gene universe. 
# This is all the genes tested for differential expression assigned a 1 for differentially expressed and 0 if not.
background_gene <- read.csv("test/results_nmp/wt_2000/beta_1/genes.txt", sep = ",", header = FALSE)[,1]

de_gene <- read.csv("test/results_nmp/wt_2000/beta_1/hub_gene_spinal_cord.txt", sep = ",", header = FALSE)[,1]
# de_gene <- read.csv("test/results_nmp/wt_2000/beta_1/hub_gene_mesoderm.txt", sep = ",", header = FALSE)[,1]
gene_universe <- as.integer(background_gene%in%de_gene)
gene_universe <- factor(gene_universe)
names(gene_universe) <- background_gene

# Create the topGO data object. 
# Only consider “Biological Process” categories(which is mostly the case) and use the Mouse Ensembl database for annotation.
go_data <- new("topGOdata",
               ontology = "BP",
               allGenes = gene_universe,
               nodeSize = 5,
               annotationFun = annFUN.org,
               mapping = "org.Mm.eg",
               ID = "symbol")

# performing enrichment test
# Use the classic algorithm and score the tests with Fisher’s exact test.

# https://bioconductor.org/packages/release/bioc/vignettes/topGO/inst/doc/topGO.pdf see all algorithm
# default is weight01
go_test <- runTest(go_data, algorithm = "weight01", statistic = "fisher")

# analysis of the result
# Keep the results with a Fisher’s exact test p-value < 0.05.
go_table <- GenTable(go_data, weightFisher = go_test,
                     orderBy = "weightFisher", ranksOf = "weightFisher",
                     topNodes = sum(score(go_test) < .05),
                     numChar=1000)
head(go_table)

# write.table(go_table, file = paste0("test/results_nmp/wt_2000/beta_1/GO_hub_gene_spinal_cord.csv"), sep = ",")
# write.table(go_table, file = paste0("test/results_nmp/wt_2000/beta_1/GO_hub_gene_mesoderm.csv"), sep = ",")

# Create a map of geneIDs to GO terms,
ann.genes <- genesInTerm(go_data)

# print the gene of a term
fisher.ann.genes <- genesInTerm(go_data, whichGO="GO:0009948")
fisher.ann.genes
