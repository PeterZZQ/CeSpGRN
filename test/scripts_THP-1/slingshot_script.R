rm(list = ls())
gc()
library(anndata)
library(SingleCellExperiment)
library(mclust, quietly = TRUE)
library(RColorBrewer)
library(slingshot)
library(umap)

setwd("/Users/ziqizhang/Documents/xsede/DynGRN/test/scripts_THP-1")

results_dir = "../results_THP-1_kt/de_edges/"
counts <- read.table(file = paste0(results_dir, "thetas.csv"), header = FALSE, sep = ",") 
X_pca <- as.matrix(read.table(file = paste0(results_dir, "thetas_pca.csv"), header = FALSE, sep = ","))
sce <- SingleCellExperiment(assays = List(counts = t(counts)))

reducedDims(sce) <- SimpleList(PCA = X_pca)

num_clust <- 2
cl2 <- kmeans(X_pca, centers = num_clust)$cluster
colData(sce)$kmeans <- cl2
group <- brewer.pal(num_clust,"Paired")[cl2]
plot(X_pca, col = group, pch=16, asp = 1)
legend("topleft", legend=seq(1,num_clust), pch=16, col=brewer.pal(num_clust,"Paired")[seq(1,num_clust)])

# find starting cluster from visualization
start = 2
sce <- slingshot(sce, clusterLabels = 'kmeans', reducedDim = 'PCA', start.clus = start)
ss <- colData(sce)$slingshot
pt <- ss@assays@data@listData$pseudotime


library(grDevices)
colors <- colorRampPalette(brewer.pal(11,'Spectral')[-6])(100)
plotcol <- colors[cut(sce$slingPseudotime_1, breaks=100)]

plot(reducedDims(sce)$PCA, col = plotcol, pch=16, asp = 1)
lines(SlingshotDataSet(sce), lwd=2, col='black')

write.table(pt, file = paste0(results_dir, "pt_slingshot.csv"), sep = ",", row.names = FALSE, col.names = FALSE)


