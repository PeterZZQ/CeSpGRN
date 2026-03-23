from scenicplus.TF_to_gene import calculate_TFs_to_genes_relationships
from scenicplus.grn_builder.modules import eRegulon
import pandas as pd
import pathlib
import anndata 
import sys, os
import numpy as np
from typing import Literal, List
from tqdm import tqdm

GBM_KWARGS = {
    'learning_rate': 0.01,
    'n_estimators': 500,
    'max_features': 0.1
}

RHO_THRESHOLD = 0.03

def calculate_regions_to_genes_relationships(
        region_to_gene_importances: pd.DataFrame,
        region_to_gene_correlation: pd.DataFrame,
        search_space: pd.DataFrame,
        out_dir: str,
        add_distance: bool = True):
    """
    # TODO: add docstrings
    """
    # transform dictionaries to pandas dataframe
    result_df = pd.concat([pd.DataFrame(data={'target': gene,
                                                'region': region_to_gene_importances[gene].index.to_list(),
                                                'importance': region_to_gene_importances[gene].to_list(),
                                                'rho': region_to_gene_correlation[gene].loc[
                                                    region_to_gene_importances[gene].index.to_list()].to_list()})
                            for gene in region_to_gene_importances.keys()
                            ]
                            )
    result_df = result_df.reset_index()
    result_df = result_df.drop('index', axis=1)
    result_df['importance_x_rho'] = result_df['rho'] * \
        result_df['importance']
    result_df['importance_x_abs_rho'] = abs(
        result_df['rho']) * result_df['importance']
    if add_distance:
        search_space_rn = search_space.rename(
            {'Name': 'region', 'Gene': 'target'}, axis=1).copy()
        result_df = result_df.merge(search_space_rn, on=['region', 'target'])
        #result_df['Distance'] = result_df['Distance'].map(lambda x: x[0])
    # log.info('Done!')
    result_df.to_csv(out_dir + "results_df.csv")


def infer_TF_to_gene(
        adata_rna: anndata.AnnData,
        temp_dir: pathlib.Path,
        adj_out_fname: pathlib.Path,
        method: Literal["GBM", "RF"],
        n_cpu: int,
        seed: int):
    
    print("Reading scRNA-seq anndata.")
    # TODO: check if the data need to be normalized
    counts_rna = adata_rna.to_df()
    tf_names = [x for x in adata_rna.var.loc[adata_rna.var["tf"] == True,:].index.values]

    print(f"Using {len(tf_names)} TFs.")
    adj = calculate_TFs_to_genes_relationships(
        df_exp_mtx=counts_rna,
        tf_names = tf_names,
        temp_dir = temp_dir,
        method = method,
        n_cpu = n_cpu,
        seed = seed)
    print(f"Saving TF to gene adjacencies to: {adj_out_fname.__str__()}")
    adj.to_csv(adj_out_fname, sep="\t", header = True, index = False)


def _format_egrns(
        eRegulons: List[eRegulon],
        tf_to_gene: pd.DataFrame):
    """Helper function to format eRegulons to a pandas dataframe."""
    REGION_TO_GENE_COLUMNS = [
        "Region",
        "Gene",
        "importance",
        "rho",
        "importance_x_rho",
        "importance_x_abs_rho"
    ]
    eRegulons_formatted = []
    for ereg in eRegulons:
        TF = ereg.transcription_factor
        is_extended = ereg.is_extended
        region_to_gene = pd.DataFrame(
            ereg.regions2genes,
            columns=REGION_TO_GENE_COLUMNS)
        n_target_regions = len(set(region_to_gene["Region"]))
        n_target_genes = len(set(region_to_gene["Gene"]))
        # TF_[extended,direct]_[+,-]/[+,-]
        eRegulon_name = TF + "_" + \
            ("extended" if is_extended else "direct") + "_" + \
            ("+" if "positive tf2g" in ereg.context else "-") + "/" + \
            ("+" if "positive r2g" in ereg.context else "-")
        # TF_[extended,direct]_[+,-]/[+,-]_(nr)
        region_signature_name = eRegulon_name + "_" + f"({n_target_regions}r)"
        # TF_[extended,direct]_[+,-]/[+,-]_(ng)
        gene_signature_name = eRegulon_name + "_" + f"({n_target_genes}g)"
        # construct dataframe
        region_to_gene["TF"] = TF
        region_to_gene["is_extended"] = is_extended
        region_to_gene["eRegulon_name"] = eRegulon_name
        region_to_gene["Gene_signature_name"] = gene_signature_name
        region_to_gene["Region_signature_name"] = region_signature_name
        eRegulons_formatted.append(region_to_gene)
    eRegulon_metadata = pd.concat(eRegulons_formatted)
    eRegulon_metadata = eRegulon_metadata.merge(
        right=tf_to_gene.rename({"target": "Gene"}, axis = 1), #TODO: rename col beforehand!
        how="left",
        on= ["TF", "Gene"],
        suffixes=["_R2G", "_TF2G"])
    return eRegulon_metadata


def infer_grn(
        tf_to_gene: pd.DataFrame,
        region_to_gene: pd.DataFrame,
        cistromes: anndata.AnnData,
        eRegulon_out_fname: pathlib.Path,
        ranking_db_fname: str,
        is_extended: bool,
        temp_dir: pathlib.Path,
        order_regions_to_genes_by: str,
        order_TFs_to_genes_by: str,
        gsea_n_perm: int,
        quantiles: List[float],
        top_n_regionTogenes_per_gene: List[float],
        top_n_regionTogenes_per_region: List[float],
        binarize_using_basc: bool,
        min_regions_per_gene: int,
        rho_dichotomize_tf2g: bool,
        rho_dichotomize_r2g: bool,
        rho_dichotomize_eregulon: bool,
        keep_only_activating: bool,
        rho_threshold: float,
        min_target_genes: int,
        n_cpu: int,
        seed: int):
    """
    Infer gene regulatory network.

    Parameters
    ----------
    TF_to_gene_adj_fname : pathlib.Path
        Path to TF to gene adjacency file.
    region_to_gene_adj_fname : pathlib.Path
        Path to region to gene adjacency file.
    cistromes_fname : pathlib.Path
        Path to cistromes file.
    eRegulon_out_fname : pathlib.Path
        Path to store output.
    ranking_db_fname : str
        Path to ranking database.
    is_extended : bool
        Whether to use extended cistromes.
    temp_dir : pathlib.Path
        Path to temporary directory.
    order_regions_to_genes_by : str
        Order regions to genes by.
    order_TFs_to_genes_by : str
        Order TFs to genes by.
    gsea_n_perm : int
        Number of permutations for GSEA.
    quantiles : List[float]
        List of quantiles used for binarizing region-to-gene adjacencies.
    top_n_regionTogenes_per_gene : List[float]
        List of top n regions per gene, used for binarizing
        region-to-gene adjacencies.
    top_n_regionTogenes_per_region : List[float]
        List of top n regions per region, used for binarizing
        region-to-gene adjacencies.
    binarize_using_basc : bool
        Whether to binarize region-to-gene adjacencies using BASC.
    min_regions_per_gene : int
        Minimum number of regions per gene.
    rho_dichotomize_tf2g : bool
        Whether to dichotomize TF-to-gene adjacencies.
    rho_dichotomize_r2g : bool
        Whether to dichotomize region-to-gene adjacencies.
    rho_dichotomize_eregulon : bool
        Whether to dichotomize eRegulons.
    keep_only_activating : bool
        Whether to keep only activating eRegulons.
    rho_threshold : float
        Threshold for dichotomizing.
    min_target_genes : int
        Minimum number of target genes.
    n_cpu : int
        Number of parallel processes to run.
    seed: int
        Random seed to use.

    """
    # from scenicplus.grn_builder.gsea_approach import build_grn
    from grn_func import build_grn
    from scenicplus.triplet_score import calculate_triplet_score
    print("Loading TF to gene adjacencies.")
    # tf_to_gene = pd.read_table(TF_to_gene_adj_fname)

    print("Loading region to gene adjacencies.")
    # region_to_gene = pd.read_table(region_to_gene_adj_fname)


    eRegulons = build_grn(
        tf_to_gene=tf_to_gene,
        region_to_gene=region_to_gene,
        cistromes=cistromes,
        is_extended=is_extended,
        temp_dir=temp_dir.__str__(),
        order_regions_to_genes_by=order_regions_to_genes_by,
        order_TFs_to_genes_by=order_TFs_to_genes_by,
        gsea_n_perm=gsea_n_perm,
        quantiles=quantiles,
        top_n_regionTogenes_per_gene=top_n_regionTogenes_per_gene,
        top_n_regionTogenes_per_region=top_n_regionTogenes_per_region,
        binarize_using_basc=binarize_using_basc,
        min_regions_per_gene=min_regions_per_gene,
        rho_dichotomize_tf2g=rho_dichotomize_tf2g,
        rho_dichotomize_r2g=rho_dichotomize_r2g,
        rho_dichotomize_eregulon=rho_dichotomize_eregulon,
        keep_only_activating=keep_only_activating,
        rho_threshold=rho_threshold,
        NES_thr=0,
        adj_pval_thr=1,
        min_target_genes=min_target_genes,
        n_cpu=n_cpu,
        merge_eRegulons=True,
        disable_tqdm=False,
        seed=seed)

    # TODO: check the functions below
    print("Formatting eGRN as table.")
    eRegulon_metadata = _format_egrns(
        eRegulons=eRegulons,
        tf_to_gene=tf_to_gene)

    # # cross-check with the database
    # print("Calculating triplet ranking.")
    # eRegulon_metadata = calculate_triplet_score(
    #     cistromes=cistromes,
    #     eRegulon_metadata=eRegulon_metadata,
    #     ranking_db_fname=ranking_db_fname)

    print(f"Saving network to {eRegulon_out_fname.__str__()}")
    eRegulon_metadata.to_csv(
        eRegulon_out_fname,
        sep="\t", header=True, index=False)



