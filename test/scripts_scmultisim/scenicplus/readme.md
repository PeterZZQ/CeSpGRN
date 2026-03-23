scenic plus running calls the sneakmake file: https://github.com/aertslab/scenicplus/blob/main/src/scenicplus/snakemake/Snakefile, with yml file https://github.com/aertslab/scenicplus/blob/main/src/scenicplus/snakemake/config.yaml
The first shell command calls:
```
scenicplus prepare_data prepare_GEX_ACC 
```
which is a wrapper for function `process_multiome_data`(https://github.com/aertslab/scenicplus/blob/main/src/scenicplus/data_wrangling/adata_cistopic_wrangling.py#L16)



## Pipeline
#### prepare_GEX_ACC_multiome
#### motif_enrichment_cistarget
```
scenicplus grn_inference motif_enrichment_cistarget
```

#### motif_enrichment_dem

#### prepare_menr
```
scenicplus prepare_data prepare_menr 
```

#### download_genome_annotations
```
scenicplus prepare_data download_genome_annotations
```

#### get_search_space
```
scenicplus prepare_data search_spance
```

#### tf_to_gene
Important
```
scenicplus grn_inference TF_to_gene
```
The function
```python
def TF_to_gene(arg):
    from scenicplus.cli.commands import infer_TF_to_gene
    infer_TF_to_gene(
        multiome_mudata_fname=arg.multiome_mudata_fname,
        tf_names_fname=arg.tf_names,
        temp_dir=arg.temp_dir,
        adj_out_fname=arg.out_tf_to_gene_adjacencies,
        method=arg.method,
        n_cpu=arg.n_cpu,
        seed=arg.seed)
```

And the script [TF to gene](https://github.com/aertslab/scenicplus/blob/main/src/scenicplus/TF_to_gene.py)

```python
def infer_TF_to_gene(
        multiome_mudata_fname: pathlib.Path,
        tf_names_fname: pathlib.Path,
        temp_dir: pathlib.Path,
        adj_out_fname: pathlib.Path,
        method: Literal["GBM", "RF"],
        n_cpu: int,
        seed: int):
    from scenicplus.TF_to_gene import calculate_TFs_to_genes_relationships
    log.info("Reading multiome MuData.")
    mdata = mudata.read(multiome_mudata_fname.__str__())
    with open(tf_names_fname) as f:
         tf_names = f.read().split("\n")
    log.info(f"Using {len(tf_names)} TFs.")
    adj = calculate_TFs_to_genes_relationships(
        df_exp_mtx=mdata["scRNA"].to_df(),
        tf_names = tf_names,
        temp_dir = temp_dir,
        method = method,
        n_cpu = n_cpu,
        seed = seed)
    log.info(f"Saving TF to gene adjacencies to: {adj_out_fname.__str__()}")
    adj.to_csv(
        adj_out_fname,
        sep="\t", header = True, index = False)
```

```python
def calculate_TFs_to_genes_relationships(
        df_exp_mtx: pd.DataFrame,
        tf_names: List[str],
        temp_dir: pathlib.Path,
        method: Literal['GBM', 'RF'] = 'GBM',
        n_cpu: int = 1,
        seed: int = 666) -> pd.DataFrame:
    """
    #TODO: Add docstrings
    """

    if(method == 'GBM'):
        method_params = [
            'GBM',      # regressor_type
            SGBM_KWARGS  # regressor_kwargs
        ]
    elif(method == 'RF'):
        method_params = [
            'RF',       # regressor_type
            RF_KWARGS   # regressor_kwargs
        ]

    exp_mtx, gene_names, tf_names = _prepare_input(
        expression_data = df_exp_mtx, gene_names = None, tf_names = tf_names)
    tf_matrix, tf_matrix_gene_names = to_tf_matrix(
        exp_mtx,  gene_names, tf_names)
            
    log.info('Calculating TF-to-gene importance')
    if temp_dir is not None:
        if type(temp_dir) == str:
            temp_dir = pathlib.Path(temp_dir)
        if not temp_dir.exists():
            Warning(f"{temp_dir} does not exist, creating it.")
            os.makedirs(temp_dir)
        
    TF_to_genes = joblib.Parallel(
        n_jobs = n_cpu,
        temp_folder = temp_dir)(
            joblib.delayed(infer_partial_network)(
                target_gene_name = gene,
                target_gene_expression = exp_mtx[:, gene_names.index(gene)],
                regressor_type = method_params[0],
                regressor_kwargs = method_params[1],
                tf_matrix = tf_matrix,
                tf_matrix_gene_names = tf_matrix_gene_names,
                include_meta = False,
                early_stop_window_length = EARLY_STOP_WINDOW_LENGTH,
                seed = seed)
            for gene in tqdm(
                gene_names, 
                total=len(gene_names), 
                desc=f'Running using {n_cpu} cores'))

    adj = pd.concat(TF_to_genes).sort_values(by='importance', ascending=False)
    log.info('Adding correlation coefficients to adjacencies.')
    adj = _add_correlation(adj, df_exp_mtx)
    adj = _inject_TF_as_its_own_target(
        TF2G_adj=adj, 
        inplace = False, 
        ex_mtx = df_exp_mtx)
    return adj
```



#### region_to_gene
```
scenicplus grn_inference region_to_gene
```
The function 
```python
from scenicplus.cli.commands import infer_region_to_gene
infer_region_to_gene(
    multiome_mudata_fname=arg.multiome_mudata_fname,
    search_space_fname=arg.search_space_fname,
    temp_dir=arg.temp_dir,
    adj_out_fname=arg.out_region_to_gene_adjacencies,
    importance_scoring_method=arg.importance_scoring_method,
    correlation_scoring_method=arg.correlation_scoring_method,
    mask_expr_dropout=arg.mask_expr_dropout,
    n_cpu = arg.n_cpu)
```

```python
def infer_region_to_gene(
        multiome_mudata_fname: pathlib.Path,
        search_space_fname: pathlib.Path,
        temp_dir: pathlib.Path,
        adj_out_fname: pathlib.Path,
        importance_scoring_method: Literal["RF", "ET", "GBM"],
        correlation_scoring_method: Literal["PR", "SR"],
        mask_expr_dropout: bool,
        n_cpu: int):
    """
    Infer region to gene relationships.

    Parameters
    ----------
    multiome_mudata_fname : pathlib.Path
        Path to multiome MuData file.
    search_space_fname : pathlib.Path
        Path to search space file.
    temp_dir : pathlib.Path
        Path to temporary directory.
    adj_out_fname : pathlib.Path
        Path to store output.
    importance_scoring_method : Literal["RF", "ET", "GBM"]
        Method to score importance.
    correlation_scoring_method : Literal["PR", "SR"]
        Method to score correlation.
    mask_expr_dropout : bool
        Whether to mask expression dropout.
    n_cpu : int
        Number of parallel processes to run.

    """
    from scenicplus.enhancer_to_gene import calculate_regions_to_genes_relationships
    log.info("Reading multiome MuData.")
    mdata = mudata.read(multiome_mudata_fname.__str__())
    log.info("Reading search space")
    search_space = pd.read_table(search_space_fname)
    adj = calculate_regions_to_genes_relationships(
        df_exp_mtx = mdata["scRNA"].to_df(),
        df_acc_mtx = mdata["scATAC"].to_df(),
        search_space = search_space,
        temp_dir = temp_dir,
        mask_expr_dropout = mask_expr_dropout,
        importance_scoring_method = importance_scoring_method,
        correlation_scoring_method = correlation_scoring_method,
        n_cpu = n_cpu)
    log.info(f"Saving region to gene adjacencies to {adj_out_fname.__str__()}")
    adj.to_csv(
        adj_out_fname,
        sep="\t", header = True, index = False)

```
And the function under script [`enhancer_to_gene.py`](https://github.com/aertslab/scenicplus/blob/main/src/scenicplus/enhancer_to_gene.py)

```python
def calculate_regions_to_genes_relationships(
        df_exp_mtx: pd.DataFrame,
        df_acc_mtx: pd.DataFrame,
        search_space: pd.DataFrame,
        temp_dir: pathlib.Path,
        mask_expr_dropout: bool = False,
        importance_scoring_method: Literal["RF", "ET", "GBM"] = 'GBM',
        importance_scoring_kwargs: dict = GBM_KWARGS,
        correlation_scoring_method: Literal["PR", "SR"] = 'SR',
        n_cpu: int = 1,
        add_distance: bool = True):
    """
    # TODO: add docstrings
    """
    # Create logger
    level = logging.INFO
    format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=format, handlers=handlers)
    log = logging.getLogger('R2G')
    # calulcate region to gene importance
    log.info(
        f'Calculating region to gene importances, using {importance_scoring_method} method')
    region_to_gene_importances = _score_regions_to_genes(
        df_exp_mtx=df_exp_mtx,
        df_acc_mtx=df_acc_mtx,
        search_space=search_space,
        mask_expr_dropout = mask_expr_dropout,
        regressor_type = importance_scoring_method,
        regressor_kwargs = importance_scoring_kwargs,
        n_cpu = n_cpu,
        temp_dir = temp_dir)

    # calculate region to gene correlation
    log.info(
        f'Calculating region to gene correlation, using {correlation_scoring_method} method')
    region_to_gene_correlation = _score_regions_to_genes(
        df_exp_mtx=df_exp_mtx,
        df_acc_mtx=df_acc_mtx,
        search_space=search_space,
        mask_expr_dropout = mask_expr_dropout,
        regressor_type = correlation_scoring_method,
        regressor_kwargs = importance_scoring_kwargs,
        n_cpu = n_cpu,
        temp_dir = temp_dir)

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
    log.info('Done!')
    return result_df
```

and the function `_score_regions_to_genes`:
```python
def _score_regions_to_genes(
        df_exp_mtx: pd.DataFrame,
        df_acc_mtx: pd.DataFrame,
        search_space: pd.DataFrame,
        mask_expr_dropout: bool,
        regressor_type: Literal["RF", "ET", "GBM", "PR", "SR"],
        regressor_kwargs: dict,
        n_cpu: int,
        temp_dir: Union[None, pathlib.Path]) -> dict:
    """
    # TODO: Add doctstrings
    """
    if len(set(df_exp_mtx.columns)) != len(df_exp_mtx.columns):
        raise ValueError("Expression matrix contains duplicate gene names")
    if len(set(df_acc_mtx.columns)) != len(df_acc_mtx.columns):
        raise ValueError("Chromatin accessibility matrix contains duplicate gene names")
    if temp_dir is not None:
        if type(temp_dir) == str:
            temp_dir = pathlib.Path(temp_dir)
        if not temp_dir.exists():
            Warning(f"{temp_dir} does not exist, creating it.")
            os.makedirs(temp_dir)
    scplus_region_names = df_acc_mtx.columns
    scplus_gene_names = df_exp_mtx.columns
    search_space = search_space[search_space['Name'].isin(scplus_region_names)]
    search_space = search_space[search_space['Gene'].isin(scplus_gene_names)]
    # Get region indeces per gene
    gene_names, acc_idx = _get_acc_idx_per_gene(
        scplus_region_names = scplus_region_names, search_space = search_space)
    EXP = df_exp_mtx[gene_names].to_numpy()
    ACC = df_acc_mtx.to_numpy()
    regions_to_genes = dict(
        joblib.Parallel(
            n_jobs = n_cpu,
            temp_folder=temp_dir)(
                joblib.delayed(_score_regions_to_single_gene)(
                    acc = ACC[:, acc_idx[idx]],
                    exp = EXP[:, idx],
                    gene_name = gene_names[idx],
                    region_names = scplus_region_names[acc_idx[idx]],
                    regressor_type = regressor_type,
                    regressor_kwargs = regressor_kwargs, 
                    mask_expr_dropout = mask_expr_dropout
                )
                for idx in tqdm(
                    range(len(gene_names)),
                    total = len(gene_names),
                    desc=f'Running using {n_cpu} cores')
                ))
    return regions_to_genes
```
And 
```python
def _get_acc_idx_per_gene(
        scplus_region_names: pd.Index,
        search_space: pd.DataFrame) -> Tuple[np.ndarray, List[List[str]]]:
    region_names = search_space["Name"].to_numpy()
    gene_names = search_space["Gene"].to_numpy()
    s = np.argsort(gene_names)
    region_names = region_names[s]
    gene_names = gene_names[s]
    region_names_to_idx = pd.DataFrame(
        index = scplus_region_names,
        data = {'idx': np.arange(len(scplus_region_names))})
    unique_gene_names, gene_idx = np.unique(gene_names, return_index = True)
    region_idx_per_gene = []
    for i in range(len(gene_idx)):
        if i < len(gene_idx) - 1:
            region_idx_per_gene.append(
                region_names_to_idx.loc[region_names[gene_idx[i]:gene_idx[i+1]], 'idx'].to_list())
        else:
            region_idx_per_gene.append(
                region_names_to_idx.loc[region_names[gene_idx[i]:], 'idx'].to_list())
    return unique_gene_names, region_idx_per_gene
```


#### eGRN_direct
```
scenicplus grn_inference eGRN
```
The key function for GRN inference, called `eGRN` (https://github.com/aertslab/scenicplus/blob/main/src/scenicplus/cli/scenicplus.py), which called the function `infer_grn` (https://github.com/aertslab/scenicplus/blob/main/src/scenicplus/cli/commands.py).

```python
def infer_grn(
        TF_to_gene_adj_fname: pathlib.Path,
        region_to_gene_adj_fname: pathlib.Path,
        cistromes_fname: pathlib.Path,
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
    from scenicplus.grn_builder.gsea_approach import build_grn
    from scenicplus.triplet_score import calculate_triplet_score
    log.info("Loading TF to gene adjacencies.")
    tf_to_gene = pd.read_table(TF_to_gene_adj_fname)

    log.info("Loading region to gene adjacencies.")
    region_to_gene = pd.read_table(region_to_gene_adj_fname)

    log.info("Loading cistromes.")
    cistromes = mudata.read(cistromes_fname.__str__())

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

    log.info("Formatting eGRN as table.")
    eRegulon_metadata = _format_egrns(
        eRegulons=eRegulons,
        tf_to_gene=tf_to_gene)

    log.info("Calculating triplet ranking.")
    eRegulon_metadata = calculate_triplet_score(
        cistromes=cistromes,
        eRegulon_metadata=eRegulon_metadata,
        ranking_db_fname=ranking_db_fname)

    log.info(f"Saving network to {eRegulon_out_fname.__str__()}")
    eRegulon_metadata.to_csv(
        eRegulon_out_fname,
        sep="\t", header=True, index=False)
```
Which calls the function

```python
def build_grn(
        tf_to_gene: pd.DataFrame,
        region_to_gene: pd.DataFrame,
        cistromes: anndata.AnnData,
        is_extended: bool,
        temp_dir: str,
        order_regions_to_genes_by='importance',
        order_TFs_to_genes_by='importance',
        gsea_n_perm=1000,
        quantiles=(0.85, 0.90),
        top_n_regionTogenes_per_gene=(5, 10, 15),
        top_n_regionTogenes_per_region=(),
        binarize_using_basc=False,
        min_regions_per_gene=0,
        rho_dichotomize_tf2g=True,
        rho_dichotomize_r2g=True,
        rho_dichotomize_eregulon=True,
        keep_only_activating=False,
        rho_threshold=RHO_THRESHOLD,
        NES_thr=0,
        adj_pval_thr=1,
        min_target_genes=5,
        n_cpu=1,
        merge_eRegulons=True,
        disable_tqdm=False,
        seed=555,
        **kwargs) -> List[eRegulon]:
    log.info('Thresholding region to gene relationships')
    # some tfs are missing from tf_to_gene because they are not 
    # preset in the gene expression matrix, so subset!
    cistromes = cistromes[
        :, cistromes.var_names[cistromes.var_names.isin(tf_to_gene['TF'])]]
    relevant_tfs, e_modules = create_emodules(
        region_to_gene=region_to_gene,
        cistromes=cistromes,
        is_extended=is_extended,
        order_regions_to_genes_by=order_regions_to_genes_by,
        quantiles=quantiles,
        top_n_regionTogenes_per_gene=top_n_regionTogenes_per_gene,
        top_n_regionTogenes_per_region=top_n_regionTogenes_per_region,
        binarize_using_basc=binarize_using_basc,
        min_regions_per_gene=min_regions_per_gene,
        rho_dichotomize=rho_dichotomize_r2g,
        keep_only_activating=keep_only_activating,
        rho_threshold=rho_threshold,
        disable_tqdm=disable_tqdm,
        n_cpu=n_cpu,
        temp_dir=temp_dir)
    log.info('Subsetting TF2G adjacencies for TF with motif.')
    TF2G_adj_relevant = tf_to_gene.loc[tf_to_gene['TF'].isin(relevant_tfs)]
    TF2G_adj_relevant.index = TF2G_adj_relevant["TF"]
    log.info('Running GSEA...')
    if rho_dichotomize_tf2g:
        log.info("Generating rankings...")
        TF2G_adj_relevant_pos = TF2G_adj_relevant.loc[TF2G_adj_relevant["rho"] > rho_threshold]
        TF2G_adj_relevant_neg = TF2G_adj_relevant.loc[TF2G_adj_relevant["rho"] < -rho_threshold]
        pos_TFs, c = np.unique(TF2G_adj_relevant_pos["TF"], return_counts=True)
        pos_TFs = pos_TFs[c >= min_target_genes]
        neg_TFs, c = np.unique(TF2G_adj_relevant_neg["TF"], return_counts=True)
        neg_TFs = neg_TFs[c >= min_target_genes]
        # The expression below will fail if there is only a single target gene (after thresholding on rho)
        # TF2G_adj_relevant_pos/neg.loc[TF] will return a pd.Series instead of dataframe
        # This should never be the case though (if min_target_genes > 1)
        # But better fix this at some point!
        TF_to_ranking_pos = {
            TF: TF2G_adj_relevant_pos.loc[TF].set_index('target')[order_TFs_to_genes_by].sort_values(ascending = False)
            for TF in tqdm(pos_TFs, total = len(pos_TFs))}
        TF_to_ranking_neg = {
            TF: TF2G_adj_relevant_neg.loc[TF].set_index('target')[order_TFs_to_genes_by].sort_values(ascending = False)
            for TF in tqdm(neg_TFs, total = len(neg_TFs))}
        pos_tf_gene_modules = joblib.Parallel(
            n_jobs=n_cpu,
            temp_folder=temp_dir)(
            joblib.delayed(_run_gsea_for_e_module)(
                e_module=e_module,
                rnk=TF_to_ranking_pos[e_module.transcription_factor],
                gsea_n_perm=gsea_n_perm,
                context=frozenset(['positive tf2g']),
                seed=seed)
            for e_module in tqdm(
                e_modules,
                total = len(e_modules),
                desc="Running for Positive TF to gene")
            if e_module.transcription_factor in pos_TFs)
        neg_tf_gene_modules = joblib.Parallel(
            n_jobs=n_cpu,
            temp_folder=temp_dir)(
            joblib.delayed(_run_gsea_for_e_module)(
                e_module=e_module,
                rnk=TF_to_ranking_neg[e_module.transcription_factor],
                gsea_n_perm=gsea_n_perm,
                context=frozenset(['negative tf2g']),
                seed=seed)
            for e_module in tqdm(
                e_modules, 
                total = len(e_modules),
                desc="Running for Negative TF to gene")
            if e_module.transcription_factor in neg_TFs)
        new_e_modules = [*pos_tf_gene_modules, *neg_tf_gene_modules]
    else:
        log.info("Generating rankings...")
        TFs, c = np.unique(TF2G_adj_relevant["TF"], return_counts=True)
        TFs = TFs[c >= min_target_genes]
        # The expression below will fail if there is only a single target gene (after thresholding on rho)
        # TF2G_adj_relevant.loc[TF] will return a pd.Series instead of dataframe
        # This should never be the case though (if min_target_genes > 1)
        # But better fix this at some point!
        TF_to_ranking = {
            TF: TF2G_adj_relevant.loc[TF].set_index('target')[order_TFs_to_genes_by].sort_values(ascending = False)
            for TF in tqdm(TFs, total = len(TFs))}
        new_e_modules = joblib.Parallel(
            n_jobs=n_cpu,
            temp_folder=temp_dir)(
            joblib.delayed(_run_gsea_for_e_module)(
                e_module=e_module,
                rnk=TF_to_ranking[e_module.transcription_factor],
                gsea_n_perm=gsea_n_perm,
                context=frozenset(['negative tf2g']))
            for e_module in tqdm(
                e_modules, 
                total = len(e_modules),
                desc="Running for Negative TF to gene")
            if e_module.transcription_factor in TFs)
    # filter out nans
    new_e_modules = [m for m in new_e_modules if not np.isnan(
        m.gsea_enrichment_score) and not np.isnan(m.gsea_pval)]

    log.info(
        f'Subsetting on adjusted pvalue: {adj_pval_thr}, minimal NES: {NES_thr} and minimal leading edge genes {min_target_genes}')
    # subset on adj_p_val
    adj_pval = p_adjust_bh([m.gsea_pval for m in new_e_modules])
    if any([np.isnan(p) for p in adj_pval]):
        Warning(
            'Something went wrong with calculating adjusted p values, early returning!')
        return new_e_modules

    for module, adj_pval in zip(new_e_modules, adj_pval):
        module.gsea_adj_pval = adj_pval

    e_modules_to_return: List[eRegulon] = []
    for module in new_e_modules:
        if module.gsea_adj_pval < adj_pval_thr and module.gsea_enrichment_score > NES_thr:
            module_in_LE = module.subset_leading_edge(inplace=False)
            if module_in_LE.n_target_genes >= min_target_genes:
                e_modules_to_return.append(module_in_LE)
    if merge_eRegulons:
        log.info('Merging eRegulons')
        e_modules_to_return = merge_emodules(
            e_modules=e_modules_to_return, inplace=False, rho_dichotomize=rho_dichotomize_eregulon)
    e_modules_to_return = [
        x for x in e_modules_to_return if not isinstance(x, list)]
    return e_modules_to_return
```
which then calles
```python

def create_emodules(
        region_to_gene: pd.DataFrame,
        cistromes: anndata.AnnData,
        is_extended: bool,
        temp_dir: str,
        order_regions_to_genes_by: str = 'importance',
        quantiles: tuple = (0.85, 0.90),
        top_n_regionTogenes_per_gene: tuple = (5, 10, 15),
        top_n_regionTogenes_per_region: tuple = (),
        binarize_using_basc: bool = False,
        min_regions_per_gene: int = 0,
        rho_dichotomize: bool = True,
        keep_only_activating: bool = False,
        rho_threshold: float = RHO_THRESHOLD,
        disable_tqdm=False,
        n_cpu=None,) -> Tuple[Set[str], List[eRegulon]]:
    # Set up multiple thresholding methods to threshold region to gene relationships
    def iter_thresholding(adj:pd.DataFrame, context: frozenset):
        grouped_adj_by_gene = Groupby(adj[TARGET_GENE_NAME].to_numpy())
        grouped_adj_by_region = Groupby(adj[TARGET_REGION_NAME].to_numpy())
        yield from chain(
            chain.from_iterable(
                _quantile_thr(
                            adjacencies=adj,
                            grouped=grouped_adj_by_gene,
                            threshold=thr,
                            min_regions_per_gene=min_regions_per_gene,
                            context=context,
                            order_regions_to_genes_by=order_regions_to_genes_by,
                            temp_dir=temp_dir,
                            n_cpu=n_cpu)
                for thr in quantiles),
            chain.from_iterable(
                _top_targets(
                    adjacencies=adj,
                    grouped=grouped_adj_by_gene,
                    n=n,
                    min_regions_per_gene=min_regions_per_gene,
                    context=context,
                    order_regions_to_genes_by=order_regions_to_genes_by,
                    temp_dir=temp_dir,
                    n_cpu=n_cpu)
                for n in top_n_regionTogenes_per_gene),
            chain.from_iterable(
                _top_regions(
                    adjacencies=adj,
                    grouped=grouped_adj_by_region,
                    n=n,
                    min_regions_per_gene=min_regions_per_gene,
                    context=context,
                    order_regions_to_genes_by=order_regions_to_genes_by,
                    temp_dir=temp_dir,
                    n_cpu=n_cpu)
                for n in top_n_regionTogenes_per_region),
            _binarize_BASC(
                    adjacencies=adj,
                    grouped=grouped_adj_by_gene,
                    min_regions_per_gene=min_regions_per_gene,
                    context=context,
                    order_regions_to_genes_by=order_regions_to_genes_by,
                    temp_dir=temp_dir,
                    n_cpu=n_cpu)
                if binarize_using_basc else [])
    # Split positive and negative correlation coefficients, if rho_dichotomize == True
    if rho_dichotomize:
        repressing_adj = region_to_gene.loc[
            region_to_gene[CORRELATION_COEFFICIENT_NAME] < -rho_threshold]
        activating_adj = region_to_gene.loc[
            region_to_gene[CORRELATION_COEFFICIENT_NAME] > rho_threshold]
        r2g_iter = chain(
            iter_thresholding(repressing_adj, frozenset(['negative r2g'])),
            iter_thresholding(activating_adj, frozenset(['positive r2g'])),)
    else:
        # don't split
        if keep_only_activating:
            r2g_iter = iter_thresholding(region_to_gene.loc[
                region_to_gene[CORRELATION_COEFFICIENT_NAME] > rho_threshold],
                context = frozenset(['positive r2g']))
        else:
            r2g_iter = iter_thresholding(
                region_to_gene,
                context = frozenset(['all r2g']))
    # Calculate the number of parameters.
    # this will be used later on in the progress bar (i.e. to know the total amount of
    # things to process)
    n_params = sum([len(quantiles) if not type(quantiles) == float else 1,
                    len(top_n_regionTogenes_per_gene) if not type(
                        top_n_regionTogenes_per_gene) == int else 1,
                    len(top_n_regionTogenes_per_region) if not type(top_n_regionTogenes_per_region) == int else 1])
    total_iter = (2 * (n_params + (binarize_using_basc * 1))
                  ) if rho_dichotomize else (n_params + (binarize_using_basc * 1))
    relevant_tfs = []
    eRegulons = []

    mtx_cistromes:pd.DataFrame = cistromes.to_df()

    for context, r2g_df in tqdm(r2g_iter, total=total_iter, disable=disable_tqdm):
        for TF_name in tqdm(
            mtx_cistromes.columns,
            total=len(mtx_cistromes.columns),
            desc=f"\u001b[32;1mProcessing:\u001b[0m {', '.join(context)}",
            leave=False,
            disable=disable_tqdm):
            regions_enriched_for_TF_motif = mtx_cistromes.index[
                mtx_cistromes[TF_name]]
            r2g_df_enriched_for_TF_motif = r2g_df.loc[list(set(
                regions_enriched_for_TF_motif) & set(r2g_df.index))]
            if len(r2g_df_enriched_for_TF_motif) > 0:
                relevant_tfs.append(TF_name)
                eRegulons.append(
                    eRegulon(
                        transcription_factor=TF_name,
                        cistrome_name=f"{TF_name}" + ("_extended" if is_extended else "_direct") + "_({len(r2g_df_enriched_for_TF_motif)}r)",
                        is_extended=is_extended,
                        regions2genes=list(r2g_df_enriched_for_TF_motif[list(
                            REGIONS2GENES_HEADER)].itertuples(index=False, name='r2g')),
                        context=context.union(frozenset(['Cistromes']))))
    return set(relevant_tfs), eRegulons

```


```python
def get_max_rank_of_motif_for_each_TF(
        cistromes: mudata.AnnData,
        ranking_db_fname: str
) -> pd.DataFrame:
        # Read database for target regions
        pr_all_target_regions = pr.PyRanges(
                region_names_to_coordinates(
                cistromes.obs_names))
        ctx_db = cisTargetDatabase(
                fname=ranking_db_fname, region_sets = pr_all_target_regions)
        l_motifs = [x.split(",") for x in cistromes.var["motifs"]]
        l_motifs_idx = [
             [ctx_db.db_rankings.index.get_loc(x) for x in m] for m in l_motifs]
        rankings = ctx_db.db_rankings.to_numpy()
        # Generate dataframe with TFs on columns and regions on rows
        # values are motif rankings. The best ranking (i.e lowest value)
        # is used across each motif annotated to the TF
        max_rank = np.array([rankings[x].min(0) for x in l_motifs_idx]).T
        # convert regions to cistrome coordinates
        db_regions_cistrome_regions = ctx_db.regions_to_db.copy() \
            .groupby("Query")["Target"].apply(lambda x: list(x))
        df_max_rank = pd.DataFrame(
              max_rank,
              index = ctx_db.db_rankings.columns, # db region names
              columns = cistromes.var_names     # TF names
        )
        df_max_rank["cistrome_region_coord"] = db_regions_cistrome_regions.loc[df_max_rank.index].values
        df_max_rank = df_max_rank.explode("cistrome_region_coord")
        df_max_rank = df_max_rank.set_index("cistrome_region_coord")
        df_max_rank = df_max_rank.groupby("cistrome_region_coord").min()
        return df_max_rank

def calculate_triplet_score(
        cistromes: mudata.AnnData,
        eRegulon_metadata: pd.DataFrame,
        ranking_db_fname: str) -> pd.DataFrame:
        eRegulon_metadata = eRegulon_metadata.copy()
        df_TF_region_max_rank = get_max_rank_of_motif_for_each_TF(
              cistromes=cistromes,
              ranking_db_fname=ranking_db_fname)
        TF_region_iter = eRegulon_metadata[["TF", "Region"]].to_numpy()
        TF_to_region_score = np.array([
              df_TF_region_max_rank.loc[region, TF]
              for TF, region in TF_region_iter])
        TF_to_gene_score = eRegulon_metadata["importance_TF2G"].to_numpy()
        region_to_gene_score = eRegulon_metadata["importance_R2G"].to_numpy()
        #rank the scores
        TF_to_region_rank = _rank_scores_and_assign_random_ranking_in_range_for_ties(
              -TF_to_region_score) #negate because lower score is better
        TF_to_gene_rank = _rank_scores_and_assign_random_ranking_in_range_for_ties(
              TF_to_gene_score)
        region_to_gene_rank = _rank_scores_and_assign_random_ranking_in_range_for_ties(
              region_to_gene_score)
        #create rank ratios
        TF_to_gene_rank_ratio = (TF_to_gene_rank.astype(np.float64) + 1) / TF_to_gene_rank.shape[0]
        region_to_gene_rank_ratio = (region_to_gene_rank.astype(np.float64) + 1) / region_to_gene_rank.shape[0]
        TF_to_region_rank_ratio = (TF_to_region_rank.astype(np.float64) + 1) / TF_to_region_rank.shape[0]
        #create aggregated rank
        rank_ratios = np.array([
              TF_to_gene_rank_ratio, region_to_gene_rank_ratio, TF_to_region_rank_ratio])
        aggregated_rank = np.zeros((rank_ratios.shape[1],), dtype = np.float64)
        for i in range(rank_ratios.shape[1]):
                aggregated_rank[i] = _calculate_cross_species_rank_ratio_with_order_statistics(rank_ratios[:, i])
        eRegulon_metadata["triplet_rank"] = aggregated_rank.argsort().argsort()
        return eRegulon_metadata
```




#### eGRN_extended
```
scenicplus grn_inference eGRN
```