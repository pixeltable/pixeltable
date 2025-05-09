#!/usr/bin/env python3
"""Script to load single-cell data into Pixeltable with transformations."""

import os
import numpy as np
import scanpy as sc
import anndata as ad
import urllib.request
import tarfile
import pixeltable as pxt

def get_data():
    """Get single-cell data from various sources"""
    # Create data and write directories (equivalent to: mkdir -p data write)
    os.makedirs('data', exist_ok=True)
    os.makedirs('write', exist_ok=True)
    
    h5ad_file = 'data/pbmc3k.h5ad'
    matrices_dir = 'data/filtered_gene_bc_matrices/hg19/'
    tar_file = 'data/pbmc3k_filtered_gene_bc_matrices.tar.gz'
    
    # Try to load from existing files first
    if os.path.exists(matrices_dir): return sc.read_10x_mtx(matrices_dir)
    if os.path.exists(h5ad_file): return ad.read_h5ad(h5ad_file)
    
    # Try to use scanpy's built-in dataset
    try:
        adata = sc.datasets.pbmc3k()
        adata.write_h5ad(h5ad_file)
        return adata
    except Exception:
        pass
    
    # Try to extract from existing tar file
    if os.path.exists(tar_file):
        print(f"Extracting from {tar_file}")
        with tarfile.open(tar_file, 'r:gz') as tar:
            tar.extractall(path='data')
        return sc.read_10x_mtx(matrices_dir)
    
    # Download and extract from URL (equivalent to:
    # curl https://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz -o data/pbmc3k_filtered_gene_bc_matrices.tar.gz
    # cd data; tar -xzf pbmc3k_filtered_gene_bc_matrices.tar.gz)
    print("Downloading 10x Genomics data")
    url = "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"
    urllib.request.urlretrieve(url, tar_file)
    with tarfile.open(tar_file, 'r:gz') as tar:
        tar.extractall(path='data')
    return sc.read_10x_mtx(matrices_dir)

@pxt.udf
def process_cell_expression(expr: pxt.Array[pxt.Float], mt_indices: list) -> pxt.Json:
    """Process a cell's gene expression data with standard transformations."""
    n_genes = int(np.count_nonzero(expr))
    if n_genes < 200: return None
    
    n_counts = float(np.sum(expr))
    pct_mt = float(np.sum(expr[mt_indices]) / n_counts * 100) if n_counts > 0 else 0.0
    
    normalized = expr * (1e4 / n_counts) if n_counts > 0 else expr
    log_norm = np.log1p(normalized)
    
    return {
        'n_genes': n_genes, 'n_counts': n_counts, 'pct_mt': pct_mt,
        'normalized': normalized.astype(float).tolist(),
        'log_norm': log_norm.astype(float).tolist()
    }

# Fetch data from AnnData
adata = get_data()
mt_indices = [i for i, name in enumerate(adata.var_names) if name.startswith(('MT-', 'mt-', 'Mt-'))]

# Pixeltable 
schema = {
    'cell_id': pxt.String,
    'raw_expression': pxt.Array[(adata.n_vars,), pxt.Float]
}
T = pxt.create_table('pbmc3k_raw', schema, if_exists='replace')

# Add computed columns directly to preform data processing
T.add_computed_column(processed=process_cell_expression(T.raw_expression, mt_indices))
T.add_computed_column(n_genes=T.processed['n_genes'])
T.add_computed_column(n_counts=T.processed['n_counts'])
T.add_computed_column(pct_mt=T.processed['pct_mt'])
T.add_computed_column(normalized=T.processed['normalized'])
T.add_computed_column(log_norm=T.processed['log_norm'])

# Insert data
expr_matrix = adata.X.toarray() 
T.insert({'cell_id': str(adata.obs_names[i]), 'raw_expression': expr_matrix[i, :]} for i in range(adata.n_obs))    

# Simple query to test data is loaded correctly
high_mt_cells = T.where(T.pct_mt > 10).select(
    T.cell_id, T.pct_mt
).order_by(T.pct_mt, asc=False).limit(3).collect()

print(f"\nTop 3 cells with high mitochondrial percentage (>10%):")
for row in high_mt_cells:
    print(f"  Cell {row['cell_id']}: {row['pct_mt']:.2f}% mitochondrial")