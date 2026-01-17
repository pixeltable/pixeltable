"""
Data wrangling script to prepare Amazon product data for the Voyage AI tutorial.

This script:
1. Loads the Amazon Product Dataset 2020 from Hugging Face
2. Splits pipe-separated image URLs into one row per image
3. Filters out placeholder/transparent-pixel images
4. Exports clean data to CSV for upload to GitHub
"""

import pandas as pd
from datasets import load_dataset

# Load the full dataset from Hugging Face
print("Loading dataset from Hugging Face...")
hf_dataset = load_dataset('calmgoose/amazon-product-data-2020', split='train')
df = hf_dataset.to_pandas()

print(f"Original dataset: {len(df)} products")
print(f"Columns: {list(df.columns)}")

# Rename columns to be Python-friendly (replace spaces with underscores)
df.columns = df.columns.str.replace(' ', '_')

# Show sample of Image column to understand the format
print("\nSample Image column values:")
for i, img in enumerate(df['Image'].head(3)):
    print(f"  {i}: {img[:200]}..." if img and len(str(img)) > 200 else f"  {i}: {img}")

# Count images per product
def count_images(img_str):
    if pd.isna(img_str) or not img_str:
        return 0
    urls = [u.strip() for u in str(img_str).split('|') if u.strip()]
    # Filter out placeholder images
    urls = [u for u in urls if 'transparent-pixel' not in u]
    return len(urls)

df['image_count'] = df['Image'].apply(count_images)
print(f"\nImage count distribution:")
print(df['image_count'].value_counts().head(10))

# Split images into separate rows
print("\nSplitting images into separate rows...")
rows = []
for _, row in df.iterrows():
    img_str = row['Image']
    if pd.isna(img_str) or not img_str:
        # Keep row with null image
        new_row = row.to_dict()
        new_row['Image'] = None
        new_row['image_idx'] = 0
        rows.append(new_row)
    else:
        urls = [u.strip() for u in str(img_str).split('|') if u.strip()]
        # Filter out placeholder images
        urls = [u for u in urls if 'transparent-pixel' not in u]
        
        if not urls:
            # No valid images, keep row with null image
            new_row = row.to_dict()
            new_row['Image'] = None
            new_row['image_idx'] = 0
            rows.append(new_row)
        else:
            # Create one row per image
            for idx, url in enumerate(urls):
                new_row = row.to_dict()
                new_row['Image'] = url
                new_row['image_idx'] = idx
                rows.append(new_row)

df_expanded = pd.DataFrame(rows)

# Remove the temporary image_count column
if 'image_count' in df_expanded.columns:
    df_expanded = df_expanded.drop(columns=['image_count'])

print(f"Expanded dataset: {len(df_expanded)} rows (from {len(df)} products)")
print(f"Average images per product: {len(df_expanded) / len(df):.2f}")

# Show distribution of image_idx
print(f"\nImage index distribution:")
print(df_expanded['image_idx'].value_counts().sort_index().head(10))

# Sample output
print("\nSample expanded rows:")
sample = df_expanded[df_expanded['Uniq_Id'] == df_expanded['Uniq_Id'].iloc[0]]
print(f"Product: {sample['Product_Name'].iloc[0][:60]}...")
print(f"Images for this product: {len(sample)}")
for _, r in sample.iterrows():
    print(f"  idx={r['image_idx']}: {r['Image'][:80]}..." if r['Image'] else f"  idx={r['image_idx']}: None")

# Take a subset for the tutorial (first 500 unique products, ~2000 rows with images)
unique_products = df_expanded['Uniq_Id'].unique()[:500]
df_subset = df_expanded[df_expanded['Uniq_Id'].isin(unique_products)]

print(f"\nSubset for tutorial: {len(df_subset)} rows from {len(unique_products)} products")

# Select only columns we need and clean null values for embedding
columns_to_keep = ['Uniq_Id', 'Product_Name', 'Category', 'Selling_Price', 
                   'About_Product', 'Image', 'image_idx']
df_subset = df_subset[columns_to_keep].copy()

# Fill null values with empty strings for text columns (required for embeddings)
text_columns = ['Category', 'About_Product']
df_subset[text_columns] = df_subset[text_columns].fillna('')

print(f"Selected {len(columns_to_keep)} columns, cleaned null values in text columns")

# Export to CSV
output_file = 'amazon_products_with_images.csv'
df_subset.to_csv(output_file, index=False)
print(f"\nExported to {output_file}")

# Also create a parquet version (more efficient)
parquet_file = 'amazon_products_with_images.parquet'
df_subset.to_parquet(parquet_file, index=False)
print(f"Exported to {parquet_file}")

# Print final stats
print("\n" + "="*60)
print("FINAL DATASET STATS")
print("="*60)
print(f"Total rows: {len(df_subset)}")
print(f"Unique products: {df_subset['Uniq_Id'].nunique()}")
print(f"Rows with images: {df_subset['Image'].notna().sum()}")
print(f"Rows without images: {df_subset['Image'].isna().sum()}")
print(f"\nColumns: {list(df_subset.columns)}")
