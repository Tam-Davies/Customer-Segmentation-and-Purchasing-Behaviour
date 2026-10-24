# Customer Segmentation Using Unsupervised Machine Learning

This project builds a customer segmentation workflow for retail data using unsupervised machine learning. It evaluates multiple clustering algorithms on customer behaviour features derived from a transactional dataset, then selects and saves the best-performing segmentation model for downstream analysis and reporting.

The analysis uses the UCI Online Retail dataset and focuses on RFM-style customer behaviour features such as recency, frequency, monetary value, quantity purchased, product diversity, and average order value.

## Project objective

The main goal is to identify groups of customers with similar purchasing behaviour and produce a profile-driven segmentation that can support marketing, retention, and customer strategy decisions.

The project pipeline includes:

- transaction-level data inspection and data quality assessment
- cleaning and filtering of invalid retail records
- customer feature engineering
- log transformation and standardisation for clustering
- PCA-based exploration for visualisation
- model comparison across K-Means, Agglomerative Clustering, and DBSCAN
- final model selection and cluster profiling
- saving model artifacts and summary tables

## Repository structure

```text
.
├── README.md
├── Customer Segmentation Using Unsupervised Machine Learning/
│   ├── data/
│   │   ├── processed/
│   │   │   ├── clean_retail.csv
│   │   │   ├── clustering_features.csv
│   │   │   └── customer_features.csv
│   │   └── raw/
│   ├── notebooks/
│   │   └── customer_segmentation_analysis.ipynb
│   ├── outputs/
│   │   ├── figures/
│   │   │   ├── dbscan_k_distance.png
│   │   │   ├── final_kmeans_pca_clusters.png
│   │   │   ├── kmeans_cluster_sizes.png
│   │   │   ├── kmeans_customer_profile_comparison.png
│   │   │   ├── kmeans_profile_heatmap.png
│   │   │   ├── kmeans_selection.png
│   │   │   └── pca_customer_space.png
│   │   ├── models/
│   │   │   ├── agglomerative_final.joblib
│   │   │   ├── dbscan_final.joblib
│   │   │   ├── kmeans_final.joblib
│   │   │   ├── pca_2d.joblib
│   │   │   └── scaler.joblib
│   │   └── tables/
│   │       ├── agglomerative_customer_clusters.csv
│   │       ├── agglomerative_selection.csv
│   │       ├── dbscan_customer_clusters.csv
│   │       ├── dbscan_evaluation.csv
│   │       ├── final_model_comparison.csv
│   │       ├── kmeans_customer_clusters.csv
│   │       └── kmeans_selection.csv
│   ├── report/
│   │   └── Customer_Segmentation_Unsupervised_Learning_Report.pdf
│   └── src/
│       ├── agglomerative_selection.py
│       ├── data_cleaning.py
│       ├── data_inspection.py
│       ├── data_quality.py
│       ├── dbscan_evaluation.py
│       ├── dbscan_selection.py
│       ├── feature_engineering.py
│       ├── final_clustering.py
│       ├── final_visualizations.py
│       ├── kmeans_selection.py
│       ├── pca_analysis.py
│       ├── preprocessing.py
│       ├── save_models.py
│       └── ...
└── ...
```

## Data and preprocessing flow

The project starts from transaction-level retail data, then converts it into customer-level features. The main feature set is:

- Recency: number of days since the last purchase
- Frequency: number of distinct invoices per customer
- Monetary: total spend per customer
- TotalQuantity: total items purchased
- ProductDiversity: number of unique products purchased
- AverageOrderValue: mean spend per order

The workflow includes:

1. Raw dataset inspection
2. Quality checks for missing values, duplicate rows, cancellations, negative quantities, and invalid prices
3. Cleaning of invalid and non-customer transactions
4. Creation of customer-level behavioural features using a reference date of one day after the latest transaction
5. Log transformation and standardisation before clustering
6. PCA for 2D visualisation of the customer feature space

## Notebook and analysis workflow

The main exploratory notebook is:

- `Customer Segmentation Using Unsupervised Machine Learning/notebooks/customer_segmentation_analysis.ipynb`

This notebook documents the complete end-to-end workflow from raw transactions to final segmentation, including:

- import setup and environment preparation
- data inspection and quality diagnostics
- cleaning and feature engineering
- scaling and PCA
- cluster algorithm selection
- model comparison and customer profiling
- final visualisations

## Model comparison and final selection

The project evaluates three clustering strategies:

- K-Means
- Agglomerative Clustering (Ward linkage)
- DBSCAN

The summary table is saved to:

- `Customer Segmentation Using Unsupervised Machine Learning/outputs/tables/final_model_comparison.csv`

### Final model comparison summary

| Model | Clusters | Noise | Silhouette | Davies-Bouldin | Calinski-Harabasz |
| --- | ---: | ---: | ---: | ---: | ---: |
| K-Means | 2 | 0 | 0.3563 | 1.0550 | 3183.4055 |
| Agglomerative | 2 | 0 | 0.3063 | 1.1336 | 2563.2299 |
| DBSCAN | 2 | 78 | 0.4724 | 0.4735 | 19.3966 |

The final model selected for the segmentation output is K-Means with 2 clusters because it provides a clean, interpretable segmentation with no noise points and strong clustering structure across the customer features. DBSCAN produces a higher silhouette score but a large share of unassigned points, which makes it less practical for a full customer segmentation story when all customers are meant to be assigned to a cohort.

## Outputs generated by the project

### Tables

- `outputs/tables/kmeans_selection.csv`
- `outputs/tables/agglomerative_selection.csv`
- `outputs/tables/dbscan_evaluation.csv`
- `outputs/tables/final_model_comparison.csv`
- `outputs/tables/kmeans_customer_clusters.csv`
- `outputs/tables/agglomerative_customer_clusters.csv`
- `outputs/tables/dbscan_customer_clusters.csv`

These tables include model selection metrics, cluster membership assignments, and customer-level profiles by cluster.

### Figures

The project saves a range of visual outputs under `outputs/figures` including:

- `pca_customer_space.png`
- `kmeans_selection.png`
- `dbscan_k_distance.png`
- `final_kmeans_pca_clusters.png`
- `kmeans_cluster_sizes.png`
- `kmeans_customer_profile_comparison.png`
- `kmeans_profile_heatmap.png`

These figures cover PCA structure, cluster selection diagnostics, customer distribution by segment, and profile comparisons between clusters.

### Saved models

The trained model artifacts are stored in `outputs/models`:

- `kmeans_final.joblib`
- `agglomerative_final.joblib`
- `dbscan_final.joblib`
- `scaler.joblib`
- `pca_2d.joblib`

These files can be reused for future scoring or similar customer segmentation workflows.

## Report

A formal project report is included at:

- `Customer Segmentation Using Unsupervised Machine Learning/report/Customer_Segmentation_Unsupervised_Learning_Report.pdf`

## How to use this project

1. Open the notebook in the `notebooks` directory for the full end-to-end analysis.
2. Run the preprocessing and feature-engineering scripts from the project directory to regenerate the processed datasets.
3. Use the selection scripts to re-create the evaluation tables and diagnostics.
4. Re-run the final clustering and visualisation scripts to regenerate the cluster summaries and plots.
5. Use the saved `.joblib` model files for inference or model deployment workflows.

## Notes

This project is designed as a data-science and business-analytics workflow rather than a deployment system. The focus is on exploratory clustering, customer profiling, and evidence-based model selection using real retail transaction behaviour.

## Summary

This repository demonstrates a practical unsupervised segmentation workflow for retail customers and highlights how behavioural clustering can support customer understanding, targeted marketing, and retention strategy. The final selected model is K-Means with two customer segments, producing actionable and interpretable customer groups.
