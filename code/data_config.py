import os
"""
configuration file includes all related datasets 
"""

root_data_folder = '../data/'
raw_data_folder = os.path.join(root_data_folder, 'raw_dat')
preprocessed_data_folder = os.path.join(root_data_folder, 'preprocessed_dat')
gene_feature_file = os.path.join(preprocessed_data_folder, 'CosmicHGNC_list.tsv')
gdsc_tcga_mapping_file = os.path.join(root_data_folder, 'tcga_gdsc_drug_mapping.csv')

#TCGA_datasets
tcga_folder = os.path.join(root_data_folder, 'tcga')
tcga_clinical_folder = os.path.join(tcga_folder, 'Clinical')
tcga_drug_name_mapping_file = os.path.join(tcga_folder, 'drug_name_mapping.csv')
tcga_first_treatment_file = os.path.join(tcga_folder, 'tcga_drug_first_treatment.csv')
tcga_first_response_file = os.path.join(tcga_folder, 'tcga_drug_first_response.csv')
#tcga_first_response_file = os.path.join(tcga_folder, 'tcga_drug_first_response_type.csv')


#Xena datasets
xena_folder = os.path.join(raw_data_folder, 'Xena')
xena_id_mapping_file = os.path.join(xena_folder, 'gencode.v23.annotation.gene.probemap')
xena_gex_file = os.path.join(xena_folder, 'tcga_RSEM_gene_tpm.gz')
xena_preprocessed_gex_file = os.path.join(preprocessed_data_folder, 'xena_gex')
xena_sample_file = os.path.join(xena_folder, 'TCGA_phenotype_denseDataOnlyDownload.tsv.gz')

#CCLE datasets
ccle_folder = os.path.join(raw_data_folder, 'CCLE')
ccle_gex_file = os.path.join(ccle_folder, 'CCLE_expression.csv')
ccle_preprocessed_gex_file = os.path.join(preprocessed_data_folder, 'ccle_gex')
ccle_sample_file = os.path.join(ccle_folder, 'sample_info.csv')

#gex features
gex_feature_file = os.path.join(preprocessed_data_folder, 'uq1000_feature.csv')

#GDSC datasets
gdsc_folder = os.path.join(raw_data_folder, 'GDSC')
gdsc_target_file1 = os.path.join(gdsc_folder, 'GDSC1_fitted_dose_response_25Feb20.csv')
gdsc_target_file2 = os.path.join(gdsc_folder, 'GDSC2_fitted_dose_response_25Feb20.csv')
gdsc_raw_target_file = os.path.join(gdsc_folder, 'gdsc_ic50flag.csv')
gdsc_sample_file = os.path.join(gdsc_folder, 'gdsc_cell_line_annotation.csv')
gdsc_preprocessed_target_file = os.path.join(preprocessed_data_folder, 'gdsc_ic50flag.csv')

#adae brain cancer datasets
adae_folder = os.path.join(root_data_folder, 'adae_data')
adae_gex_file = os.path.join(adae_folder, 'TCGA_GBM_and_LGG_PREPROCESSED_RNASEQ_EXPRESSION_500_kmeans.tsv')
adae_sex_label_file = os.path.join(adae_folder, 'TCGA_GBM_and_LGG_SEX_LABELS.tsv')
adae_subtype_label_file = os.path.join(adae_folder, 'TCGA_GBM_and_LGG_SUBTYPE_LABELS.tsv')

#PDTC datasets
pdtc_folder = os.path.join(root_data_folder, 'PDTC')
gdsc_pdtc_drug_name_mapping_file = os.path.join(root_data_folder, 'pdtc_gdsc_drug_mapping.csv')
pdtc_gex_file = os.path.join(preprocessed_data_folder, 'pdtc_uq1000_feature.csv')
pdtc_target_file = os.path.join(pdtc_folder, 'DrugResponsesAUCModels.txt')

#Celligner datasets
celligner_folder = os.path.join(root_data_folder, 'celligner')
celligner_pdtc_gex_file = os.path.join(preprocessed_data_folder, 'celligner_pdtc_uq_df.csv')
celligner_xena_gex_file = os.path.join(preprocessed_data_folder, 'celligner_xena_uq_df.csv')
