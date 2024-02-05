# If you want to generate Quilt-Instruct yourself, please first download (upload roi_data_similarmerged_facecorrected_tracesfixed_textpadded_asrcorrected_final_clustersadded_medicalcontentdetected_rerun_uniqueidentifieradded_DETAILED_DESCRIPTION.parquet into huggingface)

This file has cursors extracted in the "traces" column. You may run the clustering algorithm by running utils/cluster_singlerow.py, or directly use our extracted clusters using the "words_clustered" column in the dataframe that holds the traces. 

To generate Quilt-Instruct, please run 

TIKTOKEN_CACHE_DIR="" python pipeline_prompts_eval.py --prompt_type conversation