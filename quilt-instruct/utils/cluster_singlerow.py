from PIL import Image
import itertools
import jsonlines
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import copy
import numpy as np
import argparse

COUNTER = 0

def words_upto_t(t, text_with_utterance):
    words_count = 0
    for entry in text_with_utterance:
        if entry['end_time'] <= t:
            words_count += len(entry['utterance'].split())
    return words_count

def normalize_bbox(bbox, img_width, img_height):
        return [bbox[0]/img_width, bbox[1]/img_height, bbox[2]/img_width, bbox[3]/img_height]

def calculate_normalized_bbox_area(bbox):
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width * height    
    
    
def remove_outliers(cluster_points, column_name, multiplier=1.5):
    Q1 = cluster_points[column_name].quantile(0.25)
    Q3 = cluster_points[column_name].quantile(0.75)
    IQR = Q3 - Q1

    filtered_points = cluster_points[
        
        (cluster_points[column_name] >= (Q1 - multiplier * IQR)) & 
        (cluster_points[column_name] <= (Q3 + multiplier * IQR))
    ]
    return filtered_points

def time_decay(t, lambda_):
    return np.exp(-lambda_ * t)


def calculate_cluster_avg_time(df_clusters):

    cluster_groups = df_clusters.groupby('cluster')
    avg_times = cluster_groups['t'].mean()
    
    return avg_times

def match_clusters_to_text_v3(df_clusters, text_with_utterance):
    # Calculate avrg times for each cluster
    cluster_avg_times = calculate_cluster_avg_time(df_clusters)
    
    matched_rows = []

    for utterance in text_with_utterance:
        utterance_end_time = utterance['end_time'] 
        
        # Find the cluster whose average time is closest to the utterance's end time (we can do something else too but this is fine for now)
        cluster_id = (cluster_avg_times - utterance_end_time).abs().idxmin()

        matched_rows.append({
            't_start': utterance['start_time'], 
            't_end': utterance_end_time,
            'cluster': cluster_id,
            'utterance': utterance['utterance']
        })

    matched_df = pd.DataFrame(matched_rows)

    return matched_df

def get_cluster_word_counts(df):
    """Calculate word count for each cluster."""
    return df.groupby('cluster')['utterance'].apply(lambda x: x.str.split().str.len().sum())

def merge_with_nearest_neighbor(cluster, cluster_word_counts):
    # If it's the first cluster, merge with the next
    if cluster == cluster_word_counts.index[0]:
        return cluster_word_counts.index[1]
    # If it's the last cluster, merge with the previous
    elif cluster == cluster_word_counts.index[-1]:
        return cluster_word_counts.index[-2]
    # Else, determine which neighboring cluster has more words and merge with it
    else:
        prev_cluster = cluster_word_counts.index[cluster_word_counts.index.get_loc(cluster) - 1]
        next_cluster = cluster_word_counts.index[cluster_word_counts.index.get_loc(cluster) + 1]
        return prev_cluster if cluster_word_counts[prev_cluster] > cluster_word_counts[next_cluster] else next_cluster


def main(row, **kwargs):
    global COUNTER
    
    if COUNTER % 500 == 0:
        print(f"Processing row {COUNTER}...")
    COUNTER += 1
    use_word_uttered_feature = kwargs.get('use_word_uttered_feature')
    remove_outliers_flag = kwargs.get('remove_outliers_flag')
    lambda_ = kwargs.get('lambda_')
    min_normalized_bbox_area = kwargs.get('min_normalized_bbox_area')
    max_normalized_bbox_area = kwargs.get('max_normalized_bbox_area')
    
    cluster_found_rows = process_video_group(
        row, 
        use_word_uttered_feature, 
        remove_outliers_flag, 
        lambda_, 
        min_normalized_bbox_area, 
        max_normalized_bbox_area
    )
    
    
    return cluster_found_rows


def process_video_group(row, use_word_uttered_feature, remove_outliers_flag, lambda_, min_normalized_bbox_area, max_normalized_bbox_area):

     

    data_im_path = row['image_path'][0]
    tracers = row['traces']
    caption = row['extended_text_sentences_asrcorrected']['utterance']
    text_with_utterance = row['extended_text_time_asrcorrected']
    img = Image.open(data_im_path)
    img_width, img_height = img.size
    for i in range(len(tracers)):
        tracers[i]['x_scaled'] = int(tracers[i]['x'] * img_width)
        tracers[i]['y_scaled'] = int(tracers[i]['y'] * img_height)

    data =pd.DataFrame(pd.concat([pd.DataFrame([dic]) for dic in tracers], ignore_index=True))

    # Convrt t, x, y to float and scale t
    data['t'] = data['t'].astype(float)
    data['x'] = data['x'].astype(float)
    data['y'] = data['y'].astype(float)

    scaler = MinMaxScaler()
    data['t_adjusted'] = scaler.fit_transform(data[['t']])
    data['w'] = data['t'].apply(time_decay, args=(lambda_,))
    data['x_weighted'] = data['x'] * data['w']
    data['y_weighted'] = data['y'] * data['w']
    data['word_uttered'] = data['t'].apply(words_upto_t, args=(text_with_utterance,))
    scaler = MinMaxScaler()
    data['word_uttered_adjusted'] = scaler.fit_transform(data[['word_uttered']])

    number_of_words = len(caption.split())
    if number_of_words <= 10:
        n_clusters = 1
    elif number_of_words > 10 and number_of_words <= 40: 
        n_clusters = 2
    elif number_of_words > 40 and number_of_words <= 60:
        n_clusters = 3
    else:
        n_clusters = 4
    kmeansModel = KMeans(n_clusters=n_clusters, n_init=n_clusters)
    if use_word_uttered_feature:
        kmeansModel = kmeansModel.fit(data[['x_weighted', 'y_weighted', 'word_uttered_adjusted']])
    else:
        kmeansModel = kmeansModel.fit(data[['x_weighted', 'y_weighted']])

    data['cluster'] = kmeansModel.labels_
    img = Image.open(data_im_path)
    all_clusters = data
    all_clusters['x_scaled'] = (all_clusters['x_scaled']).astype(int)
    all_clusters['y_scaled'] = (all_clusters['y_scaled']).astype(int)

    unique_clusters = all_clusters['cluster'].unique()
    bounding_boxes = {}
    bounding_boxes_scaled = {}
    for i, cluster in enumerate(unique_clusters):
        cluster_points = all_clusters[all_clusters['cluster'] == cluster]

    #     Remove outliers for both x_scaled and y_scaled cols
        if remove_outliers_flag:
            cluster_points = remove_outliers(cluster_points, 'x_scaled')
            cluster_points = remove_outliers(cluster_points, 'y_scaled')

        x1, y1 = cluster_points['x'].min(), cluster_points['y'].min()
        x2, y2 = cluster_points['x'].max(), cluster_points['y'].max()
        bounding_boxes_scaled[cluster] = [x1, y1, x2, y2]

        # Calculate the bounding box after removing outlier points
        x1, y1 = cluster_points['x_scaled'].min(), cluster_points['y_scaled'].min()
        x2, y2 = cluster_points['x_scaled'].max(), cluster_points['y_scaled'].max()

        bounding_boxes[cluster] = [x1, y1, x2, y2] 

    img_width, img_height = img.size
    normalized_bboxes = {cluster: normalize_bbox(bbox, img_width, img_height) for cluster, bbox in bounding_boxes.items()}

 
    small_clusters = [cluster for cluster, bbox in normalized_bboxes.items() 
                         if calculate_normalized_bbox_area(bbox) < min_normalized_bbox_area 
                         or calculate_normalized_bbox_area(bbox) > max_normalized_bbox_area]
    data = data[~data['cluster'].isin(small_clusters)]
    if not data.empty: # if there are clusters left, then match the text with clusters, if not, just append the raw captions without clusters


        matched_df = match_clusters_to_text_v3(data, text_with_utterance)
        cluster_map = {old_cluster: new_cluster for new_cluster, old_cluster in enumerate(matched_df['cluster'].unique())}
        matched_df['cluster'] = matched_df['cluster'].map(cluster_map)
        
        # Filter bounding_boxes_scaled based on the keys in cluster_map

        merged_clusters = set()
        if matched_df['cluster'].nunique() > 1: 
            # Identify clusters that appear less than or equal to 3 times
            cluster_word_counts = get_cluster_word_counts(matched_df)
            infrequent_clusters = cluster_word_counts[cluster_word_counts <= 3]

            while not infrequent_clusters.empty:
                # Pick the cluster with the smallest word count
                smallest_cluster = infrequent_clusters.idxmin()
                merged_clusters.add(smallest_cluster)

                # Determine the cluster to merge with
                target_cluster = merge_with_nearest_neighbor(smallest_cluster, cluster_word_counts)

                # Apply the merge
                matched_df.loc[matched_df['cluster'] == smallest_cluster, 'cluster'] = target_cluster

                # Re-evaluate cluster word counts and infrequent clusters
                cluster_word_counts = get_cluster_word_counts(matched_df)
                infrequent_clusters = cluster_word_counts[cluster_word_counts <= 3]
                
        grouped_utterances = matched_df.groupby('cluster')['utterance'].apply(list)

        consolidated_utterances = grouped_utterances.apply(lambda x: ' '.join(x)).reset_index()
 

        consolidated_df = pd.DataFrame({
            'cluster': consolidated_utterances['cluster'],
            'consolidated_utterance': consolidated_utterances['utterance']
        })
        

        for cluster in small_clusters:#eliminate the removed clusters
            if cluster in merged_clusters:
                continue
            bounding_boxes_scaled.pop(cluster, None)

        
        filtered_bounding_boxes_scaled = {old_cluster: bbox for old_cluster, bbox in bounding_boxes_scaled.items() if old_cluster in cluster_map}
        remapped_bounding_boxes_scaled = {cluster_map[old_cluster]: bbox for old_cluster, bbox in filtered_bounding_boxes_scaled.items()}

        consolidated_df['bbox'] = consolidated_df['cluster'].map(lambda x: remapped_bounding_boxes_scaled[x])

        bounding_boxes = []

        for i, cluster in enumerate(consolidated_df['bbox'].copy()):

            x1 = np.clip(cluster[0], 0, 1)
            y1 = np.clip(cluster[1], 0, 1)
            x2 = np.clip(cluster[2], 0, 1)
            y2 = np.clip(cluster[3], 0, 1)

            consolidated_df.at[i, 'bbox'] = [x1, y1, x2, y2]
        consolidated_string = ""

        for idx, row in consolidated_df.iterrows():
            rounded_values = [round(val, 2) for val in row['bbox']]
            bbox_str = f"[{', '.join(map(str, rounded_values))}]"
            consolidated_string += f"{row['consolidated_utterance']} {bbox_str}, "

        consolidated_string = consolidated_string.rstrip(", ")
        cluster_added_captions = consolidated_string
    else: 
        cluster_added_captions = caption
        
        
          
    return cluster_added_captions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process the whole dataframe row by row')
    parser.add_argument('--use_word_uttered_feature', action='store_true', help='Use word uttered feature flag for clustering')
    parser.add_argument('--remove_outliers_flag', action='store_false', help='Remove mouse tracer outliers')
    parser.add_argument('--lambda_', type=float, default=0.2, help='Lambda value for exponential decay')
    parser.add_argument('--min_normalized_bbox_area', type=float, default=0.02, help='Minimum Normalized BBox Area, which is used to remove clusters')
    parser.add_argument('--max_normalized_bbox_area', type=float, default=0.7, help='Maximum Normalized BBox Area which is used to remove clusters')
    args = parser.parse_args()
    args_dict = vars(args)
    combined_df = pd.read_parquet('cursor.parquet')
    combined_df['clustered_whole_sentences'] = combined_df.apply(lambda row: main(row, **args_dict), axis=1)
    combined_df.to_parquet('cursor.parquet')
    
