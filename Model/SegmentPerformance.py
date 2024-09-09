import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.metrics import get_scorer

import warnings
warnings.filterwarnings('ignore')

def get_single_scorer(scorer_name):
    try:
        scorer = get_scorer(scorer_name)
        return scorer
    except KeyError:
        raise ValueError(f"Scorer '{scorer_name}' is not recognized. Please provide a valid scorer name.")

def format_number(value):
    return f"{value:.2f}"

def numeric_segmentation_edges(num_hist_dict, max_segments):
    percentile_values = np.array([min(num_hist_dict), max(num_hist_dict)])
    attempt_max_segments = max_segments
    prev_percentile_values = deepcopy(percentile_values)
    while len(percentile_values) < max_segments + 1:
        prev_percentile_values = deepcopy(percentile_values)
        percentile_values = pd.unique(
            np.nanpercentile(num_hist_dict.to_numpy(), np.linspace(0, 100, attempt_max_segments + 1))
        )
        if len(percentile_values) == len(prev_percentile_values):
            break
        attempt_max_segments *= 2

    if len(percentile_values) > max_segments + 1:
        percentile_values = prev_percentile_values

    return percentile_values

def largest_category_index_up_to_ratio(cat_hist_dict, max_segments, max_cat_proportions):
    total_values = sum(cat_hist_dict.values)
    first_less_then_max_cat_proportions_idx = np.argwhere(
        cat_hist_dict.values.cumsum() >= total_values * max_cat_proportions
    )[0][0]

    return min(max_segments, cat_hist_dict.size, first_less_then_max_cat_proportions_idx + 1)

def create_partition(dataset, column_name, max_segments=10, max_cat_proportions=0.7):
    numerical_features = dataset.select_dtypes(include='number').columns
    cat_features = dataset.select_dtypes(include='object').columns
    
    if column_name in numerical_features:
        num_hist_dict = dataset[column_name]
        percentile_values = numeric_segmentation_edges(num_hist_dict, max_segments)
        
        if len(percentile_values) == 1:
            f = lambda df, val=percentile_values[0]: (df[column_name] == val)
            label = str(percentile_values[0])
            return [{'filter': f, 'label': label}]
        
        filters = []
        for start, end in zip(percentile_values[:-1], percentile_values[1:]):
            if end == percentile_values[-1]:
                f = lambda df, a=start, b=end: (df[column_name] >= a) & (df[column_name] <= b)
                label = f'[{format_number(start)} - {format_number(end)}]'
            else:
                f = lambda df, a=start, b=end: (df[column_name] >= a) & (df[column_name] < b)
                label = f'[{format_number(start)} - {format_number(end)})'
            
            filters.append(ChecksFilter([f], label))

    elif column_name in cat_features:
        cat_hist_dict = dataset[column_name].value_counts()
        n_large_cats = largest_category_index_up_to_ratio(cat_hist_dict, max_segments, max_cat_proportions)

        filters = []
        for i in range(n_large_cats):
            f = lambda df, val=cat_hist_dict.index[i]: df[column_name] == val
            filters.append(ChecksFilter([f], str(cat_hist_dict.index[i])))

        if len(cat_hist_dict) > n_large_cats:
            f = lambda df, values=cat_hist_dict.index[:n_large_cats]: ~df[column_name].isin(values)
            filters.append(ChecksFilter([f], 'Others'))

    return filters

class ChecksFilter():
    def __init__(self, filter_functions=None, label=''):
        if not filter_functions:
            self.filter_functions = []
        else:
            self.filter_functions = filter_functions
        self.label = label

    def filter(self, dataframe, label_col=None):
        if label_col is not None:
            dataframe['temp_label_col'] = label_col
        for func in self.filter_functions:
            dataframe = dataframe.loc[func(dataframe)]

        if label_col is not None:
            return dataframe.drop(columns=['temp_label_col']), dataframe['temp_label_col']
        else:
            return dataframe

class SegmentPerformanceTest():
    def __init__(
        self,
        feature_1=None,
        feature_2=None,
        alternative_scorer='accuracy',
        max_segments=10,
        max_cat_proportions=0.9,
        random_state=42,
    ):
        
        if feature_1 and feature_1 == feature_2:
            raise ValueError("feature_1 must be different than feature_2")
        
        if feature_1 is None or feature_2 is None:
            raise ValueError("Must define both feature_1 and feature_2 or none of them")
        
        if not isinstance(max_segments, int) or max_segments < 0:
            raise ValueError("num_segments must be positive integer")

        self.feature_1 = feature_1
        self.feature_2 = feature_2
        self.random_state = random_state
        self.max_segments = max_segments
        self.max_cat_proportions = max_cat_proportions
        self.alternative_scorer = alternative_scorer

    def run(self, model, dataset, target_label):
        columns = dataset.columns

        if len(columns) < 2:
            raise ValueError('Dataset must have at least 2 features')

        if self.feature_1 not in columns or self.feature_2 not in columns:
            raise ValueError('"feature_1" and "feature_2" must be in dataset columns')

        feature_1_filters = create_partition(
            dataset, self.feature_1, max_segments=self.max_segments, max_cat_proportions=self.max_cat_proportions
        )
        feature_2_filters = create_partition(
            dataset, self.feature_2, max_segments=self.max_segments, max_cat_proportions=self.max_cat_proportions
        )

        scores = np.empty((len(feature_1_filters), len(feature_2_filters)), dtype=float)
        counts = np.empty((len(feature_1_filters), len(feature_2_filters)), dtype=int)

        for i, feature_1_filter in enumerate(feature_1_filters):
            feature_1_df = feature_1_filter.filter(dataset)
            for j, feature_2_filter in enumerate(feature_2_filters):
                feature_2_df = feature_2_filter.filter(feature_1_df)
                X = feature_2_df.drop(columns=target_label)
                y = feature_2_df[target_label]

                if feature_2_df.empty:
                    score = np.NaN
                else:
                    metrics = get_single_scorer(self.alternative_scorer)
                    score = metrics(model, X, y)

                scores[i, j] = score
                counts[i, j] = len(feature_2_df)

        x = [v.label for v in feature_2_filters]
        y = [v.label for v in feature_1_filters]

        scores_text = [[0]*scores.shape[1] for _ in range(scores.shape[0])]

        for i in range(len(y)):
            for j in range(len(x)):
                score = scores[i, j]
                if not np.isnan(score):
                    scores_text[i][j] = f'{format_number(score)}\n({counts[i, j]})'
                elif counts[i, j] == 0:
                    scores_text[i][j] = ''
                else:
                    scores_text[i][j] = f'{score}\n({counts[i, j]})'

        scores = scores.astype(object)
        mask = np.isnan(scores.astype(float))

        plt.figure(figsize=(8, 5))
        ax = sns.heatmap(scores.astype(float), mask=mask, annot=scores_text, fmt='', cmap='RdYlGn', 
                        cbar_kws={'label': self.alternative_scorer}, xticklabels=x, yticklabels=y)

        ax.set_title(f'{self.alternative_scorer} (count) by features {self.feature_1}/{self.feature_2}', fontsize=16)
        ax.set_xlabel(self.feature_2, fontsize=12)
        ax.set_ylabel(self.feature_1, fontsize=12)

        plt.xticks(rotation=-30, ha='left')
        plt.yticks(rotation=0)
        plt.gca().invert_yaxis()

        plt.tight_layout()
        plt.show()