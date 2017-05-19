# code adapted from Udacity ML Nanodegree project files for
# Titanic data exploration project and Finding Donors for CharityML project

###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import os
import matplotlib as mpl
from sklearn.manifold import MDS
import random


def feat_by_amend(data, feature, order):
    '''
    Plot feature distribution by amendment
    '''
    g = sns.FacetGrid(data, col='amendment', col_wrap=6, size=2)
    if order==None:
        g = g.map(sns.countplot, feature)
    else:
        g = g.map(sns.countplot, feature, order=order)
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(feature+' Counts for Ratified Amendments')

    # if number of categories is >2, rotate x-axis labels for readability
    if len(order) > 2:
        g.set_xticklabels(order, rotation=45, rotation_mode="anchor", ha="right")


def filter_data(data, condition):
    """
    Remove elements that do not match the condition provided.
    Takes a data list as input and returns a filtered list.
    Conditions should be a list of strings of the following format:
      '<field> <op> <value>'
    where the following operations are valid: >, <, >=, <=, ==, !=

    Example: ["legis_party == 'Democrat'", 'year < 1900']
    """

    field, op, value = condition.split(" ")

    # convert value into number or strip excess quotes if string
    try:
        value = float(value)
    except:
        value = value.strip("\'\"")

    # get booleans for filtering
    if op == ">":
        matches = data[field] > value
    elif op == "<":
        matches = data[field] < value
    elif op == ">=":
        matches = data[field] >= value
    elif op == "<=":
        matches = data[field] <= value
    elif op == "==":
        matches = data[field] == value
    elif op == "!=":
        matches = data[field] != value
    else: # catch invalid operation codes
        raise Exception("Invalid comparison operator.")

    # filter data and outcomes
    data = data[matches].reset_index(drop = True)
    return data

def passage_stats(data, outcomes, key, filters = []):
    """
    Print out selected statistics regarding passage, given a feature of
    interest and any number of filters (including no filters)
    """

    # check that the key exists
    if key not in data.columns.values :
        print("'{}' is not a feature of the amendment data.".format(key))
        return False

    # merge data and outcomes into single dataframe
    outcomes = pd.DataFrame(np.array(outcomes), columns=['Passed'])
    all_data = pd.concat([data, outcomes], axis = 1)

    # apply filters to data
    for condition in filters:
        all_data = filter_data(all_data, condition)

    # create outcomes DataFrame
    all_data = all_data[[key, 'Passed']]

    # create plotting figure
    fig, (ax1, ax2) = plt.subplots(2,1,sharex=True,figsize=(12,6))

    # 'numerical' features
    if(key in ['year','congress','legis_term','legis_start','sen_maj_pct',
                'rep_maj_pct']):

        # remove NaN values from data
        all_data = all_data[~np.isnan(all_data[key])]

        # divide the range of data into bins and count passage rates
        min_value = all_data[key].min()
        max_value = all_data[key].max()
        value_range = max_value - min_value
        bins = np.arange(min_value, max_value + value_range/20, value_range/20)

        # overlay each bin's passage rates
        nonpass_vals = all_data[all_data['Passed'] == 0][key].reset_index(drop = True)
        pass_vals = all_data[all_data['Passed'] == 1][key].reset_index(drop = True)
        y_np, x, _ = ax2.hist(nonpass_vals, bins = bins, alpha = 0.6,
                                color = 'red', label = 'Did not pass')
        y_p, x, _ = ax2.hist(pass_vals, bins = bins, alpha = 0.6,
                                color = 'green', label = 'Passed')

        # add legend to plot
        ax2.legend(framealpha = 0.8)

        # percentage passing plot
        # calculate percentage passing
        percent = 100*y_p/(y_np+y_p)
        # set the width of each bar
        bar_width = value_range/20

        percent_bar = ax1.bar(bins[:-1] + bar_width/2, percent, width = bar_width,
                                color = 'b', label='Percent Passed')
        ax1.set_xticks(bins)
        ax1.set_xticklabels(x.astype(int), rotation=45, rotation_mode="anchor",
                            ha="right")
        ax1.legend(framealpha = 0.8)
        ax1.set_title('Passage Statistics With \'%s\' Feature'%(key))

    # 'categorical' features
    else:

        # obtain categories for the key by decreasing value counts
        values = data[key].value_counts().index.tolist()
        # keep only top 10 categories
        values = values[:15]

        # create DataFrame containing categories and count of each
        frame = pd.DataFrame(index = np.arange(len(values)), columns=(key,
                                'Passed','NPassed'))
        for i, value in enumerate(values):
            frame.loc[i] = [value, \
                   len(all_data[(all_data['Passed'] == 1) & (all_data[key] == value)]), \
                   len(all_data[(all_data['Passed'] == 0) & (all_data[key] == value)])]

        # calculate percentage passing
        percent = 100*frame['Passed']/(frame['NPassed']+frame['Passed'])
        percent = percent.apply(lambda x: x if np.isfinite(x) else 0)

        # set the width of each bar
        bar_width = 6.0/len(values)

        # display each category's passage rates
        for i in np.arange(len(frame)):
            nonpass_bar = ax2.bar(i-bar_width/2, frame.loc[i]['NPassed'],
                                    width = bar_width, color = 'r')
            pass_bar = ax2.bar(i+bar_width/2, frame.loc[i]['Passed'],
                                width = bar_width, color = 'g')

            ax2.set_xticks(np.arange(len(frame)))
            ax2.set_xticklabels(values, rotation=45, rotation_mode="anchor", ha="right")
            ax2.legend((nonpass_bar[0], pass_bar[0]),('Did not pass', 'Passed'),
                            framealpha = 0.8)

            percent_bar = ax1.bar(i, percent.loc[i], width=bar_width*2, color = 'b')

            ax1.set_xticks(np.arange(len(frame)))
            ax1.set_xticklabels(values, rotation=45, rotation_mode="anchor", ha="right")

        ax1.set_title('Passage Statistics With \'%s\' Feature (15 Categories \
                        with the Most Data Shown)'%(key))


    # common attributes for plot formatting
    ax1.legend([percent_bar],['Percent Passed'],framealpha = 0.8)
    ax2.set_xlabel(key)
    ax2.set_ylabel('Number of Proposals')
    ax1.set_xlabel(key)
    ax1.set_ylabel('Percentage Passing')

    plt.show()

    # report number of proposals with missing values
    if sum(pd.isnull(all_data[key])):
        nan_outcomes = all_data[pd.isnull(all_data[key])]['Passed']
        print("Proposals with missing '{}' values: {} ({} passed, {} did not pass)".format( \
              key, len(nan_outcomes), sum(nan_outcomes == 1), sum(nan_outcomes == 0)))


def evaluate(results, accuracy, f1):
    """
    Visualization code to display results of various learners.

    inputs:
      - learners: a list of supervised learners
      - stats: a list of dictionaries of the statistic results from 'train_predict()'
      - accuracy: The score for the naive predictor
      - f1: The score for the naive predictor
    """

    # create figure
    fig, ax = plt.subplots(2, 3, figsize = (11,7))

    # constants
    bar_width = 0.4
    colors = ['#A00000','#00A0A0']

    # super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'acc_train', 'f_train',
                                        'pred_time', 'acc_test', 'f_test']):
            for i in np.arange(3):

                # creative plot code
                ax[int(j/3), j%3].bar(i+0.2+k*bar_width, results[learner][i][metric],
                                        width = bar_width, color = colors[k])
                ax[int(j/3), j%3].set_xticks([0.4, 1.4, 2.4])
                ax[int(j/3), j%3].set_xticklabels(["1%", "10%", "100%"])
                ax[int(j/3), j%3].set_xlabel("Training Set Size")
                ax[int(j/3), j%3].set_xlim((-0.1, 2.9))

    # add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")

    # add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")

    # add horizontal lines for naive predictors
    ax[0, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1,
                        color = 'k', linestyle = 'dashed')
    ax[1, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1,
                        color = 'k', linestyle = 'dashed')
    ax[0, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1,
                        color = 'k', linestyle = 'dashed')
    ax[1, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1,
                        color = 'k', linestyle = 'dashed')

    # set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))

    # create patches for the legend
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color = colors[i], label = learner))
    plt.legend(handles = patches, bbox_to_anchor = (-.80, 2.53), \
               loc = 'upper center', borderaxespad = 0., ncol = 3, fontsize = 'x-large')

    # aesthetics
    plt.suptitle("Performance Metrics for Two Supervised Learning Models",
                    fontsize = 16, y = 1.10)
    plt.tight_layout()
    plt.show()


def feature_plot(importances, X_train, y_train):

    # Display the ten most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:10]]
    values = importances[indices][:10]

    # Create the plot
    fig = plt.figure(figsize = (15,5))
    plt.title("Normalized Weights for the Ten Most Predictive Features", fontsize = 16)
    plt.bar(np.arange(10) + 0.2, values, width = 0.4, align="center", color = '#00A000', \
          label = "Feature Weight")
    plt.bar(np.arange(10) - 0.2, np.cumsum(values), width = 0.4, align = "center",
                color = '#00A0A0', label = "Cumulative Feature Weight")
    plt.xticks(np.arange(10), columns)
    plt.xlim((-0.5, 9.5))
    plt.ylabel("Weight", fontsize = 12)
    plt.xlabel("Feature", fontsize = 12)

    plt.legend(loc = 'upper center')
    plt.tight_layout()
    plt.show()


def cluster_plot(data, subset):
    '''Plot 5 random entries from each cluster for a visualization
    of the titles in each cluster
    Code reference: http://brandonrose.org/clustering'''

    # group data by cluster
    cluster_group = data.groupby('cluster', as_index=False)

    # create subset df to store 5 random entries from each cluster
    data_subset_cluster = pd.DataFrame([])

    for cluster, group in cluster_group:
        # drop duplicate titles so the plot shows a variety of titles
        indep_group = group.drop_duplicates(subset='clean_title')
        # check how many remaining entries there are in the cluster
        grlen = len(indep_group)
        # randomly select 5 entries; if there are less than 5, keep all
        data_20 = indep_group.iloc[random.sample(range(grlen), 5 if grlen >= 5 else grlen)]
        # append to placeholder df
        data_subset_cluster = pd.concat([data_subset_cluster,data_20])

    # copy subset topic data into a separate df
    topic_subset_cluster = data_subset_cluster.filter(regex='Topic')

    # calculate cosine distances between subset topics
    dist = 1-cosine_similarity(topic_subset_cluster)

    # spread out distances by applying exp function
    expfn = np.vectorize(np.exp)
    dist = expfn(dist/2)

    # convert distances into 2D array for plotting
    mds = MDS(n_components=2, dissimilarity='precomputed')
    pos = mds.fit_transform(dist)
    xs, ys = pos[:,0], pos[:,1]

    # create df for plotting
    plot_df = pd.DataFrame(dict(x=xs, y=ys, label=data_subset_cluster['cluster'],
                                title=data_subset_cluster['clean_title']))
    plot_df.reset_index(drop=True, inplace=True)

    # group by cluster
    groups = plot_df.groupby('label')

    # create figure
    fig, ax = plt.subplots(figsize=(20,20))

    # plot cluster markers
    cmap = mpl.cm.get_cmap('Vega20')
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', c=cmap(name),linestyle='',
                ms=12, mec='none', label='Cluster %s'%(name))
        ax.set_aspect('auto')

    # label markers with proposal titles
    for i in range(len(plot_df)):
        ax.text(plot_df.ix[i]['x'], plot_df.ix[i]['y'], plot_df.ix[i]['title'], size=12)

    ax.set_title('KMeans Clustering by Proposal %s'%(subset), fontsize=26)
    ax.legend(framealpha = 0.8, fontsize=16)
