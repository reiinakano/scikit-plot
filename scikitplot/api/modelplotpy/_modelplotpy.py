"""
The :mod:`~scikitplot.api.modelplotpy` module includes plots for machine learning
evaluation decile analysis e.g. Gain, Lift and Decile charts, etc.

References
----------
[1] https://github.com/modelplot/modelplotpy/blob/master/modelplotpy/functions.py
[2] https://modelplot.github.io/intro_modelplotpy.html

This package/module is designed to be compatible with both Python 2 and Python 3.
The imports below ensure consistent behavior across different Python versions by
enforcing Python 3-like behavior in Python 2.
"""
# code that needs to be compatible with both Python 2 and Python 3
from __future__ import (
    absolute_import,  # Ensures that all imports are absolute by default, avoiding ambiguity.
    division,         # Changes the division operator `/` to always perform true division.
    print_function,   # Treats `print` as a function, consistent with Python 3 syntax.
    unicode_literals  # Makes all string literals Unicode by default, similar to Python 3.
)
import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
#from matplotlib.offsetbox import (TextArea, AnnotationBbox)


## Define __all__ to specify the public interface of the module,
# not required default all etc.
__all__ = [
  # helper
  '_range01',
  '_check_input',
  # class
  'ModelPlotPy', 'modelplotpy',
  # (cumulative) Response, Lift, Gains and plots
  'plot_response',
  'plot_cumresponse',
  'plot_cumlift',
  'plot_cumgains',
  'plot_all',
  # financial
  'plot_costsrevs',
  'plot_profit',
  'plot_roi',
]
  
##########################################################################
## helper func
##########################################################################

def _range01(x):
    """
    Normalizing input
    
    Parameters
    ----------
    x : list of numeric data
        List of numeric data to get normalized

    Returns
    -------
    normalized version of x
    
    """
    return (x-np.min(x))/(np.max(x)-np.min(x))


def _check_input(
    input_list,
    check_list,
    check = '',
):
    """
    Check if the input matches any of a complete list
    
    Parameters
    ----------
    input_list : list of str
        List containing elements specified by the user.

    check_list : list of str
        List containing all possible elements defined by the model_plots object.

    check : str, default empty
        String contains the parameter that will be checked to provide an informative error.

    Returns
    -------
    error if there is no match with the complete list or the input list again
        
    Raises
    ------
    ValueError: If the elements in `input list` do not correspond with the `check_lisk`.

    """
    if len(input_list) >= 1:
        if any(elem in input_list for elem in check_list):
            input_list = input_list
        else:
            raise ValueError(
                (
                    'Invalid input for parameter %s. '
                    'The input for %s is 1 or more elements from %s '
                    'and put in a list.'
                ) % (check, check, check_list))
    return list(input_list)
  
##########################################################################
## ModelPlotPy class
##########################################################################

# class ModelPlotPy(ABC):
class ModelPlotPy(object):
    """\
    ModelPlotPy decile analysis.
    
    .. dropdown:: View aliases

        **Main aliases**
        
        `scikitplot.api.modelplotpy.ModelPlotPy`

        **Compat aliases**
        
        `scikitplot.api.modelplotpy.modelplotpy`

    Parameters
    ----------
    feature_data : list of objects (n_datasets, )
        Objects containing the X matrix for one or more different datasets.
        
    label_data : list of objects (n_datasets, )
        Objects of the y vector for one or more different datasets.
        
    dataset_labels : list of str (n_datasets, )
        Containing the names of the different `feature_data`
        and `label_data` combination pairs.
    
    models : list of objects (n_models, )
        Containing the sk-learn model objects.
    
    model_labels : list of str (n_models, )
        Names of the (sk-learn) models.
    
    ntiles : int, default 10
        The number of splits range is (2, inf]:
        
        - 10 is called `deciles`
        - 100 is called `percentiles`
        - any other value is an `ntile`
    
    seed : int, default=0
        Making the splits reproducible.
    
        .. versionchanged:: 0.3.9
            Default changed from 999 to 0.

    Raises
    ------
    ValueError: If there is no match with the complete list or the input list again

    Notes
    -----    
    ``modelplotpy`` == ``ModelPlotPy`` Alias
    for backward compatibility or convenience ``ModelPlotPy``        
    """
    def __init__(
        self,
        feature_data = [],
        label_data = [],
        dataset_labels = [],
        models = [],
        model_labels = [],
        ntiles = 10,
        seed = 0,
    ):
        """
        Create a model_plots object
        """
        super().__init__()
      
        self.feature_data = feature_data
        self.label_data = label_data
        self.dataset_labels = dataset_labels
        self.models = models
        self.model_labels = model_labels
        self.ntiles = ntiles
        self.seed = seed
    
    def get_params(self):
        """
        Get parameters of the model plots object.

        .. versionadded:: 0.3.9
        """
        return {
            "feature_data": self.feature_data,
            "label_data": self.label_data,
            "dataset_labels": self.dataset_labels,
            "models": self.models,
            "model_labels": self.model_labels,
            "ntiles": self.ntiles,
            "seed": self.seed
        }

    def set_params(self, **params):
        """
        Set parameters of the model plots object.

        .. versionadded:: 0.3.9
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter '{key}' for ModelPlotPy object.")

    def reset_params(self):
        """
        Reset all parameters to default values.

        .. versionadded:: 0.3.9
        """
        self.feature_data = []
        self.label_data = []
        self.dataset_labels = []
        self.models = []
        self.model_labels = []
        self.ntiles = 10
        self.seed = 0
    
    def reset_modules(self):
        """
        Reset the internal state of the ModelPlotPy object.

        .. versionadded:: 0.3.9
        """
        # Add any additional reset logic specific to your module here
        # For example, clearing any cached data or resetting custom variables
        self.reset_params()
        print("ModelPlotPy module has been reset.")

    def prepare_scores_and_ntiles(self):
        """
        Create eval_tot
        
        This function builds the pandas dataframe eval_tot that contains for each feature
        and label data pair given a description the actual and predicted value.
        It loops over the different models with the given model_name.
        
        Parameters
        ----------
        feature_data : list of objects (n_datasets, )
            Objects containing the X matrix for one or more different datasets.
            
        label_data : list of objects (n_datasets, )
            Objects of the y vector for one or more different datasets.
            
        dataset_labels : list of str (n_datasets, )
            Containing the names of the different feature ``feature_data``
            and label ``label_data`` data combination pairs.
        
        models : list of objects (n_models, )
            Containing the sk-learn model objects.
        
        model_labels : list of str (n_models, )
            Names of the (sk-learn) models.
            
        ntiles : int, default 10
            The number of splits 10 is called deciles, 100 is called percentiles
            and any other value is an ntile.
            Range is (2, inf].
        
        seed : int, default=0
            Making the splits reproducible.
        
            .. versionchanged:: 0.3.9
                Default changed from 999 to 0.

        Returns
        -------
        scores_and_ntiles : pandas.DataFrame
            Pandas dataframe for all given information and for each target_class it makes
            a prediction and ntile. For each ntile a small value (based on the seed) is added
            and normalized to make the results reproducible.

        Raises
        ------
        ValueError: If there is no match with the complete list or the input list again
        """
        if (len(self.models) == len(self.model_labels)) == False:
            raise ValueError(
                'The number of models and the their description model_name must be equal. '
                'The number of model = %s and model_name = %s.' % (
                    len(self.model), len(self.model_name)
                )
            )

        if (len(self.feature_data) == len(self.label_data) == len(self.dataset_labels)) == False:
            raise ValueError(
                'The number of datasets in feature_data and label_data and their description '
                'pairs must be equal. The number of datasets in feature_data = '
                '%s, label_data = %s and description = %s.' % (
                    len(self.feature_data), len(self.label_data), len(self.description)
                )
            )
        
        final = pd.DataFrame()
        for i in range(len(self.models)):
            data_set = pd.DataFrame()
            for j in range(len(self.dataset_labels)):
                y_true = self.label_data[j]
                y_true = y_true.rename('target_class')
                # probabilities and rename them
                y_pred = self.models[i].predict_proba(self.feature_data[j])
                probabilities = pd.DataFrame(
                    data = y_pred,
                    index = self.feature_data[j].index
                )
                probabilities.columns = (
                    'prob_' + self.models[i].classes_.astype('str').astype('O')
                )
                # combine the datasets
                dataset = pd.concat([self.feature_data[j], probabilities, y_true], axis = 1)
                dataset['dataset_label'] = self.dataset_labels[j]
                dataset['model_label'] = self.model_labels[i]
                # remove the feature columns
                dataset = dataset.drop(list(self.feature_data[j].columns), axis=1)
                # make deciles
                # loop over different outcomes
                n = dataset.shape[0]
                for k in self.models[i].classes_.astype('str').astype('O'):
                    #! Added small proportion to prevent equal decile bounds
                    # and reset to 0-1 range (to prevent probs > 1.0)
                    np.random.seed(self.seed)
                    prob_plus_smallrandom = _range01(
                        dataset[['prob_' + k]] + (np.random.uniform(size = (n, 1)) / 1000000)
                    )
                    prob_plus_smallrandom = np.array(
                        prob_plus_smallrandom['prob_' + k]
                    ) # cast to a 1 dimension thing
                    dataset["dec_" + k] = (
                        self.ntiles - (
                            pd.DataFrame(
                                pd.qcut(
                                    prob_plus_smallrandom,
                                    self.ntiles,
                                    labels = False,
                                ),
                                index = self.feature_data[j].index
                            )
                        )
                    )
                    # Use pd.concat to concatenate DataFrames
                data_set = pd.concat(
                    [data_set, dataset],
                    # ignore_index=False,
                )
            # If you have multiple DataFrames to concatenate, you can use a loop
            final = pd.concat(
                [final, data_set],
                # ignore_index=False,
            )
        return final
    
    def aggregate_over_ntiles(self):
        """
        Create eval_t_tot
        
        This function builds the pandas dataframe eval_t_tot and contains the aggregated output.
        The data is aggregated over datasets (feature and label-data pairs) and list of models.
        
        Parameters
        ----------
        feature_data : list of objects (n_datasets, )
            Objects containing the X matrix for one or more different datasets.
            
        label_data : list of objects (n_datasets, )
            Objects of the y vector for one or more different datasets.
            
        dataset_labels : list of str (n_datasets, )
            Containing the names of the different feature `feature_data`
            and label `label_data` data combination pairs.
        
        models : list of objects (n_models, )
            Containing the sk-learn model objects.
        
        model_labels : list of str (n_models, )
            Names of the (sk-learn) models.
            
        ntiles : int, default 10
            The number of splits 10 is called deciles,
            100 is called percentiles and any other value is an ntile.
            Range is (2, inf].
        
        seed : int, default=0
            Making the splits reproducible.
        
            .. versionchanged:: 0.3.9
                Default changed from 999 to 0.

        Returns
        -------
        Pandas dataframe with combination of all datasets, models, target values and ntiles.
        It already contains almost all necessary information for model plotting.

        Raises
        ------
        ValueError: If there is no match with the complete list or the input list again.
        """
        scores_and_ntiles = self.prepare_scores_and_ntiles()
        scores_and_ntiles['all'] = 1
        ntiles_aggregate = pd.DataFrame()
        add_origin = pd.DataFrame()
        for i in range(len(self.model_labels)):
            for j in self.models[i].classes_:
                for k in self.dataset_labels:
                    add_origin_add = pd.DataFrame({
                        'model_label': self.model_labels[i],
                        'dataset_label': [k],
                        'target_class': [j],
                        'ntile': [0],
                        'tot': [0],
                        'pos': [0],
                        'neg': [0],
                        'pct': [0],
                        'postot': [0],
                        'negtot': [0],
                        'tottot': [0],
                        'pcttot': [0],
                        'cumpos': [0],
                        'cumneg': [0],
                        'cumtot': [0],
                        'cumpct': [0],
                        'gain': [0],
                        'cumgain': [0],
                        'gain_ref': [0],
                        'pct_ref': [0],
                        'gain_opt': [0],
                        'lift': [0],
                        'cumlift': [0],
                        'cumlift_ref': [1]
                    })
                    ntiles_agg = []
                    ntiles_agg = pd.DataFrame(index=range(1, self.ntiles + 1))
                    ntiles_agg['model_label'] = self.model_labels[i]
                    ntiles_agg['dataset_label'] = k
                    ntiles_agg['target_class'] = j
                    ntiles_agg['ntile'] = range(1, self.ntiles + 1, 1)
                    relvars = ['dec_%s' % j,'all']
                    ntiles_agg['tot'] = scores_and_ntiles[
                        (scores_and_ntiles.dataset_label == k) &
                        (scores_and_ntiles.model_label == self.model_labels[i])
                    ][relvars].groupby('dec_%s' % j).agg('sum')
                    scores_and_ntiles['pos'] = scores_and_ntiles.target_class == j
                    relvars = ['dec_%s' % j, 'pos']
                    ntiles_agg['pos'] = scores_and_ntiles[
                        (scores_and_ntiles.dataset_label == k) &
                        (scores_and_ntiles.model_label == self.model_labels[i])
                    ][relvars].groupby('dec_%s' % j).agg('sum')
                    scores_and_ntiles['neg'] = scores_and_ntiles.target_class != j
                    relvars = ['dec_%s' % j, 'neg']
                    ntiles_agg['neg'] = scores_and_ntiles[
                        (scores_and_ntiles.dataset_label == k) &
                        (scores_and_ntiles.model_label == self.model_labels[i])
                    ][relvars].groupby('dec_%s' % j).agg('sum')
                    ntiles_agg['pct'] = ntiles_agg.pos / ntiles_agg.tot
                    ntiles_agg['postot'] = ntiles_agg.pos.sum()
                    ntiles_agg['negtot'] = ntiles_agg.neg.sum()
                    ntiles_agg['tottot'] = ntiles_agg.tot.sum()
                    ntiles_agg['pcttot'] = ntiles_agg.pct.sum()
                    ntiles_agg['cumpos'] = ntiles_agg.pos.cumsum()
                    ntiles_agg['cumneg'] = ntiles_agg.neg.cumsum()
                    ntiles_agg['cumtot'] = ntiles_agg.tot.cumsum()
                    ntiles_agg['cumpct'] = ntiles_agg.cumpos / ntiles_agg.cumtot
                    ntiles_agg['gain'] = ntiles_agg.pos / ntiles_agg.postot
                    ntiles_agg['cumgain'] = ntiles_agg.cumpos / ntiles_agg.postot
                    ntiles_agg['gain_ref'] = ntiles_agg.ntile / self.ntiles
                    ntiles_agg['pct_ref'] = ntiles_agg.postot / ntiles_agg.tottot
                    ntiles_agg['gain_opt'] = 1.0
                    ntiles_agg.loc[
                        (ntiles_agg.cumtot / ntiles_agg.postot) <= 1.0,
                        'gain_opt'
                    ] = (ntiles_agg.cumtot / ntiles_agg.postot)
                    ntiles_agg['lift'] = ntiles_agg.pct / (ntiles_agg.postot / ntiles_agg.tottot)
                    ntiles_agg['cumlift'] = (
                        ntiles_agg.cumpct / (ntiles_agg.postot / ntiles_agg.tottot)
                    )
                    ntiles_agg['cumlift_ref'] = 1

                    # Use pd.concat to concatenate DataFrames with ignore_index option
                    ntiles_aggregate = pd.concat(
                        [ntiles_aggregate, ntiles_agg],
                        ignore_index=True
                    )
                    add_origin = pd.concat(
                        [add_origin, add_origin_add],
                        axis = 0
                    )

        ntiles_aggregate = (
            pd.concat([add_origin, ntiles_aggregate], axis = 0)
            .sort_values(by=['model_label', 'dataset_label', 'target_class', 'ntile'])
        )
        cols = ntiles_aggregate.columns
        return ntiles_aggregate[cols]
    
    def plotting_scope(
        self,
        scope = 'no_comparison',
        select_model_label = [],
        select_dataset_label = [],
        select_targetclass = [],
        select_smallest_targetclass = True
    ):
        """
        Create plot_input
        
        This function builds the pandas dataframe plot_input wich is a subset of scores_and_ntiles.
        The dataset is the subset of scores_and_ntiles that is dependent of
        1 of the 4 evaluation types that a user can request.

        .. versionchanged:: 0.3.9
            Parameters has been reorganized.
        
        Parameters
        ----------
        
        scope : {'no_comparison', 'compare_models', 'compare_datasets', \
                 'compare_targetclasses'}, default='no_comparison'

            How is this function evaluated? There are 4 different perspectives to evaluate model plots.
            
            1. `scope='no_comparison'`
                This perspective will show a single plot that contains the viewpoint from:
                
                - 1 dataset
                - 1 model
                - 1 target class
            
            2. `scope='compare_models'` 
                This perspective will show plots that contains the viewpoint from:
                
                - 2 or more different models
                - 1 dataset
                - 1 target class
            
            3. `scope='compare_datasets'` 
                This perspective will show plots that contains the viewpoint from:
                
                - 2 or more different datasets
                - 1 model
                - 1 target class
            
            4. `scope='compare_targetclasses'`
                This perspective will show plots that contains the viewpoint from:
                
                - 2 or more different target classes
                - 1 dataset
                - 1 model
        
        select_model_label : list of str
            List of one or more elements from the model_name parameter.
        
        select_dataset_label : list of str
            List of one or more elements from the description parameter.
        
        select_targetclass : list of str
            List of one or more elements from the label data.
        
        select_smallest_targetclass : bool, default = True
            Should the plot only contain the results of the smallest targetclass.
            If True, the specific target is defined from the first dataset.

        Returns
        -------
        Pandas dataframe, a subset of scores_and_ntiles, for all dataset, model
        and target value combinations for all ntiles.
        It contains all necessary information for model plotting.

        Raises
        ------
        ValueError: If the wrong `scope` value is specified.
        """
        ntiles_aggregate = self.aggregate_over_ntiles()
        ntiles_aggregate['scope'] = scope

        if scope not in (
            'no_comparison', 'compare_models',
            'compare_datasets', 'compare_targetclasses'
        ):
            raise ValueError(
                'Invalid scope value, it must be one of the following: '
                'no_comparison, compare_models, compare_datasets or compare_targetclasses.'
            )
        
        # check parameters
        select_model_label = _check_input(
            select_model_label, self.model_labels, 'select_model_label'
        )
        select_dataset_label = _check_input(
            select_dataset_label, self.dataset_labels, 'select_dataset_label'
        )
        select_targetclass = _check_input(
            select_targetclass, list(self.models[0].classes_), 'select_targetclass'
        )

        if scope == 'no_comparison':
            print(
                'Default scope value no_comparison selected, '
                'single evaluation line will be plotted.'
            )
            if len(select_model_label) >= 1:
                select_model_label = select_model_label
            else:
                select_model_label = self.model_labels
            if len(select_dataset_label) >= 1:
                select_dataset_label = select_dataset_label
            else:
                select_dataset_label = self.dataset_labels
            if len(select_targetclass) >= 1:
                select_targetclass = select_targetclass
            elif select_smallest_targetclass == True:
                select_targetclass = [self.label_data[0].value_counts(ascending = True).idxmin()]
                print("The label with smallest class is %s" % select_targetclass[0])
            else:
                select_targetvalue = list(self.models[0].classes_)
            plot_input = ntiles_aggregate[
                (ntiles_aggregate.model_label == select_model_label[0]) & 
                (ntiles_aggregate.dataset_label == select_dataset_label[0]) & 
                (ntiles_aggregate.target_class == select_targetclass[0])]
            print(
                'Target class %s, dataset %s and model %s.' % (
                    select_targetclass[0], select_dataset_label[0], select_model_label[0]
                )
            )
        elif scope == 'compare_models':
            print('compare models')
            if len(select_model_label) >= 2:
                select_model_label = select_model_label
            else:
                select_model_label = self.model_labels
            if len(select_dataset_label) >= 1:
                select_dataset_label = select_dataset_label
            else:
                select_dataset_label = self.dataset_labels
            if len(select_targetclass) >= 1:
                select_targetclass = select_targetclass
            elif select_smallest_targetclass == True:
                select_targetclass = [self.label_data[0].value_counts(ascending = True).idxmin()]
                print("The label with smallest class is %s" % select_targetclass)
            else:
                select_targetclass = list(self.models[0].classes_)
            plot_input = ntiles_aggregate[
                (ntiles_aggregate.model_label.isin(select_model_label)) &
                (ntiles_aggregate.dataset_label == select_dataset_label[0]) &
                (ntiles_aggregate.target_class == select_targetclass[0])]
        elif scope == 'compare_datasets':
            print('compare datasets')
            if len(select_model_label) >= 1:
                select_model_label = select_model_label
            else:
                select_model_label = self.model_labels
            if len(select_dataset_label) >= 2:
                select_dataset_label = select_dataset_label
            else:
                select_dataset_label = self.dataset_labels
            if len(select_targetclass) >= 1:
                select_targetclass = select_targetclass
            elif select_smallest_targetclass == True:
                select_targetclass = [self.label_data[0].value_counts(ascending = True).idxmin()]
                print("The label with smallest class is %s" % select_targetclass)
            else:
                select_targetclass = list(self.models[0].classes_)
            plot_input = ntiles_aggregate[
                (ntiles_aggregate.model_label == select_model_label[0]) &
                (ntiles_aggregate.dataset_label.isin(select_dataset_label)) &
                (ntiles_aggregate.target_class == select_targetclass[0])]
        else: # scope == 'compare_targetclasses'
            print('compare target classes')
            if len(select_model_label) >= 1:
                select_model_label = select_model_label
            else:
                select_model_label = self.model_labels
            if len(select_dataset_label) >= 1:
                select_dataset_label = select_dataset_label
            else:
                select_dataset_label = self.dataset_labels
            if len(select_targetclass) >= 2:
                select_targetclass = select_targetclass
            else:
                select_targetclass = list(self.models[0].classes_)
            plot_input = ntiles_aggregate[
                (ntiles_aggregate.model_label == select_model_label[0]) &
                (ntiles_aggregate.dataset_label == select_dataset_label[0]) &
                (ntiles_aggregate.target_class.isin(select_targetclass))]
        return plot_input


# Alias for backward compatibility or convenience ModelPlotPy
modelplotpy = ModelPlotPy

##########################################################################
## (cumulative) Response, Lift, Gains and plots
##########################################################################

def plot_response(
    plot_input,
    save_fig = True,
    save_fig_filename = '',
    highlight_ntile = None,
    highlight_how = 'plot_text',
):
    """
    Plotting response curve
    
    Parameters
    ----------
    plot_input : pandas.DataFrame
        The result from scope_modevalplot().
        
    save_fig : bool, default=True
        Save the plot.
        
    save_fig_filename : str, optional, default=''
        Specify the path and filetype to save the plot.
        If nothing specified, the plot will be saved as jpeg
        to the current working directory.
        
    highlight_ntile : int or None, optional, default=None
        Highlight the value of the response curve at a specified ntile value.
        
        .. versionchanged:: 0.3.9
            Default changed from False to None.
        
    highlight_how : {'plot','text','plot_text'}, optional, default='plot_text'
        Highlight_how specifies where information about the model performance is printed.
        It can be shown as text, on the plot or both.
    
    Returns
    -------
    It returns a matplotlib.axes._subplots.AxesSubplot object
    that can be transformed into the same plot with the .figure command.
    The plot is by default written to disk (save_fig = True).
    The location and filetype of the file depend on the save_fig_filename parameter.
    If the save_fig_filename parameter is empty (not specified),
    the plot will be written to the working directory as png. 
    Otherwise the location and file type is specified by the user.
        
    Raises
    ------
    TypeError: If ``highlight_ntile`` is not specified as an int.
    ValueError: If the wrong ``highlight_how`` value is specified.
    """
    
    models   = plot_input.model_label.unique().tolist()
    datasets = plot_input.dataset_label.unique().tolist()
    classes  = plot_input.target_class.unique().tolist()
    scope = plot_input.scope.unique()[0]
    ntiles = plot_input.ntile.nunique() - 1
    colors = (
        "#E41A1C",
        "#377EB8",
        "#4DAF4A",
        "#984EA3",
        "#FF7F00",
        "#FFFF33",
        "#A65628",
        "#F781BF",
        "#999999"
    )
    
    if ntiles == 10:
        description_label = 'decile'
    elif ntiles == 100:
        description_label = 'percentile'
    else:
        description_label = 'ntile'

    if ntiles <= 20:
        xlabper = 1
    elif ntiles <= 40:
        xlabper = 2
    else:
        xlabper = 5

    fig, ax = plt.subplots(figsize = (12,7))
    plt.suptitle('Response', fontsize = 16)
    ax.set_xlabel(description_label)
    ax.set_ylabel("response")
    
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xticks(np.arange(0, ntiles + 1, xlabper))
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    
    ax.set_xlim([1, ntiles])
    ax.set_ylim([0, round((max(plot_input.pct)+0.05)*100,-1)/100])
    ax.grid(True)
    
    if scope == "no_comparison":
        ax.set_title(
            "model: %s & dataset: %s & target class: %s" % (
                models[0], datasets[0], classes[0]
            ),
            fontweight = 'bold'
        )
        ax.plot(
            plot_input.ntile,
            plot_input.pct,
            label = classes[0],
            color = colors[0]
        )
        ax.plot(
            plot_input.ntile,
            plot_input.pct_ref,
            linestyle = 'dashed',
            label = "overall response (%s)" % classes[0],
            color = colors[0]
        )
        ax.legend(
            loc = 'upper right',
            shadow = False,
            frameon = False
        )
    elif scope == "compare_datasets":
        for col, i in enumerate(datasets):
            ax.plot(
                plot_input.ntile[plot_input.dataset_label == i],
                plot_input.pct[plot_input.dataset_label == i],
                label = i,
                color = colors[col]
            )
            ax.plot(
                plot_input.ntile[plot_input.dataset_label == i],
                plot_input.pct_ref[plot_input.dataset_label == i],
                linestyle = 'dashed',
                label = "overall response (%s)" % i,
                color = colors[col]
            )
        ax.set_title(
            "scope: comparing datasets & model: %s & target class: %s" % (
                models[0], classes[0]
            ),
            fontweight = 'bold'
        )
        ax.legend(
            loc = 'upper right',
            shadow = False,
            frameon = False
        )
    elif scope == "compare_models":
        for col, i in enumerate(models):
            ax.plot(
                plot_input.ntile[plot_input.model_label == i],
                plot_input.pct[plot_input.model_label == i],
                label = i,
                color = colors[col]
            )
            ax.plot(
                plot_input.ntile[plot_input.model_label == i],
                plot_input.pct_ref[plot_input.model_label == i],
                linestyle = 'dashed',
                label = "overall response (%s)" % i,
                color = colors[col]
            )
        ax.set_title(
            "scope: comparing models & dataset: %s & target class: %s" % (
                datasets[0], classes[0]
            ),
            fontweight = 'bold'
        )
        ax.legend(
            loc = 'upper right',
            shadow = False,
            frameon = False
        )
    else: #compare_targetclasses
        for col, i in enumerate(classes):
            ax.plot(
                plot_input.ntile[plot_input.target_class == i],
                plot_input.pct[plot_input.target_class == i],
                label = i,
                color = colors[col]
            )
            ax.plot(
                plot_input.ntile[plot_input.target_class == i],
                plot_input.pct_ref[plot_input.target_class == i],
                linestyle = 'dashed',
                label = "overall response (%s)" % i,
                color = colors[col]
            )
        ax.set_title(
            "Scope: comparing target classes & dataset: %s & model: %s" % (
                datasets[0], models[0]
            ),
            fontweight = 'bold'
        )
        ax.legend(
            loc = 'upper right',
            shadow = False,
            frameon = False
        )
    
    if highlight_ntile is not None:
        
        if highlight_ntile not in np.linspace(1, ntiles, num = ntiles).tolist():
            raise TypeError(
                'Invalid value for highlight_ntile parameter. '
                'It must be an int value between 1 and %d' % (ntiles)
            )
            
        if highlight_how not in ('plot','text','plot_text'):
            raise ValueError(
                'Invalid highlight_how value, '
                'it must be one of the following: plot, text or plot_text.'
            )
        
        else:
            text = ''
            if scope == "no_comparison":
                cumpct = plot_input.loc[plot_input.ntile == highlight_ntile, 'pct'].tolist()
                plt.plot(
                    [1, highlight_ntile],
                    [cumpct[0]] * 2,
                    linestyle = '-.',
                    color = colors[0],
                    lw = 1.5
                )
                plt.plot(
                    [highlight_ntile] * 2,
                    [0] + [cumpct[0]],
                    linestyle = '-.',
                    color = colors[0],
                    lw = 1.5
                )
                xy = tuple(
                    [highlight_ntile] + [cumpct[0]]
                )
                ax.plot(
                    xy[0],
                    xy[1],
                    ".r",
                    ms = 20,
                    color = colors[0]
                )
                ax.annotate(
                    str(int(cumpct[0] * 100)) + "%",
                    xy = xy, xytext = (-30, -30),
                    textcoords='offset points',
                    ha = 'center',
                    va = 'bottom',
                    color = 'black',
                    bbox=dict(
                        boxstyle='round, pad = 0.4',
                        alpha = 1,
                        fc = colors[0]
                    ),  #fc = 'yellow', alpha = 0.3),
                    arrowprops = dict(
                        arrowstyle = '->',
                        color = 'black'
                    )
                )
                text += (
                    (
                        'When we select %s %d from model %s in dataset '
                        '%s the percentage of %s cases in the selection is %d'
                    ) % (
                        description_label,
                        highlight_ntile,
                        models[0],
                        datasets[0],
                        classes[0],
                        int(cumpct[0] * 100)
                    ) + '%.\n'
                )
            elif scope == "compare_datasets":
                for col, i in enumerate(datasets):
                    cumpct = (
                        plot_input.loc[
                            plot_input.ntile == highlight_ntile,
                            ['dataset_label', 'pct']
                        ]
                    )
                    cumpct = cumpct.pct[cumpct.dataset_label == i].tolist()
                    plt.plot(
                        [1, highlight_ntile],
                        cumpct * 2,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    plt.plot(
                        [highlight_ntile] * 2,
                        [0] + cumpct,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    xy = tuple([highlight_ntile] + cumpct)
                    ax.plot(
                        xy[0],
                        xy[1],
                        ".r",
                        ms = 20,
                        color = colors[col]
                    )
                    ax.annotate(
                        str(int(cumpct[0] * 100)) + "%",
                        xy = xy,
                        xytext = (-30, -30),
                        textcoords='offset points',
                        ha = 'center',
                        va = 'bottom',
                        color = 'black',
                        bbox=dict(
                            boxstyle='round, pad = 0.4',
                            fc = colors[col],
                            alpha = 0.8,
                        ), #fc = 'yellow', alpha = 0.3),
                        arrowprops = dict(
                            arrowstyle = '->',
                            color = 'black'
                        )
                    )
                    text += (
                        (
                            'When we select %s %d from model %s in dataset '
                            '%s the percentage of %s cases in the selection is %d'
                        ) % (
                            description_label,
                            highlight_ntile,
                            models[0],
                            datasets[col],
                            classes[0],
                            int(cumpct[0] * 100)
                        ) + '%.\n'
                    )
            elif scope == "compare_models":
                for col, i in enumerate(models):
                    cumpct = (
                        plot_input.loc[
                            plot_input.ntile == highlight_ntile,
                            ['model_label', 'pct']
                        ]
                    )
                    cumpct = cumpct.pct[cumpct.model_label == i].tolist()
                    plt.plot(
                        [1, highlight_ntile],
                        cumpct * 2,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    plt.plot(
                        [highlight_ntile] * 2,
                        [0] + cumpct,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    xy = tuple([highlight_ntile] + cumpct)
                    ax.plot(
                        xy[0],
                        xy[1],
                        ".r",
                        ms = 20,
                        color = colors[col]
                    )
                    ax.annotate(
                        str(int(cumpct[0] * 100)) + "%",
                        xy = xy,
                        xytext = (-30, -30),
                        textcoords='offset points',
                        ha = 'center',
                        va = 'bottom',
                        color = 'black',
                        bbox=dict(
                            boxstyle='round, pad = 0.4',
                            fc = colors[col],
                            alpha = 0.8,
                        ), #fc = 'yellow', alpha = 0.3),
                        arrowprops = dict(
                            arrowstyle = '->',
                            color = 'black')
                    )
                    text += (
                        (
                            'When we select %s %d from model %s in dataset '
                            '%s the percentage of %s cases in the selection is %d'
                        ) % (
                            description_label,
                            highlight_ntile,
                            models[col],
                            datasets[0],
                            classes[0],
                            int(cumpct[0] * 100)
                        ) + '%.\n'
                    )
            else: # compare targetvalues
                for col, i in enumerate(classes):
                    cumpct = (
                        plot_input.loc[
                            plot_input.ntile == highlight_ntile,
                            ['target_class', 'pct']
                        ]
                    )
                    cumpct = cumpct.pct[cumpct.target_class == i].tolist()
                    plt.plot(
                        [1, highlight_ntile],
                        cumpct * 2,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    plt.plot(
                        [highlight_ntile] * 2,
                        [0] + cumpct,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    xy = tuple([highlight_ntile] + cumpct)
                    ax.plot(
                        xy[0],
                        xy[1],
                        ".r",
                        ms = 20,
                        color = colors[col]
                    )
                    ax.annotate(
                        str(int(cumpct[0] * 100)) + "%",
                        xy = xy,
                        xytext = (-30, -30),
                        textcoords='offset points',
                        ha = 'center',
                        va = 'bottom',
                        color = 'black',
                        bbox=dict(
                            boxstyle='round, pad = 0.4',
                            fc = colors[col],
                            alpha = 0.8,
                        ), #fc = 'yellow', alpha = 0.3),
                        arrowprops = dict(
                            arrowstyle = '->',
                            color = 'black')
                    )
                    text += (
                        (
                            'When we select %s %d from model %s in dataset '
                            '%s the percentage of %s cases in the selection is %d'
                        ) % (
                            description_label,
                            highlight_ntile,
                            models[0],
                            datasets[0],
                            classes[col],
                            int(cumpct[0] * 100)
                        ) + '%.\n'
                    )
            if highlight_how in ('text', 'plot_text'):
                print(text[:-1])
            if highlight_how in ('plot', 'plot_text'):
                fig.text(.15, -0.001, text[:-1], ha='left')
    
    if save_fig == True:
        if not save_fig_filename:
            location = '%s/Response plot.png' % os.getcwd()
            plt.savefig(location, dpi = 300)
            print("The response plot is saved in %s" % location)
        else:
            plt.savefig(save_fig_filename, dpi = 300)
            print("The response plot is saved in %s" % save_fig_filename)
        plt.show()
        plt.gcf().clear()
    plt.show()
    return ax


def plot_cumresponse(
    plot_input,
    save_fig = True,
    save_fig_filename = '',
    highlight_ntile = None,
    highlight_how = 'plot_text',
):
    """
    Plotting cumulative response curve
    
    Parameters
    ----------
    plot_input : pandas.DataFrame
        The result from scope_modevalplot().

    save_fig : bool, default=True
        Save the plot.

    save_fig_filename : str, optional, default=''
        Specify the path and filetype to save the plot.
        If nothing specified, the plot will be saved as jpeg
        to the current working directory.

    highlight_ntile : int or None, optional, default=None
        Highlight the value of the response curve at a specified ntile value.
        
        .. versionchanged:: 0.3.9
            Default changed from False to None.

    highlight_how : str, optional, default='plot_text'
        Highlight_how specifies where information about the model performance is printed.
        It can be shown as text, on the plot or both.
        
    Returns
    -------
    It returns a matplotlib.axes._subplots.AxesSubplot object
    that can be transformed into the same plot with the .figure command.
    The plot is by default written to disk (save_fig = True).
    The location and filetype of the file depend on the save_fig_filename parameter.
    If the save_fig_filename parameter is empty (not specified),
    the plot will be written to the working directory as png. 
    Otherwise the location and file type is specified by the user.
        
    Raises
    ------
    TypeError: If ``highlight_ntile`` is not specified as an int.
    ValueError: If the wrong ``highlight_how`` value is specified.
    """
    models   = plot_input.model_label.unique().tolist()
    datasets = plot_input.dataset_label.unique().tolist()
    classes  = plot_input.target_class.unique().tolist()
    scope = plot_input.scope.unique()[0]
    ntiles = plot_input.ntile.nunique() - 1
    colors = (
        "#E41A1C",
        "#377EB8",
        "#4DAF4A",
        "#984EA3",
        "#FF7F00",
        "#FFFF33",
        "#A65628",
        "#F781BF",
        "#999999"
    )
    
    if ntiles == 10:
        description_label = 'decile'
    elif ntiles == 100:
        description_label = 'percentile'
    else:
        description_label = 'ntile'
    
    if ntiles <= 20:
        xlabper = 1
    elif ntiles <= 40:
        xlabper = 2
    else:
        xlabper = 5

    fig, ax = plt.subplots(figsize = (12,7))
    ax.set_xlabel(description_label)
    ax.set_ylabel("cumulative response")
    plt.suptitle('Cumulative response', fontsize = 16)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xticks(np.arange(0, ntiles + 1, xlabper))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.grid(True)
    ax.set_xlim([1, ntiles])
    ax.set_ylim([0, round((max(plot_input.cumpct)+0.05)*100,-1)/100])
    
    if scope == "no_comparison":
        ax.set_title(
            "model: %s & dataset: %s & target class: %s" % (
                models[0], datasets[0], classes[0]
            ),
            fontweight = 'bold'
        )
        ax.plot(
            plot_input.ntile,
            plot_input.cumpct,
            label = classes[0],
            color = colors[0]
        )
        ax.plot(
            plot_input.ntile,
            plot_input.pct_ref,
            linestyle = 'dashed',
            label = "overall response (%s)" % classes[0],
            color = colors[0]
        )
        ax.legend(
            loc = 'upper right',
            shadow = False,
            frameon = False
        )
    elif scope == "compare_datasets":
        for col, i in enumerate(datasets):
            ax.plot(
                plot_input.ntile[plot_input.dataset_label == i],
                plot_input.cumpct[plot_input.dataset_label == i],
                label = i,
                color = colors[col]
            )
            ax.plot(
                plot_input.ntile[plot_input.dataset_label == i],
                plot_input.pct_ref[plot_input.dataset_label == i],
                linestyle = 'dashed',
                label = "overall response (%s)" % i,
                color = colors[col]
            )
        ax.set_title(
            "scope: comparing datasets & model: %s & target class: %s" % (
                models[0], classes[0]
            ),
            fontweight = 'bold'
        )
        ax.legend(
            loc = 'upper right',
            shadow = False,
            frameon = False
        )
    elif scope == "compare_models":
        for col, i in enumerate(models):
            ax.plot(
                plot_input.ntile[plot_input.model_label == i],
                plot_input.cumpct[plot_input.model_label == i],
                label = i,
                color = colors[col]
            )
            ax.plot(
                plot_input.ntile[plot_input.model_label == i],
                plot_input.pct_ref[plot_input.model_label == i],
                linestyle = 'dashed',
                label = "overall response (%s)" % i,
                color = colors[col]
            )
        ax.set_title(
            "Scope: comparing models & dataset: %s & target class: %s" % (
                datasets[0], classes[0]
            ),
            fontweight = 'bold'
        )
        ax.legend(
            loc = 'upper right',
            shadow = False,
            frameon = False
        )
    else: #compare_targetclasses
        for col, i in enumerate(classes):
            ax.plot(
                plot_input.ntile[plot_input.target_class == i],
                plot_input.cumpct[plot_input.target_class == i],
                label = i,
                color = colors[col]
            )
            ax.plot(
                plot_input.ntile[plot_input.target_class == i],
                plot_input.pct_ref[plot_input.target_class == i],
                linestyle = 'dashed',
                label = "overall response (%s)" % i,
                color = colors[col]
            )
        ax.set_title(
            "Comparing target classes & dataset: %s & model: %s" % (
                datasets[0], models[0]
            ),
            fontweight = 'bold'
        )
        ax.legend(
            loc = 'upper right',
            shadow = False,
            frameon = False
        )
    
    if highlight_ntile is not None:
        
        if highlight_ntile not in np.linspace(1, ntiles, num = ntiles).tolist():
            raise TypeError(
                'Invalid value for highlight_ntile parameter. '
                'It must be an int value between 1 and %d' % (ntiles)
            )
            
        if highlight_how not in ('plot','text','plot_text'):
            raise ValueError(
                'Invalid highlight_how value, '
                'it must be one of the following: plot, text or plot_text.'
            )
        
        else:
            text = ''
            if scope == "no_comparison":
                cumpct = (
                    plot_input.loc[
                        plot_input.ntile == highlight_ntile,
                        'cumpct'
                    ].tolist()
                )
                plt.plot(
                    [1, highlight_ntile],
                    [cumpct[0]] * 2,
                    linestyle = '-.',
                    color = colors[0],
                    lw = 1.5
                )
                plt.plot(
                    [highlight_ntile] * 2 ,
                    [0] + [cumpct[0]],
                    linestyle = '-.',
                    color = colors[0],
                    lw = 1.5
                )
                xy = tuple([highlight_ntile] + [cumpct[0]])
                ax.plot(
                    xy[0],
                    xy[1],
                    ".r",
                    ms = 20,
                    color = colors[0]
                )
                ax.annotate(
                    str(int(cumpct[0] * 100)) + "%",
                    xy = xy,
                    xytext = (-30, -30),
                    textcoords = 'offset points',
                    ha = 'center',
                    va = 'bottom',
                    color = 'black',
                    bbox = dict(
                        boxstyle = 'round, pad = 0.4',
                        alpha = 1,
                        fc = colors[0]
                    ),
                    arrowprops = dict(
                        arrowstyle = '->',
                        color = 'black')
                )
                text += (
                    (
                        'When we select %ss 1 until %d according to model %s in dataset '
                        '%s the percentage of %s cases in the selection is %d'
                    ) % (
                        description_label,
                        highlight_ntile,
                        models[0],
                        datasets[0],
                        classes[0],
                        int(cumpct[0] * 100)
                    ) + '%.\n'
                )
            elif scope == "compare_datasets":
                for col, i in enumerate(datasets):
                    cumpct = (
                        plot_input.loc[
                            plot_input.ntile == highlight_ntile,
                            ['dataset_label', 'cumpct']
                        ]
                    )
                    cumpct = cumpct.cumpct[cumpct.dataset_label == i].tolist()
                    plt.plot(
                        [1, highlight_ntile],
                        cumpct * 2,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    plt.plot(
                        [highlight_ntile] * 2,
                        [0] + cumpct,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    xy = tuple([highlight_ntile] + cumpct)
                    ax.plot(
                        xy[0],
                        xy[1],
                        ".r",
                        ms = 20,
                        color = colors[col]
                    )
                    ax.annotate(
                        str(int(cumpct[0] * 100)) + "%",
                        xy = xy,
                        xytext = (-30, -30),
                        textcoords = 'offset points',
                        ha = 'center',
                        va = 'bottom',
                        color = 'black',
                        bbox = dict(
                            boxstyle = 'round, pad = 0.4',
                            fc = colors[col],
                            alpha = 0.8,
                        ),
                        arrowprops = dict(
                            arrowstyle = '->',
                            color = 'black')
                    )
                    text += (
                        (
                            'When we select %ss 1 until %d according to model %s in dataset '
                            '%s the percentage of %s cases in the selection is %d'
                        ) % (
                            description_label,
                            highlight_ntile,
                            models[0],
                            datasets[col],
                            classes[0],
                            int(cumpct[0] * 100)
                        ) + '%.\n'
                    )
            elif scope == "compare_models":
                for col, i in enumerate(models):
                    cumpct = (
                        plot_input.loc[
                            plot_input.ntile == highlight_ntile,
                            ['model_label', 'cumpct']
                        ]
                    )
                    cumpct = cumpct.cumpct[cumpct.model_label == i].tolist()
                    plt.plot(
                        [1, highlight_ntile],
                        cumpct * 2,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    plt.plot(
                        [highlight_ntile] * 2,
                        [0] + cumpct,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    xy = tuple([highlight_ntile] + cumpct)
                    ax.plot(
                        xy[0],
                        xy[1],
                        ".r",
                        ms = 20,
                        color = colors[col]
                    )
                    ax.annotate(
                        str(int(cumpct[0] * 100)) + "%",
                        xy = xy,
                        xytext = (-30, -30),
                        textcoords = 'offset points',
                        ha = 'center',
                        va = 'bottom',
                        color = 'black',
                        bbox = dict(
                            boxstyle = 'round, pad = 0.4',
                            fc = colors[col],
                            alpha = 0.8,
                        ),
                        arrowprops = dict(
                            arrowstyle = '->',
                            color = 'black')
                    )
                    text += (
                        (
                            'When we select %ss 1 until %d according to model %s in dataset '
                            '%s the percentage of %s cases in the selection is %d'
                        ) % (
                            description_label,
                            highlight_ntile,
                            models[col],
                            datasets[0],
                            classes[0],
                            int(cumpct[0] * 100)
                        ) + '%.\n'
                    )
            else: # compare targetvalues
                for col, i in enumerate(classes):
                    cumpct = (
                        plot_input.loc[
                            plot_input.ntile == highlight_ntile,
                            ['target_class', 'cumpct']
                        ]
                    )
                    cumpct = cumpct.cumpct[cumpct.target_class == i].tolist()
                    plt.plot(
                        [1, highlight_ntile],
                        cumpct * 2,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    plt.plot(
                        [highlight_ntile] * 2,
                        [0] + cumpct,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    xy = tuple([highlight_ntile] + cumpct)
                    ax.plot(
                        xy[0],
                        xy[1],
                        ".r",
                        ms = 20,
                        color = colors[col]
                    )
                    ax.annotate(
                        str(int(cumpct[0] * 100)) + "%",
                        xy = xy,
                        xytext = (-30, -30),
                        textcoords = 'offset points',
                        ha = 'center',
                        va = 'bottom',
                        color = 'black',
                        bbox = dict(
                            boxstyle = 'round, pad = 0.4',
                            fc = colors[col],
                            alpha = 0.8,
                        ),
                        arrowprops = dict(
                            arrowstyle = '->',
                            color = 'black'
                        )
                    )
                    text += (
                        (
                            'When we select %ss 1 until %d according to model %s in dataset '
                            '%s the percentage of %s cases in the selection is %d'
                        ) % (
                            description_label,
                            highlight_ntile,
                            models[0],
                            datasets[0],
                            classes[col],
                            int(cumpct[0] * 100)
                        ) + '%.\n'
                    )
            
            if highlight_how in ('text', 'plot_text'):
                print(text[:-1])
            if highlight_how in ('plot', 'plot_text'):
                fig.text(.15, -0.001, text[:-1], ha='left')
                    
    if save_fig == True:
        if not save_fig_filename:
            location = '%s/Cumulative response plot.png' % os.getcwd()
            plt.savefig(location, dpi = 300)
            print("The cumulative response plot is saved in %s" % location)
        else:
            plt.savefig(save_fig_filename, dpi = 300)
            print("The cumulative response plot is saved in %s" % save_fig_filename)
        plt.show()
        plt.gcf().clear()
    plt.show()
    return ax


def plot_cumlift(
    plot_input,
    save_fig = True,
    save_fig_filename = '',
    highlight_ntile = None,
    highlight_how = 'plot_text',
):
    """
    Plotting cumulative lift curve
    
    Parameters
    ----------
    plot_input : pandas.DataFrame
        The result from scope_modevalplot().

    save_fig : bool, default=True
        Save the plot.

    save_fig_filename : str, optional, default=''
        Specify the path and filetype to save the plot.
        If nothing specified, the plot will be saved as jpeg
        to the current working directory.

    highlight_ntile : int or None, optional, default=None
        Highlight the value of the response curve at a specified ntile value.
        
        .. versionchanged:: 0.3.9
            Default changed from False to None.

    highlight_how : str, optional, default='plot_text'
        Highlight_how specifies where information about the model performance is printed.
        It can be shown as text, on the plot or both.
    
    Returns
    -------
    It returns a matplotlib.axes._subplots.AxesSubplot object
    that can be transformed into the same plot with the .figure command.
    The plot is by default written to disk (save_fig = True). The location
    and filetype of the file depend on the save_fig_filename parameter.
    If the save_fig_filename parameter is empty (not specified),
    the plot will be written to the working directory as png. 
    Otherwise the location and file type is specified by the user.
        
    Raises
    ------
    TypeError: If ``highlight_ntile`` is not specified as an int.
    ValueError: If the wrong ``highlight_how`` value is specified.
    """
    models   = plot_input.model_label.unique().tolist()
    datasets = plot_input.dataset_label.unique().tolist()
    classes  = plot_input.target_class.unique().tolist()
    scope = plot_input.scope.unique()[0]
    ntiles = plot_input.ntile.nunique() - 1
    colors = (
        "#E41A1C",
        "#377EB8",
        "#4DAF4A",
        "#984EA3",
        "#FF7F00",
        "#FFFF33",
        "#A65628",
        "#F781BF",
        "#999999"
    )
    
    if ntiles == 10:
        description_label = 'decile'
    elif ntiles == 100:
        description_label = 'percentile'
    else:
        description_label = 'ntile'
      
    if ntiles <= 20:
        xlabper = 1
    elif ntiles <= 40:
        xlabper = 2
    else:
        xlabper = 5

    fig, ax = plt.subplots(figsize = (12,7))
    ax.set_xlabel(description_label)
    ax.set_ylabel("cumulative lift")
    plt.suptitle('Cumulative lift', fontsize = 16)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xticks(np.arange(0, ntiles + 1, xlabper))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.grid(True)
    ax.set_xlim([1, ntiles])
    ax.set_ylim([0, max(plot_input.cumlift)])
    ax.plot(
        list(range(1, ntiles + 1, 1)),
        [1] * ntiles,
        linestyle = 'dashed',
        label = "no lift",
        color = 'grey'
    )
    
    if scope == "no_comparison":
        ax.set_title(
            "model: %s & dataset: %s & target class: %s" % (
                models[0], datasets[0], classes[0]
            ),
            fontweight = 'bold'
        )
        ax.plot(
            plot_input.ntile,
            plot_input.cumlift,
            label = classes[0],
            color = colors[0]
        )
        #ax.plot(plot_input.ntile, plot_input.cumlift_ref, linestyle = 'dashed', label = "overall response (%s)" % classes[0], color = colors[0])
        ax.legend(
            loc = 'upper right',
            shadow = False,
            frameon = False
        )
    elif scope == "compare_datasets":
        for col, i in enumerate(datasets):
            ax.plot(
                plot_input.ntile[plot_input.dataset_label == i],
                plot_input.cumlift[plot_input.dataset_label == i],
                label = i,
                color = colors[col]
            )
            #ax.plot(plot_input.ntile[plot_input.dataset_label == i], plot_input.cumlift_ref[plot_input.dataset_label == i], linestyle = 'dashed', label = "overall response (%s)" % i, color = colors[col])
        ax.set_title(
            "scope: comparing datasets & model: %s & target class: %s" % (
                models[0], classes[0]
            ),
            fontweight = 'bold'
        )
        ax.legend(
            loc = 'upper right',
            shadow = False,
            frameon = False
        )
    elif scope == "compare_models":
        for col, i in enumerate(models):
            ax.plot(
                plot_input.ntile[plot_input.model_label == i],
                plot_input.cumlift[plot_input.model_label == i],
                label = i,
                color = colors[col]
            )
            #ax.plot(plot_input.ntile[plot_input.model_label == i], plot_input.cumlift_ref[plot_input.model_label == i], linestyle = 'dashed', label = "overall response (%s)" % i, color = colors[col])
        ax.set_title(
            "scope: comparing models & dataset: %s & target class: %s" % (
                datasets[0], classes[0]
            ),
            fontweight = 'bold'
        )
        ax.legend(
            loc = 'upper right',
            shadow = False,
            frameon = False
        )
    else: #compare_targetclasses
        for col, i in enumerate(classes):
            ax.plot(
                plot_input.ntile[plot_input.target_class == i],
                plot_input.cumlift[plot_input.target_class == i],
                label = i,
                color = colors[col]
            )
            #ax.plot(plot_input.ntile[plot_input.target_class == i], plot_input.cumlift_ref[plot_input.target_class == i], linestyle = 'dashed', label = "overall response (%s)" % i, color = colors[col])
        ax.set_title(
            "scope: comparing target classes & dataset: %s & model: %s" % (
                datasets[0], models[0]
            ),
            fontweight = 'bold'
        )
        ax.legend(
            loc = 'upper right',
            shadow = False,
            frameon = False
        )
    
    if highlight_ntile is not None:
        
        if highlight_ntile not in np.linspace(1, ntiles, num = ntiles).tolist():
            raise TypeError(
                'Invalid value for highlight_ntile parameter. '
                'It must be an int value between 1 and %d' % (ntiles)
            )
            
        if highlight_how not in ('plot','text','plot_text'):
            raise ValueError(
                'Invalid highlight_how value, '
                'it must be one of the following: plot, text or plot_text.'
            )
        
        else:
            text = ''
            if scope == "no_comparison":
                cumpct = (
                    plot_input.loc[
                        plot_input.ntile == highlight_ntile,
                        'cumlift'
                    ].tolist())
                plt.plot(
                    [1, highlight_ntile],
                    [cumpct[0]] * 2,
                    linestyle = '-.',
                    color = colors[0],
                    lw = 1.5
                )
                plt.plot(
                    [highlight_ntile] * 2 ,
                    [0] + [cumpct[0]],
                    linestyle = '-.',
                    color = colors[0],
                    lw = 1.5
                )
                xy = tuple([highlight_ntile] + [cumpct[0]])
                ax.plot(
                    xy[0],
                    xy[1],
                    ".r",
                    ms = 20,
                    color = colors[0]
                )
                ax.annotate(
                    str(int(cumpct[0] * 100)) + "%",
                    xy = xy, xytext = (-30, -30),
                    textcoords = 'offset points',
                    ha = 'center',
                    va = 'bottom',
                    color = 'black',
                    bbox = dict(
                        boxstyle = 'round, pad = 0.4',
                        alpha = 1,
                        fc = colors[0]
                    ),
                    arrowprops = dict(
                        arrowstyle = '->',
                        color = 'black')
                )
                text += (
                    'When we select %d' % int(
                        (float(highlight_ntile) / ntiles) * 100
                    ) + '%' + (
                        ' with the highest probability according to model %s in dataset %s, '
                        'this selection for target class %s is %s times than selecting '
                        'without a model.\n'
                    ) % (
                        models[0], datasets[0], classes[0], str(round(cumpct[0], 2))
                    )
                )
            elif scope == "compare_datasets":
                for col, i in enumerate(datasets):
                    cumpct = (
                        plot_input.loc[
                            plot_input.ntile == highlight_ntile,
                            ['dataset_label', 'cumlift']
                        ]
                    )
                    cumpct = cumpct.cumlift[cumpct.dataset_label == i].tolist()
                    plt.plot(
                        [1, highlight_ntile],
                        cumpct * 2,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    plt.plot(
                        [highlight_ntile] * 2,
                        [0] + cumpct,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    xy = tuple([highlight_ntile] + cumpct)
                    ax.plot(
                        xy[0],
                        xy[1],
                        ".r",
                        ms = 20,
                        color = colors[col]
                    )
                    ax.annotate(
                        str(int(cumpct[0] * 100)) + "%",
                        xy = xy, xytext = (-30, -30),
                        textcoords = 'offset points',
                        ha = 'center',
                        va = 'bottom',
                        color = 'black',
                        bbox = dict(
                            boxstyle = 'round, pad = 0.4',
                            fc = colors[col],
                            alpha = 0.8,
                        ),
                        arrowprops = dict(
                            arrowstyle = '->',
                            color = 'black')
                    )
                    text += (
                        'When we select %d' % int(
                            (float(highlight_ntile) / ntiles) * 100
                        ) + '%' + (
                            ' with the highest probability according to model %s in dataset %s, '
                            'this selection for target class %s is %s times than selecting '
                            'without a model.\n'
                        ) % (
                            models[0], datasets[col], classes[0], str(round(cumpct[0], 2))
                        )
                    )
            elif scope == "compare_models":
                for col, i in enumerate(models):
                    cumpct = (
                        plot_input.loc[
                            plot_input.ntile == highlight_ntile,
                            ['model_label', 'cumlift']
                        ]
                    )
                    cumpct = cumpct.cumlift[cumpct.model_label == i].tolist()
                    plt.plot(
                        [1, highlight_ntile],
                        cumpct * 2,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    plt.plot(
                        [highlight_ntile] * 2,
                        [0] + cumpct,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    xy = tuple([highlight_ntile] + cumpct)
                    ax.plot(
                        xy[0],
                        xy[1],
                        ".r",
                        ms = 20,
                        color = colors[col]
                    )
                    ax.annotate(
                        str(int(cumpct[0] * 100)) + "%",
                        xy = xy, xytext = (-30, -30),
                        textcoords = 'offset points',
                        ha = 'center',
                        va = 'bottom',
                        color = 'black',
                        bbox = dict(
                            boxstyle = 'round, pad = 0.4',
                            fc = colors[col],
                            alpha = 0.8,
                        ),
                        arrowprops = dict(
                            arrowstyle = '->',
                            color = 'black')
                    )
                    text += (
                        'When we select %d' % int(
                            (float(highlight_ntile) / ntiles) * 100
                        ) + '%' + (
                            ' with the highest probability according to model %s in dataset %s, '
                            'this selection for target class %s is %s times than selecting '
                            'without a model.\n'
                        ) % (
                            models[col], datasets[0], classes[0], str(round(cumpct[0], 2))
                        )
                    )
            else: # compare targetvalues
                for col, i in enumerate(classes):
                    cumpct = (
                        plot_input.loc[
                            plot_input.ntile == highlight_ntile,
                            ['target_class', 'cumlift']
                        ]
                    )
                    cumpct = cumpct.cumlift[cumpct.target_class == i].tolist()
                    plt.plot(
                        [1, highlight_ntile],
                        cumpct * 2,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    plt.plot(
                        [highlight_ntile] * 2,
                        [0] + cumpct,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    xy = tuple([highlight_ntile] + cumpct)
                    ax.plot(
                        xy[0],
                        xy[1],
                        ".r",
                        ms = 20,
                        color = colors[col]
                    )
                    ax.annotate(
                        str(int(cumpct[0] * 100)) + "%",
                        xy = xy, xytext = (-30, -30),
                        textcoords = 'offset points',
                        ha = 'center',
                        va = 'bottom',
                        color = 'black',
                        bbox = dict(
                            boxstyle = 'round, pad = 0.4',
                            fc = colors[col],
                            alpha = 0.8,
                        ),
                        arrowprops = dict(
                            arrowstyle = '->',
                            color = 'black')
                    )
                    text += (
                        'When we select %d' % int(
                            (float(highlight_ntile) / ntiles) * 100
                        ) + '%' + (
                            ' with the highest probability according to model %s in dataset %s, '
                            'this selection for target class %s is %s times than selecting '
                            'without a model.\n'
                        ) % (
                            models[0], datasets[0], classes[col], str(round(cumpct[0], 2))
                        )
                    )
            
            if highlight_how in ('text', 'plot_text'):
                print(text[:-1])
            if highlight_how in ('plot', 'plot_text'):
                fig.text(.15, -0.001, text[:-1], ha='left')
    
    if save_fig == True:
        if not save_fig_filename:
            location = '%s/Cumulative lift plot.png' % os.getcwd()
            plt.savefig(location, dpi = 300)
            print("The cumulative lift plot is saved in %s" % location)
        else:
            plt.savefig(save_fig_filename, dpi = 300)
            print("The cumulative lift plot is saved in %s" % save_fig_filename)
        plt.show()
        plt.gcf().clear()
    plt.show()
    return ax


def plot_cumgains(
    plot_input,
    save_fig = True,
    save_fig_filename = '',
    highlight_ntile = None,
    highlight_how = 'plot_text',
):
    """
    Plotting cumulative gains curve
    
    Parameters
    ----------
    plot_input : pandas.DataFrame
        The result from scope_modevalplot().

    save_fig : bool, default=True
        Save the plot.

    save_fig_filename : str, optional, default=''
        Specify the path and filetype to save the plot.
        If nothing specified, the plot will be saved as jpeg
        to the current working directory.

    highlight_ntile : int or None, optional, default=None
        Highlight the value of the response curve at a specified ntile value.
        
        .. versionchanged:: 0.3.9
            Default changed from False to None.
    
    highlight_how : str, optional, default='plot_text'
        Highlight_how specifies where information about the model performance is printed.
        It can be shown as text, on the plot or both.
    
    Returns
    -------
    It returns a matplotlib.axes._subplots.AxesSubplot object
    that can be transformed into the same plot with the .figure command.
    The plot is by default written to disk (save_fig = True).
    The location and filetype of the file depend on the save_fig_filename parameter.
    If the save_fig_filename parameter is empty (not specified),
    the plot will be written to the working directory as png. 
    Otherwise the location and file type is specified by the user.
        
    Raises
    ------
    TypeError: If ``highlight_ntile`` is not specified as an int.
    ValueError: If the wrong ``highlight_how`` value is specified.
    """
    models   = plot_input.model_label.unique().tolist()
    datasets = plot_input.dataset_label.unique().tolist()
    classes  = plot_input.target_class.unique().tolist()
    scope = plot_input.scope.unique()[0]
    ntiles = plot_input.ntile.nunique() - 1
    colors = (
        "#E41A1C",
        "#377EB8",
        "#4DAF4A",
        "#984EA3",
        "#FF7F00",
        "#FFFF33",
        "#A65628",
        "#F781BF",
        "#999999"
    )
    
    if ntiles == 10:
        description_label = 'decile'
    elif ntiles == 100:
        description_label = 'percentile'
    else:
        description_label = 'ntile'
    
    if ntiles <= 20:
        xlabper = 1
    elif ntiles <= 40:
        xlabper = 2
    else:
        xlabper = 5

    fig, ax = plt.subplots(figsize = (12,7))
    ax.set_xlabel(description_label)
    ax.set_ylabel("cumulative gains")
    plt.suptitle('Cumulative gains', fontsize = 16)
    ax.set_xticks(np.arange(0, ntiles + 1, xlabper))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    #ax.plot(list(range(0, ntiles + 1, 1)), np.linspace(0, 1, num = ntiles + 1).tolist(), linestyle = 'dashed', label = "minimal gains", color = 'grey')    
    ax.grid(True)
    ax.set_xlim([0, ntiles])
    ax.set_ylim([0,1])
    
    if scope == "no_comparison":
        ax.set_title(
            "model: %s & dataset: %s & target class: %s" % (
                models[0], datasets[0], classes[0]
            ),
            fontweight = 'bold'
        )
        ax.plot(
            plot_input.ntile,
            plot_input.cumgain,
            label = classes[0],
            color = colors[0]
        )
        ax.plot(
            plot_input.ntile,
            plot_input.gain_opt,
            linestyle = 'dashed',
            label = "optimal gains (%s)" % classes[0],
            color = colors[0],
            linewidth = 1.5
        )
        ax.legend(
            loc = 'lower right',
            shadow = False,
            frameon = False
        )
    elif scope == "compare_datasets":
        for col, i in enumerate(datasets):
            ax.plot(
                plot_input.ntile[plot_input.dataset_label == i],
                plot_input.cumgain[plot_input.dataset_label == i],
                label = i,
                color = colors[col]
            )
            ax.plot(
                plot_input.ntile[plot_input.dataset_label == i],
                plot_input.gain_opt[plot_input.dataset_label == i],
                linestyle = 'dashed',
                label = "optimal gains (%s)" % i,
                color = colors[col],
                linewidth = 1.5
            )
        ax.set_title(
            "scope: comparing datasets & model: %s & target class: %s" % (
                models[0], classes[0]
            ),
            fontweight = 'bold'
        )
        ax.legend(
            loc = 'lower right',
            shadow = False,
            frameon = False
        )
    elif scope == "compare_models":
        for col, i in enumerate(models):
            ax.plot(
                plot_input.ntile[plot_input.model_label == i],
                plot_input.cumgain[plot_input.model_label == i],
                label = i,
                color = colors[col]
            )
            ax.plot(
                plot_input.ntile[plot_input.model_label == i],
                plot_input.gain_opt[plot_input.model_label == i],
                linestyle = 'dashed',
                label = "optimal gains (%s)" % i,
                color = colors[col],
                linewidth = 1.5
            )
        ax.set_title(
            "scope: comparing models & dataset: %s & target class: %s" % (
                datasets[0], classes[0]
            ),
            fontweight = 'bold'
        )
        ax.legend(
            loc = 'lower right',
            shadow = False,
            frameon = False
        )
    else: #compare_targetclasses
        for col, i in enumerate(classes):
            ax.plot(
                plot_input.ntile[plot_input.target_class == i],
                plot_input.cumgain[plot_input.target_class == i],
                label = i,
                color = colors[col]
            )
            ax.plot(
                plot_input.ntile[plot_input.target_class == i],
                plot_input.gain_opt[plot_input.target_class == i],
                linestyle = 'dashed',
                label = "optimal gains (%s)" % i,
                color = colors[col],
                linewidth = 1.5
            )
        ax.set_title(
            "scope: comparing target classes & dataset: %s & model: %s" % (
                datasets[0], models[0]
            ),
            fontweight = 'bold'
        )
        ax.legend(
            loc = 'lower right',
            shadow = False,
            frameon = False
        )
    
    if highlight_ntile is not None:
        
        if highlight_ntile not in np.linspace(1, ntiles, num = ntiles).tolist():
            raise TypeError(
                'Invalid value for highlight_ntile parameter. '
                'It must be an int value between 1 and %d' % (ntiles)
            )
            
        if highlight_how not in ('plot','text','plot_text'):
            raise ValueError(
                'Invalid highlight_how value, '
                'it must be one of the following: plot, text or plot_text.'
            )
        
        else:
            text = ''
            if scope == "no_comparison":
                cumpct = (
                    plot_input.loc[
                        plot_input.ntile == highlight_ntile,
                        'cumgain'
                    ].tolist()
                )
                plt.plot(
                    [0, highlight_ntile],
                    [cumpct[0]] * 2,
                    linestyle = '-.',
                    color = colors[0],
                    lw = 1.5
                )
                plt.plot(
                    [highlight_ntile] * 2 ,
                    [0] + [cumpct[0]],
                    linestyle = '-.',
                    color = colors[0],
                    lw = 1.5
                )
                xy = tuple([highlight_ntile] + [cumpct[0]])
                ax.plot(
                    xy[0],
                    xy[1],
                    ".r",
                    ms = 20,
                    color = colors[0]
                )
                ax.annotate(
                    str(int(cumpct[0] * 100)) + "%",
                    xy = xy,
                    xytext = (-30, -30),
                    textcoords = 'offset points',
                    ha = 'center',
                    va = 'bottom',
                    color = 'black',
                    bbox = dict(
                        boxstyle = 'round, pad = 0.4',
                        alpha = 1,
                        fc = colors[0]
                    ),
                    arrowprops = dict(
                        arrowstyle = '->',
                        color = 'black')
                )
                text += (
                    'When we select %d' % int(
                        (float(highlight_ntile) / ntiles) * 100
                    ) + '%' + (
                        ' with the highest probability according to model %s, '
                        'this selection holds %d'
                    ) % (
                        models[0], int(cumpct[0] * 100)
                    ) + '%' + (
                        ' of all %s cases in dataset %s.\n'
                    )  % (
                        classes[0], datasets[0]
                    )
                )
            elif scope == "compare_datasets":
                for col, i in enumerate(datasets):
                    cumpct = (
                        plot_input.loc[
                            plot_input.ntile == highlight_ntile,
                            ['dataset_label', 'cumgain']
                        ]
                    )
                    cumpct = cumpct.cumgain[cumpct.dataset_label == i].tolist()
                    plt.plot(
                        [0, highlight_ntile],
                        cumpct * 2,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    plt.plot(
                        [highlight_ntile] * 2,
                        [0] + cumpct,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    xy = tuple([highlight_ntile] + cumpct)
                    ax.plot(
                        xy[0],
                        xy[1],
                        ".r",
                        ms = 20,
                        color = colors[col]
                    )
                    ax.annotate(
                        str(int(cumpct[0] * 100)) + "%",
                        xy = xy,
                        xytext = (-30, -30),
                        textcoords = 'offset points', ha = 'center', va = 'bottom', color = 'black',
                        bbox = dict(
                            boxstyle = 'round, pad = 0.4',
                            fc = colors[col],
                            alpha = 0.8,
                        ),
                        arrowprops = dict(
                            arrowstyle = '->',
                            color = 'black'
                        )
                    )
                    text += (
                        'When we select %d' % int(
                            (float(highlight_ntile) / ntiles) * 100
                        ) + '%' + (
                            ' with the highest probability according to model %s, '
                            'this selection holds %d'
                        ) % (
                            models[0], int(cumpct[0] * 100)
                        ) + '%' + (
                            ' of all %s cases in dataset %s.\n'
                        )  % (
                            classes[0], datasets[col]
                        )
                    )
            elif scope == "compare_models":
                for col, i in enumerate(models):
                    cumpct = (
                        plot_input.loc[
                            plot_input.ntile == highlight_ntile,
                            ['model_label', 'cumgain']
                        ]
                    )
                    cumpct = cumpct.cumgain[cumpct.model_label == i].tolist()
                    plt.plot(
                        [0, highlight_ntile],
                        cumpct * 2,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    plt.plot(
                        [highlight_ntile] * 2,
                        [0] + cumpct,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    xy = tuple([highlight_ntile] + cumpct)
                    ax.plot(
                        xy[0],
                        xy[1],
                        ".r",
                        ms = 20,
                        color = colors[col]
                    )
                    ax.annotate(
                        str(int(cumpct[0] * 100)) + "%",
                        xy = xy,
                        xytext = (-30, -30),
                        textcoords = 'offset points', ha = 'center', va = 'bottom', color = 'black',
                        bbox = dict(
                            boxstyle = 'round, pad = 0.4',
                            fc = colors[col],
                            alpha = 0.8,
                        ),
                        arrowprops = dict(
                            arrowstyle = '->',
                            color = 'black'
                        )
                    )
                    text += (
                        'When we select %d' % int(
                            (float(highlight_ntile) / ntiles) * 100
                        ) + '%' + (
                            ' with the highest probability according to model %s, '
                            'this selection holds %d'
                        ) % (
                            models[col], int(cumpct[0] * 100)
                        ) + '%' + (
                            ' of all %s cases in dataset %s.\n'
                        )  % (
                            classes[0], datasets[0]
                        )
                    )
            else: # compare targetvalues
                for col, i in enumerate(classes):
                    cumpct = (
                        plot_input.loc[
                            plot_input.ntile == highlight_ntile,
                            ['target_class', 'cumgain']
                        ]
                    )
                    cumpct = cumpct.cumgain[cumpct.target_class == i].tolist()
                    plt.plot(
                        [0, highlight_ntile],
                        cumpct * 2,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    plt.plot(
                        [highlight_ntile] * 2,
                        [0] + cumpct,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    xy = tuple([highlight_ntile] + cumpct)
                    ax.plot(
                        xy[0],
                        xy[1],
                        ".r",
                        ms = 20,
                        color = colors[col]
                    )
                    ax.annotate(
                        str(int(cumpct[0] * 100)) + "%",
                        xy = xy,
                        xytext = (-30, -30), 
                        textcoords = 'offset points',
                        ha = 'center',
                        va = 'bottom',
                        color = 'black',
                        bbox = dict(
                            boxstyle = 'round, pad = 0.4',
                            fc = colors[col],
                            alpha = 0.8,
                        ),
                        arrowprops = dict(
                            arrowstyle = '->',
                            color = 'black'
                        )
                    )
                    text += (
                        'When we select %d' % int(
                            (float(highlight_ntile) / ntiles) * 100
                        ) + '%' + (
                            ' with the highest probability according to model %s, '
                            'this selection holds %d'
                        ) % (
                            models[0], int(cumpct[0] * 100)
                        ) + '%' + (
                            ' of all %s cases in dataset %s.\n'
                        )  % (
                            classes[col], datasets[0]
                        )
                    )
            
            if highlight_how in ('text', 'plot_text'):
                print(text[:-1])
            if highlight_how in ('plot', 'plot_text'):
                fig.text(.15, -0.001, text[:-1], ha='left')
    
    if save_fig == True:
        if not save_fig_filename:
            location = '%s/Cumulative gains plot.png' % os.getcwd()
            plt.savefig(location, dpi = 300)
            print("The cumulative gains plot is saved in %s" % location)
        else:
            plt.savefig(save_fig_filename, dpi = 300)
            print("The cumulative gains plot is saved in %s" % save_fig_filename)
        plt.show()
        plt.gcf().clear()
    plt.show()
    return ax


def plot_all(
    plot_input,
    save_fig = True,
    save_fig_filename = '',
):
    """
    Plotting cumulative gains curve

    Parameters
    ----------
    plot_input : pandas.DataFrame
        The result from scope_modevalplot().

    save_fig : bool, default=True
        Save the plot.

    save_fig_filename : str, optional, default=''
        Specify the path and filetype to save the plot.
        If nothing specified, the plot will be saved as jpeg
        to the current working directory.

    Returns
    -------
    It returns a matplotlib.axes._subplots.AxesSubplot object
    that can be transformed into the same plot with the .figure command.
    The plot is by default written to disk (save_fig = True).
    The location and filetype of the file depend on the save_fig_filename parameter.
    If the save_fig_filename parameter is empty (not specified),
    the plot will be written to the working directory as png. 
    Otherwise the location and file type is specified by the user.
    """
    models   = plot_input.model_label.unique().tolist()
    datasets = plot_input.dataset_label.unique().tolist()
    classes  = plot_input.target_class.unique().tolist()
    scope = plot_input.scope.unique()[0]
    ntiles = plot_input.ntile.nunique() - 1
    colors = (
        "#E41A1C",
        "#377EB8",
        "#4DAF4A",
        "#984EA3",
        "#FF7F00",
        "#FFFF33",
        "#A65628",
        "#F781BF",
        "#999999"
    )

    if ntiles == 10:
        description_label = 'decile'
    elif ntiles == 100:
        description_label = 'percentile'
    else:
        description_label = 'ntile'
    
    if ntiles <= 20:
        xlabper = 1
    elif ntiles <= 40:
        xlabper = 2
    else:
        xlabper = 5

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, sharex = False, sharey = False, figsize = (15,10)
    )
    ax1.set_title('Cumulative gains', fontweight='bold')
    ax1.set_ylabel('cumulative gains')
    #ax1.set_xlabel('decile')
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.set_ylim(0, 1)
    ax1.set_xlim(0, ntiles)
    ax1.set_xticks(np.arange(0, ntiles + 1, xlabper))
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.grid(True)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.plot(
        list(range(0, ntiles + 1, 1)),
        np.linspace(0, 1, num = ntiles + 1).tolist(),
        linestyle = 'dashed',
        label = "minimal gains",
        color = 'grey'
    )

    ax2.set_title('Cumulative lift', fontweight='bold')
    ax2.set_ylabel('cumulative lift')
    #ax2.set_xlabel('decile')
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.set_xticks(np.arange(0, ntiles + 1, xlabper))
    ax2.set_xlim(1, ntiles)
    ax2.set_ylim([0, max(plot_input.cumlift)])
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.grid(True)
    ax2.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')
    ax2.plot(
        list(range(1, ntiles + 1, 1)),
        [1] * ntiles,
        linestyle = 'dashed',
        label = "no lift",
        color = 'grey'
    )

    ax3.set_title('Response', fontweight='bold')
    ax3.set_ylabel('response')
    ax3.set_xlabel(description_label)
    ax3.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax3.set_xticks(np.arange(0, ntiles + 1, xlabper))
    ax3.set_xlim(1, ntiles)
    ax3.set_ylim([0, round((max(plot_input.pct)+0.05)*100,-1)/100])
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.grid(True)
    ax3.yaxis.set_ticks_position('left')
    ax3.xaxis.set_ticks_position('bottom')
    
    ax4.set_title('Cumulative response', fontweight='bold')
    ax4.set_ylabel('cumulative response')
    ax4.set_xlabel(description_label)
    ax4.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax4.set_xticks(np.arange(0, ntiles + 1, xlabper))
    ax4.set_xlim(1, ntiles)
    ax4.set_ylim([0, round((max(plot_input.cumpct)+0.05)*100,-1)/100])
    ax4.spines['right'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax4.grid(True)
    ax4.yaxis.set_ticks_position('left')
    ax4.xaxis.set_ticks_position('bottom')
    
    if scope == "no_comparison":
        title = "model: %s & dataset: %s & target class: %s" % (
            models[0], datasets[0], classes[0]
        )
        ax1.plot(
            plot_input.ntile,
            plot_input.cumgain,
            label = classes[0],
            color = colors[0]
        )
        ax1.plot(
            plot_input.ntile,
            plot_input.gain_opt,
            linestyle = 'dashed',
            label = "optimal gains (%s)" % classes[0],
            color = colors[0]
        )
        ax1.legend(
            loc = 'lower right',
            shadow = False,
            frameon = False
        )
        ax2.plot(
            plot_input.ntile,
            plot_input.cumlift,
            label = classes[0],
            color = colors[0]
        )
        ax2.legend(
            loc = 'upper right',
            shadow = False,
            frameon = False
        )
        ax3.plot(
            plot_input.ntile,
            plot_input.pct,
            label = classes[0],
            color = colors[0]
        )
        ax3.plot(
            plot_input.ntile,
            plot_input.pct_ref,
            linestyle = 'dashed',
            label = "overall response (%s)" % classes[0],
            color = colors[0]
        )
        ax3.legend(
            loc = 'upper right',
            shadow = False,
            frameon = False
        )
        ax4.plot(
            plot_input.ntile,
            plot_input.cumpct,
            label = classes[0],
            color = colors[0]
        )
        ax4.plot(
            plot_input.ntile,
            plot_input.pct_ref,
            linestyle = 'dashed',
            label = "overall response (%s)" % classes[0],
            color = colors[0]
        )
        ax4.legend(
            loc = 'upper right',
            shadow = False,
            frameon = False
        )
    elif scope == "compare_datasets":
        title = "scope: comparing datasets & model: %s & target class: %s" % (
            models[0], classes[0]
        )
        for col, i in enumerate(datasets):
            ax1.plot(
                plot_input.ntile[plot_input.dataset_label == i],
                plot_input.cumgain[plot_input.dataset_label == i],
                label = i,
                color = colors[col]
            )
            ax1.plot(
                plot_input.ntile[plot_input.dataset_label == i],
                plot_input.gain_opt[plot_input.dataset_label == i],
                linestyle = 'dashed',
                label = "optimal gains (%s)" % i,
                color = colors[col]
            )
            ax2.plot(
                plot_input.ntile[plot_input.dataset_label == i],
                plot_input.cumlift[plot_input.dataset_label == i],
                label = i,
                color = colors[col]
            )
            ax3.plot(
                plot_input.ntile[plot_input.dataset_label == i],
                plot_input.pct[plot_input.dataset_label == i],
                label = i,
                color = colors[col]
            )
            ax3.plot(
                plot_input.ntile[plot_input.dataset_label == i],
                plot_input.pct_ref[plot_input.dataset_label == i],
                linestyle = 'dashed',
                label = "overall response (%s)" % i,
                color = colors[col]
            )
            ax4.plot(
                plot_input.ntile[plot_input.dataset_label == i],
                plot_input.cumpct[plot_input.dataset_label == i],
                label = i,
                color = colors[col]
            )
            ax4.plot(
                plot_input.ntile[plot_input.dataset_label == i],
                plot_input.pct_ref[plot_input.dataset_label == i],
                linestyle = 'dashed',
                label = "overall response (%s)" % i,
                color = colors[col]
            )
        ax1.legend(loc = 'lower right', shadow = False, frameon = False)
        ax2.legend(loc = 'upper right', shadow = False, frameon = False)
        ax3.legend(loc = 'upper right', shadow = False, frameon = False)
        ax4.legend(loc = 'upper right', shadow = False, frameon = False)        
    elif scope == "compare_models":
        title = "scope: comparing models & dataset: %s & target class: %s" % (
            datasets[0], classes[0]
        )
        for col, i in enumerate(models):
            ax1.plot(
                plot_input.ntile[plot_input.model_label == i],
                plot_input.cumgain[plot_input.model_label == i],
                label = i,
                color = colors[col]
            )
            ax1.plot(
                plot_input.ntile[plot_input.model_label == i],
                plot_input.gain_opt[plot_input.model_label == i],
                linestyle = 'dashed',
                label = "optimal gains (%s)" % i,
                color = colors[col]
            )
            ax2.plot(
                plot_input.ntile[plot_input.model_label == i],
                plot_input.cumlift[plot_input.model_label == i],
                label = i,
                color = colors[col]
            )
            ax3.plot(
                plot_input.ntile[plot_input.model_label == i],
                plot_input.pct[plot_input.model_label == i],
                label = i,
                color = colors[col]
            )
            ax3.plot(
                plot_input.ntile[plot_input.model_label == i],
                plot_input.pct_ref[plot_input.model_label == i],
                linestyle = 'dashed',
                label = "overall response (%s)" % i,
                color = colors[col]
            )
            ax4.plot(
                plot_input.ntile[plot_input.model_label == i],
                plot_input.cumpct[plot_input.model_label == i],
                label = i, color = colors[col]
            )
            ax4.plot(
                plot_input.ntile[plot_input.model_label == i],
                plot_input.pct_ref[plot_input.model_label == i],
                linestyle = 'dashed',
                label = "overall response (%s)" % i,
                color = colors[col]
            )            
        ax1.legend(loc = 'lower right', shadow = False, frameon = False)
        ax2.legend(loc = 'upper right', shadow = False, frameon = False)
        ax3.legend(loc = 'upper right', shadow = False, frameon = False)
        ax4.legend(loc = 'upper right', shadow = False, frameon = False)
    else: #compare_targetclasses
        title = "scope: comparing target classes & dataset: %s & model: %s" % (
            datasets[0], models[0]
        )
        for col, i in enumerate(classes):
            ax1.plot(
                plot_input.ntile[plot_input.target_class == i],
                plot_input.cumgain[plot_input.target_class == i],
                label = i,
                color = colors[col]
            )
            ax1.plot(
                plot_input.ntile[plot_input.target_class == i],
                plot_input.gain_opt[plot_input.target_class == i],
                linestyle = 'dashed',
                label = "optimal gains (%s)" % i,
                color = colors[col]
            )
            ax2.plot(
                plot_input.ntile[plot_input.target_class == i],
                plot_input.cumlift[plot_input.target_class == i],
                label = i,
                color = colors[col]
            )
            ax3.plot(
                plot_input.ntile[plot_input.target_class == i],
                plot_input.pct[plot_input.target_class == i],
                label = i,
                color = colors[col]
            )
            ax3.plot(
                plot_input.ntile[plot_input.target_class == i],
                plot_input.pct_ref[plot_input.target_class == i],
                linestyle = 'dashed',
                label = "overall response (%s)" % i,
                color = colors[col]
            )
            ax4.plot(
                plot_input.ntile[plot_input.target_class == i],
                plot_input.cumpct[plot_input.target_class == i],
                label = i,
                color = colors[col])
            ax4.plot(
                plot_input.ntile[plot_input.target_class == i],
                plot_input.pct_ref[plot_input.target_class == i],
                linestyle = 'dashed',
                label = "overall response (%s)" % i,
                color = colors[col]
            )
        ax1.legend(loc = 'lower right', shadow = False, frameon = False)
        ax2.legend(loc = 'upper right', shadow = False, frameon = False)
        ax3.legend(loc = 'upper right', shadow = False, frameon = False)
        ax4.legend(loc = 'upper right', shadow = False, frameon = False)
    plt.suptitle(title, fontsize = 16)
    if save_fig == True:
        if not save_fig_filename:
            location = '%s/Plot all.png' % os.getcwd()
            plt.savefig(location, dpi = 300)
            print("The plot all plot is saved in %s" % location)
        else:
            plt.savefig(save_fig_filename, dpi = 300)
            print("The plot all plot is saved in %s" % save_fig_filename)
        plt.show()
        plt.gcf().clear()
    plt.show()
    return ax1

##########################################################################
## modelplotpy financial
##########################################################################

def plot_costsrevs(
    plot_input,
    fixed_costs,
    variable_costs_per_unit,
    profit_per_unit,
    save_fig = True,
    save_fig_filename = '',
    highlight_ntile = None,
    highlight_how = 'plot_text',
):
    """
    Plotting costs / revenue curve
    
    Parameters
    ----------
    plot_input : pandas.DataFrame
        The result from scope_modevalplot().
    
    fixed_costs : int / float
        Specifying the fixed costs related to a selection based on the model.
        These costs are constant and do not vary with selection size (ntiles).
    
    variable_costs_per_unit : int / float
        Specifying the variable costs per selected unit for a selection based on the model.
        These costs vary with selection size (ntiles).
        
    profit_per_unit : int / float
        Specifying the profit per unit in case the selected unit converts / responds positively.

    save_fig : bool, default=True
        Save the plot.
        
    save_fig_filename : str, optional, default=''
        Specify the path and filetype to save the plot.
        If nothing specified, the plot will be saved as jpeg
        to the current working directory.
        
    highlight_ntile : int or None, optional, default=None
        Highlight the value of the response curve at a specified ntile value.
        
        .. versionchanged:: 0.3.9
            Default changed from False to None.
        
    highlight_how : str, optional, default='plot_text'
        Highlight_how specifies where information about the model performance is printed.
        It can be shown as text, on the plot or both.
    
    Returns
    -------
    It returns a matplotlib.axes._subplots.AxesSubplot object
    that can be transformed into the same plot with the .figure command.
    The plot is by default written to disk (save_fig = True).
    The location and filetype of the file depend on the save_fig_filename parameter.
    If the save_fig_filename parameter is empty (not specified),
    the plot will be written to the working directory as png. 
    Otherwise the location and file type is specified by the user.
        
    Raises
    ------
    TypeError: If ``highlight_ntile`` is not specified as an int.
    ValueError: If the wrong ``highlight_how`` value is specified.
    """
    
    models   = plot_input.model_label.unique().tolist()
    datasets = plot_input.dataset_label.unique().tolist()
    classes  = plot_input.target_class.unique().tolist()
    scope = plot_input.scope.unique()[0]
    ntiles = plot_input.ntile.nunique() - 1
    colors = (
        "#E41A1C",
        "#377EB8",
        "#4DAF4A",
        "#984EA3",
        "#FF7F00",
        "#FFFF33",
        "#A65628",
        "#F781BF",
        "#999999"
    )
    plot_input['variable_costs'] = variable_costs_per_unit * plot_input.cumtot
    plot_input['investments'] = fixed_costs + plot_input.variable_costs 
    plot_input['revenues'] = profit_per_unit * plot_input.cumpos
    
    if ntiles == 10:
        description_label = 'decile'
    elif ntiles == 100:
        description_label = 'percentile'
    else:
        description_label = 'ntile'
    
    if ntiles <= 20:
        xlabper = 1
    elif ntiles <= 40:
        xlabper = 2
    else:
        xlabper = 5
    
    fig, ax = plt.subplots(figsize = (12,7))
    ax.set_xlabel(description_label)
    ax.set_ylabel("costs / revenue")
    plt.suptitle('Costs / Revenues', fontsize = 16)
    #ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xticks(np.arange(0, ntiles + 1, xlabper))
    #ax.set_xticks(np.arange(1, ntiles + 1, 1))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.grid(True)
    ax.set_xlim([1, ntiles])
    #ax.set_ylim([0, 1])
    
    if scope == "no_comparison":
        ax.set_title(
            "model: %s & dataset: %s & target class: %s" % (
                models[0], datasets[0], classes[0]
            ),
            fontweight = 'bold'
        )
        ax.plot(
            plot_input.ntile,
            plot_input.revenues,
            label = classes[0],
            color = colors[0]
        )
        ax.plot(
            plot_input.ntile,
            plot_input.investments,
            linestyle = 'dashed',
            label = "total costs",
            color = colors[0]
        )
        ax.legend(
            loc = 'lower right',
            shadow = False,
            frameon = False
        )
    elif scope == "compare_datasets":
        for col, i in enumerate(datasets):
            ax.plot(
                plot_input.ntile[plot_input.dataset_label == i],
                plot_input.revenues[plot_input.dataset_label == i],
                label = i,
                color = colors[col]
            )
            ax.plot(
                plot_input.ntile[plot_input.dataset_label == i],
                plot_input.investments[plot_input.dataset_label == i],
                linestyle = 'dashed',
                label = "total costs (%s)" % i,
                color = colors[col]
            )
        ax.set_title(
            "scope: comparing datasets & model: %s & target class: %s" % (
                models[0], classes[0]
            ),
            fontweight = 'bold'
        )
        ax.legend(
            loc = 'upper right',
            shadow = False,
            frameon = False
        )
    elif scope == "compare_models":
        ax.plot(
            list(range(0, ntiles + 1, 1)),
            fixed_costs + variable_costs_per_unit * plot_input.cumtot.unique(),
            linestyle = 'dashed',
            label = "total costs",
            color = 'grey'
        )
        for col, i in enumerate(models):
            ax.plot(
                plot_input.ntile[plot_input.model_label == i],
                plot_input.revenues[plot_input.model_label == i],
                label = "revenues (%s)" % i,
                color = colors[col]
            )
        ax.legend(
            loc = 'lower right',
            shadow = False,
            frameon = False
        )
        ax.set_title(
            "scope: comparing models & dataset: %s & target class: %s" % (
                datasets[0], classes[0]
            ),
            fontweight = 'bold'
        )
        ax.legend(
            loc = 'lower right',
            shadow = False,
            frameon = False
        )
    else: #compare_targetclasses
        for col, i in enumerate(classes):
            ax.plot(
                plot_input.ntile[plot_input.target_class == i],
                plot_input.revenues[plot_input.target_class == i],
                label = i,
                color = colors[col]
            )
            ax.plot(
                plot_input.ntile[plot_input.target_class == i],
                plot_input.investments[plot_input.target_class == i],
                linestyle = 'dashed',
                label = "total costs (%s)" % i,
                color = colors[col]
            )
        ax.set_title(
            "Scope: comparing target classes & dataset: %s & model: %s" % (
                datasets[0], models[0]
            ),
            fontweight = 'bold'
        )
        ax.legend(
            loc = 'lower right',
            shadow = False,
            frameon = False
        )
    
    if highlight_ntile is not None:
        
        if highlight_ntile not in np.linspace(1, ntiles, num = ntiles).tolist():
            raise TypeError(
                'Invalid value for highlight_ntile parameter. '
                'It must be an int value between 1 and %d' % (ntiles)
            )
            
        if highlight_how not in ('plot','text','plot_text'):
            raise ValueError(
                'Invalid highlight_how value, '
                'it must be one of the following: plot, text or plot_text.'
            )
        
        else:
            text = ''
            if scope == "no_comparison":
                cumpct = (
                    plot_input.loc[
                        plot_input.ntile == highlight_ntile,
                        'revenues'
                    ].tolist())
                plt.plot(
                    [1, highlight_ntile],
                    [cumpct[0]] * 2,
                    linestyle = '-.',
                    color = colors[0],
                    lw = 1.5
                )
                plt.plot(
                    [highlight_ntile] * 2 ,
                    [0] + [cumpct[0]],
                    linestyle = '-.',
                    color = colors[0],
                    lw = 1.5
                )
                xy = tuple([highlight_ntile] + [cumpct[0]])
                ax.plot(
                    xy[0],
                    xy[1],
                    ".r",
                    ms = 20,
                    color = colors[0]
                )
                ax.annotate(
                    "€" + str(int(cumpct[0])),
                    xy = xy,
                    xytext = (-30, -30),
                    textcoords='offset points',
                    ha = 'center',
                    va = 'bottom',
                    color = 'black',
                    bbox=dict(
                        boxstyle='round, pad = 0.4',
                        alpha = 1,
                        fc = colors[0]
                    ), #fc = 'yellow', alpha = 0.3),
                    arrowprops = dict(
                        arrowstyle = '->',
                        color = 'black'
                    )
                )
                text += (
                    (
                        'When we select %s 1 until %d from model %s in dataset '
                        '%s the percentage of %s cases in the revenue is %d'
                    ) % (
                        description_label,
                        highlight_ntile,
                        models[0],
                        datasets[0],
                        classes[0],
                        int(cumpct[0])
                    ) + '.\n'
                )
            elif scope == "compare_datasets":
                for col, i in enumerate(datasets):
                    cumpct = (
                        plot_input.loc[
                            plot_input.ntile == highlight_ntile,
                            ['dataset_label', 'revenues']
                        ]
                    )
                    cumpct = cumpct.revenues[cumpct.dataset_label == i].tolist()
                    plt.plot(
                        [1, highlight_ntile],
                        cumpct * 2,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    plt.plot(
                        [highlight_ntile] * 2,
                        [0] + cumpct,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    xy = tuple([highlight_ntile] + cumpct)
                    ax.plot(
                        xy[0],
                        xy[1],
                        ".r",
                        ms = 20,
                        color = colors[col]
                    )
                    ax.annotate(
                        "€" + str(int(cumpct[0])),
                        xy = xy,
                        xytext = (-30, -30),
                        textcoords='offset points',
                        ha = 'center',
                        va = 'bottom',
                        color = 'black',
                        bbox=dict(
                            boxstyle='round, pad = 0.4',
                            fc = colors[col],
                            alpha = 0.8,
                        ), #fc = 'yellow', alpha = 0.3),
                        arrowprops = dict(
                            arrowstyle = '->',
                            color = 'black'
                        )
                    )
                    text += (
                        (
                            'When we select %s 1 until %d from model %s in dataset '
                            '%s the percentage of %s cases in the revenue is %d'
                        ) % (
                            description_label,
                            highlight_ntile,
                            models[0],
                            datasets[col],
                            classes[0],
                            int(cumpct[0])
                        ) + '.\n'
                    )
            elif scope == "compare_models":
                for col, i in enumerate(models):
                    cumpct = (
                        plot_input.loc[
                            plot_input.ntile == highlight_ntile,
                            ['model_label', 'revenues']
                        ]
                    )
                    cumpct = cumpct.revenues[cumpct.model_label == i].tolist()
                    plt.plot(
                        [1, highlight_ntile],
                        cumpct * 2,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    plt.plot(
                        [highlight_ntile] * 2,
                        [0] + cumpct,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    xy = tuple([highlight_ntile] + cumpct)
                    ax.plot(
                        xy[0],
                        xy[1],
                        ".r",
                        ms = 20,
                        color = colors[col]
                    )
                    ax.annotate(
                        "€" + str(int(cumpct[0])),
                        xy = xy,
                        xytext = (-30, -30),
                        textcoords='offset points',
                        ha = 'center',
                        va = 'bottom',
                        color = 'black',
                        bbox=dict(
                            boxstyle='round, pad = 0.4',
                            fc = colors[col],
                            alpha = 0.8,
                        ), #fc = 'yellow', alpha = 0.3),
                        arrowprops = dict(
                            arrowstyle = '->',
                            color = 'black'
                        )
                    )
                    text += (
                        (
                            'When we select %s 1 until %d from model %s in dataset '
                            '%s the percentage of %s cases in the revenue is %d'
                        ) % (
                            description_label,
                            highlight_ntile,
                            models[col],
                            datasets[0],
                            classes[0],
                            int(cumpct[0])
                        ) + '.\n'
                    )
            else: # compare targetvalues
                for col, i in enumerate(classes):
                    cumpct = (
                        plot_input.loc[
                            plot_input.ntile == highlight_ntile,
                            ['target_class', 'revenues']
                        ]
                    )
                    cumpct = cumpct.revenues[cumpct.target_class == i].tolist()
                    plt.plot(
                        [1, highlight_ntile],
                        cumpct * 2,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    plt.plot(
                        [highlight_ntile] * 2,
                        [0] + cumpct,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    xy = tuple([highlight_ntile] + cumpct)
                    ax.plot(
                        xy[0],
                        xy[1],
                        ".r",
                        ms = 20,
                        color = colors[col]
                    )
                    ax.annotate(
                        "€" + str(int(cumpct[0])),
                        xy = xy,
                        xytext = (-30, -30),
                        textcoords='offset points',
                        ha = 'center',
                        va = 'bottom',
                        color = 'black',
                        bbox=dict(
                            boxstyle='round, pad = 0.4',
                            fc = colors[col],
                            alpha = 0.8,
                        ), #fc = 'yellow', alpha = 0.3),
                        arrowprops = dict(
                            arrowstyle = '->',
                            color = 'black'
                        )
                    )
                    text += (
                        (
                            'When we select %s 1 until %d from model %s in dataset '
                            '%s the percentage of %s cases in the revenue is %d'
                        ) % (
                            description_label,
                            highlight_ntile,
                            models[0],
                            datasets[0],
                            classes[col],
                            int(cumpct[0])
                        ) + '.\n'
                    )
            if highlight_how in ('text', 'plot_text'):
                print(text[:-1])
            if highlight_how in ('plot', 'plot_text'):
                fig.text(.15, -0.001, text[:-1], ha='left')
    
    if save_fig == True:
        if not save_fig_filename:
            location = '%s/Costs Revenues plot.png' % os.getcwd()
            plt.savefig(location, dpi = 300)
            print("The costs / revenues plot is saved in %s" % location)
        else:
            plt.savefig(save_fig_filename, dpi = 300)
            print("The costs / revenues plot is saved in %s" % save_fig_filename)
        plt.show()
        plt.gcf().clear()
    plt.show()
    return ax


def plot_profit(
    plot_input,
    fixed_costs,
    variable_costs_per_unit,
    profit_per_unit,
    save_fig = True,
    save_fig_filename = '',
    highlight_ntile = None,
    highlight_how = 'plot_text',
):
    """
    Plotting profit curve
    
    Parameters
    ----------
    plot_input : pandas.DataFrame
        The result from scope_modevalplot().
    
    fixed_costs : int / float
        Specifying the fixed costs related to a selection based on the model.
        These costs are constant and do not vary with selection size (ntiles).
    
    variable_costs_per_unit : int / float
        Specifying the variable costs per selected unit for a selection based on the model.
        These costs vary with selection size (ntiles).
        
    profit_per_unit : int / float
        Specifying the profit per unit in case the selected unit converts / responds positively.
    
    save_fig : bool, default=True
        Save the plot.
        
    save_fig_filename : str, optional, default=''
        Specify the path and filetype to save the plot.
        If nothing specified, the plot will be saved as jpeg to the current working directory.
        
    highlight_ntile : int or None, optional, default=None
        Highlight the value of the response curve at a specified ntile value.
        
        .. versionchanged:: 0.3.9
            Default changed from False to None.
        
    highlight_how : str, optional, default='plot_text'
        Highlight_how specifies where information about the model performance is printed.
        It can be shown as text, on the plot or both.
    
    Returns
    -------
    It returns a matplotlib.axes._subplots.AxesSubplot object
    that can be transformed into the same plot with the .figure command.
    The plot is by default written to disk (save_fig = True).
    The location and filetype of the file depend on the save_fig_filename parameter.
    If the save_fig_filename parameter is empty (not specified),
    the plot will be written to the working directory as png. 
    Otherwise the location and file type is specified by the user.
        
    Raises
    ------
    TypeError: If ``highlight_ntile`` is not specified as an int.
    ValueError: If the wrong ``highlight_how`` value is specified.
    """
    
    models   = plot_input.model_label.unique().tolist()
    datasets = plot_input.dataset_label.unique().tolist()
    classes  = plot_input.target_class.unique().tolist()
    scope = plot_input.scope.unique()[0]
    ntiles = plot_input.ntile.nunique() - 1
    colors = (
        "#E41A1C",
        "#377EB8",
        "#4DAF4A",
        "#984EA3",
        "#FF7F00",
        "#FFFF33",
        "#A65628",
        "#F781BF",
        "#999999"
    )
    
    plot_input['variable_costs'] = variable_costs_per_unit * plot_input.cumtot
    plot_input['investments'] = fixed_costs + plot_input.variable_costs 
    plot_input['revenues'] = profit_per_unit * plot_input.cumpos
    plot_input['profit'] = plot_input.revenues - plot_input.investments
    
    if ntiles == 10:
        description_label = 'decile'
    elif ntiles == 100:
        description_label = 'percentile'
    else:
        description_label = 'ntile'
    
    if ntiles <= 20:
        xlabper = 1
    elif ntiles <= 40:
        xlabper = 2
    else:
        xlabper = 5
    
    fig, ax = plt.subplots(figsize = (12,7))
    ax.set_xlabel(description_label)
    ax.set_ylabel("profit")
    plt.suptitle('Profit', fontsize = 16)
    #ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xticks(np.arange(0, ntiles + 1, xlabper))
    #ax.set_xticks(np.arange(1, ntiles + 1, 1))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.grid(True)
    ax.set_xlim([1, ntiles])
    ax.plot(
        list(range(1, ntiles + 1, 1)),
        [0] * ntiles,
        linestyle = 'dashed',
        label = "break even",
        color = 'grey'
    )

    if scope == "no_comparison":
        #ax.plot(list(range(0, ntiles + 1, 1)), fixed_costs + variable_costs_per_unit * plot_input.cumtot.unique(), linestyle = 'dashed', label = "total costs", color = 'grey')
        ax.set_title(
            "model: %s & dataset: %s & target class: %s" % (
                models[0], datasets[0], classes[0]
            ),
            fontweight = 'bold'
        )
        ax.plot(
            plot_input.ntile,
            plot_input.profit,
            label = classes[0],
            color = colors[0]
        )
        #ax.plot(plot_input.ntile, plot_input.cumcosts, linestyle = 'dashed', label = "total costs", color = colors[0])
        ax.legend(
            loc = 'lower right',
            shadow = False,
            frameon = False
        )
    elif scope == "compare_datasets":
        for col, i in enumerate(datasets):
            ax.plot(
                plot_input.ntile[plot_input.dataset_label == i],
                plot_input.profit[plot_input.dataset_label == i],
                label = i,
                color = colors[col]
            )
            #ax.plot(plot_input.ntile[plot_input.dataset_label == i], plot_input.cumcosts[plot_input.dataset_label == i], linestyle = 'dashed', label = "total costs (%s)" % i, color = colors[col])
        ax.set_title(
            "scope: comparing datasets & model: %s & target class: %s" % (
                models[0], classes[0]
            ),
            fontweight = 'bold'
        )
        ax.legend(
            loc = 'upper right',
            shadow = False,
            frameon = False
        )
    elif scope == "compare_models":
        for col, i in enumerate(models):
            ax.plot(
                plot_input.ntile[plot_input.model_label == i],
                plot_input.profit[plot_input.model_label == i],
                label = "profit (%s)" % i,
                color = colors[col]
            )
        ax.legend(
            loc = 'lower right',
            shadow = False,
            frameon = False
        )
        ax.set_title(
            "scope: comparing models & dataset: %s & target class: %s" % (
                datasets[0], classes[0]
            ),
            fontweight = 'bold'
        )
        ax.legend(
            loc = 'lower right',
            shadow = False,
            frameon = False
        )
    else: #compare_targetclasses
        for col, i in enumerate(classes):
            ax.plot(
                plot_input.ntile[plot_input.target_class == i],
                plot_input.profit[plot_input.target_class == i],
                label = i,
                color = colors[col]
            )
            #ax.plot(plot_input.ntile[plot_input.target_class == i], plot_input.cumcosts[plot_input.target_class == i], linestyle = 'dashed', label = "total costs (%s)" % i, color = colors[col])
        ax.set_title(
            "Scope: comparing target classes & dataset: %s & model: %s" % (
                datasets[0], models[0]
            ),
            fontweight = 'bold'
        )
        ax.legend(
            loc = 'lower right',
            shadow = False,
            frameon = False
        )
    
    if highlight_ntile is not None:
        
        if highlight_ntile not in np.linspace(1, ntiles, num = ntiles).tolist():
            raise TypeError(
                'Invalid value for highlight_ntile parameter. '
                'It must be an int value between 1 and %d' % (ntiles)
            )
            
        if highlight_how not in ('plot','text','plot_text'):
            raise ValueError(
                'Invalid highlight_how value, '
                'it must be one of the following: plot, text or plot_text.'
            )
            
        else:
            text = ''
            if scope == "no_comparison":
                cumpct = (
                    plot_input.loc[
                        plot_input.ntile == highlight_ntile,
                        'profit'
                    ].tolist()
                )
                plt.plot(
                    [1, highlight_ntile],
                    [cumpct[0]] * 2,
                    linestyle = '-.',
                    color = colors[0],
                    lw = 1.5
                )
                plt.plot(
                    [highlight_ntile] * 2 ,
                    [0] + [cumpct[0]],
                    linestyle = '-.',
                    color = colors[0],
                    lw = 1.5
                )
                xy = tuple([highlight_ntile] + [cumpct[0]])
                ax.plot(
                    xy[0],
                    xy[1],
                    ".r",
                    ms = 20,
                    color = colors[0]
                )
                ax.annotate(
                    "€" + str(int(cumpct[0])),
                    xy = xy,
                    xytext = (-30, -30),
                    textcoords='offset points',
                    ha = 'center',
                    va = 'bottom',
                    color = 'black',
                    bbox=dict(
                        boxstyle='round, pad = 0.4',
                        alpha = 1,
                        fc = colors[0]
                    ), #fc = 'yellow', alpha = 0.3),
                    arrowprops = dict(
                        arrowstyle = '->',
                        color = 'black'
                    )
                )
                text += (
                    (
                        'When we select %s 1 until %d from model %s in dataset '
                        '%s the percentage of %s cases in the expected profit is %d'
                    ) % (
                        description_label,
                        highlight_ntile,
                        models[0],
                        datasets[0],
                        classes[0],
                        int(cumpct[0])
                    ) + '.\n'
                )
            elif scope == "compare_datasets":
                for col, i in enumerate(datasets):
                    cumpct = (
                        plot_input.loc[
                            plot_input.ntile == highlight_ntile,
                            ['dataset_label', 'profit']
                        ]
                    )
                    cumpct = cumpct.profit[cumpct.dataset_label == i].tolist()
                    plt.plot(
                        [1, highlight_ntile],
                        cumpct * 2,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    plt.plot(
                        [highlight_ntile] * 2,
                        [0] + cumpct,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    xy = tuple([highlight_ntile] + cumpct)
                    ax.plot(
                        xy[0],
                        xy[1],
                        ".r",
                        ms = 20,
                        color = colors[col]
                    )
                    ax.annotate(
                        "€" + str(int(cumpct[0])),
                        xy = xy, xytext = (-30, -30),
                        textcoords='offset points',
                        ha = 'center',
                        va = 'bottom',
                        color = 'black',
                        bbox=dict(
                            boxstyle='round, pad = 0.4',
                            fc = colors[col],
                            alpha = 0.8,
                        ), #fc = 'yellow', alpha = 0.3),
                        arrowprops = dict(
                            arrowstyle = '->',
                            color = 'black'
                        )
                    )
                    text += (
                        (
                            'When we select %s 1 until %d from model %s in dataset '
                            '%s the percentage of %s cases in the expected profit is %d'
                        ) % (
                            description_label,
                            highlight_ntile,
                            models[0],
                            datasets[col],
                            classes[0],
                            int(cumpct[0])
                        ) + '.\n'
                    )
            elif scope == "compare_models":
                for col, i in enumerate(models):
                    cumpct = (
                        plot_input.loc[
                            plot_input.ntile == highlight_ntile,
                            ['model_label', 'profit']
                        ]
                    )
                    cumpct = cumpct.profit[cumpct.model_label == i].tolist()
                    plt.plot(
                        [1, highlight_ntile],
                        cumpct * 2,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    plt.plot(
                        [highlight_ntile] * 2,
                        [0] + cumpct,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    xy = tuple([highlight_ntile] + cumpct)
                    ax.plot(
                        xy[0],
                        xy[1],
                        ".r",
                        ms = 20,
                        color = colors[col]
                    )
                    ax.annotate(
                        "€" + str(int(cumpct[0])),
                        xy = xy,
                        xytext = (-30, -30),
                        textcoords='offset points',
                        ha = 'center',
                        va = 'bottom',
                        color = 'black',
                        bbox=dict(
                            boxstyle='round, pad = 0.4',
                            fc = colors[col],
                            alpha = 0.8,
                        ), #fc = 'yellow', alpha = 0.3),
                        arrowprops = dict(
                            arrowstyle = '->',
                            color = 'black'
                        )
                    )
                    text += (
                        (
                            'When we select %s 1 until %d from model %s in dataset '
                            '%s the percentage of %s cases in the expected profit is %d'
                        ) % (
                            description_label,
                            highlight_ntile,
                            models[col],
                            datasets[0],
                            classes[0],
                            int(cumpct[0])
                        ) + '.\n'
                    )
            else: # compare targetvalues
                for col, i in enumerate(classes):
                    cumpct = (
                        plot_input.loc[
                            plot_input.ntile == highlight_ntile,
                            ['target_class', 'profit']
                        ]
                    )
                    cumpct = cumpct.profit[cumpct.target_class == i].tolist()
                    plt.plot(
                        [1, highlight_ntile],
                        cumpct * 2,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    plt.plot(
                        [highlight_ntile] * 2,
                        [0] + cumpct,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    xy = tuple([highlight_ntile] + cumpct)
                    ax.plot(
                        xy[0],
                        xy[1],
                        ".r",
                        ms = 20,
                        color = colors[col]
                    )
                    ax.annotate(
                        "€" + str(int(cumpct[0])),
                        xy = xy,
                        xytext = (-30, -30),
                        textcoords='offset points',
                        ha = 'center',
                        va = 'bottom',
                        color = 'black',
                        bbox=dict(
                            boxstyle='round, pad = 0.4',
                            fc = colors[col],
                            alpha = 0.8,
                        ), #fc = 'yellow', alpha = 0.3),
                        arrowprops = dict(
                            arrowstyle = '->',
                            color = 'black'
                        )
                    )
                    text += (
                        (
                            'When we select %s 1 until %d from model %s in dataset '
                            '%s the percentage of %s cases in the expected profit is %d'
                        ) % (
                            description_label,
                            highlight_ntile,
                            models[0],
                            datasets[0],
                            classes[col],
                            int(cumpct[0])
                        ) + '.\n'
                    )
            if highlight_how in ('text', 'plot_text'):
                print(text[:-1])
            if highlight_how in ('plot', 'plot_text'):
                fig.text(.15, -0.001, text[:-1], ha='left')
    
    if save_fig == True:
        if not save_fig_filename:
            location = '%s/Profit plot.png' % os.getcwd()
            plt.savefig(location, dpi = 300)
            print("The profit plot is saved in %s" % location)
        else:
            plt.savefig(save_fig_filename, dpi = 300)
            print("The profit plot is saved in %s" % save_fig_filename)
        plt.show()
        plt.gcf().clear()
    plt.show()
    return ax


def plot_roi(
    plot_input,
    fixed_costs,
    variable_costs_per_unit,
    profit_per_unit,
    save_fig = True,
    save_fig_filename = '',
    highlight_ntile = None,
    highlight_how = 'plot_text',
):
    """
    Plotting ROI curve
    
    Parameters
    ----------
    plot_input : pandas.DataFrame
        The result from scope_modevalplot().
    
    fixed_costs : int / float
        Specifying the fixed costs related to a selection based on the model.
        These costs are constant and do not vary with selection size (ntiles).
    
    variable_costs_per_unit : int / float
        Specifying the variable costs per selected unit for a selection based on the model.
        These costs vary with selection size (ntiles).
        
    profit_per_unit : int / float
        Specifying the profit per unit in case the selected unit converts / responds positively.
    
    save_fig : bool, default=True
        Save the plot.
        
    save_fig_filename : str, optional, default=''
        Specify the path and filetype to save the plot.
        If nothing specified, the plot will be saved as jpeg
        to the current working directory.
        
    highlight_ntile : int or None, optional, default=None
        Highlight the value of the response curve at a specified ntile value.
        
        .. versionchanged:: 0.3.9
            Default changed from False to None.
        
    highlight_how : str, optional, default='plot_text'
        Highlight_how specifies where information about the model performance is printed.
        It can be shown as text, on the plot or both.
    
    Returns
    -------
    It returns a matplotlib.axes._subplots.AxesSubplot object
    that can be transformed into the same plot with the .figure command.
    The plot is by default written to disk (save_fig = True).
    The location and filetype of the file depend on the save_fig_filename parameter.
    If the save_fig_filename parameter is empty (not specified),
    the plot will be written to the working directory as png. 
    Otherwise the location and file type is specified by the user.
        
    Raises
    ------
    TypeError: If ``highlight_ntile`` is not specified as an int.
    ValueError: If the wrong ``highlight_how`` value is specified.
    """
    
    models   = plot_input.model_label.unique().tolist()
    datasets = plot_input.dataset_label.unique().tolist()
    classes  = plot_input.target_class.unique().tolist()
    scope = plot_input.scope.unique()[0]
    ntiles = plot_input.ntile.nunique() - 1
    colors = (
        "#E41A1C",
        "#377EB8",
        "#4DAF4A",
        "#984EA3",
        "#FF7F00",
        "#FFFF33",
        "#A65628",
        "#F781BF",
        "#999999"
    )
    
    plot_input['variable_costs'] = variable_costs_per_unit * plot_input.cumtot
    plot_input['investments'] = fixed_costs + plot_input.variable_costs 
    plot_input['revenues'] = profit_per_unit * plot_input.cumpos
    plot_input['profit'] = plot_input.revenues - plot_input.investments
    plot_input['roi'] = plot_input.profit / plot_input.investments
    
    plot_input['variable_costs_tot'] = variable_costs_per_unit * plot_input.tottot
    plot_input['investments_tot'] = fixed_costs + plot_input.variable_costs_tot
    plot_input['revenues_tot'] = profit_per_unit * plot_input.postot
    plot_input['profit_tot'] = plot_input.revenues_tot - plot_input.investments_tot
    plot_input['roi_ref'] = plot_input.profit_tot / plot_input.investments_tot
    
    if ntiles == 10:
        description_label = 'decile'
    elif ntiles == 100:
        description_label = 'percentile'
    else:
        description_label = 'ntile'
    
    if ntiles <= 20:
        xlabper = 1
    elif ntiles <= 40:
        xlabper = 2
    else:
        xlabper = 5
        
    fig, ax = plt.subplots(figsize = (12,7))
    ax.set_xlabel(description_label)
    ax.set_ylabel("% roi")
    plt.suptitle('Return on Investment (ROI)', fontsize = 16)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xticks(np.arange(0, ntiles + 1, xlabper))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.grid(True)
    ax.set_xlim([1, ntiles])
    ax.plot(
        list(range(1, ntiles + 1, 1)),
        [0] * ntiles,
        linestyle = 'dashed',
        label = "break even",
        color = 'grey'
    )

    if scope == "no_comparison":
        ax.set_title(
            "model: %s & dataset: %s & target class: %s" % (
                models[0], datasets[0], classes[0]
            ),
            fontweight = 'bold'
        )
        ax.plot(
            plot_input.ntile,
            plot_input.roi,
            label = classes[0],
            color = colors[0]
        )
        ax.legend(
            loc = 'lower right',
            shadow = False,
            frameon = False
        )
    elif scope == "compare_datasets":
        for col, i in enumerate(datasets):
            ax.plot(
                plot_input.ntile[plot_input.dataset_label == i],
                plot_input.roi[plot_input.dataset_label == i],
                label = i,
                color = colors[col]
            )
        ax.set_title(
            "scope: comparing datasets & model: %s & target class: %s" % (
                models[0], classes[0]
            ),
            fontweight = 'bold'
        )
        ax.legend(
            loc = 'upper right',
            shadow = False,
            frameon = False
        )
    elif scope == "compare_models":
        for col, i in enumerate(models):
            ax.plot(
                plot_input.ntile[plot_input.model_label == i],
                plot_input.roi[plot_input.model_label == i],
                label = "roi (%s)" % i,
                color = colors[col]
            )
        ax.legend(
            loc = 'lower right',
            shadow = False,
            frameon = False
        )
        ax.set_title(
            "scope: comparing models & dataset: %s & target class: %s" % (
                datasets[0], classes[0]
            ),
            fontweight = 'bold'
        )
        ax.legend(
            loc = 'lower right',
            shadow = False,
            frameon = False
        )
    else: #compare_targetclasses
        for col, i in enumerate(classes):
            ax.plot(
                plot_input.ntile[plot_input.target_class == i],
                plot_input.roi[plot_input.target_class == i],
                label = i,
                color = colors[col]
            )
        ax.set_title(
            "Scope: comparing target classes & dataset: %s & model: %s" % (
                datasets[0], models[0]
            ),
            fontweight = 'bold'
        )
        ax.legend(
            loc = 'lower right',
            shadow = False,
            frameon = False
        )
    
    if highlight_ntile is not None:
        
        if highlight_ntile not in np.linspace(1, ntiles, num = ntiles).tolist():
            raise TypeError(
                'Invalid value for highlight_ntile parameter. '
                'It must be an int value between 1 and %d' % (ntiles)
            )
            
        if highlight_how not in ('plot','text','plot_text'):
            raise ValueError(
                'Invalid highlight_how value, '
                'it must be one of the following: plot, text or plot_text.'
            )
            
        else:
            text = ''
            if scope == "no_comparison":
                cumpct = (
                    plot_input.loc[
                        plot_input.ntile == highlight_ntile,
                        'roi'
                    ].tolist()
                )
                plt.plot(
                    [1, highlight_ntile],
                    [cumpct[0]] * 2,
                    linestyle = '-.',
                    color = colors[0],
                    lw = 1.5
                )
                plt.plot(
                    [highlight_ntile] * 2 ,
                    [0] + [cumpct[0]],
                    linestyle = '-.',
                    color = colors[0],
                    lw = 1.5
                )
                xy = tuple([highlight_ntile] + [cumpct[0]])
                ax.plot(xy[0], xy[1], ".r", ms = 20, color = colors[0])
                ax.annotate(
                    str(int(cumpct[0] * 100)) + "%",
                    xy = xy,
                    xytext = (-30, -30),
                    textcoords='offset points',
                    ha = 'center',
                    va = 'bottom',
                    color = 'black',
                    bbox=dict(
                        boxstyle='round, pad = 0.4',
                        alpha = 1,
                        fc = colors[0]
                    ), #fc = 'yellow', alpha = 0.3),
                    arrowprops = dict(
                        arrowstyle = '->',
                        color = 'black'
                    )
                )
                text += (
                    (
                        'When we select %s 1 until %d from model %s in dataset '
                        '%s the percentage of %s cases in the expected expected return on '
                        'investment is %d'
                    ) % (
                        description_label,
                        highlight_ntile,
                        models[0],
                        datasets[0],
                        classes[0],
                        int(cumpct[0] * 100)
                    ) + '%.\n'
                )
            elif scope == "compare_datasets":
                for col, i in enumerate(datasets):
                    cumpct = (
                        plot_input.loc[
                            plot_input.ntile == highlight_ntile,
                            ['dataset_label', 'roi']
                        ]
                    )
                    cumpct = cumpct.roi[cumpct.dataset_label == i].tolist()
                    plt.plot(
                        [1, highlight_ntile],
                        cumpct * 2,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    plt.plot(
                        [highlight_ntile] * 2,
                        [0] + cumpct,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    xy = tuple([highlight_ntile] + cumpct)
                    ax.plot(
                        xy[0],
                        xy[1],
                        ".r",
                        ms = 20,
                        color = colors[col]
                    )
                    ax.annotate(
                        str(int(cumpct[0] * 100)) + "%",
                        xy = xy,
                        xytext = (-30, -30),
                        textcoords='offset points',
                        ha = 'center',
                        va = 'bottom',
                        color = 'black',
                        bbox=dict(
                            boxstyle='round, pad = 0.4',
                            fc = colors[col],
                            alpha = 0.8,
                        ), #fc = 'yellow', alpha = 0.3),
                        arrowprops = dict(
                            arrowstyle = '->',
                            color = 'black'
                        )
                    )
                    text += (
                        (
                            'When we select %s 1 until %d from model %s in dataset '
                            '%s the percentage of %s cases in the expected expected '
                            'return on investment is %d'
                        ) % (
                            description_label,
                            highlight_ntile,
                            models[0],
                            datasets[col],
                            classes[0],
                            int(cumpct[0] * 100)
                        ) + '%.\n'
                    )
            elif scope == "compare_models":
                for col, i in enumerate(models):
                    cumpct = (
                        plot_input.loc[
                            plot_input.ntile == highlight_ntile,
                            ['model_label', 'roi']
                        ]
                    )
                    cumpct = cumpct.roi[cumpct.model_label == i].tolist()
                    plt.plot(
                        [1, highlight_ntile],
                        cumpct * 2,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    plt.plot(
                        [highlight_ntile] * 2,
                        [0] + cumpct,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    xy = tuple([highlight_ntile] + cumpct)
                    ax.plot(
                        xy[0],
                        xy[1],
                        ".r",
                        ms = 20,
                        color = colors[col]
                    )
                    ax.annotate(
                        str(int(cumpct[0] * 100)) + "%",
                        xy = xy,
                        xytext = (-30, -30),
                        textcoords='offset points',
                        ha = 'center',
                        va = 'bottom',
                        color = 'black',
                        bbox=dict(
                            boxstyle='round, pad = 0.4',
                            fc = colors[col],
                            alpha = 0.8,
                        ), #fc = 'yellow', alpha = 0.3),
                        arrowprops = dict(
                            arrowstyle = '->',
                            color = 'black'
                        )
                    )
                    text += (
                        (
                            'When we select %s 1 until %d from model %s in dataset '
                            '%s the percentage of %s cases in the expected expected '
                            'return on investment is %d'
                        ) % (
                            description_label,
                            highlight_ntile,
                            models[col],
                            datasets[0],
                            classes[0],
                            int(cumpct[0] * 100)
                        ) + '%.\n'
                    )
            else: # compare targetvalues
                for col, i in enumerate(classes):
                    cumpct = (
                        plot_input.loc[
                            plot_input.ntile == highlight_ntile,
                            ['target_class', 'roi']
                        ]
                    )
                    cumpct = cumpct.roi[cumpct.target_class == i].tolist()
                    plt.plot(
                        [1, highlight_ntile],
                        cumpct * 2,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    plt.plot(
                        [highlight_ntile] * 2,
                        [0] + cumpct,
                        linestyle = '-.',
                        color = colors[col],
                        lw = 1.5
                    )
                    xy = tuple([highlight_ntile] + cumpct)
                    ax.plot(
                        xy[0],
                        xy[1],
                        ".r",
                        ms = 20,
                        color = colors[col]
                    )
                    ax.annotate(
                        str(int(cumpct[0] * 100)) + "%",
                        xy = xy,
                        xytext = (-30, -30),
                        textcoords='offset points', ha = 'center', va = 'bottom', color = 'black',
                        bbox=dict(
                            boxstyle='round, pad = 0.4',
                            fc = colors[col],
                            alpha = 0.8,
                        ), #fc = 'yellow', alpha = 0.3),
                        arrowprops = dict(
                            arrowstyle = '->',
                            color = 'black'
                        )
                    )
                    text += (
                        (
                            'When we select %s 1 until %d from model %s in dataset '
                            '%s the percentage of %s cases in the expected return on '
                            'investment is %d'
                        ) % (
                            description_label,
                            highlight_ntile,
                            models[0],
                            datasets[0],
                            classes[col],
                            int(cumpct[0] * 100)
                        ) + '%.\n'
                    )
            if highlight_how in ('text', 'plot_text'):
                print(text[:-1])
            if highlight_how in ('plot', 'plot_text'):
                fig.text(.15, -0.001, text[:-1], ha='left')
    
    if save_fig == True:
        if not save_fig_filename:
            location = '%s/ROI plot.png' % os.getcwd()
            plt.savefig(location, dpi = 300)
            print("The roi plot is saved in %s" % location)
        else:
            plt.savefig(save_fig_filename, dpi = 300)
            print("The roi plot is saved in %s" % save_fig_filename)
        plt.show()
        plt.gcf().clear()
    plt.show()
    return ax