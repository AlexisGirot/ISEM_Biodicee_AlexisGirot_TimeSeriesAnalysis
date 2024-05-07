#Import python packages
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
import json
import os
import inspect
import sklearn.metrics
import multiprocessing
import functools

current_dir = os.path.dirname(os.path.abspath(inspect.getsourcefile(lambda:0)))


# Import R functions and packages
importr("tidyverse")
importr("chngpt")
importr("asdetect")
source_file = "functions_trajclass.R"
robjects.r.source(current_dir + "/" + source_file)



def rpy2_to_python(rpy2_object):
    """
    Transform a rpy2 object into nested dict to make it easier to manipulate and save using json
    Computes the result recursively
    
    Parameters
    ----------
    rpy2_object : some rpy2 container
    
    Returns
    -------
    return_container : a python container (dict, np.ndarray, etc)
    """
    
    if str(type(rpy2_object)) == "<class 'rpy2.rinterface_lib.sexp.NACharacterType'>":
        return_container = "NA"
    elif str(type(rpy2_object)) in ("<class 'rpy2.rlike.container.OrdDict'>", "<class 'pandas.core.frame.DataFrame'>"):
        try:
            return_container = {k:rpy2_to_python(rpy2_object[k]) for k in rpy2_object.keys()}
        except ValueError:
            return_container = list(rpy2_object)
    elif str(type(rpy2_object)) == "<class 'pandas.core.series.Series'>":
        return_container = list(rpy2_object)[0]
        if str(type(return_container)) == "<class 'rpy2.rinterface_lib.sexp.NACharacterType'>":
            return_container = "NA"
    elif str(type(rpy2_object)) in ("<class 'numpy.ndarray'>", "<class 'rpy2.robjects.vectors.BoolVector'>"):
        return_container = list(rpy2_object)
    else:
        raise Exception(f"Non implemented type: {type(rpy2_object)}")

    
    return return_container

def load_data(file, method, delimiter = ',', cols = None, rows = None, to_keep = None):
    """
    Load the data in file f and format it in a form suitable for analysis
    
    Parameters
    ----------
    file : file object
            Should point to a .csv file
    method : str
            Possible values: "by_col", 'by_row'
    delimiter : char
    cols : {"id":(str, str, ...), "time":str, "Y":str}
            When using the "by_col" method
    rows = {"id:(str, str, ...)", "time":[str, str, str]}
            When using the "by_row" method
    to_keep = (str, str, ...)
            A list of columns to keep
    
    Returns
    -------
    dict{dicts{numpy.ndarrays}} = {id:{"scen":np.ndarray, "X":np.ndarray, "Y":np.ndarray}}
    """
    
    reader = csv.DictReader(file, delimiter = ',')
    
    data = {}
    
    if method == "by_col":
        if cols["id"] is None:
            raise ValueError("No identifying column was provided.")
        
        
        for row in reader:
            ide = tuple([row[r_ide] for r_ide in cols["id"]])
            
            if "NA" not in ide: #NAs in the id are mistakes
                y = row[cols["Y"]]
                
                if "NULL" != y and "NA" != y:
                    if ide not in data:
                        data[ide] = {"id":[], "X":[], "Y":[]}
                        
                        if to_keep is not None:
                            for column in to_keep:
                                data[ide][column] = []
                    
                    data[ide]["id"].append(str(ide))
                    data[ide]["X"].append(int(row[cols["time"]]))
                    data[ide]["Y"].append(float(y))
                    
                    if to_keep is not None:
                        for column in to_keep:
                            data[ide][column].append(row[column])
    
    elif method == "by_row":
        if rows["id"] is None:
            raise ValueError("No identifying column was provided.")
        
        for row in reader:
            ide = tuple([row[r_ide] for r_ide in rows["id"]])
            
            if "NA" not in ide: #NAs in the id are mistakes
                if ide in data:
                    raise Exception(f"Two rows have the same identifier {ide}.")
                
                else:
                    X = [int(t) for t in rows["time"]]
                    Y = [row[str(t)] for t in rows["time"]]
                    
                    Z = [z for z in zip(X, Y) if (z[1] != "NA" and z[1] != "NULL" and z[1] != "")]
                    
                    if len(Z) > 0:
                        X, Y = zip(*Z)
                        data[ide] = {"id":[str(ide)] * len(X),\
                                 "X":X,\
                                 "Y":[float(y) for y in Y]}
                        
                        if to_keep is not None:
                            for column in to_keep:
                                data[ide][to_keep] = row[to_keep]
    
    else:
        raise ValueError(f"Method {method} not implemented.")
    
    return data

def dict_to_dataframes(data):
    """
    Convert Python dictionary of dictionary to R list of dataframes, suitable for classification
    
    Parameters
    ----------
    data : dict
        Each element is a dict containing
        a time series in the form {"id":list, "X":list, "Y":list, ...}
    
    Returns
    -------
    list of R dataframes
    """
    dataframes = {}
    for key, value in data.items():
        if len(value["X"]) > 0: # Remove empty time series to avoid errors
            df = pd.DataFrame.from_dict({"id":value["id"],\
                                        "X":value["X"],\
                                        "Y":value["Y"]},\
                                        orient='index').T
            df.columns = ["id", "X", "Y"]
            df['X'] = df['X'].astype(int) #For some reason these conversions are needed
            df['Y'] = df['Y'].astype(float)
        
            with(robjects.default_converter + pandas2ri.converter).context():
                R_df = robjects.conversion.get_conversion().py2rpy(df)

            dataframes[value["id"][0]] = R_df

    return robjects.ListVector(dataframes)

def unstr_name(name):
    """
    The names of the time series have been converted from tuple to str for JSON serialization. This function does the inverse process
    
    Parameters
    ----------
    name : str
    
    Returns
    -------
    tuple
    """
    
    tup = name[1:-1].split(',')
    
    if tup[-1] == "":
        return tuple(tup[:-1])
    else:
        return tuple(tup)

class TS_list(object):
    def __init__(self, data, method = None, delimiter = ',', cols = None, rows = None, to_keep = None, name = "Some list of time series", is_log_transformed = False):
        self.name = name
        self.df_list = None
        self.asd_thr = None
        self.classification = None
        self.time_series = None
        self.saved_time_series = None
        self.is_log_transformed = is_log_transformed
        
        if type(data) == dict:
            self.time_series = data
        
        elif str(type(data)) == "<class '_io.TextIOWrapper'>": # Load the data from a file, either CSV (raw time series) or JSON (already processed)
            extension = data.name.split(".")[-1]
            
            
            if extension == "csv":
                self.time_series = load_data(file = data, method = method, delimiter = delimiter, cols = cols, rows = rows, to_keep = to_keep)
            
            
            elif extension == "json":
                recovered_data = json.load(data)
                
                if "name" in recovered_data:
                    self.name = recovered_data["name"]
                if "asd_thr" in recovered_data:
                    self.asd_thr = recovered_data["asd_thr"]
                if "classification" in recovered_data:
                    self.classification = recovered_data["classification"]
                if "saved time series" in recovered_data:
                    self.saved_time_series = recovered_data["saved time series"]
                    self.time_series = {unstr_name(name):self.saved_time_series[name] for name in self.saved_time_series.keys()}
                    
                
            else:
                raise ValueError(f"Non implemented extension: {extension}")
        
        else:
            raise ValueError(f"Type {type(data)} not implemented.")
    
    def hist(self, ax = None, nb_bins = 20):
        """
        Provides a histogram in the form of a subplot
        
        Parameters
        ----------
        ax : ax object
        nb_bins : int
        
        Returns
        -------
        Subplot
        """
        
        if ax is None:
            ax = plt.gca()
        
        len_TS = [len(ts["X"]) for ts in self.time_series.values()]
        
        ax.hist(len_TS, bins = nb_bins)
        ax.set_title(self.name)
        ax.set_xlabel("TS length (points)")
        ax.set_ylabel("Number of time series")
        
        return ax
    
    def classify(self, method = "aic_asd", asd_thr = 0.1):
        """
        Classify the time series using Mathieu's code
        
        Parameters
        ----------
        method : str
        asd_thr : float in [0,1]
        
        Returns
        -------
        dict : Need to return for the parralel computation to work
        """
        
        if self.df_list is None:
            self.df_list = dict_to_dataframes(self.time_series)
        
        self.asd_thr = asd_thr
        self.classification = robjects.r.run_classif_data(\
                                  df_list = self.df_list,\
                                  str = method,\
                                  asd_thr = asd_thr,\
                                  run_loo = False,\
                                  two_bkps = False,\
                                  smooth_signif = True,\
                                  group = "id",\
                                  time = "X",\
                                  variable = "Y",\
                                  save_plot=False)
        
        with(robjects.default_converter + pandas2ri.converter).context():
            self.classification = robjects.conversion.get_conversion().rpy2py(self.classification)
        
        self.classification = rpy2_to_python(self.classification)
        
        return self.classification
    
    def save(self, file, saving_name = None):
        """
        Saves the object to a JSON file, including the time series and classifications
        
        Parameters
        ----------
        file : a JSON file object
        
        
        Returns
        -------
        None
        """
        # Keys must be str, not tuples for JSON serialization
        
        self.saved_time_series = {str(k):self.time_series[k] for k in self.time_series.keys()}
        
        if saving_name is None:
            saving_name = self.name
        
        json.dump({"name":saving_name,\
                   "saved time series":self.saved_time_series,\
                   "asd_thr":self.asd_thr,\
                   "classification":self.classification}\
                  , file)
    
    def bar_plot(self, ax = None):
        """
        Make a plot of the different infered classes
        
        Parameters
        ----------
        ax : subplot.ax
        
        Returns
        -------
        Subplot
        """
        
        if self.classification is None:
            raise Exception("The data has not been classified yet. Please run the classification before trying to analyse its results.")
        
        else:
            # Count
            l_class = [ts["best_traj"]["class"] for ts in self.classification["outlist"].values()]
            classes = []
            counts = []

            for cl in set(l_class):
                classes.append(cl)
                counts.append(len([x for x in l_class if x == cl]))
            
            # Order the list for more understandable plot
            order = ["no_change", "linear", "quadratic", "abrupt"]
            classes, counts = zip(*sorted(zip(classes, counts), key = lambda x:order.index(x[0])))
            
            # Bar plot
            if ax is None:
                ax = plt.gca()
            
            graph = ax.bar(classes, counts)
            
            i = 0
            for bar in graph:
                width = bar.get_width()
                height = bar.get_height()
                x, y = bar.get_xy()
                ax.text(x+width/2,
                        y+height*1.01,
                        f"{counts[i]/np.sum(counts)*100:.2f}%",
                        ha="center",
                        weight="bold")
                i += 1
            
            ax.set_title(self.name)
            ax.set_xlabel("Infered class")
            ax.set_ylabel("Number of time series")
            
            return ax
    
    def confusion_matrix(self, expected_class_column, ax = None):
        """
        Make a confusion matrix of the different infered classes
        
        Parameters
        ----------
        expected_class_column = str
            The column in which to fetch the expected class
            It should have been kept !
        ax : subplot.ax
        
        Returns
        -------
        Subplot
        """
        
        if self.classification is None:
            raise Exception("The data has not been classified yet. Please run the classification before trying to analyse its results.")
            
        
        l_expected = []
        l_infered = []
        
        for ts_id in self.classification["outlist"].keys(): #Heavy syntax but robust to changes in the dict orders
            l_expected.append(self.time_series[unstr_name(ts_id)][expected_class_column][0])
            l_infered.append(self.classification["outlist"][ts_id]["best_traj"]["class"])
        
        
        labels = [] # Labels in the right order
        for x in l_expected:
            if len(labels) < len(set(l_expected)) and x not in labels:
                labels.append(x)
                
        cm = sklearn.metrics.confusion_matrix(y_true = l_expected, y_pred = l_infered, normalize = "true", labels=labels)
        
        if ax is None:
                ax = plt.gca()
        
        #im = ax.imshow(cm, cmap = plt.cm.Blues)
        cmd = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = labels)

        pl = cmd.plot(ax=ax, cmap = plt.cm.Blues)
        ax.set_title(self.name)
        ax.set_xlabel("Infered class")
        ax.set_ylabel("Expected class")
        ax.set_xticks(ticks = range(len(labels)), labels = labels)
        ax.set_yticks(ticks = range(len(labels)), labels = labels)
        
        return pl

    def plot(self, time_series_id, ax = None, linetype = 'o-'):
        """
        Make a plot of the specified time series
        
        Parameters
        ----------
        ax : subplot.ax
        
        Returns
        -------
        Subplot
        """
        
        X = self.time_series[time_series_id]["X"]
        Y = self.time_series[time_series_id]["Y"]
            
        if ax is None:
            ax = plt.gca()
            
        ax.plot(X, Y, linetype)
        if self.is_log_transformed:
            ax.set_title(time_series_id + " (log transformed)")
        else:
            ax.set_title(time_series_id)
        ax.set_xlabel("Time (arb. u.)")
        ax.set_ylabel("State (arb. u.)")
            
        return ax

    def confusion_matrix_abruptness(self, expected_class_column, ax = None):
        """
        Make a confusion matrix of abrupt VS non abrupt
        
        Parameters
        ----------
        expected_class_column = str
            The column in which to fetch the expected class
            It should have been kept !
        ax : subplot.ax
        
        Returns
        -------
        Subplot
        """
        
        if self.classification is None:
            raise Exception("The data has not been classified yet. Please run the classification before trying to analyse its results.")
            
        
        l_expected = []
        l_infered = []
        
        for ts_id in self.classification["outlist"].keys(): #Heavy syntax but robust to changes in the dict orders
            l_expected.append(self.time_series[unstr_name(ts_id)][expected_class_column][0])
            l_infered.append(self.classification["outlist"][ts_id]["best_traj"]["class"])
        
        for i in range(len(l_expected)):
            if l_expected[i] != "abrupt":
                l_expected[i] = "non-abrupt"
            if l_infered[i] != "abrupt":
                l_infered[i] = "non-abrupt"
        
        labels = [] # Labels in the right order
        for x in l_expected:
            if len(labels) < len(set(l_expected)) and x not in labels:
                labels.append(x)
                
        cm = sklearn.metrics.confusion_matrix(y_true = l_expected, y_pred = l_infered, normalize = "true", labels=labels)
        
        if ax is None:
                ax = plt.gca()
        
        #im = ax.imshow(cm, cmap = plt.cm.Blues)
        cmd = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = labels)

        pl = cmd.plot(ax=ax, cmap = plt.cm.Blues)
        ax.set_title(self.name)
        ax.set_xlabel("Infered abruptness")
        ax.set_ylabel("Expected abruptness")
        ax.set_xticks(ticks = range(len(labels)), labels = labels)
        ax.set_yticks(ticks = range(len(labels)), labels = labels)
        
        return pl
    
    def mc_classify(self, method = "aic_asd", asd_thr = 0.1):
        """
        Classify the time series using Mathieu's code in a multicore framework
        
        Parameters
        ----------
        method : str
        asd_thr : float in [0,1]
        
        Returns
        -------
        None
        """
        
        # Divide the time series list in the number of cpus, then run the classification as usual
        nb_cpu = multiprocessing.cpu_count()
        TS_sublists = []
        l_id = list(self.time_series.keys())
        for i in range(nb_cpu-1):
            sublist = {ide:self.time_series[ide] for ide in l_id[i * len(l_id) // nb_cpu:(i+1) * len(l_id) // nb_cpu - 1] }
            TS_sublists.append(TS_list(sublist))

        sublist = {ide:self.time_series[ide] for ide in l_id[(nb_cpu - 1) * len(l_id) // nb_cpu:]}
        TS_sublists.append(TS_list(sublist))
        

        with multiprocessing.Pool() as pool:
            l_classif = pool.map(functools.partial(TS_list.classify, method = method, asd_thr = asd_thr), TS_sublists)
        

        
        # Now bring back together the classification results
        self.classification = l_classif[0]
        
        for i in range(1,len(TS_sublists)):
            self.classification["outlist"].update(l_classif[i]["outlist"])
            for k in self.classification["traj_ts_full"].keys():
                self.classification["traj_ts_full"][k] += l_classif[i]["traj_ts_full"][k]
            
            
    def log_transform(self):
        """
        (Inplace) log transforms the Y data /!\ The initial data is not preserved /!\
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        
        for ts_id in self.time_series.keys():
            self.time_series[ts_id]["Y"] = list(np.log(1 + np.array(self.time_series[ts_id]["Y"])))
    
        self.name = self.name + " (log transformed)"
    
    
    
    
    
    