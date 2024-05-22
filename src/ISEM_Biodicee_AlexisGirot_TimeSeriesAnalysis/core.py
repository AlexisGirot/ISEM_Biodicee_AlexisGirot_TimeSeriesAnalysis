"""
The core of the package. Defines the TS_list class as well as utilitary functions.
"""


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
        return_container = list([x for x in rpy2_object])
        for i in range(len(return_container)):
            if str(type(return_container[i])) in ("<class 'rpy2.rinterface_lib.sexp.NACharacterType'>", "<class 'rpy2.rinterface_lib.sexp.NALogicalType'>"):
                return_container[i] = "NA"
        if len(return_container) == 1:
            return_container = return_container[0]
        
    elif str(type(rpy2_object)) in ("<class 'numpy.ndarray'>", "<class 'rpy2.robjects.vectors.BoolVector'>"):
        return_container = list(rpy2_object)
        for i in range(len(return_container)):
            if str(type(return_container[i])) in ("<class 'rpy2.rinterface_lib.sexp.NALogicalType'>", "<class 'rpy2.rinterface_lib.sexp.NACharacterType'>"):
                return_container[i] = "NA"
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
                                data[ide][column] = row[column]
    
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
        
        
        elif str(type(data)) == "<class 'ISEM_Biodicee_AlexisGirot_TimeSeriesAnalysis.core.TS_list'>":
            self.asd_thr = data.asd_thr
            self.classification = data.classification
            self.time_series = data.time_series
            self.is_log_transformed = data.is_log_transformed
            if self.name == "Some list of time series":
                self.name = data.name
        
        
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
    
    def classify(self, method = "aic_asd", asd_thr = 0.1, min_len = 20):
        """
        Classify the time series using Mathieu's code
        
        Parameters
        ----------
        method : str
        asd_thr : float in [0,1]
        min_len : int
        
        Returns
        -------
        dict : Need to return for the parralel computation to work
        """
        if self.df_list is None:
            print(f"{self.name}: Building self.df_list")
            self.df_list = dict_to_dataframes(self.time_series)
        
        self.asd_thr = asd_thr
        print(f"{self.name}: Running the classification")
        self.classification = robjects.r.run_classif_data(\
                                  df_list = self.df_list,\
                                  min_len = min_len,\
                                  str = method,\
                                  asd_thr = asd_thr,\
                                  run_loo = False,\
                                  two_bkps = False,\
                                  smooth_signif = True,\
                                  group = "id",\
                                  time = "X",\
                                  variable = "Y",\
                                  save_plot=False)
        
        print(f"{self.name}: Converting back to python")
        with(robjects.default_converter + pandas2ri.converter).context():
            self.classification = robjects.conversion.get_conversion().rpy2py(self.classification)
        
        print(f"{self.name}: Unnesting data frames")
        self.classification = rpy2_to_python(self.classification)
        
        print(f"{self.name}: Returning the result")
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
        Make a plot of the different infered classes. The color can indicate the potentially missed shifts
        
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
            l_id = self.classification["outlist"].keys()
            l_class = [self.classification["outlist"][ts_id]["best_traj"]["class"] for ts_id in l_id]
#             l_class = [ts["best_traj"]["class"] for ts in self.classification["outlist"].values()]
            classes = []
            counts = []
            missed_counts = []

            for cl in set(l_class):
                classes.append(cl)
                counts.append(len([x for x in l_class if x == cl]))
                missed_counts.append(len([ts_id for ts_id in l_id if self.classification["outlist"][ts_id]["best_traj"]["class"] == cl and self.classification["outlist"][ts_id]["res"]["n_brk_asd"] > 0]))
                
            
            # Order the list for more understandable plot
            order = ["no_change", "linear", "quadratic", "abrupt"]
            classes, counts, missed_counts = zip(*sorted(zip(classes, counts, missed_counts), key = lambda x:order.index(x[0])))
            
            # Bar plot
            if ax is None:
                ax = plt.gca()
            
            counts = np.array(counts)
            missed_counts = np.array(missed_counts)
            
            bottom = 0
            graph1 = ax.bar(classes, counts - missed_counts, bottom = bottom)
            if max(missed_counts) != 0: # To render nier plots when there is no missed shift (eg if asd_thr = 1)
                bottom = counts - missed_counts
                graph2 = ax.bar(classes, missed_counts, bottom = bottom)
            
            
            #i = 0
            for i in range(len(classes)):
                bar1 = graph1[i]
                if max(missed_counts) != 0:
                    bar2 = graph2[i]
                width = bar1.get_width()
                if max(missed_counts) != 0:
                    height = bar1.get_height() + bar2.get_height()
                else:
                    height = bar1.get_height()
                x, y = bar1.get_xy()
                ax.text(x+width/2,
                        y+height*1.01,
                        f"{counts[i]/np.sum(counts)*100:.2f}%",
                        ha="center",
                        weight="bold")
                #i += 1
            
            ax.legend(["No brkpoint", "Brkpoint"])
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

    def plot(self, time_series_id, ax = None, xlabel = "Time (arb. u.)", linetype = 'o-', title = None):
        """
        Make a plot of the specified time series
        
        Parameters
        ----------
        time_series_id : tuple
        ax : subplot.ax
        xlabel : str
        linetype : str
        title : str (the to_keep columb in which to fetch the title). If None, the time series id will be the title
        
        Returns
        -------
        Subplot
        """
        
        X = self.time_series[time_series_id]["X"]
        Y = self.time_series[time_series_id]["Y"]
        
        if title is None:
            title = str(time_series_id)
            
        else:
            if type(self.time_series[time_series_id][title]) is list:
                title = self.time_series[time_series_id][title][0]
            else:
                title = self.time_series[time_series_id][title]
            
        if ax is None:
            ax = plt.gca()
            
        ax.plot(X, Y, linetype)
        if self.is_log_transformed:
            ax.set_title(title + " (log transformed)")
            ax.set_ylabel("State, log transformed (arb. u.)")
        else:
            ax.set_title(title)
            ax.set_ylabel("State (arb. u.)")
        ax.set_xlabel(xlabel)
            
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
    
    def mc_classify(self, method = "aic_asd", asd_thr = 0.1, min_len = 20):
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
        
        self.asd_thr = asd_thr
        
        #Keep only long enough time series
        to_classify = {ts_id:ts for ts_id, ts in self.time_series.items() if len(ts["id"]) >= min_len}
        
        # Divide the time series list in the number of cpus, then run the classification as usual
        nb_cpu = multiprocessing.cpu_count()
        nb_cpu = min(nb_cpu, len(to_classify)) #In case there are very few time series
        TS_sublists = []
        l_id = list(to_classify.keys())
        
        for i in range(nb_cpu-1):
            sublist = {ide:to_classify[ide] for ide in l_id[i * (len(l_id) // nb_cpu):(i+1) * (len(l_id) // nb_cpu)] }
            TS_sublists.append(TS_list(data = sublist, name = f"Sublist {i}"))


        sublist = {ide:to_classify[ide] for ide in l_id[(nb_cpu - 1) * (len(l_id) // nb_cpu):]}
        TS_sublists.append(TS_list(data = sublist, name = f"Sublist {nb_cpu-1}"))
        

        with multiprocessing.Pool() as pool:
            l_classif = pool.map(functools.partial(TS_list.classify, method = method, asd_thr = asd_thr, min_len = min_len), TS_sublists)
        

        
        # Now bring back together the classification results
        self.classification = l_classif[0]
        
        for i in range(1,len(TS_sublists)):
            self.classification["outlist"].update(l_classif[i]["outlist"])
            for k in self.classification["traj_ts_full"].keys():
                if k in l_classif[i]["traj_ts_full"]:
                    self.classification["traj_ts_full"][k] += l_classif[i]["traj_ts_full"][k]
        
        return self.classification
            
            
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
    
    def __getitem__(self, index):
        """
        Subset the time series list, depending on the argument type
        
        Parameters
        ----------
        index : * slice : slices all the time series
                * dict : keeps only the time series fulfilling the conditions (eg : {"expected_traj":"quadratic"})
                * str : keeps only the time series for which the index appears in the id
        
        Returns
        -------
        None
        """
        
        
        if type(index) == slice: #Subset every time series
            new_ts_list =  TS_list(data = {ts_id:{col_id:col[index] for col_id, col in ts.items()} for ts_id, ts in self.time_series.items()},
                           name = self.name,
                           is_log_transformed = self.is_log_transformed)
            
            new_ts_list.df_list = None
            new_ts_list.asd_thr = self.asd_thr
            new_ts_list.classification = None
            new_ts_list.saved_time_series = None
        
        elif type(index) == dict:
            new_data = {}
            for ts_id, ts in self.time_series.items():
                fulfills_condition = True
                for col_id, col_val in index.items():
                    if ts[col_id][0] != col_val:
                            fulfills_condition = False

                if fulfills_condition:
                    new_data[ts_id] = ts
                    
            new_ts_list = TS_list(data = new_data,
                           name = self.name,
                           is_log_transformed = self.is_log_transformed)
        
        elif type(index) == str:
            new_data = {}
            for ts_id, ts in self.time_series.items():
                if index in ts_id:
                    new_data[ts_id] = ts
                    
            new_ts_list = TS_list(data = new_data,
                           name = self.name,
                           is_log_transformed = self.is_log_transformed)
        
        else:
            raise TypeError(f"Non implemented type: {type(index)}")
            
        
        return new_ts_list
    
    
    def __truediv__(self, other):
        """
        Provides the confusion matrix comparing the classes infered from two classifications of the same time series
        
        Parameters
        ----------
        other : TS_list
        
        Returns
        -------
        np.ndarray
        """
        
        if self.classification is None or other.classification is None:
            raise Exception("At least one of the time series lists has not been classified yet. Please run the classification before trying to analyse its results.")
        
        
        if self.classification["outlist"].keys() != other.classification["outlist"].keys():
            raise Exception("The two time series lists are not identical.")
        
        l_other = []
        l_self = []
        
        for ts_id in self.classification["outlist"].keys(): #Heavy syntax but robust to changes in the dict orders
            l_other.append(other.classification["outlist"][ts_id]["best_traj"]["class"])
            l_self.append(self.classification["outlist"][ts_id]["best_traj"]["class"])
        
        
        labels = [] # Labels in the right order
        for x in l_other:
            if len(labels) < len(set(l_other)) and x not in labels:
                labels.append(x)
                
        cm = sklearn.metrics.confusion_matrix(y_true = l_other, y_pred = l_self, normalize = "true", labels=labels)
        
        ax = plt.gca()
        
        #im = ax.imshow(cm, cmap = plt.cm.Blues)
        cmd = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = labels)

        pl = cmd.plot(ax=ax, cmap = plt.cm.Blues)
        #ax.set_title("Compare two classifications")
        ax.set_xlabel(self.name)
        ax.set_ylabel(other.name)
        ax.set_xticks(ticks = range(len(labels)), labels = labels)
        ax.set_yticks(ticks = range(len(labels)), labels = labels)
        
        return pl
        
        return cm

    def __floordiv__(self, other):
        """
        Provides the confusion matrix comparing the abruptness infered for two classifications of a same time series list
        
        Parameters
        ----------
        other : TS_list
        
        Returns
        -------
        np.ndarray
        """
        
        if self.classification is None or other.classification is None:
            raise Exception("At least one of the time series lists has not been classified yet. Please run the classification before trying to analyse its results.")
        
        
        if self.classification["outlist"].keys() != other.classification["outlist"].keys():
            raise Exception("The two time series lists are not identical.")
        
        l_other = []
        l_self = []
        
        for ts_id in self.classification["outlist"].keys(): #Heavy syntax but robust to changes in the dict orders
            l_other.append(other.classification["outlist"][ts_id]["best_traj"]["class"])
            l_self.append(self.classification["outlist"][ts_id]["best_traj"]["class"])
        
        for i in range(len(l_other)):
            if l_other[i] != "abrupt":
                l_other[i] = "non-abrupt"
            if l_self[i] != "abrupt":
                l_self[i] = "non-abrupt"
        
        labels = [] # Labels in the right order
        for x in l_other:
            if len(labels) < len(set(l_other)) and x not in labels:
                labels.append(x)
                
        cm = sklearn.metrics.confusion_matrix(y_true = l_other, y_pred = l_self, normalize = "true", labels=labels)
        
        ax = plt.gca()
        
        #im = ax.imshow(cm, cmap = plt.cm.Blues)
        cmd = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = labels)

        pl = cmd.plot(ax=ax, cmap = plt.cm.Blues)
        #ax.set_title("Compare two classifications")
        ax.set_xlabel(self.name)
        ax.set_ylabel(other.name)
        ax.set_xticks(ticks = range(len(labels)), labels = labels)
        ax.set_yticks(ticks = range(len(labels)), labels = labels)
        
        return pl
