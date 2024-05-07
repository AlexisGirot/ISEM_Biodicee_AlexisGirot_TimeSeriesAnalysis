# Some functions to correct mistakes (eg when we have forgotten to keep a column or change the name when saving)

from . import core

def resave(f_out, TS_l, data, delimiter = ",", method = None, cols = None, rows = None, to_keep = None, saving_name = None):
    """
    Save again when a file has not been saved properly
    
    Parameters
    ----------
    f_out : file object
    TS_l : TS_list
    data : same as data arg of TS_list
    to_keep : ["str", "str", "str"]
    saving_name : str
    
    Returns
    -------
    None
    """
    
    ts = core.TS_list(data = data, method = method, delimiter = delimiter, cols = cols, rows = rows,  to_keep = to_keep)
    TS_l.time_series = ts.time_series
    TS_l.save(file = f_out, saving_name = saving_name)
