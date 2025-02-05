import os
from numpy_typing import np, ax
from typing import Self, TextIO
from datetime  import datetime
import time

def str_(x:float):
    if (isinstance(x, str)):
        return x
    # check if x is np.nan
    if x != x:
        return ""
    if x is None:
        return ""
    if np.isnan(x):
        return ""
    
    # if end with .0 
    if x == int(x):
        return str(int(x))
    
    # if has
        
    return str(x)

def str_date_to_str(date:str):
    return "/".join(date.split("/")[::-1])

        
def full(shape, value, dtype):
    if (dtype == "U64" and np.isnan(value)):
        value = ""
        return np.full(shape, value, dtype=dtype)
        
    return np.full(shape, value, dtype=dtype)

def find(s, ch):
    for i, ltr in enumerate(s):
        if ltr == ch:
            return i
    return -1

def unit_digit(s:str):
    unit_digit = find(s, ".")
    if (unit_digit == -1):
        unit_digit = len(s)
    return unit_digit - 1

def h_line(col_widths):
    str_array = "+"
    
    for width in col_widths:
        if (width == 0): continue
        str_array += "-" * (width + 2) + "+"
    return str_array

class Group:
    name:str
    i:int
    columns:"list[str]"
    columns_indices:"dict[str, int]"
    
    def __init__(self, name:str, data_i:int):
        self.name = name
        self.i = data_i
        self.columns = []
        self.columns_indices = {}
        
    def add_column(self, column:str):
        self.columns.append(column)
        self.columns_indices[column] = len(self.columns) - 1
        
    def remove_column(self, column:str):
        i = self.columns_indices.get(column, None)
        self.columns.pop(i)
        self.columns_indices.pop(column)
        return i
        
    def pop_column(self, i:int):
        column = self.columns.pop(i)
        self.columns_indices.pop(column)
        
    def rename_column(self, old_column:str, new_column:str):
        i = self.columns_indices[old_column]
        self.columns[i] = new_column
        self.columns_indices.pop(old_column)
        self.columns_indices[new_column] = i
        
    def get_column_index(self, column:str):
        return self.columns_indices.get(column, None)
        
    def copy(self):
        res = Group(self.name, self.i)
        res.columns = self.columns.copy()
        res.columns_indices = self.columns_indices.copy()
        return res
    
    def  __len__(self):
        return len(self.columns)

class RunLogger:

    data:"list[np.float64_2d[ax.sample, ax.feature]]"
    groups:"dict[str, Group]"
    columns_group:"dict[str, str]"
    ENTREE = ["i", "model", "date", "time"]
    DEFAULT_GROUP = "metrics"
    ENTREE_GROUP = "run info"
    grouped_by = []
    
    def __init__(self, file=None):
        
        self.data = [np.zeros((0, len(self.ENTREE)), dtype=object)]
        
        entree_group = Group(self.ENTREE_GROUP, 0)
        entree_group.columns = self.ENTREE
        entree_group.columns_indices = {col: i for i, col in enumerate(self.ENTREE)}
        self.groups = {self.ENTREE_GROUP:  entree_group}
        
        self.columns_group = {entree: self.ENTREE_GROUP for entree in self.ENTREE}
        self.grouped_by = []
        self.last_run_id = -1
        
        self.file = file
        self.__load__(file)
        
# |====================================================================================================================
# | PRIVATE GROUPS
# |====================================================================================================================

    def __create_group__(self, group:str, dtype:object=np.float64):
        if group not in self.groups:
            self.groups[group] = Group(group, len(self.data))
            self.data.append(full((len(self.data[0]), 0), np.nan, dtype=dtype))
        return self.groups[group]
            
    def __remove_group__(self, group:str):
        if group in self.groups:
            ith = self.groups[group].i
            self.data.pop(ith)
            self.groups.pop(group)
            # reindex groups
            for group in self.groups.values():
                if group.i > ith:
                    group.i -= 1
        else:
            raise ValueError(f"The group {group} does not exist")
    
    def __rename_group__(self, old_group:str, new_group:str):
        if old_group not in self.groups or new_group in self.groups:
            return
        
        self.groups[new_group] = self.groups[old_group]
        self.groups[new_group].name = new_group
        self.groups.pop(old_group)
    
    def __get_group__(self, group_name:str):
        if (group_name is None):
            group_name = "metrics"
        
        return self.groups.get(group_name, None), group_name
    
# |====================================================================================================================
# | PRIVATE COLUMNS
# |====================================================================================================================

    def __create_column__(self, column:str, group_name:str=None, dtype:object=np.float64) -> tuple[Group, int]:
        if (column in self.columns_group):
            raise ValueError(f"The column {column} already exists")
        group, group_name = self.__get_group__(group_name)
        if (group is None):
            group = self.__create_group__(group_name, dtype)
            
        group.add_column(column)
        self.columns_group[column] = group_name
        self.data[group.i] = np.concatenate((self.data[group.i], full((len(self.data[group.i]), 1), np.nan, dtype=self.data[group.i].dtype)), axis=1)
        return group, len(group) - 1
    
    def __get_or_create_column__(self, column:str, group_name:str=None, dtype:object=np.float64) -> tuple[Group, int]:
        if column not in self.columns_group:
            return self.__create_column__(column, group_name, dtype)
        group = self.groups[self.columns_group[column]]
        return group, group.get_column_index(column)
        
    def __remove_column__(self, column:str):
        group_name = self.columns_group.get(column, None)
        if (group_name is None):
            raise ValueError(f"The column {column} does not exist") 
               
        group = self.groups[group_name]
        i = group.remove_column(column)
        self.columns_group.pop(column)
        self.data[group.i] = np.concatenate((self.data[group.i][:, :i], self.data[group.i][:, i+1:]), axis=1)
        
        if (len(group) == 0):
            self.__remove_group__(group_name)
            
           
    def __rename_column__(self, old_column:str, new_column:str):
        group_name = self.columns_group.get(old_column, None)
        if (group_name is None):
            raise ValueError(f"The column {old_column} does not exist")
        
        if (new_column in self.columns_group):
            raise ValueError(f"The column {new_column} already exists")
        
        group = self.groups[group_name]
        group.rename_column(old_column, new_column)
        self.columns_group[new_column] = group_name
        self.columns_group.pop(old_column)
        
    def __set_column_group__(self, column:str, group_name:str):
        if column not in self.columns_group:
            raise ValueError(f"The column {column} does not exist")
        
        group, group_name = self.__get_group__(group_name)
        if (group is None):
            group = self.__create_group__(group_name)
        
        old_group_name = self.columns_group[column]
        old_group = self.groups[old_group_name]
        
        
        # remove from last group
        old_i = old_group.remove_column(column)
        # save data and remove column
        data = self.data[old_group.i][:, old_i:old_i+1]
        self.data[old_group.i] = np.concatenate((self.data[old_group.i][:, :old_i], self.data[old_group.i][:, old_i+1:]), axis=1)
        
        # add to new group 
        group.add_column(column)
        self.data[group.i] = np.concatenate((self.data[group.i], data), axis=1)
        
        # update the columns
        self.columns_group[column] = group_name
        

# |====================================================================================================================
# | OVERLOADS
# |====================================================================================================================

        
    def __len__(self):
        return len(self.data[0])
    
    def copy(self):
        res = RunLogger()
        res.data = [data.copy() for data in self.data]
        res.groups = {group.name: group.copy() for group in self.groups.values()}
        res.columns_group = self.columns_group.copy()
        res.file = self.file
        res.last_run_id = self.last_run_id
        return res
    
    def __to_str_array__(self):
        col_index = {}
        str_array = np.zeros((len(self) + 1, len(self.columns_group)), dtype="U64")
        # header
        header_widths = np.zeros((len(self.columns_group)), dtype=int)
        for i, (col, group) in enumerate(self.columns_group.items()):
            str_array[0, i] = col            
            col_index[col] = i
            header_widths[i] = len(col)

        # data
        data_widths = np.zeros((len(self.columns_group)), dtype=int)
        data_largest_widths = np.zeros((len(self.columns_group)), dtype=int)
        for i, (col, group) in enumerate(self.columns_group.items()):
            col_i = self.groups[group].get_column_index(col)
            data = self.data[self.groups[group].i]
            for l in range(len(self)):
                str_array[l+1, i] = str_(data[l, col_i])
                
                # compute the largest width
                if (len(str_array[l+1, i]) > data_widths[i]):
                    data_widths[i] = len(str_array[l+1, i])
                    data_largest_widths[i] = l
                    
        # align data on the unit digit
        for i, (col, group) in enumerate(self.columns_group.items()):
            group = self.groups[group]
            if (self.data[group.i].dtype == "U64" and col != "i"):
                continue
            
            col_i = group.get_column_index(col)
            largest = str_array[data_largest_widths[i]+1, i]
            unit_loc = unit_digit(largest)
            
            
            for l in range(len(self)):
                str_unit_loc = unit_digit(str_array[l+1, i])
                if (str_unit_loc < unit_loc):
                    diff = unit_loc - str_unit_loc
                    str_array[l+1, i] = " " * diff + str_array[l+1, i]
                str_array[l+1, i] = str_array[l+1, i] + " " * (data_widths[i] - len(str_array[l+1, i]))
                    
            
        
        col_widths = np.maximum(header_widths, data_widths)
        
        return str_array, col_index, col_widths, header_widths, data_widths, data_largest_widths
    
    def __compute_groups_width__(self, col_widths, col_index):
        groups_width = {}
        SPACES_BETWEEN = 3
        
        for group in self.groups.values():
            if (len(group) == 0):
                groups_width[group.name] = 0
                continue
            
            total_width = 0
            nb_columns = 0
            for i in range(len(group.columns)):
                if (group.columns[i] not in self.grouped_by):
                    total_width += col_widths[col_index[group.columns[i]]]
                    nb_columns += 1
            
            if (nb_columns == 0):
                groups_width[group.name] = 0
                continue
            
            total_width += (nb_columns - 1) * SPACES_BETWEEN
            
            if (total_width < len(group.name)):
                diff = len(group.name) - total_width
                # add one point of diff to the first smallest column
                for _ in range(diff):
                    min_i = -1
                    min_v = 1000000
                    for i in range(0, len(group.columns)):
                        col_i = col_index[group.columns[i]]
                        if col_widths[col_i] < min_v and group.columns[i] not in self.grouped_by:
                            min_v = col_widths[col_i]
                            min_i = col_i
                    col_widths[min_i] += 1
                    total_width += 1
                
            groups_width[group.name] = total_width
            
        return groups_width, col_widths
                
            
    def __compute_group_by__(self, col_widths, col_index, str_array:np.ndarray):
        grouped_by = self.grouped_by[::-1]
        group_by_widths = np.zeros((len(self.grouped_by)), dtype=int)
        groups_locs = []
        groups_names = []
        # reversed enumerate
        for i in range(len(grouped_by)):
            group_by = grouped_by[i]
            group_by_widths[i] = col_widths[col_index[group_by]]
            
            all_locs = []
            all_names = []
            zones = [(0, len(str_array))]
            if (len(groups_locs) > 0):
                zones = []
                for j in range(len(groups_locs[-1])):
                    for k in range(len(groups_locs[-1][j])-1):
                        zones.append((groups_locs[-1][j][k], groups_locs[-1][j][k+1]))
                        
            for zone_i in range(len(zones)):
                zone_start, zone_end = zones[zone_i]
                
                locs = []
                names = []
                col_i = col_index[group_by]
            
                locs.append(zone_start)
                names.append(str_array[zone_start, col_i])
                for l in range(zone_start, zone_end-1):
                    if (str_array[l+1, col_i] != str_array[l, col_i]):
                        locs.append(l+1)
                        names.append(str_array[l+1, col_i])
                locs.append(zone_end)
                all_locs.append(locs)
                all_names.append(names)
                
            groups_locs.append(all_locs)
            groups_names.append(all_names)
            
        # remove col_widths of group by
        col_widths = col_widths.copy()
        for i in range(len(grouped_by)):
            col_i = col_index[grouped_by[i]]
            col_widths[col_i] = 0
                
        return grouped_by, group_by_widths, groups_locs, groups_names, col_widths
    
    
        
            
    
    def __str__(self, title=""):
        # render like:
        # +-----------+--------------------------+
        # | run info  |       Metrics            |
        # +---+-------+---------------+----------+
        # | i | model | long_col_name | shrt     |
        # +---+-------+---------------+----------+
        # | 0 | 0.1   | 0.2           | 10000000 |
        # | 1 | 0.5   | 0.3           | 10000000 |
        # ...
        

        str_array, col_index, col_widths, header_widths, data_widths, data_largest_widths = self.__to_str_array__()
        
        groups_width, col_widths = self.__compute_groups_width__(col_widths, col_index)
        grouped_by, group_by_widths, groups_locs, groups_names, col_widths = self.__compute_group_by__(col_widths, col_index, str_array[1:])
        
       
            
        res = ""
        if (title != ""):
            res += title + "\n"
            
        # preprocessing
        group_line = h_line([groups_width[group] for group in self.groups.keys()]) + "\n"
        col_line = h_line(col_widths) + "\n"
        grouped_by_lines = []
        for i in range(len(grouped_by)):
            l = ""
            for j in range(0, i):
                l += "| " + " " * group_by_widths[j] + " "
            l += h_line(group_by_widths[i:])[:-1]
            grouped_by_lines.append(l)
            
        header_spacing = "" 
        if (len(grouped_by_lines) > 0):
            header_spacing = " " * (len(grouped_by_lines[0]))
            
        
        # render groups
        res += header_spacing+group_line+header_spacing
        for group in self.groups.values():
            if (groups_width[group.name] == 0): continue
            res += "| "
            total_spaces = groups_width[group.name] - len(group.name)
            res += " " * (total_spaces // 2)
            res += group.name
            res += " " * (total_spaces - total_spaces // 2)
            res += " "
        res += "|\n"
            
        # render header
        if (len(grouped_by) > 0):
            res += grouped_by_lines[0]
        res += col_line
        for i in range(len(grouped_by)):
            res += "| "
            total_spaces = group_by_widths[i] - len(grouped_by[i])
            res += " " * (total_spaces // 2)
            res += grouped_by[i]
            res += " " * (total_spaces - total_spaces // 2)
            res += " "
            
        for i, col in enumerate(str_array[0]):
            if (col_widths[i] == 0): continue
            res += "| "
            total_spaces = col_widths[i] - len(col)
            res += " " * (total_spaces // 2)
            res += col
            res += " " * (total_spaces - total_spaces // 2)
            res += " "
        res += "|\n"
        
        if (len(grouped_by) > 0):
            res += grouped_by_lines[0]
        res += col_line
        used_zone = [-1] * len(grouped_by)
        for row_i, row in enumerate(str_array[1:]):

            zone_changed_at_level = -1
            line_str = "| "
            for group_i in range(len(grouped_by)):
                
                for z in range(len(groups_locs[group_i])):
                    for p in range(len(groups_locs[group_i][z]) - 1):
                        if (row_i >= groups_locs[group_i][z][p] and row_i < groups_locs[group_i][z][p+1]):
                            
                            if (row_i == (groups_locs[group_i][z][p] + groups_locs[group_i][z][p+1]-1)//2):
                                total_spaces = group_by_widths[group_i] - len(groups_names[group_i][z][p])
                                line_str += " " * (total_spaces // 2)
                                line_str += groups_names[group_i][z][p]
                                line_str += " " * (total_spaces - total_spaces // 2)
                                line_str += " | "
                            else:
                                line_str += " " * group_by_widths[group_i]
                                line_str += " | "
                            
                            if (used_zone[group_i] == -1):
                                used_zone[group_i] = p
                            elif (used_zone[group_i] != p):
                                used_zone[group_i] = p
                                if (zone_changed_at_level == -1):
                                    zone_changed_at_level = group_i
                            break
            

            if (zone_changed_at_level != -1):
                res+= grouped_by_lines[zone_changed_at_level] + col_line
            res += line_str
            
            for i, col in enumerate(row):
                if (col_widths[i] == 0): continue
                
                total_spaces = col_widths[i] - len(col)
                res += " " * (total_spaces // 2)
                res += col
                res += " " * (total_spaces - total_spaces // 2)
                res += " | "
                
                
            res += "\n"
        
        if (len(grouped_by) > 0):
            res += grouped_by_lines[0]
        res += col_line
        
        return res

    
    def __repr__(self):
        return self.__str__()
    
    def __empty_like__(self, size:int):
        res = RunLogger()
        res.data = [np.zeros((size, self.data[i].shape[1]), dtype=self.data[i].dtype) for i in range(len(self.data))]
        res.groups = {group.name: group.copy() for group in self.groups.values()}
        res.columns_group = self.columns_group.copy()
        return res
    
# |====================================================================================================================
# | PUBLIC
# |====================================================================================================================

    def __add_metadata__(self, run:dict[str, float], group_name:dict[str, str]):
        # add the metadata
        dt = datetime.now()
        date = dt.strftime("%Y/%m/%d")
        time = dt.strftime("%H:%M:%S") 
        
        self.last_run_id += 1
        run["date"] = date
        group_name["date"] = self.ENTREE_GROUP
        run["time"] = time
        group_name["time"] = self.ENTREE_GROUP
        run["i"] = str(self.last_run_id)
        group_name["i"] = self.ENTREE_GROUP
        
        return run, group_name
        
    def add_run(self, run:dict[str, float], group_name:"str|dict[str, str]"=None, dtypes:dict[str, object]=None):
        if (group_name is None):
            group_name = {col: self.DEFAULT_GROUP for col in run.keys()}
        elif isinstance(group_name, str):
            group_name = {col: group_name for col in run.keys()}
        else:
            for col in run.keys():
                if col not in group_name:
                    group_name[col] = self.DEFAULT_GROUP 
                    
        groups = set(group_name.values())
        if (dtypes is None):
            dtypes = {group: np.float64 for group in groups}
        elif not isinstance(dtypes, dict):
            dtypes = {group: dtypes for group in groups}
        else:
            for group in groups:
                if group not in dtypes:
                    dtypes[group] = np.float64
        # remplace each dtype str by U64
        dtypes = {group: "U64" if dtype == str else dtype for group, dtype in dtypes.items()}
            

        if "model" not in run:
            raise ValueError("The run must have a 'model' key")
        
        self.__load__(self.file)
        run, group_name = self.__add_metadata__(run, group_name)
        
        # create columns
        used_columns:dict[str, tuple[Group, int]] = {}
        for col, value in run.items():
            group, i = self.__get_or_create_column__(col, group_name[col], dtypes.get(group_name[col], None))
            used_columns[col] = (group, i)
            
        # add a row in each group
        for i in range(len(self.data)):
            self.data[i] = np.concatenate((self.data[i], full((1, self.data[i].shape[1]), np.nan, dtype=self.data[i].dtype)), axis=0)
            
        # add the values
        for col, value in run.items():   
            group, i = used_columns[col] 
            self.data[group.i][-1, i] = value
        
        self.__save__(self.file)
        
        return self.last_run_id
        
    def remove_run(self, i:int):
        i = str(i)
        # find the run with the index i = i
        entree_group = self.groups[self.ENTREE_GROUP]
        col_i = entree_group.columns.index("i")
        find = False
        for row_i in range(len(self.data[entree_group.i])):
            if self.data[entree_group.i][row_i, col_i] == i:
                find = True
                break
            
        if (find):
            for group in self.groups.values():
                self.data[group.i] = np.concatenate((self.data[group.i][:row_i, :], self.data[group.i][row_i+1:, :]), axis=0)
                
        # check every column. if a column is empty remove it
        if (len(self) > 0):
            to_remove = []
            for group in self.groups.values():
                for i in range(len(group.columns)-1, -1, -1):
                    if (self.data[group.i].dtype == "U64"):
                        if np.all(self.data[group.i][:, i] == ""):
                            to_remove.append(group.columns[i])
                    else:
                        if np.all(np.isnan(self.data[group.i][:, i])):
                            to_remove.append(group.columns[i])

            for col in to_remove:
                self.__remove_column__(col)
        
        self.__save__(self.file)
        
    def remove_column(self, column:str):
        self.__remove_column__(column)
        self.__save__(self.file)
            

    def render(self, file:TextIO, title=""):
        file.write(self.__str__(title))
        
    
    def get_columns(self) -> list:
        return list(self.columns_group.keys())
        
        
        
        
# |====================================================================================================================
# | FILTERS
# |====================================================================================================================

        
    def group_by(self, column:str, inplace=True, reverse=False) -> Self:
        
        res = self if inplace else self.copy()
        
        group_name = res.columns_group.get(column, None)
        if (group_name is None):
            raise ValueError(f"The column {column} does not exist")
        
        col_i = res.groups[group_name].columns.index(column)
        
        groups:"dict[object, list[int]]" = {}
        for i in range(len(res)):
            value = res.data[res.groups[group_name].i][i, col_i]
            if value not in groups:
                groups[value] = []
            groups[value].append(i)
            
        # sort values
        groups_name = list(groups.keys())
        # if dtype is str sort normally but put "" at the end
        if (res.data[res.groups[group_name].i].dtype == "U64" and not reverse):
            groups_name = sorted(groups_name, key=lambda x: x if x != "" else "z"*64, reverse=reverse)
        else:
            groups_name = sorted(groups_name, reverse=reverse)
        
        
        order = []
        for name in groups_name:
            order.extend(groups[name])
            
        # reorder rows
        for group in res.groups.values():
            res.data[group.i] = res.data[group.i][order, :]
            
        res.grouped_by.append(column)
                
        return res
             
             
    def filter(self, columns:"list[str]", inplace=True) -> Self:      
        res = self if inplace else self.copy()
        
        # only keep the columns in the list
        for group in res.groups.values():
            for i in range(len(group.columns)-1, -1, -1):
                if group.columns[i] not in columns:
                    res.__remove_column__(group.columns[i])
                    
        return res
    
    def split_by(self, column:str) -> "list[RunLogger]":
        res = []
        
        group_name = self.columns_group.get(column, None)
        if (group_name is None):
            raise ValueError(f"The column {column} does not exist")
        
        col_i = self.groups[group_name].columns.index(column)
        
        groups:"dict[object, list[int]]" = {}
        for i in range(len(self)):
            value = self.data[self.groups[group_name].i][i, col_i]
            if value not in groups:
                groups[value] = []
            groups[value].append(i)
        
        for value, order in groups.items():
            run = self.copy()
            for group in run.groups.values():
                run.data[group.i] = run.data[group.i][order, :]
            res.append(run)
            
        return res
        
    def join(runs:"list[RunLogger]") -> "RunLogger":
        res = RunLogger()
        
        for r in range(0, len(runs)):
            
            for group in runs[r].groups.values():
                for col in group.columns:
                    res.__get_or_create_column__(col, group.name, runs[r].data[group.i].dtype)

        for r in range(0, len(runs)):
            for group_res in res.groups.values():
                res.data[group_res.i] = np.concatenate((res.data[group_res.i], np.full((len(runs[r]), len(group_res.columns)), np.nan, dtype=res.data[group_res.i].dtype)), axis=0)

                
            for group in runs[r].groups.values():
                group_res = res.groups[group.name]  

                for col in group.columns:
                    col_i = group_res.columns.index(col)
                    
                    res.data[group_res.i][-len(runs[r]):, col_i] = runs[r].data[group.i][:, group.columns.index(col)]
                    
        # for each group cast object into U64 TODO not normal
        for group in res.groups.values():
            if (res.data[group.i].dtype == object):
                res.data[group.i] = res.data[group.i].astype("U64")
            
        
        return res
    
    def get_best_groupes_by(self, column:str, group_by:str, maximize=False, inplace=True) -> "list[RunLogger]":
        groups = {}
        
        gb_group_name = self.columns_group.get(group_by, None)
        if (gb_group_name is None):
            raise ValueError(f"The column {group_by} does not exist")
        
        gb_col_i = self.groups[gb_group_name].columns.index(group_by)
        
        col_group_name = self.columns_group.get(column, None)
        if (col_group_name is None):
            raise ValueError(f"The column {column} does not exist")
        
        col_col_i = self.groups[col_group_name].columns.index(column)
        
        for i in range(len(self)):
            gb_value = self.data[self.groups[gb_group_name].i][i, gb_col_i]
            col_value = self.data[self.groups[col_group_name].i][i, col_col_i]
            
            if gb_value not in groups:
                groups[gb_value] = (-1, None)
            
            if (groups[gb_value][0] == -1 or (maximize and col_value > groups[gb_value][1]) or (not maximize and col_value < groups[gb_value][1])):
                groups[gb_value] = (i, col_value)
            
        
        # sort groups by col_value
        groups = sorted(groups.items(), key=lambda x: x[1][1], reverse=maximize)
        if (inplace):
            rows = [g[1][0] for g in groups]
            for group in self.groups.values():
                self.data[group.i] = self.data[group.i][rows, :]
            return self
            
        else:
            res = self.__empty_like__(len(groups))
            
            for i, g in enumerate(groups):
                i_run = g[1][0]
                
                for group in self.groups.values():
                    res.data[group.i][i, :] = self.data[group.i][i_run, :]
                    
            return res
        
    def where(self, column:str, eq:object=None, gt:object=None, lt:object=None, inplace=True) -> Self:
        res = self if inplace else self.copy()
        
        group_name = res.columns_group.get(column, None)
        if (group_name is None):
            raise ValueError(f"The column {column} does not exist")
        
        col_i = res.groups[group_name].columns.index(column)
        
        rows_i = []
        for i in range(len(res)):
            value = res.data[res.groups[group_name].i][i, col_i]
            if ((eq is None or value == eq) and (gt is None or value > gt) and (lt is None or value < lt)):
                rows_i.append(i)
                
        for group in res.groups.values():
            res.data[group.i] = res.data[group.i][rows_i, :]
            
        return res
        
        
         
        
# |====================================================================================================================
# | SAVE/LOAD
# |====================================================================================================================

    
    def __save__(self, file):
        if (file is None):
            return
        
        with open(file, "wb") as f:
            
            f.write(np.int32(self.last_run_id).tobytes())
            f.write(np.int8(len(self.groups)).tobytes())
            for group in self.groups.values():
                f.write(group.name.encode("utf-8").ljust(128))
                dtype = int(self.data[group.i].dtype == np.float64)
                f.write(np.int8(dtype).tobytes())
                f.write(np.int16(len(group.columns)).tobytes())
                for col in group.columns:
                    f.write(col.encode("utf-8").ljust(64))
                f.write(np.int32(self.data[group.i].shape[0]).tobytes())
                if dtype == 1:
                    f.write(self.data[group.i].tobytes())
                else:
                    f.write(self.data[group.i].astype("S64").tobytes())
                
            
    
    def __load__(self, file):
        if (file is None):
            return self
        if not os.path.exists(file):
            return self
        
        with open(file, "rb") as f:
            self.last_run_id = np.frombuffer(f.read(4), dtype=np.int32)[0]
            nb_groups = np.frombuffer(f.read(1), dtype=np.int8)[0]
            self.groups = {}
            self.data = []
            for i in range(nb_groups):
                group_name = f.read(128).decode("utf-8").strip()
                dtype = np.frombuffer(f.read(1), dtype=np.int8)[0]
                nb_columns = np.frombuffer(f.read(2), dtype=np.int16)[0]
                columns = [f.read(64).decode("utf-8").strip() for _ in range(nb_columns)]
                nb_rows = np.frombuffer(f.read(4), dtype=np.int32)[0]
                if dtype == 1:
                    self.data.append(np.frombuffer(f.read(8 * nb_columns * nb_rows), dtype=np.float64).reshape((nb_rows, nb_columns)))
                else:
                    self.data.append(np.frombuffer(f.read(64 * nb_columns * nb_rows), dtype="S64").reshape((nb_rows, nb_columns)))
                    # convert to simple string
                    self.data[-1] = self.data[-1].astype("U64")
                
                self.groups[group_name] = Group(group_name, i)
                self.groups[group_name].columns = columns
                self.groups[group_name].columns_indices = {col: i for i, col in enumerate(columns)}
                
                
                for col in columns:
                    self.columns_group[col] = group_name
        
        return self
             
if __name__ == "__main__":
    
    LOGGER = RunLogger("../_Artifacts/logs.pkl")
    file = open("../_Artifacts/log.txt", "w")
    
    loggers_ = LOGGER.split_by("EVAL")
    loggers:"list[RunLogger]" = []
    for i in range(len(loggers_)):
        loggers.extend(loggers_[i].split_by("TRAIN"))
    for i in range(len(loggers)):
        loggers[i].get_best_groupes_by("RMSE_24", "model", maximize=False)
    logger:RunLogger = RunLogger.join(loggers)    

    logger.group_by("TRAIN").group_by("EVAL")


    logger.render(file, "Best models by dataset")
    file.write("\n\n")
    LOGGER.render(file, "All models")
    file.close()
    
