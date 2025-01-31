import os
from numpy_typing import np, ax
from typing import Self, TextIO
from datetime  import datetime
import time

def str_(x:float):
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

class RunLogger:

    data:np.float64_2d[ax.sample, ax.feature]
    columns:"list[str]"
    entree:"list[str]"
    ENTREE = ["i", "model", "date", "time"]
    
    def __init__(self, file=None):
        
        self.data = np.zeros((0,0), dtype=np.float64)
        self.columns = []
        self.entree = []
        
        self.file = file
        self.__load__(file)
        
        
    def __insert_column__(self, column:str):
        i = 0
        while i < len(self.columns) and self.columns[i] < column:
            i += 1
        self.columns.insert(i, column)
        self.data = np.concatenate([self.data, np.full((self.data.shape[0], 1), np.nan, dtype=np.float64)], axis=1)
        return i
        
    def __remove_column__(self, column:str):
        i = self.columns.index(column)
        self.columns.pop(i)
        self.data = np.concatenate((self.data[:, :i], self.data[:, i+1:]), axis=1)
        
    def __rename_column__(self, old_column:str, new_column:str):
        new_i = self.insert_column(new_column)
        old_i = self.columns.index(old_column)
        self.data[:, new_i] = self.data[:, old_i]
        self.remove_column(old_column)
        
        
    def add_run(self, run:dict[str, float]):
        if "model" not in run:
            raise ValueError("The run must have a 'model' key")
        
        self.__load__(self.file)
        
        dt = datetime.now()
        date = dt.strftime("%Y/%m/%d")
        time = dt.strftime("%H:%M:%S") 
        
        columns_i = {}
        for key in run.keys():
            if key == "model":
                continue
            
            if key not in self.columns:
                # add the column but keep the order sorted
                i = self.__insert_column__(key)
                for k in columns_i:
                    if columns_i[k] >= i:
                        columns_i[k] += 1
                columns_i[key] = i

                
            else:
                columns_i[key] = self.columns.index(key)
                
        # add the run to the dataframe
        data_i = np.full((1, len(self.columns)), np.nan, )
        for key, value in run.items():
            if key != "model":
                data_i[0, columns_i[key]] = value
        
        self.data = np.concatenate((self.data, data_i), axis=0)
        self.entree.append([run["model"], len(self.data)-1, date, time])
        
        self.__save__(self.file)
        
    def render(self, file:TextIO, title=""):
        
        # render like:
        # +---+-------+---------------+----------+
        # | i | model | long_col_name | shrt     |
        # +---+-------+---------------+----------+
        # | 0 | 0.1   | 0.2           | 10000000 |
        # | 1 | 0.5   | 0.3           | 10000000 |
        # ...
        
        def h_line(col_widths):
            str_repr = "+"
            for width in col_widths:
                str_repr += "-" * (width + 2) + "+"
            return str_repr + "\n"
        
        
        str_repr = []
        
        # header
        str_repr.append(["i", "model", "date", "time"])
        for col in self.columns:
            str_repr[-1].append(col)     
            
        # data
        for i, run in enumerate(self.entree):
            str_repr.append([str_(run[1]), run[0], str_date_to_str(run[2]), run[3]])
            for j in range(len(self.columns)):
                str_repr[-1].append(str_(self.data[i, j]))
                
        
        # compute column widths
        col_widths = [0] * len(str_repr[0])
        for row in str_repr:
            for i in range(len(row)):
                col_widths[i] = max(col_widths[i], len(row[i]))
        
        # render header
        if (title != ""):
            file.write(title + "\n")
        file.write(h_line(col_widths))
        for i, col in enumerate(str_repr[0]):
            file.write("| ")
            file.write(col)
            file.write(" " * (col_widths[i] - len(col)))
            file.write(" ")
        file.write("|\n")
        file.write(h_line(col_widths))
        
        # render data
        for row in str_repr[1:]:
            for i, col in enumerate(row):
                file.write("| ")
                file.write(col)
                file.write(" " * (col_widths[i] - len(col)))
                file.write(" ")
            file.write("|\n")
            
        file.write(h_line(col_widths))
        file.write("\n")
        
        
        
        
    def sort_by(self, column:str, desc=True) -> Self:
        sort_by_entree = column in self.ENTREE
        if sort_by_entree:
            column_i = self.ENTREE.index(column)
        else:
            column_i = self.columns.index(column)
            
        
        indices = list(range(len(self.entree)))
        
        if sort_by_entree:
            if (column == "date" or column == "time"):
                indices.sort(key=lambda i: self.entree[i][2] + self.entree[i][3], reverse=desc)
            else:
                indices.sort(key=lambda i: self.entree[i][column_i], reverse=desc)
        else:
            
            indices_without_nan = [i for i in indices if not np.isnan(self.data[i, column_i])]
            indices_nan = [i for i in indices if np.isnan(self.data[i, column_i])]
            
            indices_without_nan.sort(key=lambda i: self.data[i, column_i], reverse=desc)
            
            indices = indices_without_nan + indices_nan
        
        data = np.zeros((0, len(self.columns)))
        entree = []
        
        for i in indices:
            data = np.concatenate((data, self.data[i:i+1, :]), axis=0)
            entree.append(self.entree[i])
            
        res = RunLogger()
        res.data = data
        res.columns = self.columns
        res.entree = entree
        
        return res
            
        
    def get_runs(self, model:str) -> Self:
        columns = set()
        data = np.zeros((0, len(self.columns)))
        entree = []
        
        for i, run in enumerate(self.entree):
            run_model, run_index, date, time = run
            if run_model == model:
                data = np.concatenate((data, self.data[i:i+1, :]), axis=0)
                used_columns = [j for j in range(len(self.columns)) if not np.isnan(self.data[i, j])]
                columns.update(used_columns)
                entree.append([run_model, run_index, date, time])
                
        data = data[:, list(columns)]
        
        res = RunLogger()
        res.data = data
        res.columns = [self.columns[j] for j in columns]
        res.entree = entree
        
        return res
    
    def get_best_run_per_model(self, metric, maximize=True) -> Self:
        best_runs = {}
        
        metric_i = self.columns.index(metric)        
        
        for i, run in enumerate(self.entree):
            run_model = run[0]
            if run_model not in best_runs:
                best_runs[run_model] = i
            else:
                if maximize:
                    if self.data[i, metric_i] > self.data[best_runs[run_model], metric_i]:
                        best_runs[run_model] = i
                else:
                    if self.data[i, metric_i] < self.data[best_runs[run_model], metric_i]:
                        best_runs[run_model] = i
                        
        columns = set()
        data = np.zeros((0, len(self.columns)))
        entree = []
        
        for model, i in best_runs.items():
            data = np.concatenate((data, self.data[i:i+1, :]), axis=0)
            used_columns = [j for j in range(len(self.columns)) if not np.isnan(self.data[i, j])]
            columns.update(used_columns)
            entree.append(self.entree[i])
            
        data = data[:, list(columns)]

        res = RunLogger()
        res.data = data
        res.columns = [self.columns[j] for j in columns]
        res.entree = entree       
            
        return res.sort_by(metric, desc=maximize)
    
    def __save__(self, file):
        if (file is None):
            return
        
        with open(file, "wb") as f:
            # nb_columns (int32)
            # columns (str 128)
            # nb_entree (int32)
            # entree (str 128, int32, str 128, str 128)
            # data (float64 * nb_columns * nb_entree)
            
            f.write(np.int32(len(self.columns)).tobytes())
            for col in self.columns:
                f.write(col.ljust(64).encode("utf-8"))
            
            f.write(np.int32(len(self.entree)).tobytes())
            for run in self.entree:
                f.write(run[0].ljust(64).encode("utf-8"))
                f.write(np.int32(run[1]).tobytes())
                f.write(run[2].ljust(16).encode("utf-8"))
                f.write(run[3].ljust(16).encode("utf-8"))
                
            # convert to float64 and save
            f.write(self.data.astype(np.float64).tobytes())
                
            
    
    def __load__(self, file):
        if (file is None):
            return self
        if not os.path.exists(file):
            return self
        
        with open(file, "rb") as f:
            # nb_columns (int32)
            # columns (str 128)
            # nb_entree (int32)
            # entree (str 128, int32, str 128, str 128)
            # data (float64 * nb_columns * nb_entree)
            
            nb_columns = np.frombuffer(f.read(4), dtype=np.int32)[0]
            self.columns = []
            for i in range(nb_columns):
                self.columns.append(f.read(64).decode("utf-8").strip())
                
            nb_entree = np.frombuffer(f.read(4), dtype=np.int32)[0]
            self.entree = []
            for i in range(nb_entree):
                model = f.read(64).decode("utf-8").strip()
                index = np.frombuffer(f.read(4), dtype=np.int32)[0]
                date = f.read(16).decode("utf-8").strip()
                time = f.read(16).decode("utf-8").strip()
                self.entree.append([model, index, date, time])
            
            print(self.columns)
            print(self.entree)
            d = np.frombuffer(f.read(), dtype=np.float64)
            self.data = d.reshape((nb_entree, nb_columns))
            
        return self
             
if __name__ == "__main__":
    
    logger = RunLogger("./test.pkl")
    logger.add_run({"model": "model1", "loss": 0.1, "acc": 0.5})
    logger.render(open("test.txt", "w"))
    