import os
import csv
import json
import pandas as pd


def csv_creator():
    for folder in os.listdir("classification_reports"):
        for json_file in os.listdir(f"classification_reports/{folder}"):
            with open(f"classification_reports/{folder}/{json_file}") as f:
                report = json.load(f)
                report.update({"accuracy": {"precision": None, "recall": None, "f1-score": report["accuracy"], "support": report['macro avg']['support']}})
                df = pd.DataFrame(report).transpose()
                df.to_csv(f'{json_file}.csv')


def fix_extensions():
    folders = ["seed42", "seed36836"]
    for folder in folders:
        for exper in os.listdir(folder):
            for file in os.listdir(f"{folder}/{exper}"):
                _, ext = os.path.splitext(file)
                if(ext == ""):
                    os.rename(f"{folder}/{exper}/{file}",f"{folder}/{exper}/{file}.json")


def create_csv_files_serverdata():
    folders = ["seed42", "seed36836"]
    for folder in folders:
        for exper in os.listdir(folder):
            
            with open(f"{folder}/{exper}/evaluate_metrics_clientapp.json") as client_eval:
                with open(f"{folder}/{exper}/evaluate_metrics_serverapp.json") as server_eval:
                    cl_obj = json.load(client_eval)
                    ser_obj = json.load(server_eval)
                    content = [None for i in range(0,151)]
                    for key in ser_obj.keys():
                        
                        if(key == "0"):
                            ser_obj[key].update({"eval_loss":None,"eval_acc":None})
                        else:
                            ser_obj[key].update(cl_obj[key])
                        content[int(key)] = ser_obj[key]
                    df = pd.DataFrame(content).transpose()
                    df.to_csv(f'{exper}_metrics.csv')
                    
csv_creator()
fix_extensions()              
create_csv_files_serverdata()
                        

                    