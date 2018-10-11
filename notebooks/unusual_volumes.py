import csv
import json


import csv
import json



def top_stocks():
    csvfile = open('csv_stocks.csv', 'r')
    
    fieldnames = ("Symbol", "Name", "Industry", "Sector", "Exchange", "Cap mln", "Last", "Change", "Change %", "Volume")

    reader = csv.DictReader( csvfile, fieldnames)
    json_values = ""
    for row in reader:
        json_values += (json.dumps(row))
        json_values += "\n"
        
        
    return json_values

    
