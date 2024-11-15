import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as sps
from scipy.optimize import minimize
import argparse

from project import Normal, Bernoulli, Poisson

parse = argparse.ArgumentParser(description = "Instructions")
parse.add_argument('source', help = "data source", 
                   choices = ['loc', 'r', 'sm'])
parse.add_argument('--dset', help = "dataset")
parse.add_argument('--package', help = "package in R datasets")
parse.add_argument('--path', help = "location/ path of csv file")
parse.add_argument('--Y', help = "select Y column in data")
parse.add_argument('--X', nargs = '*', help = "select X columns in data")
parse.add_argument('--add_intercept', action = 'store_true', help = "add intercept to X")   
parse.add_argument('--model', help = "distribution of responses : normal, bernoulli, poisson",
                   choices = ["normal","bernoulli","poisson"]) 
parse.add_argument('--predictor', nargs = '*', help = "predict new values", type = float)

def load_data(args):
    if args.source == "r":
        data = sm.datasets.get_rdataset(args.dset, package = args.package).data
    elif args.source == "sm":
        data = getattr(sm.datasets, args.dset).load_pandas().data    
    elif args.source == 'loc':
        data = pd.read_csv(args.path)    
    else:
        raise ValueError ("No data found")
    return data

def test():
    args = parse.parse_args()
    try:
        data = load_data(args)
        print(data.info())        
        Y = data[args.Y]    
        X = data[args.X]
    except ValueError as e:
        print (e)
    
    if args.add_intercept:
        X = sm.add_constant(X)

    model_selection = {'normal': Normal, 'bernoulli': Bernoulli, 'poisson': Poisson}
    
    model = model_selection[args.model]
    beta = model(X,Y)
    beta.mle()
    if args.predictor:
        print('Expected value Y: ',beta.predict(args.predictor))           

test()

# Example terminal command line arguments:
## r --dset Duncan --package carData --Y income --X education prestige --add_intercept --model normal --predictor 1 90 80
## sm --dset spector --Y GRADE --X GPA TUCE PSI --add_intercept --model bernoulli --predictor 1 3.3 20 1
## loc --path https://raw.githubusercontent.com/BI-DS/GRA-4152/refs/heads/master/warpbreaks.csv --Y breaks --X wool tension --add_intercept --model poisson --predictor 1 0 2