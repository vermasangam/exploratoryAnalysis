import pandas as pd
import numpy as np
from matplotlib import use
use("Agg")
import matplotlib.pyplot as plt
import csv,argparse,os,shutil

def plot_final_graph(inputdf, target_rate,population,variable,title):
    fig = plt.figure(figsize=(15,10))
    ax1 = fig.add_subplot(221)
    ax1.bar(range(0,len(inputdf)), inputdf[population],color='c',label='#Customer',width=0.3)
    ax1.set_ylabel('Population')
    ax1.set_xlabel(variable)
    plt.title(title)
    plt.xticks(range(0,len(inputdf)), inputdf.index, rotation=45)
    
    ax1.text(1.1,0.30,'Lower',transform=ax1.transAxes)
    ax1.text(1.1,0.24,'Targets',transform=ax1.transAxes)
    ax1.annotate('', xy=(1.15,0.05), xycoords='axes fraction', xytext=(1.15,0.22),arrowprops=dict(arrowstyle="simple", color='r'))
    
    ax1.text(1.1,0.70,'Higher',transform=ax1.transAxes)
    ax1.text(1.1,0.65,'Targets',transform=ax1.transAxes)
    ax1.annotate('', xy=(1.15,0.94), xycoords='axes fraction', xytext=(1.15,0.75),arrowprops=dict(arrowstyle="simple", color='g'))
    
    ax2 = ax1.twinx()   
    ax2.plot(range(0,len(inputdf)), inputdf[target_rate],color='g',label='Log Odds')
    ax2.set_ylabel('Log Odds')
    fig.savefig('Plots/%s.png'%variable, bbox_inches = 'tight')

def get_risk_table_categorical(inputdf, variable , target, variable_desc, cutoff = 1000):
    df1 = inputdf[[variable,target]]
    df1[variable][df1[variable].isnull()]="MISSING"
    
    freq1 = df1[variable].value_counts()
    dict1 = {}
    for key,value in zip(list(freq1.index), list(freq1)):
        if value < cutoff:
            dict1[key] = "OTHER"
        else:
            dict1[key] = str(key)
 
    df1["Clean_%s"%variable] = df1[[variable]].applymap(dict1.get)
    df2 = pd.crosstab(df1["Clean_%s"%variable], df1[target],colnames=[target])
    df2.columns = ['Bads','Goods']
    df2['LOG ODDS'] = np.log(df2['Goods']/df2['Bads'])
    df2['Population Percentage'] = (df2['Goods']+df2['Bads'])/(df2['Goods']+df2['Bads']).sum()
    df2.sort_values('LOG ODDS', inplace=True)
    df2['Variable'] = variable
    
    plot_final_graph(df2,'LOG ODDS','Population Percentage',variable,variable_desc)
    return df2

def get_next_range(arr,group_range,start):
    if group_range + start >=100:
        return 100
    elif (100 - group_range/2) < start + group_range:
        return 100
    elif arr[-1] == arr[start]:
        return 100
    elif (arr[start+group_range] == arr[start]) or (arr[start] < 0):
        return np.max([np.min(np.where(arr > arr[start])),np.min(np.where(arr >= 0))])
    else:
        return group_range + start

def get_risk_table_numeric(df,var,target,groups,title,special_values):
    df1 = df[[var,target]]
    df2 = df[[var,target]]
    if len(special_values) > 0:
        df1.replace(special_values,[np.nan for x in special_values],inplace=True)
    df3 = df2[df1[var].isnull()]
    df3.fillna(-999,inplace=True)
    df1.dropna(inplace=True)
    
    bins = []
    begin_traverse = 0
    percentiles = np.around(np.array([np.percentile(df1[var],p) for p in range(0,100)]), decimals = 5)
    group_range = int(100/groups)
    
    while (begin_traverse <100):
        bins += [percentiles[begin_traverse]]
        begin_traverse = get_next_range(percentiles,group_range,begin_traverse)
    
    bins.append(np.max(df1[var])+1)
    df1.loc[:,'BINS'] = pd.cut(df1[var], bins, right=False, labels=None, retbins=False, precision=3, include_lowest=False)
    df4 = df1.groupby('BINS').agg([ np.mean, sum, np.size ])
    df5 = df4[target][['sum','size']]
    df5['Bin Mean'] = df4[var][['mean']]
    df5['LOG ODDS'] = np.log(df5['sum']/(df5['size']-df5['sum']))
    df5['Population Percentage'] = 1.0*df5['size']/len(df)
    df5.sort_values(by='Bin Mean', inplace=True) 
    
    if len(df3) > 0:
        df4 = df3.groupby(var).agg([ np.mean, sum, np.size ])
        df6 = df4[target][['sum','size']]
        df6['Bin Mean'] = df4.index
        df6['LOG ODDS'] = np.log(df6['sum']/(df6['size']-df6['sum']))
        df6['Population Percentage'] = 1.0*df6['size']/len(df)
        df6.sort_values('Bin Mean', inplace=True) 
        df5 = pd.concat([df6,df5])
    plot_final_graph(df5,'LOG ODDS','Population Percentage',var,title)
    df5['Variable'] = var
    return df5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ifile", help="Input file")
    parser.add_argument("--d", help="Delimiter. Default: Comma")
    parser.add_argument("--target", help="Dependent Variable. Default: target")
    parser.add_argument("--invars", help="File with independent variables")
    parser.add_argument("--numeric_bins", help="Bins for numerical variables. Defualt: 10")
    parser.add_argument("--categorical_threshold", help="Threshold below which categorical values grouped to others. Defualt: 0.05")
    parser.add_argument("--output_csv", help="Path for output summary csv. If not given, no csv output saved")
    
    args = parser.parse_args()
    infile = args.ifile
    target = args.target if args.target else 'target'
    delimiter = args.d if args.d else ','
    numeric_bins = args.numeric_bins if args.numeric_bins else 10
    categorical_threshold = args.categorical_threshold if args.categorical_threshold else 0.05
    invars = [x.strip().split(':')[0] for x in open(args.invars,'rb')]
    labels = [x.strip().split(':')[-1] for x in open(args.invars,'rb')]
    
    df=pd.read_csv(infile,delimiter=delimiter,usecols=invars+[target])
    
    if os.path.exists('Plots'):
        shutil.rmtree('Plots')
    os.makedirs('Plots')   
    outdf = [] 
    for col,label in zip(invars,labels):
        if df[col].dtypes==np.int64 or df[col].dtypes==np.float64:
            outdf.append(get_risk_table_numeric(df,col,target,numeric_bins,label,[]))
        else:
            outdf.append(get_risk_table_categorical(df, col ,target, label,int(categorical_threshold*len(df))))
        
    if args.output_csv:
        pd.concat(outdf).to_csv(args.output_csv)
        
