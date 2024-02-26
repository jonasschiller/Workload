import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import set_matplotlib_formats

def init_plotting():
    label_size = 8
    # plt.style.use('fivethirtyeight')
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r'\usepackage{amsmath}')
    
    pr ={"axes.labelsize": 8,               # LaTeX default is 10pt font.
        "font.size": 8,
        "legend.fontsize": 8,               # Make the legend/label fonts
        "xtick.labelsize": 8,               # a little smaller
        "ytick.labelsize": 8,
        "errorbar.capsize": 2.5,
        "font.family": "serif",
        "font.serif": [],                   # blank entries should cause plots
        "font.sans-serif": [],    
        }
    for k, v in pr.items():
        mpl.rcParams[k] = v
    set_matplotlib_formats('retina')
    mpl.rcParams["font.family"] = "computer modern roman"
    # mpl.rcParams["font.serif"] = ["STIX"]
    mpl.rcParams["mathtext.fontset"] = "stix"

def load_data(resultspath,frameworks,experiment):
    data=[]
    for framework in frameworks:
        frame=pd.read_csv(resultspath+'\\PrimitiveOperations\\'+ framework + '\\' + experiment + '\\data\short_results.csv' ,header=0,sep=';')
        frame['framework']=framework
        data.append(frame)
        if framework=="MPyC":
            # assuming your dataframe is df and the time column is 'time'
            data[-1]['runtime_internal(s)'] = pd.to_datetime(data[-1]['runtime_internal(s)'], format='%H:%M:%S.%f').dt.time

            # convert time to seconds
            data[-1]['runtime_internal(s)'] = data[-1]['runtime_internal(s)'].apply(lambda t: t.hour*3600 + t.minute*60 + t.second + t.microsecond*1e-6)
            data[-1]['RAMused(MB)']= data[-1]['RAMused(MB)']/1024
        if framework=="HPMPC":
            factor = 1
            if data[-1]['splitroles'].max() == 1:
                factor = 6	
            elif data[-1]['splitroles'].max() == 2:
                factor = 24
            else:
                factor=1
            data[-1]['runtime_external(s)'] = pd.to_numeric(data[-1]['runtime_external(s)'])* data[-1]['threads'].max()*factor
    return data

def plot_line(x_values,y_values,labels,file_path,experiment,x_axis,y_axis):
    fig, ax = plt.subplots()
    max_value=0
    for x_value, y_value,label in zip(x_values,y_values,labels):
        max_value=max(np.max(y_value),max_value)
        ax.plot(x_value,y_value, label=label, marker='s')
    ax.set_xlabel(x_axis)
    plt.yticks(np.arange(0, round(max_value,2), round(max_value/10,2)))
    #ax.set_yscale('log')  # Set y-axis as logarithmic
    ax.set_ylabel(y_axis)
    ax.set_title(y_axis + ' for '+ experiment[3:])
    ax.figure.set_size_inches(10, 6)
    ax.legend()
    fig.tight_layout()
    plt.savefig(file_path,dpi=300,bbox_inches='tight')
    plt.show()

def basic_bar_plot(y_values,labels,experiment, titel, y_label):
    fig, ax = plt.subplots()
    #ax.set_yscale('log')  # Set y-axis as logarithmic
    bars=ax.bar(labels,y_values)
    ax.set_ylabel(y_label)
    ax.set_title(titel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')

    # Second, remove the ticks as well.
    ax.tick_params(bottom=False, left=False)

    # Third, add a horizontal grid (but keep the vertical grid hidden).
    # Color the lines a light gray as well.
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)

    fig.tight_layout()
    # Rotate labels
    ax.legend()
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, yval, ha='center', va='bottom',weight='bold')

    plt.xticks(rotation='vertical')
    plt.savefig(experiment+'_bar.png',dpi=300, bbox_inches='tight')
    plt.show()

def get_latency(data):
    y_values = []
    x_values = []
    labels = []
    for i in range(len(data)):
        if(data[i]['framework'][0] in 'HPMPC'):
            for function in data[i]['function'].unique():
                y_values.append(data[i].loc[(data[i]['function']==function) & (data[i]['bandwidths(Mbs)'] == 25000) & (data[i]['packetdrops(%)'] == 0) & (data[i]['preprocess']==0), ['runtime_external(s)']].to_numpy().reshape(-1)[0:4])
                x_values.append(data[i]['latencies(ms)'].unique()[0:4])
                labels.append("HPMPC passive adversary " +str(function)) 
        elif(data[i]['framework'][0] in 'MPyC'):
            y_values.append(data[i].loc[(data[i]['bandwidths(Mbs)'] == 25000) & (data[i]['packetdrops(%)'] == 0), ['runtime_external(s)']].to_numpy().reshape(-1)[0:4])
            x_values.append(data[i]['latencies(ms)'].unique()[0:4])
            labels.append("MPyC")
        elif(data[i]['framework'][0] in 'MOTION'):
            for protocol in data[i]['protocol'].unique():
                y_values.append(data[i].loc[(data[i]['bandwidths(Mbs)'] == 25000) &(data[i]['packetdrops(%)'] == 0) & (data[i]['protocol']==protocol), ['runtime_external(s)']].to_numpy().reshape(-1)[:4])
                x_values.append(data[i]['latencies(ms)'].unique()[0:4])
                labels.append("MOTION " + protocol)
        else:
            for protocol in data[i]['protocol'].unique():
                if protocol not in ["ps-rep-field","ps-rep-ring"]:
                    y_values.append(data[i].loc[(data[i]['bandwidths(Mbs)'] == 25000) &(data[i]['packetdrops(%)'] == 0) & (data[i]['protocol']==protocol), ['runtime_external(s)']].to_numpy().reshape(-1)[0:4])
                    x_values.append(data[i]['latencies(ms)'].unique()[0:4])
                    labels.append("MP-SPDZ " + protocol)
    return x_values,y_values,labels

def get_packetdrop(data):
    y_values = []
    x_values = []
    labels = []
    for i in range(len(data)):
        if(data[i]['framework'][0] in 'HPMPC'):
            for function in data[i]['function'].unique():
                if function == 41:
                    y_values.append(data[i].loc[(data[i]['function']==function) & (data[i]['bandwidths(Mbs)'] == 25000) & (data[i]['latencies(ms)'] == 0) & (data[i]['preprocess']==0), ['runtime_external(s)']].to_numpy().reshape(-1)[0:4])
                    x_values.append(data[i]['packetdrops(%)'].unique()[0:4])
                    labels.append("HPMPC passive adversary")
        elif(data[i]['framework'][0] in 'MPyC'):
            y_values.append(data[i].loc[(data[i]['bandwidths(Mbs)'] == 25000) & (data[i]['latencies(ms)'] == 0), ['runtime_external(s)']].to_numpy().reshape(-1)[0:4])
            x_values.append(data[i]['packetdrops(%)'].unique()[0:4])
            labels.append("MPyC")
        else:
            for protocol in data[i]['protocol'].unique():
                y_values.append(data[i].loc[(data[i]['bandwidths(Mbs)'] == 25000) & (data[i]['latencies(ms)'] == 0) & (data[i]['protocol']==protocol), ['runtime_external(s)']].to_numpy().reshape(-1)[0:4])
                x_values.append(data[i]['packetdrops(%)'].unique()[0:4])
                labels.append(data[i]['framework'][0]+ " " + protocol)
    return x_values,y_values,labels

    
def get_bandwidth(data):
    y_values = []
    x_values = []
    labels = []
    for i in range(len(data)):
        if(data[i]['framework'][0] in 'HPMPC'):
            for function in data[i]['function'].unique():
                if function == 41:
                    y_values.append(data[i].loc[(data[i]['function']==function) & (data[i]['packetdrops(%)'] == 0) & (data[i]['latencies(ms)'] == 0) & (data[i]['preprocess']==0), ['runtime_external(s)']].to_numpy().reshape(-1)[0:4])
                    x_values.append(data[i]['bandwidths(Mbs)'].unique()[0:4])
                    labels.append("HPMPC passive adversary")
        elif(data[i]['framework'][0] in 'MPyC'):
            y_values.append(data[i].loc[(data[i]['packetdrops(%)'] == 0) & (data[i]['latencies(ms)'] == 0), ['runtime_external(s)']].to_numpy().reshape(-1)[0:4])
            x_values.append(data[i]['bandwidths(Mbs)'].unique()[0:4])
            labels.append("MPyC")
        else:
            for protocol in data[i]['protocol'].unique():
                y_values.append(data[i].loc[(data[i]['packetdrops(%)'] == 0) & (data[i]['latencies(ms)'] == 0) & (data[i]['protocol']==protocol), ['runtime_external(s)']].to_numpy().reshape(-1)[0:4])
                x_values.append(data[i]['bandwidths(Mbs)'].unique()[0:4])
                if protocol in "arithmetic_gmw":
                    protocol="arithmetic gmw"
                if protocol in "boolean_gmw":
                    protocol="boolean gmw"
                if protocol in "boolean_bmr":
                    protocol="boolean bmr"
                labels.append(data[i]['framework'][0]+ " " + protocol)
    return x_values,y_values,labels

