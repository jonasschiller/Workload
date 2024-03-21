import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import set_matplotlib_formats

def init_plotting():
    #plt.style.use('fivethirtyeight')
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r'\usepackage{amsmath}')
    
    pr ={"axes.labelsize": 9,               # LaTeX default is 10pt font.
        "font.size": 9,
        "legend.fontsize": 7,               # Make the legend/label fonts
        "xtick.labelsize": 9,               # a little smaller
        "ytick.labelsize": 9,
        "errorbar.capsize": 2.5,
        "font.family": "STIXGeneral",
        "font.serif": [],                   # blank entries should cause plots
        "font.sans-serif": [],    
        }
    for k, v in pr.items():
        mpl.rcParams[k] = v
    #set_matplotlib_formats('retina')
    #mpl.rcParams["font.family"] = "computer modern serif"
    # mpl.rcParams["font.serif"] = ["STIX"]
    mpl.rcParams["mathtext.fontset"] = "stix"
    

def load_data(resultspath,frameworks,experiment,usecase):
    data=[]
    if not usecase:
        help="\\PrimitiveOperations\\"
    else:
        help="\\UseCases\\"
    for framework in frameworks:
        frame=pd.read_csv(resultspath+help+ framework + '\\' + experiment + '\\data\short_results.csv' ,header=0,sep=';')
        frame['framework']=framework
        data.append(frame)
        if framework=="MPyC":
            # assuming your dataframe is df and the time column is 'time'
            data[-1]['runtime_internal(s)'] = pd.to_datetime(data[-1]['runtime_internal(s)'], format='%H:%M:%S.%f').dt.time

            # convert time to seconds
            data[-1]['runtime_internal(s)'] = data[-1]['runtime_internal(s)'].apply(lambda t: t.hour*3600 + t.minute*60 + t.second + t.microsecond*1e-6)
            data[-1]['peakRAM(MiB)']= data[-1]['peakRAM(MiB)']/1024
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

def plot_line(x_values,y_values,labels,file_path,experiment,x_axis,y_axis,log=False,x_ticks=[]):
    fig, ax = plt.subplots()
    max_value=0
    for x_value, y_value,label in zip(x_values,y_values,labels):
        max_value=max(np.max(y_value),max_value)
        if label == "MOTION pas. adv./ dishst. maj. BMR":
            ax.plot(x_value,y_value, label=label, marker='s', color='black')
        else: 
            ax.plot(x_value,y_value, label=label, marker='s')
    ax.set_xlabel(x_axis, fontsize=15)
    #np.arange(0, round(max_value,2), round(max_value/10,2))
    plt.yticks(fontsize=10)
    if len(x_ticks) > 0:
        plt.xticks(x_ticks,fontsize=13)
    else:
        plt.xticks(fontsize=13)
    if log:
        ax.set_yscale('log')  # Set y-axis as logarithmic
    ax.set_ylabel(y_axis,fontsize=17)
    ax.set_title(y_axis + ' for '+ experiment[3:])
    ax.figure.set_size_cm(10, 6)
    ax.legend(fontsize=13)
    fig.tight_layout()
    plt.savefig(file_path,dpi=300,bbox_inches='tight')
    plt.show()
    
def plot_deviation_bandwidth(data,labels,file_path,experiment,x_axis,y_axis,log=False,x_ticks=[],right_push=0.56,up_push=1.132):
    fig, ax = plt.subplots()
    max_value=0
    for i in range(len(data)):
        max_value=max(data[i]['mean'].max(),max_value)
        sns.lineplot(x=data[i].index, y='mean', label=labels[i], data=data[i], marker="s")
        plt.fill_between(data[i].index, data[i]['mean'] - data[i]['std'], data[i]['mean'] + data[i]['std'], alpha=0.3)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    #np.arange(0, round(max_value,0), round(max_value/10,0))
    plt.yticks()
    if len(x_ticks) > 0:
        plt.xticks(x_ticks)
    else:
        plt.xticks()
    if log:
        ax.set_yscale('log')  # Set y-axis as logarithmic
    ax.figure.set_size_inches(5.91, 3.8)
    if len(labels) > 8:
        ax.set_title(y_axis + ' for '+ experiment[3:], y=1.33)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.37),framealpha=1, ncol=2)
    elif len(labels) > 6:
        ax.set_title(y_axis + ' for '+ experiment[3:], y=up_push - 0.021)
        ax.legend(loc='upper center', bbox_to_anchor=(right_push, up_push),framealpha=0.8, ncol=2)
    else:
        ax.set_title(y_axis + ' for '+ experiment[3:], y=up_push-0.04)
        ax.legend(loc='upper center', bbox_to_anchor=(right_push, up_push),framealpha=1, ncol=2)
    ax.figure.set_size_inches(5.91, 3.8)
    ax.grid(True, which='major', axis='y', linestyle='--', color='lightgray')
    #plt.tick_params(axis='y', which='both', right=True, left=True, labelright=True)
    fig.tight_layout()
    plt.savefig(file_path,dpi=300,bbox_inches='tight')
    plt.show()
    
def plot_deviation(data,labels,file_path,experiment,x_axis,y_axis,log=False,x_ticks=[]):
    fig, ax = plt.subplots()
    max_value=0
    for i in range(len(data)):
        max_value=max(data[i]['mean'].max(),max_value)
        sns.lineplot(x=data[i].index, y='mean', label=labels[i], data=data[i], marker="s")
        plt.fill_between(data[i].index, data[i]['mean'] - data[i]['std'], data[i]['mean'] + data[i]['std'], alpha=0.3)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    #np.arange(0, round(max_value,0), round(max_value/10,0))
    plt.yticks()
    if len(x_ticks) > 0:
        plt.xticks(x_ticks)
    else:
        plt.xticks()
    if log:
        ax.set_yscale('log')  # Set y-axis as logarithmic
    ax.set_title(y_axis + ' for '+ experiment[3:])
    ax.figure.set_size_inches(5.91, 3.8)
    ax.legend(framealpha=0.7)
    ax.grid(True, which='major', axis='y', linestyle='--', color='lightgray')
    #plt.tick_params(axis='y', which='both', right=True, left=True, labelright=True)
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
    ax.tick_params(bottom=False, left=False,labelsize=10)

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
    ax.figure.set_size_inches(5.91, 3.8)
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
            y_values.append(data[i].loc[ (data[i]['bandwidths(Mbs)'] == 25000) & (data[i]['packetdrops(%)'] == 0), ['runtime_external(s)']].to_numpy().reshape(-1)[0:4])
            x_values.append(data[i]['latencies(ms)'].unique()[0:4])
            labels.append("MPyC")
        elif(data[i]['framework'][0] in 'MOTION'):
            for protocol in data[i]['protocol'].unique():
                y_values.append(data[i].loc[(data[i]['bandwidths(Mbs)'] == 25000) &(data[i]['packetdrops(%)'] == 0) & (data[i]['protocol']==protocol), ['runtime_external(s)']].to_numpy().reshape(-1)[:4])
                x_values.append(data[i]['latencies(ms)'].unique()[0:4])
                labels.append("MOTION " + protocol)
        else:
            for protocol in data[i]['protocol'].unique():
                if protocol not in ["ps-rep-field","sy-rep-field","malicious-rep-field"]:
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
                y_values.append(data[i].loc[(data[i]['function']==function) & (data[i]['bandwidths(Mbs)'] == 25000) & (data[i]['latencies(ms)'] == 0) & (data[i]['preprocess']==0), ['runtime_external(s)']].to_numpy().reshape(-1)[0:4])
                x_values.append(data[i]['packetdrops(%)'].unique()[0:4])
                labels.append("HPMPC passive adversary")
        elif(data[i]['framework'][0] in 'MPyC'):
            y_values.append(data[i].loc[(data[i]['bandwidths(Mbs)'] == 25000) & (data[i]['latencies(ms)'] == 0), ['runtime_external(s)']].to_numpy().reshape(-1)[0:4])
            x_values.append(data[i]['packetdrops(%)'].unique()[0:4])
            labels.append("MPyC")
        else:
            for protocol in data[i]['protocol'].unique():
                if protocol not in ["ps-rep-field","sy-rep-field","malicious-rep-field"]:
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
                    y_values.append(data[i].loc[(data[i]['function']==function) & (data[i]['packetdrops(%)'] == 0) & (data[i]['latencies(ms)'] == 0) & (data[i]['preprocess']==0), ['runtime_external(s)']].to_numpy().reshape(-1)[0:4])
                    x_values.append(data[i]['bandwidths(Mbs)'].unique()[0:4])
                    labels.append("HPMPC passive adversary")
        elif(data[i]['framework'][0] in 'MPyC'):
            y_values.append(data[i].loc[ (data[i]['packetdrops(%)'] == 0) & (data[i]['latencies(ms)'] == 0), ['runtime_external(s)']].to_numpy().reshape(-1)[0:4])
            x_values.append(data[i]['bandwidths(Mbs)'].unique()[0:4])
            labels.append("MPyC")
        else:
            for protocol in data[i]['protocol'].unique():
                if protocol not in ["ps-rep-field","sy-rep-field","malicious-rep-field"]:
                    y_values.append(data[i].loc[(data[i]['packetdrops(%)'] == 0) & (data[i]['latencies(ms)'] == 0) & (data[i]['protocol']==protocol), ['runtime_external(s)']].to_numpy().reshape(-1)[0:4])
                    x_values.append(data[i]['bandwidths(Mbs)'].unique()[0:4])
                    labels.append(data[i]['framework'][0]+ " " + protocol)
    return x_values,y_values,labels


def get_latency_deviation(data):
    ret = []
    label=[]
    for i in range(len(data)):
        framework = data[i]['framework'].iloc[0]
        bandwidth_filter = (data[i]['bandwidths(Mbs)'] == 25000) & (data[i]['packetdrops(%)'] == 0)
        
        if framework == "HPMPC":
            for function in data[i]['function'].unique():
                for input_size in data[i]['input_size'].unique():
                    function_filter = data[i]['function'] == function
                    help = data[i].loc[bandwidth_filter & function_filter & (data[i]['input_size'] == input_size) & (data[i]['preprocess'] == 0)]
                    ret.append(get_mean_std(help,'latencies(ms)')) 
                    label.append("HPMPC "+ str(function) + " "+ str(input_size))
        elif framework == "MPyC":
            help = data[i].loc[bandwidth_filter]
            ret.append(get_mean_std(help,'latencies(ms)'))
            label.append("MPyC")
        elif framework == "MP-SPDZ":
            for protocol in data[i]['protocol'].unique():
                for input_size in data[i]['input_size'].unique():
                    protocol_filter = data[i]['protocol'] == protocol
                    help = data[i].loc[bandwidth_filter & protocol_filter & (data[i]['input_size'] == input_size)]
                    ret.append(get_mean_std(help,'latencies(ms)')) 
                    label.append("MP-SPDZ "+ protocol + " " + str(input_size))
        else:
            for protocol in data[i]['protocol'].unique():
                for input_size in data[i]['input_size'].unique():
                    protocol_filter = data[i]['protocol'] == protocol
                    help = data[i].loc[bandwidth_filter & protocol_filter & (data[i]['input_size'] == input_size)]
                    ret.append(get_mean_std(help,'latencies(ms)')) 
                    label.append("MOTION "+ protocol + " " + str(input_size))
    return ret,label

def get_mean_std(data,column):
    mean = data.groupby([column])['runtime_external(s)'].mean()
    std = data.groupby([column])['runtime_external(s)'].std()
    mean.name = 'mean'
    std.name = 'std'
    help_dev = pd.concat([mean, std], axis=1)
    help_dev[column] = help_dev.index
    return help_dev
                
def get_bandwidth_deviation(data):
    ret = []
    label=[]
    for i in range(len(data)):
        framework = data[i]['framework'].iloc[0]
        bandwidth_filter = (data[i]['latencies(ms)'] == 0) & (data[i]['packetdrops(%)'] == 0)
        
        if framework == "HPMPC":
            for function in data[i]['function'].unique():
                function_filter = data[i]['function'] == function
                help = data[i].loc[bandwidth_filter & function_filter & (data[i]['preprocess'] == 0)]
                ret.append(get_mean_std(help,'bandwidths(Mbs)'))
                label.append("HPMPC "+ str(function))
        elif framework == "MPyC":
            help = data[i].loc[bandwidth_filter]
            ret.append(get_mean_std(help,'bandwidths(Mbs)'))
            label.append("MPyC")
        elif framework == "MP-SPDZ":
            for protocol in data[i]['protocol'].unique():
                for input_size in data[i]['input_size'].unique():
                    protocol_filter = data[i]['protocol'] == protocol
                    help = data[i].loc[bandwidth_filter & protocol_filter & (data[i]['input_size'] == input_size)]
                    ret.append(get_mean_std(help,'bandwidths(Mbs)'))
                    label.append("MP-SPDZ "+ protocol+" "+str(input_size))
        else:
            for protocol in data[i]['protocol'].unique():
                protocol_filter = data[i]['protocol'] == protocol
                help = data[i].loc[bandwidth_filter & protocol_filter]
                ret.append(get_mean_std(help,'bandwidths(Mbs)'))
                label.append("MOTION "+ protocol)
        
    return ret,label

def get_packetdrop_deviation(data):
    ret = []
    label=[]
    for i in range(len(data)):
        framework = data[i]['framework'].iloc[0]
        
        if framework == "HPMPC":
            for function in data[i]['function'].unique():
                help = data[i].loc[(data[i]['latencies(ms)'] == 0) & (data[i]['function'] == function) & (data[i]['preprocess'] == 0) & (data[i]['bandwidths(Mbs)'] == 25000)]
                ret.append(get_mean_std(help,'packetdrops(%)'))
                label.append("HPMPC passive adversary "+ str(function))
        elif framework == "MPyC":
            help = data[i].loc[(data[i]['latencies(ms)'] == 0) & (data[i]['bandwidths(Mbs)'] == 25000)]
            ret.append(get_mean_std(help,'packetdrops(%)'))
            label.append("MPyC")
        elif framework == "MP-SPDZ":
            for protocol in data[i]['protocol'].unique():
                for input_size in data[i]['input_size'].unique():
                    help = data[i].loc[(data[i]['latencies(ms)'] == 0) & (data[i]['protocol'] == protocol) & (data[i]['bandwidths(Mbs)'] == 25000)& (data[i]['input_size'] == input_size)]
                    ret.append(get_mean_std(help,'packetdrops(%)'))
                    label.append("MP-SPDZ "+ protocol+" "+str(input_size))
        else:
            for protocol in data[i]['protocol'].unique():
                help = data[i].loc[(data[i]['latencies(ms)'] == 0) & (data[i]['protocol'] == protocol) & (data[i]['bandwidths(Mbs)'] == 25000)]
                ret.append(get_mean_std(help,'packetdrops(%)'))
                label.append("MOTION "+ protocol )
        
    
    return ret,label