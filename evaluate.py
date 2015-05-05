#!/usr/bin/env python
# encoding: utf-8

from optparse import OptionParser
import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_parameters():
    parser = OptionParser()
    parser.add_option("--result", type="string",dest="result_dir",default="results")
    #parser.add_option("--config", type="string",dest="conffile",default="config/func-eval.tcl")
    parser.add_option("-w","--whiskers",  type="string",dest="whiskers", default=os.path.join(cwd,"/home/lxa/F.3.23"))
    parser.add_option("--func", type="string", dest="evaluate_func", default = "light")
    (config, args) = parser.parse_args()
    return config

def light_eval(result_base):
    """light 5 TCP flows simulate 100s"""
    result_dir = os.path.join(result_base, 'light')
    onoff_eval(result_dir, nTCP=5)

def heavy_eval(resut_base):
    """heavy 50 TCP flows simulate 100s"""
    result_dir = os.path.join(result_base, 'heavy')
    onoff_eval(result_dir, nTCP=50)

def mix_eavl(result_base):
    """mix 5 TCP + 2 UDP flows simulate 100s"""
    result_dir = os.path.join(result_base, 'mix')
    onoff_eval(result_dir, nTCP=50)

def burst_eval():
    pass

def onoff_eval(result_dir, nTCP=8, nUDP=0, bw=15, rtt=100):
    candidates = ["RED", "CoDel", "PIE", "KEMY"]
    do_eveal = True
    if  os.path.exists(result_dir):
        while True:
            ret = raw_input("Results already existed, remove and continue ? y/n\t")
            if ret == 'y':
                os.system("rm -rf "+result_dir)
                os.makedirs(result_dir)
                break
            elif ret == 'n':
                do_eveal = False
                break
    else:
        os.makedirs(result_dir)
    if do_eveal == True:
        for i in xrange(128):
            evaluate(result_dir, "./config/func-eval.tcl", candidates, "Application/OnOff",nTCP, nUDP, run= i + 1, delay = rtt/2)
        for candidate in candidates:
            result_file = os.path.join(result_dir, candidate)
            subprocess.call(["awk -f ./awks/onoff_throughput.awk " + result_file+" >"  + result_file+".throughput" ], shell=True)
            subprocess.call(["awk -v rtt="+str(rtt)+" -f ./awks/onoff_delay.awk " + result_file+" >"  + result_file+".delay" ], shell=True)

    graph_box(result_dir,candidates, "")

def rtt_eval(result_base):
    candidates = ["RED", "CoDel", "PIE", "KEMY"]
    result_dir = os.path.join(result_base, 'rtt')
    do_eveal = True
    if  os.path.exists(result_dir):
        while True:
            ret = raw_input("Results already existed in " + result_dir + ", remove and continue ? y/n\t")
            if ret == 'y':
                os.system("rm -rf "+result_dir)
                os.makedirs(result_dir)
                break
            elif ret == 'n':
                do_eveal = False
                break
    else:
        os.makedirs(result_dir)

    rtts = [10, 50, 100,120, 150, 180, 200, 250]
    bw = 15
    iter_times = 32
    if do_eveal == True:
        for rtt in rtts:
            for i in xrange(iter_times):
                evaluate(result_dir, "./config/func-eval.tcl", candidates, "Application/OnOff", 8, 0 ,run= i+1 , bw=bw, delay=rtt/2)
        for candidate in candidates:
            result_file = os.path.join(result_dir, candidate)
            subprocess.call(["awk -v senders=8 -v iter="+str(iter_times)+" -f  ./awks/bw_throughput.awk " + result_file+" >"  + result_file+".throughput" ], shell=True)
            subprocess.call(["awk -v senders=8 -v iter="+str(iter_times)+" -f ./awks/rtt_delay.awk " + result_file+" >"  + result_file+".delay" ], shell=True)

    #graph_box(result_dir,candidates, "")
    graph_base_rtt(result_dir, candidates, "", rtts)

def bw_eval(result_base):
    candidates = ["RED", "CoDel", "PIE", "KEMY"]
    result_dir = os.path.join(result_base, 'bw')
    do_eveal = True
    if  os.path.exists(result_dir):
        while True:
            ret = raw_input("Results already existed in" + result_dir + ", remove and continue ? y/n\t")
            if ret == 'y':
                os.system("rm -rf "+result_dir)
                os.makedirs(result_dir)
                break
            elif ret == 'n':
                do_eveal = False
                break
    else:
        os.makedirs(result_dir)

    bws = [ 5,10, 40, 60, 80, 100, 200]
    rtt=100
    sender_num = 8
    iter_times = 128
    if do_eveal == True:
        for bw in bws:
            for i in xrange(iter_times):
                evaluate(result_dir, "./config/func-eval.tcl", candidates, "Application/OnOff", sender_num, 0 ,run= i+1, bw=bw, delay=rtt/2)

    graph_type = 1
    if graph_type == 0:
        awk_throughput = "./awks/bw_throughput.awk"
        awk_delay = "./awks/bw_delay.awk"
    else:
        awk_throughput = "./awks/onoff_throughput.awk"
        awk_delay = "./awks/onoff_delay.awk"

    for candidate in candidates:
        result_file = os.path.join(result_dir, candidate)
        subprocess.call(["awk -v senders=8 -v iter="+str(iter_times)+" -f  "+awk_throughput+" " + result_file+" >"  + result_file+".throughput" ], shell=True)
        subprocess.call(["awk -v senders=8 -v iter="+str(iter_times)+" -v rtt="+str(rtt)+" -f "+awk_delay+" " + result_file+" >"  + result_file+".delay" ], shell=True)
    #graph_multi_box(result_dir, candidates, iter_times, sender_num, bws)
    #graph_box(result_dir,candidates, "")
    if(graph_type == 0):
        graph_base_bw(result_dir, candidates, "", bws)
    else:
        graph_group_box(result_dir, candidates, iter_times, sender_num, bws)
        #graph_multi_box(result_dir, candidates, iter_times, sender_num, bws)
import pandas as pd
def construct_Dataframe(result_dir, candidates, M, xticks):
    xs = []
    ncandidates = len(candidates)
    nxticks = len(xticks)
    for i in xrange(ncandidates):
        for xtick in xticks:
            xs = xs + [xtick]*M
    throughputs = []
    delays = []
    aqms = []
    for candidate in candidates:
        result_file = os.path.join(result_dir, candidate + "." + "throughput")
        throughputs = throughputs + np.loadtxt(result_file, unpack = True).tolist()
        result_file = os.path.join(result_dir, candidate + "." + "delay")
        delays = delays + np.loadtxt(result_file, unpack = True).tolist()
        aqms = aqms + [candidate]*( M * nxticks)

    nlen = len(xs)
    if (nlen!= len(throughputs)) or (nlen != len(delays)) or(nlen != len(aqms)):
        print "Dataframe: len error!"
        print nlen, len(throughputs), len(delays), len(aqms)
        sys.exit(1)
    return pd.DataFrame(dict(xtick=xs, throughput=throughputs, delay=delays, AQM=aqms))
def graph_group_box(result_dir, candidates, iter_times,sender_num, xticks):
    #first construct Dataframe
    datas = construct_Dataframe(result_dir, candidates, iter_times, xticks)
    metrics = ['throughput', 'delay']
    xlabel = "Bottleneck Bandwidth [Mbps]"
    ylabels = {'throughput':'Goodput [Mbps]','delay':'Queueing Delay [msec]'}
    for metric in metrics:
        boxes = sns.factorplot("xtick", metric, "AQM", datas, kind="box", hue_order=candidates, palette="husl");
        boxes.set_axis_labels(xlabel, ylabels[metric])
        plt.savefig(os.path.join(result_dir, metric+"-group-box.eps"), format='eps')
        sns.plt.show()
def graph_multi_box(result_dir, candidates, iter_times,sender_num, xticks):
    metrics = ['throughput', 'delay']
    ylabels = {}
    ylabels['throughput'] = 'Goodput [Mbps]'
    ylabels['delay'] = 'Queueing Delay [msec]'
    for metric in metrics:
        plt.grid(True)
        plt.ylabel(ylabels[metric],fontsize=18)
        i = 0
        ncandidates = len(candidates)
        data = [None]*(ncandidates*len(xticks))
        for candidate in candidates:
            result_file = os.path.join(result_dir, candidate + "." + metric)
            tmp = np.loadtxt(result_file, unpack = True)
            if not tmp.size ==  iter_times*len(xticks):
                print "result file error"
                sys.exit(1)
            t_start = 0
            for j in xrange(len(xticks)):
                data[j*ncandidates+i] = tmp[t_start:t_start+iter_times:1]
                t_start = t_start + iter_times
            if t_start != iter_times * len(xticks):
                print "t_start error"
                sys.exit(1)
            i = i + 1
        #boxes= plt.boxplot(data, showfliers = False)
        sns.boxplot(data)
        #plt.xticks(range(1,len(candidates)+1), candidates,fontsize=18)
        plt.savefig(os.path.join(result_dir, metric+"-box.eps"), format='eps')
        plt.show()

def evaluate(result_dir, conffile, candidates, tcp_app, ntcpsrc, nudpsrc,run=1, bw=15, delay=50):
    """evaluate an AQM by run run-test.tcl """

    if tcp_app == "Application/OnOff":
        trace_type = '-onoff_out'
    else :
        #trace_type = '-qtr'
        trace_type = '-qmon'

    child_ps = []
    for candidate in candidates:
        tcl_args = ['./run-test.tcl', \
                                          conffile,\
                                          '-nTCPsrc', str(ntcpsrc),\
                                          '-tcp_app', tcp_app,\
                                          '-nUDPsrc', str(nudpsrc),\
                                          '-bw', str(bw),\
                                          '-delay', str(delay),\
                                          '-gw', candidate,\
                                          '-run', str(run),\
                                          trace_type, result_dir+"/"+candidate, \
                                          #'-nam', candidate+'.nam',\
                                          ]
#        if onoff_out != "":
            #tcl_args = tcl_args + ['-onoff_out', os.path.join(onoff_out,"onoff-"+candidate)]
        child_ps.append(subprocess.Popen(tcl_args))
        print " ".join(tcl_args)

    for child_p in child_ps:
        child_p.wait()

def graph_base_rtt(result_dir, candidates, graph_title, rtts):
    """graph base simulation rtt"""
    metrics = ['throughput', 'delay']
    ylabels = {}
    ylabels['throughput'] = 'Goodput [Mbps]'
    ylabels['delay'] = 'Queueing Delay [msec]'
    styles={}
    styles['KEMY'] = '-s'
    styles['RED'] = '-*'
    styles['CoDel'] = '-^'
    styles['PIE'] = '-o'
    #row = 0
    for metric in metrics:
        #row = row + 1
        #plt.subplot(len(metrics),1,row)
        plt.title(graph_title)
        plt.grid()
        plt.ylabel(ylabels[metric],fontsize=18)
        plt.xlabel("Bottleneck RTT [ms]",fontsize=18)
        for candidate in candidates:
            result_file = os.path.join(result_dir, candidate) +"." + metric
            data = np.loadtxt(result_file, unpack = True)
            line, =plt.plot(data,styles[candidate], label = candidate, markersize=10)

        plt.legend()

        plt.xticks(range(len(rtts)), rtts)
        plt.savefig( os.path.join(result_dir,metric) +".eps", format="eps")
        plt.show()

def graph_base_bw(result_dir, candidates, graph_title, bws):
    """graph base simulation bandwidth"""
    metrics = ['throughput', 'delay']
    ylabels = {}
    ylabels['throughput'] = 'Goodput [Mbps]'
    ylabels['delay'] = 'Queueing Delay [msec]'
    styles={}
    styles['KEMY'] = '-s'
    styles['RED'] = '-*'
    styles['CoDel'] = '-^'
    styles['PIE'] = '-o'
    #row = 0
    for metric in metrics:
        #row = row + 1
        #plt.subplot(len(metrics),1,row)
        plt.title(graph_title)
        plt.grid()
        plt.ylabel(ylabels[metric],fontsize=18)
        plt.xlabel("Bottleneck Bandwidth [Mbps]",fontsize=18)
        for candidate in candidates:
            result_file = os.path.join(result_dir, candidate) +"." + metric
            data = np.loadtxt(result_file, unpack = True)
            line, =plt.plot(data,styles[candidate], label = candidate, markersize=10)

        if metric == "throughput":
            plt.legend(loc=2)
        else:
            plt.legend()
        plt.xticks(range(len(bws)), bws)
        plt.savefig( os.path.join(result_dir,metric) +".eps", format="eps")
        #plt.savefig( os.path.join(result_dir,metric) +".svg", format="svg")
        plt.show()

def graph_base_simtime(result_dir, candidates, graph_title):
    """graph base simulation time"""
    #graph x:simulation time, y: metric
    #metrics = ['throughput', 'delay', 'drop_rate']
    #metrics = ['throughput', 'delay']
    metrics = ['qlen' ]
    ylabels = {}
    ylabels['qlen'] = 'Queue length [packets]'
    ylabels['throughput'] = 'Throughput [Mbps]'
    ylabels['delay'] = 'Queueing Delay [msec]'
    ylabels['drop_rate'] = 'Drop Rate'
    for metric in metrics:
        plt.title(graph_title)
        plt.grid()
        plt.xlabel("Simulation Time [Sec]",fontsize=18)
        plt.ylabel(ylabels[metric],fontsize=18)

        for candidate in candidates:
            result_file = os.path.join(result_dir, candidate) +"." + metric
            data = np.loadtxt(result_file, unpack = True)
            line, =plt.plot(data[0],data[1], label = candidate)

        plt.legend()
        plt.savefig( os.path.join(result_dir,metric) +".eps", format="eps")
        plt.show()

def graph_box(result_dir, candidates, graph_title):
    metrics = ['throughput', 'delay']
    ylabels = {}
    ylabels['throughput'] = 'Goodput [Mbps]'
    ylabels['delay'] = 'Queueing Delay [msec]'
    #colors = ["purple", "pink", "blue"]
    for metric in metrics:
        plt.title(graph_title)
        plt.grid(True)
        plt.ylabel(ylabels[metric],fontsize=18)
        data = []
        for candidate in candidates:
            result_file = os.path.join(result_dir, candidate + "." + metric)
            tmp = np.loadtxt(result_file, unpack = True)
            data.append( tmp )
        boxes= plt.boxplot(data, showfliers = False)

#        for patch, color in zip(boxes['boxes'], colors):
            #patch.set_alpha(0.8)

        #boxs= plt.boxplot(data)
        plt.xticks(range(1,len(candidates)+1), candidates,fontsize=18)
        plt.savefig(os.path.join(result_dir, metric+"-box.eps"), format='eps')
        plt.show()
def graph_cdf(result_dir, candidates, graph_title):
    metrics = ['delay','throughput']
    for metric in metrics:
        plt.title(graph_title)
        plt.grid(True)
        for candidate in candidates:
            result_file = os.path.join(result_dir, candidate + "." + metric)
            data = np.loadtxt(result_file, unpack = True)
            sorted_data = np.sort(data[1])
            yvals = np.arange(len(sorted_data)) / float(len(sorted_data))
            line, = plt.plot(sorted_data, yvals,label = candidate)
        plt.legend()
        plt.savefig(os.path.join(result_dir, metric+"-cdf.eps",format = 'eps'))
        plt.show()







if __name__ == '__main__':
    cwd = os.getcwd()
    config = get_parameters()
    #candidates = ["KEMY", "RED","PIE","CoDel"]
    candidates = ["KEMY", "CoDel", "PIE"]
    #candidates = ["KEMY","RED"]
    result_base = os.path.join(cwd, config.result_dir)
    os.environ['WHISKERS'] = config.whiskers
    print config.whiskers

    evaluate_func = config.evaluate_func
    if evaluate_func == "light":
        light_eval(result_base)
    elif evaluate_func == "heavy":
        heavy_eval(result_base)
    elif evaluate_func == "mix":
        mix_eavl(result_base)
    elif evaluate_func == "onoff":
        onoff_eval(result_base)
    elif evaluate_func == "bw":
        bw_eval(result_base)
    elif evaluate_func == "rtt":
        rtt_eval(result_base)
