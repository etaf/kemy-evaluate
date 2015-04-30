#!/usr/bin/env python
# encoding: utf-8

from optparse import OptionParser
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt

def get_parameters():
    parser = OptionParser()
    parser.add_option("--result", type="string",dest="result_dir",default="results")
    #parser.add_option("--config", type="string",dest="conffile",default="config/func-eval.tcl")
    parser.add_option("-w","--whiskers",  type="string",dest="whiskers", default=os.path.join(cwd,"/home/lxa/F.3.23"))
    parser.add_option("--func", type="string", dest="evaluate_func", default = "onoff")
    (config, args) = parser.parse_args()
    return config

def func_evaluate(result_dir, candidates, nTCPsrc, nUDPsrc):
    """functional evaluate"""
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
    if do_eveal:
        evaluate(result_dir, "./config/func-eval.tcl", candidates, "Application/FTP", nTCPsrc, nUDPsrc)

    for candidate in candidates:
        awk_process(result_dir+"/"+candidate)

def light_eval(result_base):
    """light 5 TCP flows simulate 100s"""
    result_dir = os.path.join(result_base, 'light')
    candidates = ['KEMY','RED','CoDel']
    func_evaluate(result_dir, candidates, 5, 0)

    graph_base_simtime(result_dir, candidates, "light: 5 TCP Flows")

def heavy_eval(resut_base):
    """heavy 50 TCP flows simulate 100s"""
    result_dir = os.path.join(result_base, 'heavy')
    candidates = ['KEMY','RED']
    func_evaluate(result_dir, candidates, 50, 0)
    graph_base_simtime(result_dir, candidates, "Heavy: 50 TCP Flows")
    #graph_box(result_dir, "Heavy: 50 TCP Flows")
    #graph_cdf(result_dir, "Heavy: 50 TCP Flows")

def mix_eavl():
    """mix 5 TCP + 2 UDP flows simulate 100s"""
    result_dir = os.path.join(result_base, 'mix')
    candidates = ['KEMY','RED']
    func_evaluate(result_dir, candidates, 5, 2 )
    graph_base_simtime(result_dir, candidates, "Mixture: 5 TCP + 2 UDP Flows")

def burst_eval():
    pass

def onoff_eval(result_base):
    candidates = ["RED", "CoDel", "PIE", "KEMY"]
    result_dir = os.path.join(result_base, 'onoff')
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
            evaluate(result_dir, "./config/func-eval.tcl", candidates, "Application/OnOff", 8, 0 ,run= i + 1)
    for candidate in candidates:
        result_file = os.path.join(result_dir, candidate)
        subprocess.call(["awk -f ./awks/onoff_throughput.awk " + result_file+" >"  + result_file+".throughput" ], shell=True)
        subprocess.call(["awk -f ./awks/onoff_delay.awk " + result_file+" >"  + result_file+".delay" ], shell=True)

    graph_box(result_dir,candidates, "")
    #graph_cdf(result_dir, "Heavy: 50 TCP Flows")

def bw_eval(result_base):
    candidates = ["RED", "CoDel", "PIE", "KEMY"]
    result_dir = os.path.join(result_base, 'bw')
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

    bws = [1,5,10,20,50,70,100,120]

    if do_eveal == True:
        for bw in bws:
            evaluate(result_dir, "./config/func-eval.tcl", candidates, "Application/OnOff", 8, 0 ,run= 1, bw=bw)

    for candidate in candidates:
        result_file = os.path.join(result_dir, candidate)
        subprocess.call(["awk -v senders=8 -f  ./awks/bw_throughput.awk " + result_file+" >"  + result_file+".throughput" ], shell=True)
        subprocess.call(["awk -v senders=8 -f ./awks/bw_delay.awk " + result_file+" >"  + result_file+".delay" ], shell=True)

    #graph_box(result_dir,candidates, "")
    graph_base_bw(result_dir, candidates, "", bws)
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



def awk_process(result_file):
    """awk process and graph"""

    # throughput
    #subprocess.call(["awk -f ./awks/inst_throughput.awk " + result_file+" >"  + result_file+".throughput" ], shell=True)
    # delay
    #subprocess.call(["awk -f ./awks/inst_delay.awk " + result_file+" >"  + result_file+".delay" ], shell=True)
    # drop_rate
    #subprocess.call(["awk -f ./awks/drop_rate.awk " + result_file+" >"  + result_file+".drop_rate" ], shell=True)
    #queue length
    subprocess.call(["awk -f ./awks/inst_qlen.awk " + result_file+" >"  + result_file+".qlen" ], shell=True)


def graph_base_bw(result_dir, candidates, graph_title, bws):
    """graph base simulation bandwidth"""
    metrics = ['throughput', 'delay']
    ylabels = {}
    ylabels['throughput'] = 'Throughput [Mbps]'
    ylabels['delay'] = 'Queueing Delay [msec]'

    for metric in metrics:
        plt.title(graph_title)
        plt.grid()
        plt.ylabel(ylabels[metric])
        for candidate in candidates:
            result_file = os.path.join(result_dir, candidate) +"." + metric
            data = np.loadtxt(result_file, unpack = True)
            line, =plt.plot(data, label = candidate)

        plt.legend()
        plt.xticks(range(len(bws)), bws)
        plt.savefig( os.path.join(result_dir,metric) +".eps", format="eps")
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
        plt.xlabel("Simulation Time [Sec]")
        plt.ylabel(ylabels[metric])

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
    ylabels['throughput'] = 'Average Throughput [Mbps]'
    ylabels['delay'] = 'Average Queueing Delay [msec]'
    colors = ["purple", "pink", "blue"]
    for metric in metrics:
        plt.title(graph_title)
        plt.grid(True)
        plt.ylabel(ylabels[metric])
        data = []
        for candidate in candidates:
            result_file = os.path.join(result_dir, candidate + "." + metric)
            tmp = np.loadtxt(result_file, unpack = True)
            data.append( tmp )
        boxes= plt.boxplot(data, showfliers = False)

#        for patch, color in zip(boxes['boxes'], colors):
            #patch.set_alpha(0.8)

        #boxs= plt.boxplot(data)
        plt.xticks(range(1,len(candidates)+1), candidates)
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
