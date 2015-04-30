#!/usr/bin/env python
# encoding: utf-8

from optparse import OptionParser
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
cwd = os.getcwd()

parser = OptionParser()
parser.add_option("--result", type="string",dest="result_dir",default="results/func")
parser.add_option("-w","--whiskers",  type="string",dest="whiskers", default=os.path.join(cwd,"../../../queue/kemy-train/src/result/E.10"))
(config, args) = parser.parse_args()

#condidates = ["KEMY", "RED","PIE","CoDel"]
#condidates = ["KEMY", "CoDel", "PIE"]
condidates = ["KEMY","PIE"]

conffile = "./config/func-eval.tcl"

RESULT_DIR = os.path.join(cwd, config.result_dir)
os.environ['WHISKERS'] = config.whiskers
print config.whiskers

def light_eval():
    """light 5 TCP flows simulate 100s"""
    result_dir = os.path.join(RESULT_DIR, 'light')
    evaluate(result_dir, 5, 0)
    graph_base_simtime(result_dir, "light: 5 TCP Flows")

def heavy_eval():
    """heavy 50 TCP flows simulate 100s"""
    result_dir = os.path.join(RESULT_DIR, 'heavy')
    evaluate(result_dir, 50, 0)
    graph_base_simtime(result_dir, "Heavy: 50 TCP Flows")
    #graph_box(result_dir, "Heavy: 50 TCP Flows")
    #graph_cdf(result_dir, "Heavy: 50 TCP Flows")

def mix_eavl():
    """mix 5 TCP + 2 UDP flows simulate 100s"""
    result_dir = os.path.join(RESULT_DIR, 'mix')
    evaluate(result_dir, 25,25 )
    graph_base_simtime(result_dir, "Mixture: 5 TCP + 2 UDP Flows")

def evaluate(result_dir, ntcpsrc, nudpsrc):
    """evaluate an AQM by run run-test.tcl """
    if  os.path.exists(result_dir):
        while True:
            ret = raw_input("Results already existed, remove and continue ? y/n\t")
            if ret == 'y':
                break
            elif ret == 'n':
                return 0

        print "pre results existed .\n"
    else:
        os.makedirs(result_dir)

    child_ps = []
    for condidate in condidates:
        tcl_args = ['./run-test.tcl', \
                                          conffile,\
                                          '-nTCPsrc', str(ntcpsrc),\
                                          '-tcp_app', 'Application/FTP',\
                                          '-nUDPsrc', str(nudpsrc),\
                                          '-bw', '15',\
                                          '-delay', '100',\
                                          '-gw', condidate,\
                                          '-qtr', result_dir+"/"+condidate, \
                                          #'-nam', condidate+'.nam',\
                                          ]
        child_ps.append(subprocess.Popen(tcl_args))
        print " ".join(tcl_args)

    for child_p in child_ps:
        child_p.wait()

    for condidate in condidates:
        result_process(result_dir+"/"+condidate)



def result_process(result_file):
    """awk process and graph"""
    tic="0.1"
    # throughput
    subprocess.call(["awk -f ./awks/inst_throughput.awk tic="+tic+" " + result_file+" >"  + result_file+".throughput" ], shell=True)
    # delay
    subprocess.call(["awk -f ./awks/inst_delay.awk tic="+tic +" "+result_file +" >"  + result_file+".delay" ], shell=True)
    # drop_rate
    #subprocess.call(["awk -f ./awks/drop_rate.awk " + result_file+" >"  + result_file+".drop_rate" ], shell=True)


def graph_base_simtime(result_dir, graph_title):
    """graph base simulation time"""
    #graph x:simulation time, y: metric
    #metrics = ['throughput', 'delay', 'drop_rate']
    metrics = ['throughput', 'delay']
    ylabels = {}
    ylabels['throughput'] = 'Throughput [Mbps]'
    ylabels['delay'] = 'Queueing Delay [msec]'
    ylabels['drop_rate'] = 'Drop Rate'
    for metric in metrics:
        plt.title(graph_title)
        plt.grid()
        plt.xlabel("Simulation Time [Sec]")
        plt.ylabel(ylabels[metric])

        for condidate in condidates:
            result_file = os.path.join(result_dir, condidate) +"." + metric
            data = np.loadtxt(result_file, unpack = True)
            line, =plt.plot(data[0],data[1], label = condidate)

        plt.legend()
        plt.savefig( os.path.join(result_dir,metric) +".svg", format="svg")
        plt.show()

def graph_box(result_dir, graph_title):
    metrics = ['throughput', 'delay']
    for metric in metrics:
        plt.title(graph_title)
        plt.grid(True)
        data = []
        for condidate in condidates:
            result_file = os.path.join(result_dir, condidate + "." + metric)
            tmp = np.loadtxt(result_file, unpack = True)
            data.append( tmp[1] )
        boxs= plt.boxplot(data, showfliers = False)
        plt.xticks(range(1,len(condidates)+1), condidates)
        plt.savefig(os.path.join(result_dir, metric+"-box.svg"))
        plt.show()
def graph_cdf(result_dir, graph_title):
    metrics = ['delay','throughput']
    for metric in metrics:
        plt.title(graph_title)
        plt.grid(True)
        for condidate in condidates:
            result_file = os.path.join(result_dir, condidate + "." + metric)
            data = np.loadtxt(result_file, unpack = True)
            sorted_data = np.sort(data[1])
            yvals = np.arange(len(sorted_data)) / float(len(sorted_data))
            line, = plt.plot(sorted_data, yvals,label = condidate)
        plt.legend()
        plt.savefig(os.path.join(result_dir, metric+"-cdf.svg"))
        plt.show()

def eval_dynamic_bw():
    bws = [0.128, 0.256, 0.512, 1.5, 3, 10, 45, 100]
    result_dir = os.path.join(RESULT_DIR,"dyn_bw")
    if  os.path.exists(result_dir):
        while True:
            ret = raw_input("Results already existed, remove and continue ? y/n\t")
            if ret == 'y':
                break
            elif ret == 'n':
                return 0
    else:
        os.makedirs(result_dir)

    child_ps = []
    for condidate in condidates:
        for bw in bws: tcl_args = ['./run-test.tcl', conffile,\
                                          '-nTCPsrc', 16,\
                                          '-tcp_app', 'Application/OnOff',\
                                          '-nUDPsrc', 0,\
                                          '-bw', bw,\
                                          '-delay', '100',\
                                          '-gw', condidate,\
                                          '-qtr', result_dir+"/"+condidate + "-bw-"+bw, \
                                          #'-nam', condidate+'.nam',\
                                          ]







if __name__ == '__main__':
    light_eval()
    heavy_eval()
    mix_eavl()
    #eval_dynamic_bw()
