#!/usr/bin/env python
# encoding: utf-8

from optparse import OptionParser
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
cwd = os.getcwd()

parser = OptionParser()
parser.add_option("--result", type="string",dest="result_dir",default="results/cmp")
parser.add_option("-w","--whiskers",  type="string",dest="whiskers", default=os.path.join(cwd,"../../../queue/kemy-train/src/result/E.10"))
(config, args) = parser.parse_args()

#condidates = ["KEMY", "RED","PIE","CoDel"]
condidates = ["KEMY", "CoDel", "PIE"]

conffile = "./config/func-eval.tcl"

RESULT_DIR = os.path.join(cwd, config.result_dir)
os.environ['WHISKERS'] = config.whiskers
print config.whiskers

def eval_onoff():
    """evaluate an AQM by run run-test.tcl """
    result_dir = os.path.join(RESULT_DIR,"onoff")
    evaluate(result_dir, 32, 0)

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
                                          '-tcp_app', 'Application/OnOff',\
                                          '-nUDPsrc', str(nudpsrc),\
                                          '-bw', '10',\
                                          '-delay', '100',\
                                          '-gw', condidate,\
                                          '-qtr', result_dir+"/"+condidate, \
                                          '-onavg', '1000', \
                                          '-offavg', '2',\
                                          #'-nam', condidate+'.nam',\
                                          ]
        child_ps.append(subprocess.Popen(tcl_args))
        print " ".join(tcl_args)

    for child_p in child_ps:
        child_p.wait()

    for condidate in condidates:
        result_onoff_process(result_dir+"/"+condidate)

def result_onoff_process(result_file):

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





if __name__ == '__main__':
    eval_onoff()
