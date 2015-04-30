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

def evaluate(result_dir, conffile, candidates, tcp_app, ntcpsrc, nudpsrc,run=1, bw=15, delay=50):
    """evaluate an AQM by run run-test.tcl """

    if tcp_app == "Application/OnOff":
        trace_type = '-onoff_out'
    else :
        trace_type = '-qtr'
        #trace_type = '-qmon'

    child_ps = []
    for candidate in candidates:
        tcl_args = ['./dyn-test.tcl', \
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

if __name__ == '__main__':
    cwd = os.getcwd()
    config = get_parameters()
    candidates = ["RED", "KEMY"]
    result_base = os.path.join(cwd, config.result_dir)
    os.environ['WHISKERS'] = config.whiskers
    print config.whiskers
    result_dir = os.path.join(result_base,'dyn-bw')
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
        evaluate(result_dir, "./config/func-eval.tcl", candidates, "Application/FTP", 8, 0)
        for candidate in candidates:
            result_file = os.path.join(result_dir, candidate)
            print result_file
            subprocess.call(["awk -f ./awks/inst_delay.awk " + result_file+" >"  + result_file+".delay" ], shell=True)
            subprocess.call(["awk -f ./awks/cmuthput.awk " + result_file+" >"  + result_file+".cmuthput" ], shell=True)
    #graph
    metrics = ['cmuthput']
    #plt.title("Per-packet queueing delay for dynamic bandwidth")
    plt.grid()
    f,(ax,ax2) = plt.subplots(2,1,sharex=True)
    plt.xlabel("Simulation Time [Sec]",fontsize=18)
    #plt.ylabel("Instantaneous Queueing Delay[msec]",fontsize=12)
    ax.set_ylabel("Instantaneous Queueing Delay[msec]",fontsize=12)
    for candidate in candidates:
        result_file = os.path.join(result_dir, candidate) +"." + "delay"
        data = np.loadtxt(result_file, unpack = True)
        ax.plot(data[0],data[1], label = candidate)
        ax2.plot(data[0],data[1])
    ax.legend()
    ax.set_ylim(40,700)
    ax2.set_ylim(0,40)

    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop='off') # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    d = .015 # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d,+d),(-d,+d), **kwargs)      # top-left diagonal
    ax.plot((1-d,1+d),(-d,+d), **kwargs)    # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d,+d),(1-d,1+d), **kwargs)   # bottom-left diagonal
    ax2.plot((1-d,1+d),(1-d,1+d), **kwargs) # bottom-right diagonal
    plt.savefig( os.path.join(result_dir,"delay") +".eps", format="eps")
    plt.show()
    for candidate in candidates:
        result_file = os.path.join(result_dir, candidate) +"." + "cmuthput"
        data = np.loadtxt(result_file, unpack = True)
        line, =plt.plot(data[0],data[1], label = candidate)

    plt.xlabel("Simulation Time [Sec]",fontsize=18)
    plt.ylabel("Cumulative bytes transferred [Byte]",fontsize=18)
    plt.legend(loc=2)
    plt.savefig( os.path.join(result_dir,"cmuthput") +".eps", format="eps")
    plt.show()

