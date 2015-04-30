# config file for kemy  simulations
# this one is where all the sources are identical
#

global opt


# source, sink, and app types
set opt(nTCPsrc) 32
set opt(tcp) TCP/Newreno
set opt(sink) TCPSink

# AQM details
set opt(gw) KEMY;            # queueing at bottleneck
set opt(rcvwin) 65536
set opt(qlimit) 65536
# app parameters
set opt(tcp_app) Application/OnOff
set opt(pktsize) 1000;           # doesn't include proto headers (bytes)
set opt(hdrsize) 50

#UDP
set opt(nUDPsrc) 0
set opt(udp_app) Application/Traffic/CBR
set opt(CBR_rate) 6Mb
# random on-off times for sources
# 
set opt(run) 1
set opt(onrand) Exponential
set opt(offrand) Exponential
#set opt(onavg) 5.0;              # mean on and off time
set opt(onavg) 10000.0;              # mean on and off time ( x opt(pktsize)bytes)
#set opt(offavg) 0.2;             # mean on and off time (s)
set opt(offavg) 5;             # mean on and off time (s)
set opt(ontype) "bytes";         # valid options: "time", "bytes", and "flowcdf"
#set opt(ontype) "time";         # valid options: "time", "bytes", and "flowcdf"

# simulator parameters
set opt(simtime) 100.0;          # total simulated time (s)
#set opt(tr) kemyout;            # output trace in opt(tr).out
set opt(partialresults) false;   # show partial throughput, delay, and utility?
set opt(verbose) false;          # verbose printing for debugging (esp stats)
set opt(checkinterval) 0.0001;    # check stats every 5 ms



# utility and scoring
#set opt(alpha) 1.0
#set opt(tracewhisk) "none";      # give a connection ID to print for that flow, or give "all"


# tcp details
Agent/TCP set tcpTick_ .0001
Agent/TCP set timestamps_ true
set opt(flowoffset) 40

