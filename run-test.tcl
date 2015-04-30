#!/bin/sh
# the next line finds ns \
nshome=`dirname $0`; [ ! -x $nshome/ns ] && [ -x ../../../ns ] && nshome=../../..
# the next line starts ns \
export nshome; exec $nshome/ns "$0" "$@"

if [info exists env(nshome)] {
	set nshome $env(nshome)
} elseif [file executable ../../../ns] {
	set nshome ../../..
} elseif {[file executable ./ns] || [file executable ./ns.exe]} {
	set nshome "[pwd]"
} else {
	puts "$argv0 cannot find ns directory"
	exit 1
}
set env(PATH) "$nshome/bin:$env(PATH)"


proc Usage {} {
    global opt argv0
    puts "Usage: $argv0 \[-simtime seconds\] \[-nsrc numSources\]"
    puts "\t\[-tr tracefile\]"
    puts "\t\[-bw $opt(bneck)] \[-delay $opt(delay)\]"
    exit 1
}

proc Getopt {} {
    global opt argc argv
    if {$argc == 0} Usage
    for {set i 1} {$i < $argc} {incr i} {
        set key [lindex $argv $i]
        if ![string match {-*} $key] continue
        set key [string range $key 1 end]
        set val [lindex $argv [incr i]]
        set opt($key) $val
        #puts "$key = $val"
        if [string match {-[A-z]*} $val] {
            incr i -1
            continue
        }
    }
}

proc create-dumbbell-topology {} {
    global ns opt node_array bt_aqm

    Queue/KEMY set mean_pktsize_ [expr $opt(pktsize) + $opt(hdrsize)]
    #create bottle link node_array(0) -> node_array(1)
    set node_array(0) [$ns node]
    set node_array(1) [$ns node]
    $ns duplex-link $node_array(0) $node_array(1) ${opt(bw)}Mb ${opt(delay)}ms $opt(gw)
    #set qsize [expr (5 * ${opt(bw)}  * 2 * ${opt(delay)} * 1000.0) / (8 * ($opt(pktsize) + $opt(hdrsize)))]
    $ns queue-limit $node_array(0) $node_array(1) $opt(qlimit)
    $ns queue-limit $node_array(1) $node_array(0) 65536
    set bt_aqm [[$ns link $node_array(0) $node_array(1)] queue]

    if { [info exists opt(qmon)] } {

        #puts "$opt(qmon)"
        set f_qmon [open $opt(qmon) w]
        set qmon [$ns monitor-queue $node_array(0) $node_array(1) $f_qmon 0.1]
        [$ns link $node_array(0) $node_array(1)] queue-sample-timeout
    }

    if {[info exists opt(qtr)] } {

        #puts "$opt(qtr)"
        set f_qtr [open $opt(qtr) w]
        $ns trace-queue $node_array(0) $node_array(1) $f_qtr
    }

    if { [info exists opt(trace4split)] } {
        $bt_aqm trace4split
    }
    for {set i 1} {$i <= $opt(nsrc)} {incr i} {
        set node_array([expr $i+$i]) [$ns node]
        set node_array([expr $i+$i+1]) [$ns node]
        $ns duplex-link $node_array([expr $i+$i]) $node_array(0) 1000Mb 1ms DropTail
        $ns duplex-link $node_array([expr $i+$i+1]) $node_array(1) 1000Mb 1ms DropTail

        $ns queue-limit $node_array([expr $i+$i]) $node_array(0) 65536
        $ns queue-limit $node_array(0) $node_array([expr $i+$i]) 65536
        $ns queue-limit $node_array([expr $i+$i+1]) $node_array(1) 65536
        $ns queue-limit $node_array(1) $node_array([expr $i+$i+1]) 65536
    }
}

proc create-sources-destinations {} {
    global ns opt node_array tcp_app_src tp linuxcc f
    if { [string range $opt(tcp) 0 9] == "TCP/Linux/"} {
        set linuxcc [ string range $opt(tcp) 10 [string length $opt(tcp)] ]
        set opt(tcp) "TCP/Linux"
    }
    for {set i 1} {$i <= $opt(nTCPsrc)} {incr i} {
        set src [expr $i + $i]
        set dst [expr $src + 1]
        set tp($i) [$ns create-connection-list $opt(tcp) $node_array($src) $opt(sink) $node_array($dst) $i]
        set tcpsrc [lindex $tp($i) 0]
        set tcpsink [lindex $tp($i) 1]

    	$tcpsrc set fid_ [expr $i%1024]
        $tcpsrc set packetSize_ $opt(pktsize)
        $tcpsrc set window_ $opt(rcvwin)
        $tcpsrc set syn_ 0
        $tcpsrc set delay_growth_ 0
        if { $opt(tcp_app) == "Application/OnOff" } {
            set tcp_app_src($i) [new Application/OnOff $opt(ontype) $i $opt(pktsize) $opt(hdrsize) $opt(run) $opt(onavg) $opt(offavg) $tcpsrc]
            $tcp_app_src($i) attach-agent $tcpsrc
        } else {
            set tcp_app_src($i) [ new $opt(tcp_app) ]
            $tcp_app_src($i) attach-agent $tcpsrc
            $ns at 0 "$tcp_app_src($i) start"
        }
    }
    for {set i 1} {$i <= $opt(nUDPsrc)} {incr i} {
        set src [expr ($i + $opt(nTCPsrc))*2]
        set dst [expr $src + 1]
        set kp($i) [$ns create-connection-list UDP $node_array($src) Null $node_array($dst) $i]
        set udpsrc [lindex $kp($i) 0]
        set udpnull [lindex $kp($i) 1]
        $udpsrc set fid [expr ($i+$opt(nTCPsrc))%1024]
        set udp_app_src($i) [ new $opt(udp_app) ]
        if { $opt(udp_app) == "Application/Traffic/CBR"} {
            $udp_app_src($i) set packetSize_ opt(pktsize)
            $udp_app_src($i) set rate_ $opt(CBR_rate)
        }
        $udp_app_src($i) attach-agent $udpsrc
        $ns at 0 "$udp_app_src($i) start"

    }

}

proc finish {} {
    global ns opt tcp_app_src bt_aqm
    global f_tr f_qmon f_qtr f_nam
    if { $opt(tcp_app) == "Application/OnOff" } {
        for {set i 1} {$i <= [array size tcp_app_src]} {incr i} {
            if { [info exists opt(onoff_out)] } {
                $tcp_app_src($i) stats $opt(onoff_out)
            } else {
                $tcp_app_src($i) stats
            }
        }
    }
    #$bt_aqm  printstats

    $ns flush-trace

    if { [info exists f_tr] } {
        close $f_tr
    }

    if { [info exists f_qtr] } {
        close $f_qtr
    }

    if { [info exists f_nam] } {
        close $f_nam
    }
    if { [info exists f_qmon]} {
        close $f_qmon
    }
    exit 0
}
## MAIN ##

set conffile [lindex $argv 0]
puts $conffile
source $conffile
puts "Reading params from $conffile"

Getopt
set ns [new Simulator]

global defaultRNG

RandomVariable/Pareto set shape_ 0.5

if { [info exists opt(tr)] } {
    # if we don't set up tracing early, trace output isn't created!!
    set f [open $opt(tr) w]
    $ns trace-all $f
}

if { [info exists opt(nam)] } {
    set f_nam [open $opt(nam) w]
    $ns namtrace-all $f_nam
}

set opt(nsrc) [expr $opt(nTCPsrc) + $opt(nUDPsrc)]
create-dumbbell-topology
create-sources-destinations

$ns at $opt(simtime) "finish"
$ns run
