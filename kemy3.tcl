# Experiments with on-off sources that transmit data for a certain
# "on" time and then are silent for a certain "off" time. The on and
# off times come from exponential distributions at specifiable rates.
# During the "on" period, the data isn't sent at a constant bit rate
# as in the existing exponential on-off traffic model in
# tools/expoo.cc but is instead sent according to the underlying
# transport (agent) protocol, such as TCP.  The "off" period is the
# same as in that traffic model.

#!/bin/sh
# the next line finds ns \
nshome=`dirname $0`; [ ! -x $nshome/ns ] && [ -x ../../../../ns ] && nshome=../../../..
# the next line starts ns \
export nshome; exec $nshome/ns "$0" "$@"

if [info exists env(nshome)] {
	set nshome $env(nshome)
} elseif [file executable ../../../../ns] {
	set nshome ../../../..
} elseif {[file executable ./ns] || [file executable ./ns.exe]} {
	set nshome "[pwd]"
} else {
	puts "$argv0 cannot find ns directory"
	exit 1
}
set env(PATH) "$nshome/bin:$env(PATH)"

source ./sender-app.tcl
source logging-app2.tcl
source stats.tcl

set conffile [lindex $argv 0]
#set conffile remyconf/vz4gdown.tcl
#set conffile remyconf/equisource.tcl

proc Usage {} {
    global opt argv0
    puts "Usage: $argv0 \[-simtime seconds\] \[-seed value\] \[-nsrc numSources\]"
    puts "\t\[-tr tracefile\]"
    puts "\t\[-bw $opt(bneck)] \[-delay $opt(delay)\]"
    exit 1
}

proc Getopt {} {
    global opt argc argv
#    if {$argc == 0} Usage
    for {set i 1} {$i < $argc} {incr i} {
        set key [lindex $argv $i]
        if ![string match {-*} $key] continue
        set key [string range $key 1 end]
        set val [lindex $argv [incr i]]
        set opt($key) $val
        if [string match {-[A-z]*} $val] {
            incr i -1
            continue
        }
    }
}
proc create-adhoc-topology {bneckbw delay} {
    global  ns s d opt wireless_config  accessrate accessdelay global
    set topology       [new Topography]
    $topology load_flatgrid $wireless_config(x) $wireless_config(y)
    create-god $opt(nsrc)

    set chan [new $wireless_config(chan)]
    $ns node-config -adhocRouting $wireless_config(rp) \
		-llType $wireless_config(ll) \
		-macType $wireless_config(mac) \
		-ifqType $wireless_config(ifq) \
		-ifqLen $wireless_config(ifqlen) \
		-antType $wireless_config(ant) \
		-propType $wireless_config(prop) \
		-phyType $wireless_config(netif) \
		-topoInstance $topology \
		-agentTrace ON \
		-routerTrace ON \
		-macTrace ON \
		-movementTrace OFF \
		-channel $chan

    #create src nodes
    for { set i 0 } { $i < $opt(nsrc) } { incr i } {
        set node_($i) [$ns node]
        $node_($i) random-motion 0

        # 30 defines the node size for nam
        $ns initial_node_pos $node_($i) 30

        # Define node initial position in nam

        # Telling nodes when the simulation ends
        $ns at $opt(simtime) "$node_($i) reset"
    }
    set god_ [God instance]
    source ./scen-n8-p0-M20-t100-x500-y500

    ###create destination node
    set d [$ns node]
    $d random-motion 0
    $ns initial_node_pos $d 50
    $d set X_ 150
    $d set Y_ 150
    $d set Z_ 0
    $ns at 0.1 "$node_(0) setdest 208.308566787323 258.102679738160 7.039599019158"
    $god_ set-dist  0 1 2
    ####

    for { set i 0 } { $i < $opt(nsrc) } { incr i } {

        set s($i) $node_($i)
    }

}
#
# Create a square topology.
#
#       O                  O
#        \                /
#          O------------O
#        / |            | \
#       O  |            |  O
#          |            |
#       O  |            |  O
#        \ |            | /
#          O------------O
#        /                \
#       O                  O
#
#
proc create-square-topology {bneckbw delay} {
    global ns opt tp src recvapp

    set nsender $opt(nsrc)
   #create gw
    for {set i 0} {$i < 4} {incr i} {
        set gws($i) [$ns node]
    }
    #create senders
    for {set i 0} {$i < $nsender} {incr i} {
        set senders($i) [$ns node]
    }
    #create receivers
    for {set i 0} {$i < 4} {incr i} {
        set receivers($i) [$ns node]
    }
    #create link between gws
    for {set i 0} {$i < 4} {incr i} {
        $ns duplex-link $gws($i) $gws([expr ($i+1)%4 ]) $bneckbw $delay  $opt(gw)
    }
    #create link between gw and sender
    for {set i 0} {$i < $nsender} {incr i} {
        $ns duplex-link $senders($i) $gws([expr $i/($nsender/4)]) 1000Mb 1ms $opt(gw)
    }
    #create link between gw and receiver
    for {set i 0} {$i < 4} {incr i} {
        $ns duplex-link $receivers($i) $gws($i) 1000Mb 1ms $opt(gw)
    }
   #apply tcl sink
    for {set i 0} {$i < $nsender} {incr i} {

         if {[expr $i%2] == 0} {
             set tp($i) [$ns create-connection-list $opt(tcp) $senders($i) $opt(sink) $receivers([expr (($i/($nsender/4))+2)%4]) $i]
         } else {
             set tp($i) [$ns create-connection-list $opt(tcp) $senders($i) $opt(sink) $receivers([expr (($i/($nsender/4))+1)%4]) $i]
         }
         set tcpsrc [lindex $tp($i) 0]
         set tcpsink [lindex $tp($i) 1]
         $tcpsrc set fid_ [expr $i%256]
         $tcpsrc set window_ $opt(rcvwin)
         $tcpsrc set packetSize_ $opt(pktsize)

         set src($i) [ $tcpsrc attach-app $opt(app) ]
         $src($i) setup_and_start $i $tcpsrc
         set recvapp($i) [new LoggingApp $i]
         $recvapp($i) attach-agent $tcpsink
         $ns at 0.0 "$recvapp($i) start"
    }


}

# Create a simple dumbbell topology.
#
proc create-dumbbell-topology {bneckbw delay} {
    global ns opt s gw d accessrate accessdelay nshome
    for {set i 0} {$i < $opt(nsrc)} {incr i} {
#        $ns duplex-link $s($i) $gw 10Mb 1ms DropTail
#        $ns duplex-link $gw $d $bneckbw $delay DropTail
        $ns duplex-link $s($i) $gw $accessrate($i) $accessdelay($i) $opt(gw)
        $ns queue-limit $s($i) $gw $opt(maxq)
        $ns queue-limit $gw $s($i) $opt(maxq)
        if { $opt(gw) == "XCP" } {
            # not clear why the XCP code doesn't do this automatically
            set lnk [$ns link $s($i) $gw]
            set q [$lnk queue]
            $q set-link-capacity [ [$lnk set link_] set bandwidth_ ]
            set rlnk [$ns link $gw $s($i)]
            set rq [$rlnk queue]
            $rq set-link-capacity [ [$rlnk set link_] set bandwidth_ ]
        }
    }
    if { $opt(link) == "trace" } {
        $ns simplex-link $d $gw [ bw_parse $bneckbw ] $delay $opt(gw)
#        [ [ $ns link $d $gw ] link ] trace-file "$nshome/link/tracedata/uplink-verizon4g.pps"
        source $nshome/link/trace.tcl
        $ns simplex-link $gw $d [ bw_parse $bneckbw ] $delay $opt(gw)
        [ [ $ns link $gw $d ] link ] trace-file $opt(linktrace)
    } else {
        $ns duplex-link $gw $d $bneckbw $delay $opt(gw)
    }
    if { [info exists opt(tr)] } {

	$ns trace-queue $gw $d
    set qm_file "qlens/"
    append qm_file $opt(gw)
    append qm_file ".out"
    set qm [open $qm_file w]
    set qmon [$ns monitor-queue $gw $d $qm 0.1]
    [$ns link $gw $d] queue-sample-timeout

    }

    #$ns trace-queue $gw $d
    #set qm_file $opt(gw)
    #append qm_file ".out"
    #set qm [open $qm_file w]
    #set qmon [$ns monitor-queue $gw $d $qm 0.1]
    #[$ns link $gw $d] queue-sample-timeout

    $ns queue-limit $gw $d $opt(maxq)
    $ns queue-limit $d $gw $opt(maxq)
    if { $opt(gw) == "XCP" } {
        # not clear why the XCP code doesn't do this automatically
        set lnk [$ns link $gw $d]
        set q [$lnk queue]
        $q set-link-capacity [ [$lnk set link_] set bandwidth_ ]
        set rlnk [$ns link $d $gw]
        set rq [$rlnk queue]
        $rq set-link-capacity [ [$rlnk set link_] set bandwidth_ ]
    }
}

proc create-adhoc-sources-sinks {} {
    global ns opt s src recvapp tp protocols protosinks f
    set numsrc $opt(nsrc)
    for {set i 0} {$i < $numsrc} {incr i} {
        set tp($i) [$ns create-connection-list $opt(tcp) $s($i) $opt(sink) $s([expr ($i + 1)%$numsrc  ]) $i]
        set tcpsrc [lindex $tp($i) 0]
        set tcpsink [lindex $tp($i) 1]
        $tcpsrc set fid_ [expr $i%256]

        $tcpsrc set window_ $opt(rcvwin)
        $tcpsrc set packetSize_ $opt(pktsize)
        if { [info exists opt(tr)] } {
            #$tcpsrc trace cwnd_
            #$tcpsrc trace rtt_
            #$tcpsrc trace maxseq_
            #$tcpsrc trace ack_
            #$tcpsrc attach $f
        }

        set src($i) [ $tcpsrc attach-app $opt(app) ]
        $src($i) setup_and_start $i $tcpsrc
        set recvapp($i) [new LoggingApp $i]
        $recvapp($i) attach-agent $tcpsink
        $ns at 0.0 "$recvapp($i) start"
    }
}
proc create-sources-sinks {} {
    global ns opt s d src recvapp tp protocols protosinks f linuxcc

    set numsrc $opt(nsrc)
    if { [string range $opt(tcp) 0 9] == "TCP/Linux/"} {
        set linuxcc [ string range $opt(tcp) 10 [string length $opt(tcp)] ]
        set opt(tcp) "TCP/Linux"
    }

    for {set i 0} {$i < $numsrc} {incr i} {

        set tp($i) [$ns create-connection-list $opt(tcp) $s($i) $opt(sink) $d $i]
        set tcpsrc [lindex $tp($i) 0]
        set tcpsink [lindex $tp($i) 1]

        $tcpsrc set fid_ [expr $i%256]
        $tcpsrc set window_ $opt(rcvwin)
        $tcpsrc set packetSize_ $opt(pktsize)

        if { [info exists opt(tr)] } {
            $tcpsrc trace cwnd_
            $tcpsrc trace rtt_
            $tcpsrc trace maxseq_
            $tcpsrc trace ack_
            if { $opt(tcp) == "TCP/Rational" } {
                $tcpsrc trace _intersend_time
            }
            $tcpsrc attach $f
        }

        set src($i) [ $tcpsrc attach-app $opt(app) ]
        $src($i) setup_and_start $i $tcpsrc
        set recvapp($i) [new LoggingApp $i]
        $recvapp($i) attach-agent $tcpsink
        $ns at 0.0 "$recvapp($i) start"
    }
}

proc finish {} {
    global ns opt stats src recvapp linuxcc
    global f nf
    for {set i 0} {$i < $opt(nsrc)} {incr i} {
        set sapp $src($i)
        $sapp dumpstats
        set rcdbytes [$recvapp($i) set nbytes_]
        set rcd_nrtt [$recvapp($i) set nrtt_]
        if { $rcd_nrtt > 0 } {
            set rcd_avgrtt [expr 1000.0*[$recvapp($i) set cumrtt_] / $rcd_nrtt ]
        } else {
            set rcd_avgrtt 0.0
        }
        if {$i == 0} {
            if {$opt(cycle_protocols) != "true"} {
                if { [info exists linuxcc] } {
                    puts "Results for $opt(tcp)/$linuxcc $opt(gw) $opt(sink) over $opt(simtime) seconds:"
                } else {
                    puts "Results for $opt(tcp) $opt(gw) $opt(sink) over $opt(simtime) seconds:"
                }
            } else {
                puts "Results for mix of protocols:"
            }
        }

        [$sapp set stats_] showstats $rcdbytes $rcd_avgrtt
    }

    if { [info exists f] } {
        $ns flush-trace
        close $f
    }
    if { [info exists nf] } {
        #$ns flush-trace
        close $nf
        #exec nam adhoc.nam &
    }
    exit 0
}

## MAIN ##

Agent/TCP set tcpTick_ .0001
Agent/TCP set timestamps_ true
set opt(hdrsize) 50
set opt(flowoffset) 40

source $conffile
puts "Reading params from $conffile"

Getopt
if { [info exists opt(gw)] } {
    set wireless_config(ifq) "Queue/$opt(gw)"
}
#puts "AMQ: $wireless_config(ifq)"
#set opt(rcvwin) [expr int(32*$opt(maxq))]

if { ![info exists opt(spike)] } {
    set opt(spike) false
}

if {$opt(spike) == "true"} {
    set opt(ontype) "time"
}

if { ![info exists opt(reset)] } {
    set opt(reset) true;    # reset TCP connection on end of ON period
}

set_access_params $opt(nsrc)

puts "case: $opt(bneck)"

global defaultRNG
$defaultRNG seed $opt(seed)

#    ns-random $opt(seed)

set ns [new Simulator]
Queue set limit_ $opt(maxq)
RandomVariable/Pareto set shape_ 0.5
set flowfile flowcdf-allman-icsi.tcl
if { $opt(ontype) == "flowcdf" } {
    source $flowfile
}

if {$opt(topo) == "square"} {
   if { [info exists opt(nam)] } {
        set nf [open square.nam w]
        $ns namtrace-all $nf
   }
   create-square-topology $opt(bneck) $opt(delay)

} elseif { $opt(topo) == "adhoc"} {
    if { [info exists opt(nam)] } {
        set nf [open adhoc.nam w]
        $ns namtrace-all-wireless $nf $wireless_config(x) $wireless_config(y)

    }
    set f     [open adhoc.tr w]
    $ns trace-all $f

#    set nf [open adhoc.nam w]
    #$ns namtrace-all-wireless $nf $wireless_config(x) $wireless_config(y)

    create-adhoc-topology $opt(bneck) $opt(delay)
    create-adhoc-sources-sinks
} else {
    if { [info exists opt(nam)] } {
        set nf [open out.nam w]
        $ns namtrace-all $nf
    }

    if { [info exists opt(tr)] } {
        # if we don't set up tracing early, trace output isn't created!!
        set f [open $opt(tr).tr w]
        $ns trace-all $f
    }


    # create sources, destinations, gateways
    for {set i 0} {$i < $opt(nsrc)} {incr i} {
        set s($i) [$ns node]
    }
    set d [$ns node];               # destination for all the TCPs
    set gw [$ns node];              # bottleneck router
    create-dumbbell-topology $opt(bneck) $opt(delay)
    create-sources-sinks
}


if { $opt(cycle_protocols) == true } {
    for {set i 0} {$i < $opt(nsrc)} {incr i} {
        puts "$i: [lindex $protocols $i]"
    }
}

$ns at $opt(simtime) "finish"
$ns run
