# Experiments with on-off sources that transmit data for a certain
# "on" time and then are silent for a certain "off" time. The on and # off times come from exponential distributions at specifiable rates.
# During the "on" period, the data isn't sent at a constant bit rate
# as in the existing exponential on-off traffic model in
# tools/expoo.cc but is instead sent according to the underlying
# transport (agent) protocol, such as TCP.  The "off" period is the
# same as in that traffic model.

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

#Create a simulator object
set ns [new Simulator]

#Define different colors for data flows (for NAM)
$ns color 1 Blue
$ns color 2 Red

#Open the NAM trace file
#set nf [open out.nam w]
#set tr [open out.tr w]
#$ns namtrace-all $nf
#$ns trace-all $tr
#Define a 'finish' procedure
proc finish {} {
        global ns nf kemyq onoff_ftp mq_f tr

        #get utiltiy
        #set utility 0
        #for ... get utility
        #$onoff_ftp stats
        #$kemyq printstats $utility
        #
        #$onoff_ftp stats
        $onoff_ftp stats "/home/lxa/test.out"
        $ns flush-trace
        #Close the NAM trace file
        #close $nf
        close $mq_f
        #close $tr
        #Execute NAM on the trace file
        #exec ../../../../bin/nam out.nam &
        exit 0
}

#Create four nodes
set n0 [$ns node]
set n1 [$ns node]
set n2 [$ns node]
set n3 [$ns node]

set gw "PIE"
#Create links between the nodes
$ns duplex-link $n0 $n2 2000Mb 10ms $gw
$ns duplex-link $n1 $n2 2000Mb 10ms $gw
$ns duplex-link $n2 $n3 0.1Mb 20ms $gw
set kemyq [[$ns link $n2 $n3] queue]
set mylink [[$ns link $n2 $n3] link]
$mylink set bandwidth_ 1000Mb
set mq_f [open mq.out w]
$ns trace-queue $n2 $n3 $mq_f

#Set Queue Size of link (n2-n3) to 10
$ns queue-limit $n2 $n3 10000000

#Give node position (for NAM)
$ns duplex-link-op $n0 $n2 orient right-down
$ns duplex-link-op $n1 $n2 orient right-up
$ns duplex-link-op $n2 $n3 orient right

#Monitor the queue for link (n2-n3). (for NAM)
$ns duplex-link-op $n2 $n3 queuePos 0.5


#Setup a TCP connection
set tcp [new Agent/TCP]
$tcp set class_ 2
$ns attach-agent $n0 $tcp
set sink [new Agent/TCPSink]
$ns attach-agent $n3 $sink
$ns connect $tcp $sink
$tcp set fid_ 1

#Setup a FTP over TCP connection
set onoff_ftp [new Application/OnOff "bytes" 1 1200 50 1  50000.0 0.2 $tcp]

$onoff_ftp attach-agent $tcp


#Setup a UDP connection
set udp [new Agent/UDP]
$ns attach-agent $n1 $udp
set null [new Agent/Null]
$ns attach-agent $n3 $null
$ns connect $udp $null
$udp set fid_ 2

#Setup a CBR over UDP connection
set cbr [new Application/Traffic/CBR]
$cbr attach-agent $udp
$cbr set type_ CBR
$cbr set packet_size_ 1000
$cbr set rate_ 100mb
$cbr set random_ false


#Schedule events for the CBR and FTP agents
$ns at 0.1 "$cbr start"
$ns at 1.0 "$onoff_ftp start"
$ns at 100.0 "$mylink set bandwidth_ 10Mb"
$ns at 400.0 "$onoff_ftp stop"
$ns at 400.5 "$cbr stop"

#Detach tcp and sink agents (not really necessary)
$ns at 400.5 "$ns detach-agent $n0 $tcp ; $ns detach-agent $n3 $sink"

#Call the finish procedure after 5 seconds of simulation time
$ns at 401 "finish"

#Print CBR packet size and interval
#puts "CBR packet size = [$cbr set packet_size_]"
#puts "CBR interval = [$cbr set interval_]"

#Run the simulation
$ns run

