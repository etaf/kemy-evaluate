BEGIN{
}
{
    event = $1
    curr_time = $2
    pkt_id = $12
    if( event == "+" && send_times[pkt_id] == 0){
        send_times[pkt_id] = curr_time
    }
    if( event == "r"){
        recv_times[pkt_id] = curr_time
        printf("%f %f\n", curr_time, (curr_time - send_times[pkt_id] - 0.1)*1000)
    }
}

END{
    total_delay = 0
    recv_num = 0
    for( pid in recv_times){
        if (send_times[pid] == 0) {
            printf("\nError in delay.awk: receiving a packet that wasn't sent %g\n",pid)
        }
        delay = recv_times[pid] - send_times[pid]
        recv_num++
        total_delay += delay
        #printf("%f %f\n",recv_times[pid],delay*1000)
    }
    if(recv_num == 0){
        avg_delay = 0
    }
    else{
        avg_delay = (float)total_delay / recv_num
        }
    #printf("average dealy: %f\n",avg_delay)
}
