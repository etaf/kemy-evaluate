#Author:        ETAF
#input:         queue trace file
#output:        <time,througput> arrs

BEGIN{
    start_time = 1e6
    recv_sum = 0
}
{
    event = $1
    curr_time = $2
    pkt_size = $6

    if(curr_time < start_time){
        start_time = curr_time
    }

    #receive
    if(event == "r"){
        recv_sum += pkt_size
        printf("%f %f\n", curr_time, (float)recv_sum / (curr_time - start_time)*8/1000000)
    }
}

