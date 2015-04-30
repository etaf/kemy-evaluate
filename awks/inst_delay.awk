{
    event = $1
    curr_time = $2
    pkt_id = $12
    if( event == "+" && send_times[pkt_id] == 0){
        send_times[pkt_id] = curr_time
    }
    if( event == "r"){
        printf("%f %f\n", curr_time, (curr_time - send_times[pkt_id])*1000 - 50)
    }
}
#BEGIN{
    #tic = 0.1
    #delay_sum = 0
    #cnt = 0
    #last_start =0

#}
#{
    #event = $1
    #curr_time = $2
    #pkt_id = $12
    #if( event == "+" && send_times[pkt_id] == 0){
        #send_times[pkt_id] = curr_time
    #}
    #if( event == "r"){
        #cnt = cnt + 1
        #delay_sum = delay_sum + (curr_time - send_times[pkt_id])*1000 - 50
        #if( (curr_time - last_start) >= tic){
            #printf("%f %f\n", curr_time, delay_sum/cnt)
            #last_start = curr_time
            #delay_sum = 0
            #cnt = 0
        #}
    #}
#}
