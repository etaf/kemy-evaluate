#Author:        ETAF
#input:         queue trace file
#output:        <time,througput> arrs

BEGIN{
    start_time = -1
    end_time = 0
    sum = 0
}
{
    event = $1
    curr_time = $2
    pkt_size = $6
    
    #receive
    if( event == "+")
    {
        if(start_time == -1)
        {
            start_time = curr_time
        }
    }
    if(event == "r"){
        sum += pkt_size
    }

    if(end_time < curr_time)
        end_time = curr_time
}
END{
    throughput = 0
#    for(i=1;i<cnt;++i){
        #througput = (float)pkt_sum[i] /( times[i]-start_time) * 8 /1024/1024
        #printf("%f %f\n",times[i],througput)
    #}
    printf("%fMbps\n",sum/(end_time - start_time)*8/1000000 )
}
