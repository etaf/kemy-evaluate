# Author:       ETAF
# input:        queue trace file
# output:       <time -> drop_rate > array
#               totall drop_rate
BEGIN {
    arrive_num = 0
    drop_num = 0
    cnt = 0
}

{
    event = $1
    curr_time = $2
    if( event == "+" || event == "d"){
        # packet arrive
        if(event == "+") {
            arrive_num++
        }
        # packet droped
        if ( event == "d") {
            drop_num++
        }
        times[cnt] = curr_time
        #printf("drop=%d arrive=%d",drop_num, arrive_num)
        drop_rates[cnt] = (float)drop_num/arrive_num
        cnt++
    }
}

END {
    for(i=0; i<cnt; ++i){
        printf("%f %f\n", times[i],drop_rates[i])
    }
    if( cnt == 0){
        drop_rate = 0
    }
    else{
        drop_rate = drop_rates[cnt-1]
        }
    #printf("%f\n",drop_rate)
}
