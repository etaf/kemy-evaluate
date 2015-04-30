#!/usr/bin/awk -f
#===============================================================================
#
#          File:  bw_throughput.awk
# 
#   Description:  
# 
#   VIM Version:  7.0+
#        Author:  YOUR NAME (), 
#  Organization:  
#       Version:  1.0
#       Created:  Wednesday, April 22, 2015 07:43
#      Revision:  ---
#       License:  
#===============================================================================
BEGIN{
    M = iter * senders
}
{
    sum += $2
    if(NR%M == 0) {
        printf("%f\n",sum/M);
        sum = 0;
    }
}
