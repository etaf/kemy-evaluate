#!/usr/bin/awk -f
#===============================================================================
#
#          File:  bw_delay.awk
# 
#   Description:  
# 
#   VIM Version:  7.0+
#        Author:  YOUR NAME (), 
#  Organization:  
#       Version:  1.0
#       Created:  Wednesday, April 22, 2015 07:57
#      Revision:  ---
#       License:  
#===============================================================================
BEGIN{
    M = iter * senders
}
{
    sum += $5
    if(NR%M == 0) {
        printf("%f\n",sum/M * 1000 - rtt);
        sum = 0;
    }
}
