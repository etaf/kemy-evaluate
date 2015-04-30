#!/usr/bin/awk -f
#===============================================================================
#
#          File:  onoff_delay.awk
# 
#   Description:  
# 
#   VIM Version:  7.0+
#        Author:  YOUR NAME (), 
#  Organization:  
#       Version:  1.0
#       Created:  Saturday, April 18, 2015 15:48
#      Revision:  ---
#       License:  
#===============================================================================

{
    printf("%f\n",$5 * 1000 - rtt)
}
