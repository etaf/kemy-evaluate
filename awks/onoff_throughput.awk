
{
    #print $2
    sum = sum + $2
    if(NR%senders == 0){
        printf("%f\n",sum/senders)
        sum = 0
    }
}

