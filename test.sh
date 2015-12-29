#!/bin/sh

function assert_eq() {
    a=`echo $1 | cut -d: -f$2`
    b=`echo $3 | cut -d: -f$4`

    [ "$a" != "$b" ] && {
        echo "assert() failed"
        echo $1
        echo $a
        echo $3
        echo $b
        return 1
    }
    return 0
}


function validateSimple() {
    assert_eq "$1" 3 "$2" 4 || return 1
    return 0
}

function doSimpleTest() {
    spec=$1
    shift

    cpu=`$CMD $*`
    abort=`echo $cpu | cut -d: -f2 | cut -d, -f2`
    [ "$abort" = "1" ] && {
        echo "simplex aborted..."
        echo "$CMD $*"
        return 0
    }
    gpu=`$CMD $GPU $*`
    validateSimple "$cpu" "$gpu" || {
        echo "$CMD $*"
        return 1
    }

    cpu_itr=`echo $cpu | cut -d: -f2 | cut -d, -f1`
    gpu_itr=`echo $gpu | cut -d: -f2 | cut -d, -f1`
    cpu_abt=`echo $cpu | cut -d: -f2 | cut -d, -f2`
    gpu_abt=`echo $gpu | cut -d: -f2 | cut -d, -f2`
    cpu_opt=`echo $cpu | cut -d: -f2 | cut -d, -f3`
    gpu_opt=`echo $gpu | cut -d: -f2 | cut -d, -f3`
    cpu_time=`echo $cpu | cut -d: -f1`
    gpu_time=`echo $gpu | cut -d: -f1`

    echo $spec,$cpu_itr,$gpu_itr,$cpu_abt,$gpu_abt,$cpu_opt,$gpu_opt,$cpu_time,$gpu_time >> stats.log

    cpu_t=`echo $cpu_time | cut -d, -f1`
    gpu_t=`echo $gpu_time | cut -d, -f1`
    echo -e "    itr (cpu/gpu): $cpu_itr / $gpu_itr\t`awk \"BEGIN {printf \\\"%.2f\\\",${cpu_t}/${gpu_t}}\"`"
    return 0
}


CMD="./simplex -o /dev/null -W"
GPU="-p1 -c"

########################################################

echo -n > stats.log

seed=3495
v=100
while [ $v -lt 1000 ]; do
    c=$(((4*v)/5))
    max=$((v+v/2))
    while [ $c -le $max ]; do
        echo "--Rv${v}c${c}--"
        doSimpleTest $v,$c -R $v,$c,20,I,0.4,$seed -M$((v*200)) # || exit 1
        c=$((c+v/10))
    done
    
    v=$((v+10))
done

while [ $v -lt 15000 ]; do
    c=$(((4*v)/5))
    max=$((v+v/2))
    while [ $c -le $max ]; do
        echo "--Rv${v}c${c}--"
        doSimpleTest $v,$c -R $v,$c,20,I,0.4,$seed -M$((v*200)) # || exit 1
        c=$((c+v/10))
    done
    
    v=$((v+100))
done

exit 0

