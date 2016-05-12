#!/usr/bin/env bash

echo starting script

EXP_CTR=0
while [ $EXP_CTR -lt 65 ]; do
    CTR=0
    while [ $CTR -lt 100 ]; do
        echo attempting to convert and delete image #$CTR
        convert experiment#$EXP_CTR\/images/image#$CTR experiment#$EXP_CTR/images/image#$CTR.png

        convert experiment#$EXP_CTR\/pseudopatterns_I/image#$CTR\_input experiment#$EXP_CTR\/pseudopatterns_I/image#$CTR\_input.png
        convert experiment#$EXP_CTR\/pseudopatterns_I/image#$CTR\_output experiment#$EXP_CTR\/pseudopatterns_I/image#$CTR\_output.png

        convert experiment#$EXP_CTR\/pseudopatterns_II/image#$CTR\_input experiment#$EXP_CTR\/pseudopatterns_II/image#$CTR\_input.png
        convert experiment#$EXP_CTR\/pseudopatterns_II/image#$CTR\_output experiment#$EXP_CTR\/pseudopatterns_II/image#$CTR\_output.png
        rm experiment#$EXP_CTR\/images/image#$CTR
        rm experiment#$EXP_CTR\/pseudopatterns_I/image#$CTR\_input
        rm experiment#$EXP_CTR\/pseudopatterns_I/image#$CTR\_output
        rm experiment#$EXP_CTR\/pseudopatterns_II/image#$CTR\_input
        rm experiment#$EXP_CTR\/pseudopatterns_II/image#$CTR\_output
        let CTR=CTR+1
        done
    let EXP_CTR=EXP_CTR+1
    done

echo SUCCESS!

