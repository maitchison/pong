#!/bin/bash

export MKL_NUM_THREADS=${2:-1}
export NUMEXPR_NUM_THREADS=${2:-1}                                        export OMP_NUM_THREADS=${2:-1}

python3 pong.py train $1

