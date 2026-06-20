#!/bin/bash

seed_list=()  
echo "  IC      Seed  " >> "IC_List.txt"
for sim in {2..50}
do
        while : 
        do
                seed=$((10000 + RANDOM % 99999))
                # Check if the seed is already in the list
                if [[ ! " ${seed_list[@]} " =~ " ${seed} " ]]; then
                seed_list+=("$seed")  # Add the unique seed to the list
                break
                fi
        done

        echo "Generating ICs" ${sim}
        echo "Current Seed Number:" ${seed}
        printf "%6d %8d\n" "$sim" "$seed">> "IC_List.txt"	
        mkdir -p "$sim" 
        cd "$sim" || exit 1
        cp ../1/{geometry.xyz,hessian.hdf5,INPORB,start.py,restart.py,Slurm.job} .		
        sed -i "s/seed=69884/seed=${seed}/" start.py
        sed -i "s/--job-name=IC1/--job-name=IC${sim}/" Slurm.job
        sbatch Slurm.job
        cd ..
        echo ""
done
