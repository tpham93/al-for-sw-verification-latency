#!/bin/bash
#
#SBATCH --job-name=delayed_feedback_al
#SBATCH --output=log_slurm/slurm-%A.log
#SBATCH --get-user-env
#SBATCH --partition=run
#SBATCH --ntasks=100
#SBATCH --mem-per-cpu=8G

source ../../environment/bin/activate

# rm log/*

folder=H4
echo $folder > testname
mkdir 'result_'$folder


for budgetperc in 8 2 32
do
    for dataset_index in {0..11}
    do
        for algo in "rand" "var_uncer" "split" "pal" "FO+BI+var_uncer" "FO+BI+split" "FO+BI+pal" "FO+FI+var_uncer" "FO+FI+split" "FO+FI+pal"
        do
            for delay in "variable"
            do
                for i_rep in {0..9}
                do
                    for clf in "PWC"
                    do
                        for K in 3
                        do
                            for prior_exp in 0
                            do
                                experiment_name="D"$dataset_index"_d"$delay"_i"$i_rep"_a"$algo"_b"$budgetperc"_c"$clf"_p"$prior_exp"_K"$K
                                log_file="log/"$experiment_name".log"
                                export_file="result_"$folder"/"$experiment_name
                                param_string="with delay="$delay" budgetperc="$budgetperc" i_rep="$i_rep" algo="$algo" dataset_name=D"$dataset_index" clf_name="$clf" prior_exp="$prior_exp" K="$K" testdate="$folder" pkl_output_path="$export_file
                                
                                if test ! -f "$export_file" ; then
                                    srun -N1 -n1 -o $log_file python '-u' 'run_streamal.py' --name=$experiment_name $param_string  &
                                fi
                            done
                        done
                    done
                done
            done
        done
        wait
    done
done
wait
