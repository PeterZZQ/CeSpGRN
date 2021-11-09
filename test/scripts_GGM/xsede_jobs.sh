kt=0

for interval in 1
do
    for seed in 0 1 2
    do 
        for param in {0..8}
        do
            # cp test_GGM_200-${kt}-${interval}-${seed}-${param}.job test_GGM_200-${kt}-1-${seed}-${param}.job 
            sbatch test_GGM_200-${kt}-${interval}-${seed}-${param}.job
        done
    done
done