source activate l2sn
for epoch in 1 2 3
do
    for batch_train in  256 128 64 32 16 8 1
    do
        for lr in 0.005 0.001
        do
            python learning/train_fcmcnf.py -batch_train $batch_train -n_epoch $epoch -lr $lr -problem FCMCNF
            rm -rf ./stats
            python main.py -n_cpu 8 -data_partition test -problem FCMCNF
        done
    done
done