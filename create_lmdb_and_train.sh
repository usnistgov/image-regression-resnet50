#!/bin/bash

lmdb_dir="/lmdb"
db_name="regression"

#arguments to create the lmdb and add csv file name to csv folder path
buildParam="$1 $2 $3 $4/$6"

#arguments for training
trainParam=""

args_index=0
for i in $@;
do
	if [ $args_index -ge 6 ]
	then
		trainParam="${trainParam} $i"
	fi
        args_index=$(($args_index + 1))	
done

python3 build_lmdb.py --output_folder ${lmdb_dir} --dataset_name ${db_name} ${buildParam} &&  python3 train.py --train_database ${lmdb_dir}/train-${db_name}.lmdb --test_database ${lmdb_dir}/test-${db_name}.lmdb ${trainParam}
