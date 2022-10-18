#!/bin/bash

# define the experiment
experiment(){
    # root=$1
    per_class_num=$1
    sound_save_path=$2
    mnist_save_path=$3
    soundmnist_save_path=$4
    sound_mean_save_path=$5
    sound_mean_name=$6
    metadropout_save_path=$7
    finetune_save_path=$8
    batch_size=$9
	vis_device=$10
    
	printf "==> start training with missing modality\n"
	printf "\n==> start training with missing modality\n" >> experiment-$vis_device:per-class-num-$per_class_num.txt   
	
	python train_missing_eval_missing.py \
		--checkpoint $metadropout_save_path \
		--per_class_num $per_class_num \
		--sound_mean_path $sound_mean_save_path \
		--sound_mean_name $sound_mean_name \
		--batch_size $batch_size \
		--vis_device $vis_device \
		# --soundmnist_model_path $best_soundmnist_path \
		--soundmnist_model_name $best_soundmnist_name >> experiment-$vis_device:per-class-num-$per_class_num.txt   
	
	printf "==> end training with missing modality\n"
	printf "\n==> end training with missing modality\n" >> experiment-$vis_device:per-class-num-$per_class_num.txt   

	printf "==========> end experiment $per_class_num: %(%Y-%m-%d %H:%M:%S)T <==========\n" >> experiment-$vis_device:per-class-num-$per_class_num.txt    
	printf "==========> end experiment $per_class_num: %(%Y-%m-%d %H:%M:%S)T <==========\n"

}

run (){
	experiment "$@"
}

run 15 ./save/sound/450/new/15 ./save/mnist/450/new/15 ./save/soundmnist/new/15 ../Output/ meta_mean.npy ./save/metadrop/feature/new/15 ./save/finetune/15 64 4