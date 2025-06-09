#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=cpu
#SBATCH -J validator
#SBATCH -n 40                 # 总核数 40
#SBATCH --ntasks-per-node=40   # 每节点核数
#SBATCH --qos=cpu     


# this shell use CPU and just validate one checkpoint
source activate retro_branch_dt

root_dir=$(pwd)
echo "root_dir:"${root_dir}

# 使用date命令生成带日期时间的字符串
current_time=$(date +"%Y-%m-%d_%H-%M-%S")

# 构建带有日期时间的文件名
filename="validator_log__${current_time}.txt"

# checkpoint_path
total_path=/lab/shiyh_lab/12332470/code/retro_branching_offline/outputs/2024-07-19/02-47-16/gail_learner/gail_gnn/gail_gnn_0/

# 遍历文件夹，对每个checkpoint validate
#for item in "$total_path"/*; do
#倒序遍历，从最新的checkpoint开始
for item in $(find "$total_path" -mindepth 1 -maxdepth 1 -type d -name "checkpoint_*" | sort -V -r); do
  if [ -d "$item" ] && [[ $item == *"checkpoint"* ]]; then  # 如果是目录
    #echo "Found directory: $item"
    path="${item}/"
    #echo ${path}
    if [ ! -d "$path/rl_validator" ]; then
	echo "begin validate $path"
    	python experiments/validator.py --config-path=configs --config-name=validator.yaml \
      environment.observation_function=default  \
      instances.co_class=set_covering instances.co_class_kwargs.n_rows=500 instances.co_class_kwargs.n_cols=1000 \
      experiment.agent_name=gail experiment.path_to_load_agent=${path} \
      experiment.path_to_load_instances=${root_dir}/retro_branching_paper_validation_instances \
      experiment.path_to_save=${path} \
      experiment.device=cpu  | tee "${path}/${filename}"
    fi    
  fi
done


