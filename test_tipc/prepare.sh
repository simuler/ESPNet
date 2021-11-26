#!/bin/bash
source test_tipc/common_func.sh

FILENAME=$1

# MODE be one of ['lite_train_lite_infer' 'lite_train_whole_infer' 'whole_train_whole_infer',  
#                 'whole_infer', 'klquant_whole_infer',
#                 'cpp_infer', 'serving_infer']

MODE=$2

dataline=$(cat ${FILENAME})

# parser params
IFS=$'\n'
lines=(${dataline})

# The training params
model_name=$(func_parser_value "${lines[1]}")

trainer_list=$(func_parser_value "${lines[14]}")

# MODE be one of ['lite_train_infer' 'whole_infer' 'whole_train_infer']
MODE=$2
if [ ${MODE} = "infer" ];then
    model_path=test_tipc/output/norm_gpus_0_autocast_null/
    rm -rf $model_path
    wget -nc -P $model_path  https://paddleseg.bj.bcebos.com/dygraph/humanseg/train/fcn_hrnetw18_small_v1_humanseg_192x192.zip
    cd $model_path && unzip fcn_hrnetw18_small_v1_humanseg_192x192.zip && mv fcn_hrnetw18_small_v1_humanseg_192x192/model.pdparams . && cd ../../../
fi
rm -rf ./test_tipc/data/mini_city
#wget -nc -P ./test_tipc/data/ https://paddleseg.bj.bcebos.com/humanseg/data/mini_supervisely.zip
cd .. && cp -r ./data/data11077/ ./PaddleSeg/test_tipc/data/ && cd ./PaddleSeg
# cd ./test_tipc/data/ && unzip mini_city.zip && cd ../../