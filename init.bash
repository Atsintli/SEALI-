#/bin/bash
export models=(test_model)

puerto=8501
for model in ${models[@]}
do
  tensorflow_model_server --rest_api_port=$puerto --model_name=improv_class --model_base_path=$(pwd)/$model &
  ((puerto=puerto+10)) #to increment port number
done

sleep 1
python3 05_extract_features_in_realtime+OSC_sender.py
