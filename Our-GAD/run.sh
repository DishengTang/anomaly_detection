nohup python -u run_GNN.py --epochs 1000 --net SupCL --lambda_encoder gcn --device 1 > yelp_pretrain.log &
nohup python -u run_GNN.py --epochs 1000 --net SupCL --lambda_encoder gcn --dataset Amazon --device 0 > amazon_pretrain.log &
