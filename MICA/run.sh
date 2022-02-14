nohup python -u run_GNN.py --epochs 1000 --net SupCL --lambda_encoder gcn --device 1 --RPMAX 20 > yelp_add.log &
nohup python -u run_GNN.py --epochs 1200 --net SupCL --lambda_encoder gcn --dataset Amazon --device 0 --lr 0.0001 --RPMAX 20 > amazon_add.log &


nohup python -u run_GNN.py --epochs 1200 --net SupCL --lambda_encoder gcn --dataset Amazon --device 0 --lr 0.0001 --max_freqs 5 --RPMAX 20 > amazon.log &
nohup python -u run_GNN.py --epochs 1000 --net SupCL --lambda_encoder gcn --device 1 --lr 0.002 --max_freqs 80 --RPMAX 20 > yelp_multi.log &