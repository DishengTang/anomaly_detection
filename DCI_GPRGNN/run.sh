# nohup python -u run_main.py --net COLA --RPMAX 5 --subgraph_size 5 --dataset YelpChi --device 0 > cola_yelp_5.log &
# nohup python -u run_main.py --net COLA --RPMAX 5 --subgraph_size 10 --dataset Amazon --device 1 > cola_amazon_10.log &
nohup python -u run_main.py --RPMAX 5 --dataset YelpChi --device 1 > dci_yelp.log &