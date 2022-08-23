#!/bin/bash
echo "--"
echo "--Destroy VNX scenario--"
cd ~   
cd ~/Escritorio/Load-Balancer-vnx
sudo vnx -f load-balacer11_lxc_ubuntu64.xml -v  --destroy 

echo "--"
echo "--Destroy VNFs--"
sudo docker rm --force vnfT1
sudo docker rm --force vnfT2
sudo docker rm --force vnfT1-2

echo "--"
echo "--Destroy OVSs--"
sudo ovs-vsctl del-br BalancerNet
sudo ovs-vsctl del-br TransportNet
sudo ovs-vsctl del-br Transport2
sudo ovs-vsctl del-br ExtNet

