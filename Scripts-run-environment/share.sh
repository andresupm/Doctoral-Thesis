#!/bin/bash
echo "--"
echo "-- Create VNF-Share Proccess..."

## 1. Iniciar VNFs como Dockers:
sudo docker run --name vnfT1-2 -t -d --privileged vnf-img

## 3. Conectar VNFT1-2 a OVS
sudo ovs-docker add-port BalancerNet veth0 vnfT1-2
sudo ovs-docker add-port TransportNet veth1 vnfT1-2

##VNFT1-2 settings
sudo docker exec -it vnfT1-2 /sbin/ifconfig veth1 192.168.100.4/24

## 3. Setting IP Address in VNFT1-2
sudo docker exec -it vnfT1-2 /etc/init.d/openvswitch-switch restart
sudo docker exec -it vnfT1-2 ovs-vsctl add-br br0
sudo docker exec -it vnfT1-2 ovs-vsctl set-controller br0 tcp:172.17.0.1:6633
sudo docker exec -it vnfT1-2 ovs-vsctl set bridge br0 other_config:datapath-id=0000000000000003
#sudo docker exec -it vnfT2 ovs-vsctl set-fail-mode br0 secure
#sudo docker exec -it vnfT1-2 ovs-vsctl set bridge br0 stp_enable=true
sudo docker exec -it vnfT1-2 ovs-vsctl add-port br0 veth0
sudo docker exec -it vnfT1-2 ovs-vsctl add-port br0 vxlan2 -- set interface vxlan2 type=vxlan options:remote_ip=192.168.100.3 options:key=inet options:dst_port=8742

## 3. Setting OVSs in VNFT2

#sudo docker exec -it vnfT2 ovs-vsctl set bridge br0 stp_enable=true
sudo docker exec -it vnfT2 ovs-vsctl add-port br0 vxlan2 -- set interface vxlan2 type=vxlan options:remote_ip=192.168.100.4 ofport=3 options:key=inet options:dst_port=8742
#sudo docker exec -it vnfT2 ovs-vsctl add-port br0 vxlan2 -- set interface vxlan2 type=vxlan options:remote_ip=192.168.100.4 ofport=3

################ ARP OVS #####################3

#sudo docker exec -it vnfT2 ovs-ofctl add-flow br0 "priority=1000,arp,nw_dst=10.2.2.20,in_port=LOCAL,actions=output:1"
#sudo docker exec -it vnfT2 ovs-ofctl add-flow br0 "priority=1000,arp,nw_dst=10.2.2.21,in_port=LOCAL,actions=output:1"
#sudo docker exec -it vnfT2 ovs-ofctl add-flow br0 "priority=1000,arp,nw_dst=10.2.2.22,in_port=LOCAL,actions=output:1"
#sudo docker exec -it vnfT2 ovs-ofctl add-flow br0 "priority=1000,arp,nw_dst=10.2.2.23,in_port=LOCAL,actions=output:1"

############ ARP BalancerNet OVS ################3

#sudo ovs-ofctl add-flow BalancerNet "priority=1000,arp,nw_dst=10.2.2.20,in_port=LOCAL,actions=output:1"
#sudo ovs-ofctl add-flow BalancerNet "priority=1000,arp,nw_dst=10.2.2.21,in_port=LOCAL,actions=output:2"
#sudo ovs-ofctl add-flow BalancerNet "priority=1000,arp,nw_dst=10.2.2.22,in_port=LOCAL,actions=output:3"
#sudo ovs-ofctl add-flow BalancerNet "priority=1000,arp,nw_dst=10.2.2.23,in_port=LOCAL,actions=output:4"

#sudo docker exec -it vnfT2 chmod 777 /usr/bin/vnx_config_nat
#sudo docker exec -it vnfT2 /usr/bin/vnx_config_nat br0 veth1


##################### OVS Flowentries Settings #####################
echo "--"
echo "--Setting flows in LoadBalancerNet OVS..."

#sudo ovs-ofctl -Oopenflow13 add-group BalancerNet "group_id=100 type=select selection_method=dp_hash bucket=output:5 bucket=output:6"

#sudo ovs-ofctl -Oopenflow13 add-flow BalancerNet "table=0,ip,nw_src=10.2.2.0/24,actions=group:100"

#sudo ovs-ofctl -Oopenflow13 add-flow BalancerNet "priority=100,in_port=1,actions=group:100"
#sudo ovs-ofctl -Oopenflow13 add-flow BalancerNet "priority=100,in_port=2,actions=group:100"
#sudo ovs-ofctl -Oopenflow13 add-flow BalancerNet "priority=100,in_port=3,actions=group:100"
#sudo ovs-ofctl -Oopenflow13 add-flow BalancerNet "priority=100,in_port=4,actions=group:100"

#sudo ovs-ofctl add-flow BalancerNet "priority=1000,arp,nw_dst=10.2.2.20,in_port=LOCAL,actions=output:1"
#sudo ovs-ofctl add-flow BalancerNet "priority=1000,arp,nw_dst=10.2.2.21,in_port=LOCAL,actions=output:1"
#sudo ovs-ofctl add-flow BalancerNet "priority=1000,arp,nw_dst=10.2.2.22,in_port=LOCAL,actions=output:1"
#sudo ovs-ofctl add-flow BalancerNet "priority=1000,arp,nw_dst=10.2.2.23,in_port=LOCAL,actions=output:1"


#sudo ovs-ofctl add-flow BalancerNet "priority=100,ip,nw_dst=10.2.2.20,actions=output:1"
#sudo ovs-ofctl add-flow BalancerNet "priority=100,ip,nw_dst=10.2.2.21,actions=output:2"
#sudo ovs-ofctl add-flow BalancerNet "priority=100,ip,nw_dst=10.2.2.22,actions=output:3"
#sudo ovs-ofctl add-flow BalancerNet "priority=100,ip,nw_dst=10.2.2.23,actions=output:4"

#sudo ovs-ofctl -Oopenflow13 add-flow BalancerNet "priority=1000,ip,nw_dst=10.2.2.20,in_port=LOCAL,actions=output:1"
#sudo ovs-ofctl -Oopenflow13 add-flow BalancerNet "priority=1000,ip,nw_dst=10.2.2.21,in_port=LOCAL,actions=output:2"
#sudo ovs-ofctl -Oopenflow13 add-flow BalancerNet "priority=1000,ip,nw_dst=10.2.2.22,in_port=LOCAL,actions=output:3"
#sudo ovs-ofctl -Oopenflow13 add-flow BalancerNet "priority=1000,ip,nw_dst=10.2.2.23,in_port=LOCAL,actions=output:4"


################# Flows filter based on IP source ################

#sudo ovs-ofctl -Oopenflow13 add-flow BalancerNet "table=0,ip,nw_src=10.2.2.20/24,actions=output:5"
#sudo ovs-ofctl -Oopenflow13 add-flow BalancerNet "table=0,ip,nw_src=10.2.2.21/24,actions=output:5"

#sudo ovs-ofctl -Oopenflow13 add-flow BalancerNet "table=0,ip,nw_src=10.2.2.22/24,actions=output:6"
#sudo ovs-ofctl -Oopenflow13 add-flow BalancerNet "table=0,ip,nw_src=10.2.2.23/24,actions=output:6"



#sudo ovs-ofctl -Oopenflow13 add-flow balancer "table=0,ip,nw_dst=192.168.30.0/24,actions=group:100"

