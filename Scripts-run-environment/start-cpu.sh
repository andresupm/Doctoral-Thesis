#!/bin/bash

echo "--"
echo "-- Create Load Balancer OVS..."
cd ~/Escritorio/Load-Balancer-vnx
./init.sh

#Locate in the directory where it contains the VNX .xml file

cd ~   
cd ~/Escritorio/Load-Balancer-vnx
sudo vnx -f load-balacer11_lxc_ubuntu64.xml -v -t

sleep 10

cd ~

## Copy test file into the h1 container

#sudo scp ~/Escritorio/Load-Balancer-vnx/container-metrics/vnf-sharing-metrics.sh root@19$



sudo ovs-vsctl set bridge BalancerNet other_config:datapath-id=0000000000000001
sudo ovs-vsctl set-controller BalancerNet tcp:127.0.0.1:6633
#sudo ovs-vsctl set bridge BalancerNet stp_enable=true
#sudo ovs-vsctl set bridge TransportNet stp_enable=true
#sudo ovs-vsctl set bridge ExtNet stp_enable=true

## 1. Iniciar VNFs como Dockers:
echo "--"
echo "-- Docker Controller Starting..."
sudo docker run --name vnfT1 -t -d --cpus="1.0" --privileged vnf-img
sudo docker run --name vnfT2 -t -d --cpus="1.0" --privileged vnf-img

##################### VNFs Connections #####################
echo "--"
echo "--Connecting VNF-Dockers with OVS..."

## 1. Conectar VNFT1 a OVS
sudo ovs-docker add-port BalancerNet veth0 vnfT1
sudo ovs-docker add-port TransportNet veth1 vnfT1

## 2. Conectar VNFT2 a OVS
sudo ovs-docker add-port TransportNet veth0 vnfT2
sudo ovs-docker add-port ExtNet veth1 vnfT2

#sudo ovs-ofctl show LoadBalancerNet

##################### VNFs IP Settings #####################
echo "--"
echo "--Setting IP Address in VNFs..."

## 1. Setting IP Address in VNFT1
sudo docker exec -it vnfT1 /sbin/ifconfig veth1 192.168.100.2/24

## 2. Setting IP Address in VNFT2
sudo docker exec -it vnfT2 /sbin/ifconfig veth0 192.168.100.3/24

## 3. Setting OVSs in VNFT1
sudo docker exec -it vnfT1 /etc/init.d/openvswitch-switch restart
sudo docker exec -it vnfT1 ovs-vsctl add-br br0
sudo docker exec -it vnfT1 ovs-vsctl set bridge br0 other_config:datapath-id=0000000000000002
#sudo docker exec -it vnfT2 ovs-vsctl set-fail-mode br0 secure
sudo docker exec -it vnfT1 ovs-vsctl set-controller br0 tcp:172.17.0.1:6633
#sudo docker exec -it vnfT1 ovs-vsctl set bridge br0 stp_enable=true
sudo docker exec -it vnfT1 ovs-vsctl add-port br0 veth0
sudo docker exec -it vnfT1 ovs-vsctl add-port br0 vxlan -- set interface vxlan type=vxlan options:remote_ip=192.168.100.3 


## 4. Setting OVSs in VNFT2
sudo docker exec -it vnfT2 /etc/init.d/openvswitch-switch restart
sudo docker exec -it vnfT2 ovs-vsctl add-br br0
sudo docker exec -it vnfT2 ovs-vsctl set bridge br0 other_config:datapath-id=0000000000000004
#sudo docker exec -it vnfT2 ovs-vsctl set-fail-mode br0 secure
sudo docker exec -it vnfT2 ovs-vsctl set-controller br0 tcp:172.17.0.1:6633
#sudo docker exec -it vnfT2 ovs-vsctl set bridge br0 stp_enable=true
sudo docker exec -it vnfT2 ovs-vsctl add-port br0 veth1 -- set interface veth1 ofport=1
#sudo docker exec -it vnfT2 ovs-vsctl add-port br0 vxlan -- set interface vxlan type=vxlan options:remote_ip=192.168.100.2 #options:key=inet options:dst_port=8742
sudo docker exec -it vnfT2 ovs-vsctl add-port br0 vxlan -- set interface vxlan type=vxlan options:remote_ip=192.168.100.2 ofport=2

#sudo docker exec -it vnfT2 route add -net 192.100.10.0 netmask 255.255.255.0 gw 192.168.30.3




#################### Share Section ###################

## 1. Iniciar VNFs como Dockers:
sudo docker run --name vnfT1-2 -t -d --cpus="1.0" --privileged vnf-img

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
sudo docker exec -it vnfT1-2 ovs-vsctl add-port br0 vxlan2 -- set interface vxlan2 type=vxlan options:remote_ip=192.168.100.3 #options:key=inet options:dst_port=8742

## 3. Setting OVSs in VNFT2


#sudo docker exec -it vnfT2 ovs-vsctl set bridge br0 stp_enable=true
sudo docker exec -it vnfT2 ovs-vsctl add-port br0 vxlan2 -- set interface vxlan2 type=vxlan options:remote_ip=192.168.100.4 ofport=3 #options:key=inet options:dst_port=8742
#sudo docker exec -it vnfT2 ovs-vsctl add-port br0 vxlan2 -- set interface vxlan2 type=vxlan optio$





##################### VNFs Settings #####################
#echo "--"
#echo "--Setting Iperf in VNFs--"
#sudo docker exec -it vnfT2 chmod 777 /usr/bin/vnx_config_nat
#sudo docker exec -it vnfT2 /usr/bin/vnx_config_nat br0 veth1

echo "--"
echo "--Done--"

#######################Iperf Test #########################3
#iperf -u -c 10.2.3.30 -T -b 2M -t 10 -i1 -d


