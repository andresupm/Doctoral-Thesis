#ovs-vsctl -- set Port 8b92ab2732bf4_l qos=@newqos \
#	 -- --id=@newqos create QoS type=linux-htb other-config:max-rate=250000000 queues=0=@q0\
#	 -- --id=@q0 create Queue other-config:min-rate=8000000 other-config:max-rate=150000000\

#ovs-vsctl -- set Port 8c4fbac2bbe34_l qos=@defaultqos \
#	-- --id=@defaultqos create QoS type=linux-htb other-config:max-rate=300000000 queues=1=@q1\
#	 -- --id=@q1 create Queue other-config:min-rate=5000000 other-config:max-rate=200000000

echo "-- Setting bandwidth: 100Mb in VNFT1 and 50Mb in VNFT1-2---"
echo "----"
echo "---- Set queue in port 2 from OVS-BalancerNet "
echo "----"
ovs-vsctl -- set Port 8ba52bd949fb4_l qos=@newqos\
 -- --id=@newqos create QoS type=linux-htb other-config:max-rate=300000000 queues=0=@q0\
 -- --id=@q0 create Queue other-config:min-rate=3000000 other-config:max-rate=290000000\

echo "----"
echo "---- Set queue in port 3 from OVS-BalancerNet "
echo "----"
ovs-vsctl -- set Port ceac488cb62f4_l qos=@defaultqos\
 -- --id=@defaultqos create QoS type=linux-htb other-config:max-rate=150000000 queues=1=@q1\
 -- --id=@q1 create Queue other-config:min-rate=8000000 other-config:max-rate=140000000

echo "----"
echo "---- List queues from OVS-BalancerNet "
echo "----"

ovs-vsctl list queue
