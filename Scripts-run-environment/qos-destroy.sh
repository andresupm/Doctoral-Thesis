### List QoS and Queues

sudo ovs-vsctl list queue
sudo ovs-vsctl list qos

#### Destroy Qos All

sudo ovs-vsctl -- --all destroy queue
sudo ovs-vsctl -- --all destroy qos

## Destroy each uuid

sudo ovs-vsctl  --  destroy  qos 0b62cfb0-c335-45ca-8910-ca4030ea13af
sudo ovs-vsctl  --  destroy  queue 24ff16d8-926e-4c12-aad8-b7b06c6ec575
