#!/bin/bash

sudo ovs-vsctl del-br BalancerNet
sudo ovs-vsctl del-br TransportNet
sudo ovs-vsctl del-br ExtNet
sudo ovs-vsctl del-br Transport2
sudo ovs-vsctl add-br BalancerNet
sudo ovs-vsctl add-br TransportNet
sudo ovs-vsctl add-br ExtNet
sudo ovs-vsctl add-br Transport2
