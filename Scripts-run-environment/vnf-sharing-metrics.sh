### This script run several tests over multipath and loadbalancing SDN-Applications 
### to test the CPU consuming of each VNF (VNFT1 and VNFT1-2) 

# We apply the follow metodology:
# 1. The script increase the bandwidth in a exponencial way (multiplying by 2) 
# 2. the script runs each test 120 seconds and wait 120 seconds between tests

apt-get install iperf

echo "-----"
echo "---- Test Ping ---"

ping 10.2.2.30 -c 4
sleep 5

echo "--- Starting Iperf test ---"
echo "----"
echo "--- Start iperf with 4Mb of bandwidth --"

iperf -c 10.2.2.30 -i 30 -t 60 -b 4M -P 10 #> iperf-client-4.csv
#sleep 30

echo "-- Start iperf with 8Mb of bandwidth --"

iperf -c 10.2.2.30 -i 30 -t 60 -b 8M -P 10 #> iperf-client-8.csv
#sleep 30

echo "-- Start iperf with 16Mb of bandwidth --"

iperf -c 10.2.2.30 -i 30 -t 60 -b 16M -P 10 #> iperf-client-16.csv
#sleep 30

echo "-- Start iperf with 32Mb of bandwidth --"

iperf -c 10.2.2.30 -i 30 -t 60 -b 32M -P 10 #> iperf-client-32.csv
#sleep 30

echo "-- Start iperf with 64Mb of bandwidth --"

iperf -c 10.2.2.30 -i 30 -t 60 -b 64M -P 10 #> iperf-client-64.csv
#sleep 30

echo "-- Start iperf with 128Mb of bandwidth --"

iperf -c 10.2.2.30 -i 30 -t 60 -b 128M -P 10 #> iperf-client-128.csv
#sleep 30

echo "-- Start iperf with 256Mb of bandwidth --"

iperf -c 10.2.2.30 -i 30 -t 60 -b 256M -P 10 #> iperf-client-256.csv
#sleep 30

echo "-- Start iperf with 512Mb of bandwidth --"

iperf -c 10.2.2.30 -i 30 -t 60 -b 512M -P 10 #> iperf-client-512.csv
#sleep 30



