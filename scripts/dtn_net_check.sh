#!/bin/bash
#SBATCH -J mdt-net-check
#SBATCH -M es
#SBATCH -p dtn_f5_f6
#SBATCH -A bil-fire3
#SBATCH --qos=dtn
#SBATCH --constraint=f6
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:05:00
#SBATCH -o dtn_net_check_%j.out

echo "== hostname =="
hostname
echo "== interfaces =="
ip -o addr show | awk '{print $2, $3, $4}'
echo "== TCP probe: login hsn0 172.25.95.5:22 =="
timeout 5 bash -c 'echo > /dev/tcp/172.25.95.5/22' && echo HSN_OK || echo HSN_FAIL
echo "== TCP probe: login bond0 172.25.62.54:22 =="
timeout 5 bash -c 'echo > /dev/tcp/172.25.62.54/22' && echo BOND_OK || echo BOND_FAIL
echo "== TCP probe: login fen 192.188.179.74:22 =="
timeout 5 bash -c 'echo > /dev/tcp/192.188.179.74/22' && echo FEN_OK || echo FEN_FAIL