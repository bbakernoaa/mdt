#!/bin/bash
#SBATCH -J mdt-log-check
#SBATCH -M es
#SBATCH -p dtn_f5_f6
#SBATCH -A bil-fire3
#SBATCH --qos=dtn
#SBATCH --constraint=f6
#SBATCH --nodelist=dtn03
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH -t 00:05:00
#SBATCH -o dtn_worker_log_check_%j.out

echo "== worker scratch dirs =="
ls -lt /tmp/dask-scratch-space/ 2>/dev/null | head
echo "== newest worker log tails =="
for d in /tmp/dask-scratch-space/worker-*; do
  echo "--- $d ---"
  ls -lt "$d" | head -5
  for f in "$d"/*.log; do
    echo "--- tail $f ---"
    tail -30 "$f"
  done
done
echo "== dask worker processes =="
ps -eo pid,etime,pcpu,pmem,args | grep dask_worker | grep -v grep