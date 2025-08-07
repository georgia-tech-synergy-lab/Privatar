conda activate pica37
cd /scratch2/jianming/work/Privatar_prj/multiface_partition_bdct4x4_ibdct_hp
python launch_test_accuracy.py >> /scratch2/jianming/work/Privatar_prj/multiface_partition_bdct4x4_ibdct_hp/log_partition_bdct4x4_ibdct_hp.txt 2>&1

python3 /scratch2/jianming/work/Privatar_prj/memory_clean_gpu.py

cd /scratch2/jianming/work/Privatar_prj/multiface_quantization
python launch_test_accuracy.py>> /scratch2/jianming/work/Privatar_prj/multiface_quantization/log_quantization.txt 2>&1

python3 /scratch2/jianming/work/Privatar_prj/memory_clean_gpu.py
cd /scratch2/jianming/work/Privatar_prj/multiface_sparse
python launch_test_accuracy.py>> /scratch2/jianming/work/Privatar_prj/multiface_sparse/log_sparse.txt 2>&1

python3 /scratch2/jianming/work/Privatar_prj/memory_clean_gpu.py

cd /scratch2/jianming/work/Privatar_prj/multiface
python launch_test_accuracy.py>> /scratch2/jianming/work/Privatar_prj/multiface/log_multiface.txt 2>&1
