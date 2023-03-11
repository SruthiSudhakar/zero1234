# run this after current experiments are done
CUDA_VISIBLE_DEVICES=1 \
python load_inference_test_views.py \
--data_root "/home/rliu/Desktop/cvfiler04/datasets/GoogleScannedObjects" \
--res_path "experiments/googlescan_test_new_baseline"

CUDA_VISIBLE_DEVICES=1 \
python load_inference_test_views.py \
--data_root "/home/rliu/Desktop/cvfiler04/datasets/RTMV/google_scanned" \
--res_path "experiments/rtmv_baseline"