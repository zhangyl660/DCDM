# train diffusion
CUDA_VISIBLE_DEVICES=0 python scripts/image_train.py --data_dirs office31/amazon pseudo/AtoW --log_dir /data/log --attention_resolutions 32,16,8 --learn_sigma True --image_size 256 --num_channels 256 --class_cond True --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --diffusion_steps 1000 --noise_schedule linear --lr 3e-5 --batch_size 8 --end_step 50000 --num_classes 31

# train domain classifier
TRAIN_FLAGS="--iterations 2000 --anneal_lr True --batch_size 32 --lr 1e-4 --save_interval 500 --weight_decay 0.05"
CLASSIFIER_FLAGS="--image_size 256 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"
CUDA_VISIBLE_DEVICES=0 python scripts/domain_discriminator_train.py --data_dirs office31/amazon office31/webcam $TRAIN_FLAGS $CLASSIFIER_FLAGS --log_dir /data/log

# sample with domain
MODEL_FLAGS="--attention_resolutions 32,16,8 --learn_sigma True --class_cond True --diffusion_steps 1000 --dropout 0.0 --image_size 256 --noise_schedule linear --num_channels 256 --num_res_blocks 2 --num_head_channels 64 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
CLASSIFIER_FLAGS="--image_size 256 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --classifier_scale 2.5 --classifier_use_fp16 True"
SAMPLE_FLAGS="--batch_size 8 --num_samples 200"
CUDA_VISIBLE_DEVICES=0 python scripts/domain_sample_dpm_solver.py --num_classes 31 --specified_domain 1 --log_dir /data/log --model_path office31/AtoW/ema_0.9999_050000.pt --classifier_path domain_classifier/A_W/model001999.pt $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS