python3 -m experiments.Unet.pre_training.train
python3 -m experiments.Unet.fine_tuning.train
python3 -m experiments.Unet.test


python3 -m experiments.Swin_UNETR.pre_training.train
python3 -m experiments.Swin_UNETR.fine_tuning.train
python3 -m experiments.Swin_UNETR.test

python3 -m experiments.Single_step_PDPM.pre_training.train
python3 -m experiments.Single_step_PDPM.fine_tuning.train
python3 -m experiments.Single_step_PDPM.test_sino
python3 -m experiments.Single_step_PDPM.test_osem