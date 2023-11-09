bash MF_Frame_pos1_1.sh 5e-4 1e-3 1024 1024 0.14 1.00 6 reweight amazon drop Pos_DROLoss 256
bash lgn_Frame_pos1_1.sh 1e-3 1e-1 4096 800 0.30  0.80 6 reweight nodrop Pos_DROLoss amazon 3 50 no_cosine no_sample 256
bash lgn_Frame_pos1_1.sh 1e-3 1e-1 4096 800 0.30  0.80 6 reweight nodrop Pos_DROLoss amazon 3 50 no_cosine no_sample 512
