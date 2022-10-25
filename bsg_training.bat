@echo off
call conda activate BSG
cd "D:\Projects\query-luminary\src\prep\train\bsg"
call python run_bsg_invoice.py --epochs=%1 --alpha=%2 --max_vocab_size=%3 --batch_size=%4 --nr_neg_samples=%5 --embedding_size=%6
exit 0