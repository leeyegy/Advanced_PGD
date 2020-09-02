for step_size in 0.03137
do
  for num_steps in 1
  do
    for factor in 100000
    do
			python train_APGD_cifar10.py --random_start --factor $factor --model-dir checkpoint/factor_$factor --epsilon 0.03137 --step-size $step_size  --num-steps $num_steps | tee log/train-apgd.txt
done
done
done
