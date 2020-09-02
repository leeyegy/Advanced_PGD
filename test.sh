for factor in 1 10 100 1000 10000000
do
	python test_pgd_cifar10.py --factor $factor --num-steps 20  --test_model_path checkpoint/res18-pgd-8.pt | tee log/factor_$factor.txt
done
