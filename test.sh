#/bin/bash

results_file=test_results.txt
width=4096
height=4096
sum_size=3

echo x y time > $results_file

res=$(./main res.ppm $width $height 1 1 t | grep -oh [0-9]*)
echo 1 1 \t$res >> $results_file

for x in $(seq 1 32)
do
	for y in 1 2 4
	do
		if [ "$x" -ne "$y" ]
		then
			echo Size: $x $y
			sum=0
			for i in $(seq 1 $sum_size)
			do
				res=$(./main res.ppm $width $height $x $y t | grep -oh [0-9]*)
				sum=$(($sum+$res))
			done
			echo $x $y $(($sum / $sum_size)) >> $results_file
		fi
	done
done

