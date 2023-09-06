k_max=20
r_max=5
step_size=5
for rg in $(ls outputs/*rg*.gpickle);
do
  name=$(echo $rg | cut -d '/' -f 2 | cut -d '.' -f 1)
  for i in $(seq -w 0 $step_size $k_max);
  do
    if [ $i -le 0 ];
    then
      i=1
    fi
    for seed in $(seq -w 1 $r_max);
    do
      echo structure: $name training k=$i/$k_max seed $seed/$r_max
      python train_einet_with_em.py \
        --train-dir dataset/coco2017-cooccurences-train.csv \
        --valid-dir dataset/coco2017-cooccurences-valid.csv \
        --structure-dir $rg \
        --export-dir outputs/$name \
        --export-name mixture-$i-seed-$seed \
        --num-input-distributions $i \
        --num-sums $i \
        --batch-size 2048 \
        --step-size 0.05 \
        --device cuda \
        --num-workers 8 \
        --max-tolerance 20 \
        --num-iterations 200 \
        --random-seed $seed
    done
  done
done
