for i in $(seq -w 1 20);
do
  echo training k=$i/20
  python train_mixture.py \
    --train-dir dataset/coco2017-cooccurences-train.csv \
    --valid-dir dataset/coco2017-cooccurences-valid.csv \
    --export-dir outputs/mixture \
    --export-name mixture-$i \
    --num-mixture $i \
    --batch-size 8196 \
    --step-size 0.05 \
    --device cuda \
    --num-workers 8
done
