export PYTHONPATH=$(pwd)
echo "Training on task ${2}";
echo "run: ${3}"
perl -pi -e "s/^run: \d+/run: ${3}/" "./config/${2}.yaml"
CUDA_VISIBLE_DEVICES=${1} python -u ./codes/main.py --config "./config/${2}.yaml"