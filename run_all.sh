cd examples/flux
bash ./run.sh
cd ../hunyuan
bash ./run.sh
cd ../wan
bash ./run.sh
cd ../../

mkdir -p results_all
mkdir -p results_all/flux
mkdir -p results_all/hunyuan
mkdir -p results_all/wan

mv examples/flux/output/* results_all/flux/
mv examples/hunyuan/output/* results_all/hunyuan/
mv examples/wan/*.mp4 results_all/wan/