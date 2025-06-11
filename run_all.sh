cd examples/flux
bash ./run.sh
cd ../hunyuan
bash ./run.sh
cd ../wan
bash ./run.sh
cd ../../
mkdir -p results_all

mv examples/flux/output/* results_all/
mv examples/hunyuan/output/* results_all/
mv examples/wan/*.mp4 results_all/