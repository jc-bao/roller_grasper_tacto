cd ..
for j in {A..G} 
do
  for i in {0..6} 
  do
    echo "===================$j$i==================="
    cp /juno/u/chaoyi/rl/egad/data/egad_eval_set/processed_meshes/tamp.urdf /juno/u/chaoyi/rl/egad/data/egad_eval_set/processed_meshes/temp.urdf
    sed -i "s/XXX/$j$i/g" /juno/u/chaoyi/rl/egad/data/egad_eval_set/processed_meshes/temp.urdf 
    python roller_env.py --obj-urdf /juno/u/chaoyi/rl/egad/data/egad_eval_set/processed_meshes/temp.urdf --file-name "$j$i" --n 32 
  done
done