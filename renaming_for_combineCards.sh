mkdir output_2223_rapidity/Combine

cp -r output_2022_rapidity/Combine/Datacard_rapidity_2022.txt output_2223_rapidity/Combine

cp -r output_2023_rapidity/Combine/Datacard_rapidity_2023.txt output_2223_rapidity/Combine

# cp -r output_2022_rapidity/Combine/Models_rapidity output_2223_rapidity/Combine

# mv output_2223_rapidity/Combine/Models_rapidity/ output_2223_rapidity/Combine/Models_rapidity_2022/

# cp -r output_2023_rapidity/Combine/Models_rapidity output_2223_rapidity/Combine

# mv output_2223_rapidity/Combine/Models_rapidity/ output_2223_rapidity/Combine/Models_rapidity_2023/

cd output_2223_rapidity/Combine

combineCards.py Y22=Datacard_rapidity_2022.txt Y23=Datacard_rapidity_2023.txt > Datacard_rapidity_2223.txt

sed -i -E '/^shapes\s+\S+\s+Y22_/ s|(\s)\./Models_rapidity/|\1./Models_rapidity_2022/|g' Datacard_rapidity_2223.txt

sed -i -E '/^shapes\s+\S+\s+Y23_/ s|(\s)\./Models_rapidity/|\1./Models_rapidity_2023/|g' Datacard_rapidity_2223.txt

rm -rf Datacard_rapidity_2022.txt Datacard_rapidity_2023.txt
