mkdir output_2223_inclusive

mkdir output_2223_inclusive/Combine

cp -r output_2022_inclusive/Combine/Datacard_2022.txt output_2223_inclusive/Combine

cp -r output_2023_inclusive/Combine/Datacard_2023.txt output_2223_inclusive/Combine

cp -r output_2022_inclusive/Combine/Models output_2223_inclusive/Combine

mv output_2223_inclusive/Combine/Models/ output_2223_inclusive/Combine/Models_2022/

cp -r output_2023_inclusive/Combine/Models output_2223_inclusive/Combine

mv output_2223_inclusive/Combine/Models/ output_2223_inclusive/Combine/Models_2023/

cd output_2223_inclusive/Combine

combineCards.py Y22=Datacard_2022.txt Y23=Datacard_2023.txt > Datacard_2223.txt

sed -i -E '/^shapes\s+\S+\s+Y22_/ s|(\s)\./Models/|\1./Models_2022/|g' Datacard_2223.txt

sed -i -E '/^shapes\s+\S+\s+Y23_/ s|(\s)\./Models/|\1./Models_2023/|g' Datacard_2223.txt

rm -rf Datacard_2022.txt Datacard_2023.txt
