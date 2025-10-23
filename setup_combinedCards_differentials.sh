#!/bin/bash
set -euo pipefail

VAR="$1"
VERSION="${VAR}_v1"

mv "output_2223_${VAR}/Combine/" "output_2223_${VAR}/Combine_hadd/"
mv "output_2223_${VAR}/Datacards/" "output_2223_${VAR}/Datacards_hadd/"

mkdir -p "output_2223_${VAR}/Combine"
mkdir -p "output_2223_${VAR}/Datacards"

cp -r output_2223_${VAR}/Datacards_hadd/* "output_2223_${VAR}/Datacards/"
rm -f output_2223_${VAR}/Datacards/Datacard_${VAR}_2223.txt

# Copy datacards
cp "output_2022_${VAR}/Combine/Datacard_${VAR}_2022.txt" "output_2223_${VAR}/Combine"
cp "output_2023_${VAR}/Combine/Datacard_${VAR}_2023.txt" "output_2223_${VAR}/Combine"

# Copy and rename model directories
cp -r "output_2022_${VAR}/Combine/Models_${VAR}" "output_2223_${VAR}/Combine"
mv "output_2223_${VAR}/Combine/Models_${VAR}" "output_2223_${VAR}/Combine/Models_${VAR}_2022"

cp -r "output_2023_${VAR}/Combine/Models_${VAR}" "output_2223_${VAR}/Combine"
mv "output_2223_${VAR}/Combine/Models_${VAR}" "output_2223_${VAR}/Combine/Models_${VAR}_2023"

cd "output_2223_${VAR}/Combine"

# Combine datacards
combineCards.py Y22="Datacard_${VAR}_2022.txt" Y23="Datacard_${VAR}_2023.txt" > "Datacard_${VAR}_2223.txt"

# Fix model paths for each year
sed -i -E "/^shapes\s+\S+\s+Y22_/ s|(\s)\./Models_${VAR}/|\1./Models_${VAR}_2022/|g" "Datacard_${VAR}_2223.txt"
sed -i -E "/^shapes\s+\S+\s+Y23_/ s|(\s)\./Models_${VAR}/|\1./Models_${VAR}_2023/|g" "Datacard_${VAR}_2223.txt"

# Cleanup temporary datacards and copy datacard in Datacards directory
rm -f "Datacard_${VAR}_2022.txt" "Datacard_${VAR}_2023.txt"
cp -r Datacard_${VAR}_2223.txt ../Datacards/

# Run law tasks
law run RunText2Workspace --year 2223 --variable "${VAR}" --batch-flavor local --version "${VERSION}" --workflow local --workers 36
law run CreateDiffSpectra --year 2223 --variable "${VAR}" --batch-flavor local --workers 36
