#! /bin/bash

read -p "How many files: " FILES

for ((i=0;i<=FILES;++i));
do
    printf -v j "%05d" $i

    ./join_vtk++ -o TI.uov.$j.vtk TI.block0.uov.$j.vtk TI.block1.uov.$j.vtk TI.block2.uov.$j.vtk TI.block3.uov.$j.vtk

    rm TI.block0.uov.$j.vtk TI.block1.uov.$j.vtk TI.block2.uov.$j.vtk TI.block3.uov.$j.vtk
done
