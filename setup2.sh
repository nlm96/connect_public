#!/bin/bash

# This script installs all dependencies and sets up a conda environment.
# The user must have the loadable kernel modules 'intel' and 'mkl' installed
#
# Author: Andreas Nygaard (2022)



NO='\033[0;31mno\033[0m'
YES='\033[0;32myes\033[0m'

echo -e "--------------------------------------------------------------\n\n"
cat source/assets/logo_colour.txt
echo -e "\n--------------------------------------------------------------\n"

echo -e "Running setup script for connect\n"

echo -e "The following things can be done (but can also be skipped):"
echo -e "    - create conda environment with all dependencies"
echo -e "    - install and setup Monte Python or use previous installation"
echo -e "    - install and setup CLASS or use previous installation"
echo -e "    - install Cobaya and CAMB (in environment)\n\n"
echo -e "--------------------------------------------------------------\n\n"


while [ -z $create_env ]
do
    echo "Creating conda environment. Proceed? [yes, skip]"
    read create_env
done

if [ $create_env == "yes" ]
then
    echo "Enter name of conda environment to create, or leave blank to use"
    echo "default name 'ConnectEnvironment':"
    read env_name
    if [ -z $env_name ]
    then
	env_name="ConnectEnvironment"
    fi
fi

if ! [ $create_env == "yes" ]
then
    echo "Enter name of conda environment to use, or leave blank to not use"
    echo "an environment:"
    read env_name
fi


while [ -z $class ]
do
    echo "Do you want to install CLASS in the environment? [yes, no]"
    read class
done

if [ $class == "yes" ]
then
    echo "If you already have a CLASS installation, enter the absolute"
    echo "path. Otherwise, leave blank and CLASS repo will be cloned:"
    read class_path
fi

echo -e "\n--------------------------------------------------------------\n"

if [ $create_env == "yes" ]
then
    Ans1=$YES
    env_name_string="\n    Name of conda environment:\n    \033[0;34m${env_name}\033[0m"
else
    Ans1=$NO
fi


if [ $class == "yes" ]
then
    Ans4=$YES
    if ! [ -z $class_path ]
    then
	path_class="\n    Path to CLASS:\n    \033[0;34m${class_path}\033[0m"
    else
	path_class="\n    Cloning CLASS repo to \033[0;34mconnect/resources\033[0m"
    fi
else
    Ans4=$NO
fi


echo -e "You have selected the following:\n"
echo -e "Create conda environment             :                   ${Ans1}${env_name_string}"
echo -e "Setup link to Monte Python           :                   ${Ans2}${path_mp}${clik_mp}"
echo -e "Install CLASS                        :                   ${Ans4}${path_class}"

echo -e "\n--------------------------------------------------------------\n"

while [ -z $proceed ]
do
    echo -e "\nProceed? [yes, abort]"
    read proceed
done

if [ $proceed == "abort" ]
then
    echo -e "\nYou have aborted the setup. Please try again\n"
    exit 0
fi



##################################################################################

##############################     Actual setup     ##############################

##################################################################################



source ~/.bashrc 2> /dev/null
source ~/.bash_profile 2> /dev/null
source "$(conda info | grep -i 'base environment' | awk '{for(i=1;i<=NF;i++) if($i ~ /\//) print $i}')/etc/profile.d/conda.sh"

# Load correct compiler and set paths
module load gcc/12.2.0 openmpi cmake mkl 2> /dev/null
export CC=/comm/swstack/core/gcc/12.2.0/bin/gcc
export CXX=/comm/swstack/core/gcc/12.2.0/bin/g++
export CPP=/comm/swstack/core/gcc/12.2.0/bin/cpp
export LD_LIBRARY_PATH=/comm/swstack/core/gcc/12.2.0/lib64:$LD_LIBRARY_PATH
export PATH=/comm/swstack/core/gcc/12.2.0/bin:$PATH
export CMAKE_C_COMPILER=/comm/swstack/core/gcc/12.2.0/bin/gcc
export CMAKE_CXX_COMPILER=/comm/swstack/core/gcc/12.2.0/bin/g++
export FC=/comm/swstack/core/gcc/12.2.0/bin/gfortran


conda init


if [ $Ans1 == $YES ]
then
    echo "--> Creating Conda environment, this will take a few minutes..."
    conda clean --index-cache -y
    # Remove ConnectEnvironment if it exists
    conda env remove -y --name $env_name
    conda create -y --name $env_name python=3.10 cython=3.0 scipy=1.11 numpy=1.26 astropy=5.1 pip=23.2 numexpr=2.8 pandas=2.0


    conda activate $env_name
    export LD_LIBRARY_PATH=/lib64:$LD_LIBRARY_PATH
    export CC=/comm/swstack/core/gcc/12.2.0/bin/gcc
    export CXX=/comm/swstack/core/gcc/12.2.0/bin/g++
    export CPP=/comm/swstack/core/gcc/12.2.0/bin/cpp
    export LD_LIBRARY_PATH=/comm/swstack/core/gcc/12.2.0/lib64:$LD_LIBRARY_PATH
    export PATH=/comm/swstack/core/gcc/12.2.0/bin:$PATH
    export CMAKE_C_COMPILER=/comm/swstack/core/gcc/12.2.0/bin/gcc
    export CMAKE_CXX_COMPILER=/comm/swstack/core/gcc/12.2.0/bin/g++
    export FC=/comm/swstack/core/gcc/12.2.0/bin/gfortran


    pip install matplotlib==3.7
    pip install mpi4py==3.1.4
    pip install tensorflow==2.10
    pip install tensorflow-probability==0.18.0
    pip install sshkeyboard
    pip install playsound

    echo "--> ..done!"
fi



if [ $Ans4 == $YES ]
then
    if ! [ -z $env_name ]
    then
        conda activate $env_name
    fi
    if ! [ -z $class_path ]
    then
	echo "--> Building classy wrapper..."
	connect_path=$PWD
	cd $class_path
    make clean
    make -j CC=$CC CXX=$CXX FC=$FC
	cd $connect_path
	echo "--> ...done!"
fi



python -c "from source.assets.animate import play; play()"

echo -e "\nSetup is all done!\n"
