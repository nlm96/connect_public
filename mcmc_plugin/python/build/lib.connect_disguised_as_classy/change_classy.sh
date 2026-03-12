#!/bin/bash

ss=$(ls /home/maanson/Speciale/connectv2/mcmc_plugin/python/build/lib.connect_disguised_as_classy/)

if [[ "$ss" == *"combi"* ]]
then
    mv /home/maanson/Speciale/connectv2/mcmc_plugin/python/build/lib.connect_disguised_as_classy/classy.py /home/maanson/Speciale/connectv2/mcmc_plugin/python/build/lib.connect_disguised_as_classy/classy_single.py
    mv /home/maanson/Speciale/connectv2/mcmc_plugin/python/build/lib.connect_disguised_as_classy/classy_combi.py /home/maanson/Speciale/connectv2/mcmc_plugin/python/build/lib.connect_disguised_as_classy/classy.py
fi

if [[ "$ss" == *"single"* ]]
then
    mv /home/maanson/Speciale/connectv2/mcmc_plugin/python/build/lib.connect_disguised_as_classy/classy.py /home/maanson/Speciale/connectv2/mcmc_plugin/python/build/lib.connect_disguised_as_classy/classy_combi.py
    mv /home/maanson/Speciale/connectv2/mcmc_plugin/python/build/lib.connect_disguised_as_classy/classy_single.py /home/maanson/Speciale/connectv2/mcmc_plugin/python/build/lib.connect_disguised_as_classy/classy.py
fi