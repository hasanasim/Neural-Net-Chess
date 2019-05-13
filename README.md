# Reproducing results
 
1. Figures 1 & 2 are reproduced by running the code as it is given.
2. Figres 3 & 4 are reproduced by changing the parameter `gamma = 0.85` to `gamma = 0.085` 
3. Figures 5 & 6 are reproduced by changing the parameter `beta = 0.00005` to `beta = 0.0000025`
4. Figures 7 & 8 are reproduced by changing the parameter `sarsa = 0` to `sarsa = 1`. Only do this at initialization (line 83).

Note : Please remember when moving from one experiment to the other to set the previously changed parameters back to their original.

Runtime: 1,2,3 took approximately 30 minutes each on a dual-core i5 MacBook Pro. Experiment 4 took 1 hour. For quick testing change the number of episodes to 1000 or 100 for instant plots.

Output of running is an interactive plot in Python. The plot is also saved as a .png file.