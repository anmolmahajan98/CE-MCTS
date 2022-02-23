This repository contains the code I have worked upon in my Master's thesis. I have designed a Conceptual Expansion based Monte Carlo Tree Search (CE-MCTS) to discover individual specific Machine Learning model to predict their behaviour on the tasks with insufficient data and utilizing knowledge from some secondary but unrelated task.


Task -Financial Time Series Prediction
***Datasets for this task are not provided because of an NDA signed with the financial institution
whose data has been used.
There are 7 sub-folders:
● Transfer Learned models
○ Contains the script for generating transfer learned models for 5 cross validation
folds on task.
Random Search
○ Contains the scripts of 5 cross validation folds for Random Search.
○ These scripts can be executed by putting corresponding transfer learned models
at the correct path.
● Greedy Search
○ Contains the scripts of 5 cross validation folds for Greedy Search.
○ These scripts can be executed by putting corresponding transfer learned models
at the correct path.
● Beam Search
○ Contains the scripts of 5 cross validation folds for Beam Search.
○ These scripts can be executed by putting corresponding transfer learned models
at the correct path.
● Aggregate
○ Contains the script for calculating the performance of naive transfer learned
models across 5 cross-validation folds
● 2nd Task-trained
○ Contains the script for calculating the performance of naive transfer learned
model further trained on the T1 data set (details explained in the research paper)
● CE-MCTS
○ Contains the scripts of 5 cross validation folds for CE-MCTS.
○ These scripts can be executed by putting corresponding transfer learned models
at the correct path.
● Script ‘ mse.py ’ is also included which can be used to calculate the Mean Square Error
(MSE) and Standard Deviation (SD). To execute this script, simply name the directory
containing the prediction outputs of the search approach to obtain corresponding MSE
and SD.
