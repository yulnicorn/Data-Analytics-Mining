# Data-Analytics-Mining
Enivronment Setup:
Anaconda - https://docs.anaconda.com/anaconda/user-guide/getting-started/

Python 3.7 - https://www.anaconda.com/python-3-7-package-build-out-miniconda-release/

1. DAM NN.ipynb
Commands to Run:
  1. conda activate "environment containing the following python dependencies"
  2. jupyter notebook
  3. (Open DAM NN.ipynb in browser)
  4. (Run the notebook)
  
Python Dependencies:
  - jupyter 1.0.0
  - pandas 0.25.1
  - TensorFlow 1.14.0
  - matplotlib.pyplot        
  - Matplotlib 3.1.1
  - numpy 1.16.5
  - sklearn 0.21.3
Installation:
Conda install -c condo-forge numpy pandas matplotlib tensorflow==1.14.0 jupyter sklearn

Output: Markdowns cells are in placed to explain the output.

+Graphs
   + training and test accuracy
   + training and test loss
==================================================================================
2. Cleaning.ipynb
Commands to Run:
  1. conda activate "environment containing the following python dependencies"
  2. jupyter notebook
  3. (Open cleaning.ipynb in browser)
  4. (Run the notebook)

Python Dependencies:
  - jupyter 1.0.0
  - pandas 0.25.1
  - numpy 1.16.5
  - matplotlib.pyplot
  - Matplotlib 3.1.1
  - missingno 0.4.2
  - impyute 0.0.7

Installation:
Conda install -c condo-forge numpy pandas matplotlib jupyter missingno
pip3 install impyute

Output: Markdowns cells are in placed to explain the output.
  + Dataframe of clean dataset
  + Dataframe of female and male cleaned dataset
==================================================================================
3. DecisionTree.ipynb
Commands to Run:
  1. conda activate "environment containing the following python dependencies"
  2. jupyter notebook
  3. (Open DecisionTree.ipynb in browser)
  4. (Run the notebook)

Python Dependencies:
  - jupyter 1.0.0
  - pandas 0.25.1
  - numpy 1.16.5
  - matplotlib.pyplot
  - Matplotlib 3.1.1
  - sklearn 0.21.3

Installation:
Conda install -c condo-forge numpy pandas matplotlib jupyter sklearn

Output: Markdowns cells are in placed to explain the output.
  + Accuracy of the prediction
  + Graphs
  + Bar chart showing mean values of Decision made by partner for second date base on attr_o, sinc_o, shar_o, amb_o, fun_o, intel_o, like_o
==================================================================================
4. EDA(female_imputed/male_imputed/6 attributes).ipynb
Commands to Run:
  1. conda activate "environment containing the following python dependencies"
  2. jupyter notebook
  3. (Open EDA(female_imputed/male_imputed/6 attributes)).ipynb in browser)
  4. (Run the notebook)

Python Dependencies:
  - jupyter 1.0.0
  - pandas 0.25.1
  - numpy 1.16.5
  - matplotlib.pyplot
    - Matplotlib 3.1.1
  - seaborn 0.9.0

Installation:
Conda install -c condo-forge numpy pandas matplotlib jupyter seaborn

Output: Markdowns cells are in placed to explain the output.
  + Graphs
  + Heat map of dataset correspondence
  + Pair plot of pairwise of dataset
  + Bar chart showing mean score of the attributes based on partner’s decision
  + Histogram based on the rating for same and other race
  + Line chart of mean score of attributes across different age group
  + Line chart of preferred score of attributes across different age group
==================================================================================
5. 
