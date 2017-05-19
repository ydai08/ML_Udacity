# Machine Learning Engineer Nanodegree
# Capstone Project
# Yang Dai
## Project: Predicting Passage of US Constitutional Amendments

### Install

This project requires **Python 3**, [iPython Notebook](http://ipython.org/notebook.html), and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [seaborn] (https://seaborn.pydata.org/)
- [yaml] (http://pyyaml.org/)
- [nltk] (http://www.nltk.org/)
- [PIL] (http://www.pythonware.com/products/pil/)
- [wordcloud] (https://github.com/amueller/word_cloud)
- [palettable] (https://jiffyclub.github.io/palettable/)


### Code

Code is provided in the `amendments_capstone.ipynb` notebook file, which requires the `visuals.py` Python file, the `usa.png` graphic file, and the following dataset files: 
        `us-nara-amending-america-dataset-amend.csv`, 
        `party_divisions.csv`, 
        `legislators-current.yaml`,  
        `legislators-historical.yaml`


### Run

In a terminal or command window, navigate to the top-level project directory `amendments/` (that contains this README) and run one of the following commands:

```bash
ipython notebook amendments_capstone.ipynb
```  
or
```bash
jupyter notebook amendments_capstone.ipynb
```

This will open the iPython Notebook software and project file in your browser.
`

### Data

The `us-nara-amending-america-dataset-amend.csv` dataset consists of 11,797 proposed Constitutional amendments introduced from 1787 to 2014. It is a modified version of the dataset retrieved from (https://www.archives.gov/open/dataset-amendments.html), with the `amendment` column manually populated to indicate whether a proposal is associated with a ratified amendment or an unratified but passed amendment.
