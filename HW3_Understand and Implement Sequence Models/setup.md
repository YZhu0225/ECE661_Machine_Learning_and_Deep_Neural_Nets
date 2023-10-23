##(1) Install the latest Pytorch, functools, regex, matplotlib, tqdm, nltk, numpy.


### Pip envoriment
``pip install torch``

``pip install functools regex matplotlib tqdm numpy pandas``

``pip install -U nltk``

``pip install -U scikit-learn``


### Conda virtual enviroment 
``conda install -c pytorch pytorch``

``conda install -c travis functools``

``conda install -c anaconda regex``

``conda install -c conda-forge matplotlib``

``conda install -c conda-forge tqdm``

``conda install -c anaconda nltk``

``conda install -c anaconda numpy``

``conda install -c anaconda pandas``

``conda install -c anaconda scikit-learn``


##(2) After installing nltk, you need to further run a python command to install stopwords. 

``python -m nltk.downloader stopwords``


##(3) You are free to install your own package. If not specifically stated (e.g., you should not use Pytorch's internal nn package), you can use potentially any package. 
