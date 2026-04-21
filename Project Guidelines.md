- Project topic - regression (supervised learning) – use regression methods (e.g. ridge regression, Gaussian processes) to model data or predict from data. 
- Project Report – The project report is essentially the project proposal with all the details filled in. The report should have the following contents: 1) introduction – what is the problem? why is important?; 2) methodology – what algorithms did you use and what are the technical details? what are the advantages and disadvantages?; 3) experimental setup – what data did you use? how did you pre-process the data? which algorithms did you run on the data? what is the metric for evaluation?; 4) experimental results – what were the results? what insight do you get from these results? what are some typical success and failure cases? – The project report should be at least 4 pages. There is no upper page limit, but probably it should not be more than 8 pages long. For group projects, the project report must state the level of contribution from each project member. 
If you use 3rd party code, you must acknowledge it with an appropriate reference. 
- Presentation Video - 5 minutes long

- Grading – The marks for this project will be distributed as follows: 

- 16.7% – Project proposal. 

- 16.7% – Technical correctness (whether you used the algorithms correctly) 

- 16.7% – Experiments. More points for thoroughness and testing interesting cases (e.g., different parameter settings). 

- 16.7% – Analysis of the experiments. More points for insightful observations and analysis. 

- 16.7% – Quality of the written report (organized, complete descriptions, etc). 

- 16.7% – Project poster presentation. 

Note: Here $1 6 . 7 \%$ means 5/30 

# 2 Default Course Project – Digit Classification

The default project is handwritten digit classification on a subset of the MNIST digits dataset. 

• Dataset – The provided dataset is a subset of the MNIST digits. The dataset has 10 classes (digits 0 through 9) with 4000 images (400 images per class). Each feature vector is a vectorized image (784 dimensions), containing grayscale values [0, 255]. The original image dimensions are $2 8 \times 2 8$ . Here is an example montage of the digits: 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-20/81621455-46f7-4f12-9bb0-394e76b92d7f/2252185a44b4f7f3122d214bbd483f9dad205a2f163d05335728fb87921fe26e.jpg)


The MATLAB file digits4000.mat (or digits4000 *.txt for non-MATLAB users) contains the following data: 

– digits vec – a $7 8 4 \times 4 0 0 0$ matrix, where each column is a vectorized image, i.e. the feature vector $x _ { i } \in \mathbb { R } ^ { 7 8 4 }$ . 

– digits labels – a $1 \times 4 0 0 0$ matrix with the corresponding labels $y _ { i } \in \{ 0 , \cdot \cdot \cdot , 9 \}$ . 

– trainset – a $2 \times 2 0 0 0$ matrix, where each row is a set of indices to be used for training the classifier. 

– testset – a $2 \times 2 0 0 0$ matrix, where each row is the corresponding set of indices to be used for testing the classifier. 

The image above was generated with the following MATLAB code: 

testX $=$ digits_vec(:,testset(1,:)); % get test data (trial 1) 

testXimg $=$ reshape(testX, [28 28 1 2000]); % turn into an image sequence 

montage(uint8(testXimg), ’size’, [40 50]); % view as a montage 

• Methodology – You can use any technique from the course material, e.g., Bayes classifiers, Fisher’s Discriminant, SVMs, logistic regression, perceptron, kernel functions, etc. You may also use other classification techniques not learned in class, but you will need to describe them in detail in your report. Two useful libraries for classification are “libsvm” and “liblinear”. You can also pre-process the feature vectors, e.g., using PCA or kPCA to reduce the dimension, or apply other processing techniques (e.g., normalization or some image processing). 

Finally, a common trick for doing multi-class classification using only binary classifiers (e.g. SVMs) is to use a set of 1-vs-all binary classifiers. Each binary classifier is trained to distinguish one digit (+1) vs. the rest of the digits (-1). In this case, there are 10 binary classifiers total. Given a test example, each binary classifier makes a prediction. Hopefully, only one classifier has a positive prediction, which can then be selected as the class. If not, then the classifier that has the most confidence in its prediction is selected. For example, for SVMs the classifier that places the test example furthest from the margin would be selected. For logistic regression, the selection would be based on the calculated class probability. 

• Evaluation – The classifiers are evaluated over 2 experiment trials. In each trial, 50% of the data has been set aside for training (and cross-validation of parameters), and the remaining $5 0 \%$ is held out for testing only. The indices of the training set and test sets are given in the trainset and testset matrices. For a given trial, the same writer does not appear in both the training and test sets. 

For each trial, train a classifier using only the training set data (images and labels). You may also use the training set to select the optimal model parameters using cross-validation. After training the classifier, apply the classifier to the test data (images only) to predict the class. Record the accuracy (number correct predictions / total number) for that trial. Do not tune the parameters to optimize the test accuracy directly! You can only tune the parameters using the training set. 

As a baseline, a simple nearest-neighbors classifier with Euclidean distance was used on the test data. The resulting classification accuracy for each experiment trial is: 

<table><tr><td>trial</td><td>1</td><td>2</td><td>mean (std)</td></tr><tr><td>1-NN</td><td>0.9135</td><td>0.9185</td><td>0.9160 (0.0035)</td></tr></table>

In your experiments, which classifier does better? What feature pre-processing helps or hurts the performance? How does the performance vary with parameter values? 

• Bonus Challenge – In the bonus challenge, I will give you a new test set containing my own handwritten digits, and you will try to classify them using your trained classifiers. Whoever gets the best performance wins a prize! 