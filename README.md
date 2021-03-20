# Counterfeit Bank Note Detection
 *location: `Counterfeit-Banknote-Detection.md`*
 
# Synopsis:

#### Introduction and Objective
In this project, we will attempt to predict/detect counterfeit Swiss 1000-franc bank notes using a variety of methods, such as Linear Discriminant Analysis, Logistic Regression, and Factor Analysis, and then select the best performing model for final prediction/detection. Our models will be the following:

- Linear discriminant analysis (LDA)
- Logistic regression
- Factor analysis and LDA
- Factor analysis and logistic regression

However before doing so, we will explore our data with scatterplots, level plots, and boxplots and then proceed to prepare our data for cross-fold validation. At the end, we will evaluate our models' performances across all folds.

#### Conclusion
Overall, our several models yielded similar results in terms of validation accuracies, only varying between 95% and 100% across folds and across methodologies. Our baseline LDA model appears to perform the best across all folds; it has the **highest accuracy of approximately 100%** and does not require additional computation and data manipulation as with factor analysis. By adopting this final model which strictly uses linear discriminant analysis, we are able to successfully **predict/detect counterfeit Swiss 1000-franc bank notes at an accuracy rate between 95% and 100%**.
 
#### Methods Used
- Data Visualization
- Cross-Fold Validation
- Linear Discriminant Analysis
- Logistic Regression
- Factor Analysis

#### Technologies
- R Programming 

#### Contributors
- Jared Thach - github/jared8thach
- Dataset from “Multivariate Statistics: A practical approach”, by Bernhard Flury and Hans Riedwyl, Chapman and Hall, 1988.
