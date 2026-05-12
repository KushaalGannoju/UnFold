# Student Depression Detection System Report Content

This document is structured as a 30-page content draft. You can use one numbered section per page, or combine some shorter pages during formatting.

## Page 1: Title Page
Student Depression Detection System Using Machine Learning

Minor Project Report submitted to the Department of Computer Science and Engineering, Maulana Azad National Institute of Technology.

Prepared by:
Himanshu Verma
Kushaal Sai Gannoju
Ritika Rani Ujjainia
Rayyan Zameer Baikadi

Project guides:
Dr. R.K. Pateriya
Dr. Archana Balmik

## Page 2: Certificate / Approval Page
This page should contain the standard departmental certificate stating that the work titled “Student Depression Detection System Using Machine Learning” is an original minor-project submission carried out under the supervision of the guides and submitted in partial fulfillment of the academic requirements.

## Page 3: Acknowledgment
We sincerely express our gratitude to our project guides, Dr. R.K. Pateriya and Dr. Archana Balmik, for their valuable guidance, technical suggestions, and constant encouragement throughout the development of this project. Their support helped us refine the system from a basic predictive model into a more structured and application-oriented student mental health screening platform. We also thank the Department of Computer Science and Engineering for providing the academic environment and resources needed to complete this work. We acknowledge the Kaggle dataset contributors for making the student depression dataset publicly available, which formed the experimental foundation of the project. Finally, we thank our families and friends for their encouragement and support.

## Page 4: Abstract
Student mental health has become an urgent concern in modern educational institutions. Depression among students affects academic performance, social functioning, emotional stability, and long-term well-being. However, early identification remains difficult because many students do not seek professional help due to stigma, low awareness, or limited counseling access. This project proposes a Student Depression Detection System that uses machine learning to estimate depression risk based on academic, behavioral, lifestyle, and psychological indicators. The system is trained on the Kaggle Student Depression Dataset containing 27,901 records. The final workflow includes robust preprocessing, feature encoding, model comparison, and deployment through an interactive Streamlit interface.

Three classifiers were considered in the final pipeline: Gradient Boosting, Random Forest, and Logistic Regression. Gradient Boosting was selected as the deployed model because it produced the strongest overall performance. The final Gradient Boosting model achieved 84.87% cross-validation accuracy and 84.70% test accuracy, with 85.84% precision, 88.46% recall, and 87.13% F1-score on the depressed class. Feature analysis showed that suicidal thoughts, academic pressure, and financial stress were the strongest determinants of prediction. The resulting system serves as an educational screening aid and is not intended to replace professional clinical diagnosis.

Keywords:
Student Depression, Mental Health Screening, Machine Learning, Gradient Boosting, Risk Prediction, Streamlit

## Page 5: Table of Contents
Suggested main sections:
1. Introduction
2. Problem Statement and Objectives
3. Literature Background
4. Dataset Description
5. Methodology
6. Data Preprocessing
7. Model Development
8. System Architecture
9. Implementation
10. Results and Analysis
11. Applications, Limitations, and Future Scope
12. Conclusion
13. References

## Page 6: Introduction
Depression is one of the most significant mental health challenges affecting students in higher education. Academic expectations, competition, uncertainty about future careers, social adjustment, family pressure, financial burden, and unhealthy routines combine to create a high-risk environment. Depression in students often manifests not only as sadness, but also as fatigue, low concentration, withdrawal, loss of motivation, disturbed sleep, reduced academic engagement, and self-harm ideation in severe cases. Because many of these signals develop gradually, institutions often fail to identify vulnerable students at an early stage.

Machine learning offers a practical way to analyze large student datasets and identify patterns associated with mental health risk. By learning from historical records, machine learning models can support early warning systems that help counselors and academic institutions perform proactive screening. This project builds such a system using student profile information and supervised classification methods.

## Page 7: Background and Motivation
The motivation for this work arises from the gap between mental health need and mental health access. Traditional depression assessment methods depend heavily on in-person interviews and self-reporting after substantial symptom escalation. These methods are clinically valid but are difficult to scale across large student populations. Institutions often lack the manpower to continuously monitor thousands of students. At the same time, many students hesitate to approach counselors until their condition deteriorates.

A computational system cannot replace professional diagnosis, but it can assist with triage. A well-designed risk-prediction tool can flag students who may benefit from supportive follow-up. This makes the screening process faster, broader, and more consistent, especially in environments where counseling resources are limited.

## Page 8: Problem Statement
The core problem addressed by this project is the absence of an accessible, data-driven, and scalable screening mechanism for student depression risk. Educational institutions usually rely on manual observation or voluntary disclosure, both of which are incomplete and inconsistent. Large student populations make continuous personalized monitoring impractical. In addition, student data contains a mixture of numeric, ordinal, and binary variables that require structured preprocessing before model training.

The project therefore aims to design a machine learning pipeline that can ingest student information, transform it into a suitable format, evaluate multiple predictive models, and expose the best model through an easy-to-use application interface.

## Page 9: Objectives
Primary objectives:
Develop a reliable machine learning model for depression-risk prediction among students.
Compare multiple classifiers and identify the most suitable final model.
Build an interactive user interface for real-time prediction.

Secondary objectives:
Create a robust preprocessing pipeline for mixed student data.
Improve interpretability through feature analysis and contribution visualization.
Evaluate the model using standard classification metrics.
Present a deployable educational prototype suitable for demonstration and academic study.

## Page 10: Literature Background
Research in educational mental health consistently shows that depression correlates with academic pressure, poor sleep quality, financial stress, reduced life satisfaction, social isolation, and prior psychological vulnerabilities. In applied machine learning research, depression-risk prediction often benefits from nonlinear models because mental health patterns are rarely explained by a single linear relationship. Ensemble methods are frequently preferred because they capture threshold effects and interactions among features.

Feature interpretability is especially important in health-related prediction because stakeholders must understand why the model is producing high-risk outputs. This project aligns with that principle by including feature-ranking and contribution-analysis mechanisms alongside predictive performance comparison.

## Page 11: Dataset Source and Overview
The dataset used in this project is the Kaggle Student Depression Dataset, containing 27,901 student records and 18 original columns. The dataset includes demographic, academic, behavioral, and psychological factors such as age, gender, CGPA, academic pressure, study satisfaction, work-study hours, financial stress, dietary habits, sleep duration, suicidal thoughts, and family history of mental illness. The target variable indicates whether the student is depressed.

After cleaning, the class distribution in the dataset is:
Depressed: 16,336
Not Depressed: 11,565

The dataset provides a suitable balance of size, diversity, and behavioral relevance for developing a practical classification system.

## Page 12: Data Understanding and Feature Audit
Before training, the dataset was examined for column consistency, feature relevance, and practical usability. Certain columns such as ID, city, degree, and profession do not meaningfully support depression-risk prediction and were therefore excluded. In addition, the project-specific dataset revealed that work pressure and job satisfaction were effectively non-informative because their values were dominated by zeros. Retaining such fields would add noise rather than signal, so they were removed from the final workflow.

The resulting effective student-focused feature set covered:
Gender
Age
Academic pressure
CGPA
Study satisfaction
Sleep duration
Dietary habits
Have you ever had suicidal thoughts
Work-study hours
Financial stress
Family history of mental illness

## Page 13: Data Preprocessing Pipeline
The preprocessing pipeline begins by standardizing column names. All column names are converted to lowercase, spaces are replaced with underscores, and punctuation such as question marks and slashes is normalized. This simplifies downstream feature matching and avoids errors caused by inconsistent naming.

The target column is then identified using the keyword “depression.” The target values are converted to numeric form, invalid entries are dropped, and the final target is cast as an integer binary label. This ensures that the prediction task is treated as a clean supervised classification problem.

## Page 14: Encoding and Cleaning Strategy
Different features required different encoding strategies. Sleep duration was mapped from category labels such as “5-6 hours” or “7-8 hours” into meaningful numeric hour values. Dietary habits were converted to an ordinal scale representing unhealthy, moderate, and healthy behavior. Binary responses such as yes/no and gender values were converted into numeric indicators. After encoding, only numeric columns were retained for training.

Missing values, if any remained after conversion, were imputed using median values. This preserves robustness while avoiding the instability of dropping a large number of records. The final feature matrix was therefore fully numeric, clean, and suitable for machine learning.

## Page 15: Train-Test Split and Validation Strategy
The dataset was divided into training and testing portions using an 80:20 stratified split. The stratification step preserves the depressed and not-depressed class ratio in both subsets. This is important in mental health prediction because class imbalance can otherwise distort evaluation.

The training set contained 22,320 records and the test set contained 5,581 records. Model selection and tuning were performed using 5-fold stratified cross-validation on the training set. This allows a reliable estimate of generalization performance while minimizing dependence on a single random split.

## Page 16: Model Development Overview
Three classifiers were included in the final workflow:
Gradient Boosting
Random Forest
Logistic Regression

Gradient Boosting and Random Forest were selected because they are strong ensemble methods capable of capturing nonlinear feature interactions. Logistic Regression was included as a simpler linear baseline. Each model was trained and evaluated under the same overall data split, allowing meaningful comparison across methods.

## Page 17: Gradient Boosting Model
Gradient Boosting builds an additive sequence of weak learners, where each new tree focuses on correcting the residual errors made by the previous trees. This makes it especially effective in datasets where relationships are nonlinear, threshold-driven, and interaction-heavy. Student depression prediction fits this profile well because risk does not increase uniformly. Instead, a combination of high academic pressure, strong financial stress, suicidal thoughts, poor sleep, and low satisfaction can sharply shift prediction outcomes.

In this project, Gradient Boosting was also paired with a feature-refinement stage, and the final deployed model used 10 selected predictors. The model ultimately delivered the best balance of cross-validation accuracy, test accuracy, precision, recall, and interpretability.

## Page 18: Random Forest and Logistic Regression Baselines
Random Forest served as a robust ensemble baseline. It combines many decision trees trained on random subsets of features and samples, then aggregates their predictions. In this project it produced solid performance but did not match the final Gradient Boosting model.

Logistic Regression served as the linear baseline. It provides a useful comparison point because it assumes a simpler decision boundary. Although its recall remained high for the depressed class, its overall generalization performance was lower than the tree-based models, confirming that the student-depression problem is not purely linear in structure.

## Page 19: Hyperparameter Tuning
Hyperparameter tuning was performed through randomized search with stratified 5-fold cross-validation. For Gradient Boosting, tuning considered estimators, learning rate, tree depth, subsampling ratio, and minimum leaf size. Random Forest and Logistic Regression were also configured and evaluated as tuned baselines. This tuning process ensured that the final model selection was based on measured generalization performance rather than default settings.

The final deployed Gradient Boosting configuration achieved the strongest outcome:
CV Accuracy: 84.87%
CV Standard Deviation: 0.19%

## Page 20: Final Performance Results
The final model comparison is as follows:

Gradient Boosting:
Training Accuracy: 85.57%
CV Accuracy: 84.87%
CV Std Dev: 0.19%
Test Accuracy: 84.70%
Precision: 85.84%
Recall: 88.46%
F1 Score: 87.13%

Random Forest:
Training Accuracy: 79.41%
CV Accuracy: 79.46%
CV Std Dev: 1.66%
Test Accuracy: 78.77%
Precision: 79.78%
Recall: 85.37%
F1 Score: 82.48%

Logistic Regression:
Training Accuracy: 78.53%
CV Accuracy: 75.82%
CV Std Dev: 0.25%
Test Accuracy: 78.01%
Precision: 73.51%
Recall: 97.64%
F1 Score: 83.87%

These results justify selecting Gradient Boosting as the final deployed model.

## Page 21: Classification Report Analysis
The final Gradient Boosting model produced balanced and practically useful performance. Precision of 85.84% means that most students flagged as depressed were genuinely high-risk according to the dataset labels. Recall of 88.46% is particularly important because it shows the model successfully captures a large portion of depressed students. The F1-score of 87.13% confirms that the model maintains a strong trade-off between precision and recall.

For screening-oriented systems, recall is especially important because false negatives represent missed students who may need help. Therefore, the final model’s recall strength makes it suitable as a preliminary support tool.

## Page 22: Confusion Matrix Analysis
The confusion matrix divides outcomes into four quadrants:
True Positive: Depressed students correctly identified as depressed
True Negative: Not depressed students correctly identified as not depressed
False Positive: Not depressed students incorrectly flagged as depressed
False Negative: Depressed students incorrectly classified as not depressed

This analysis helps move beyond raw accuracy. In health-related prediction, the cost of false negatives is usually more serious than the cost of false positives. Therefore, a model with strong depressed-class recall is preferable for an early-screening use case.

## Page 23: Feature Importance Analysis
Feature-importance analysis highlighted the strongest predictors in the final model. The most influential factor was “Have You Ever Had Suicidal Thoughts,” followed by academic pressure and financial stress. These findings align with both domain understanding and real-world student mental health patterns. Additional influential features included age, work-study hours, dietary habits, study satisfaction, sleep duration, family history of mental illness, and CGPA.

This ranking suggests that depression risk is shaped by a combination of direct psychological warning signs, academic context, lifestyle quality, and family vulnerability rather than by any single isolated factor.

## Page 24: Key Insights from the Results
Several project insights emerged from the experiments:
Academic pressure is a major structural stressor in student life and strongly influences model prediction.
Financial stress is not merely an economic concern; it is a major mental-health risk factor.
Psychological indicators such as suicidal thoughts sharply increase prediction confidence.
Lifestyle variables like sleep and dietary habits make meaningful contributions to classification quality.
Tree-based ensemble models are better suited than linear models for this problem because the data contains nonlinear thresholds and interactions.

These insights support both technical conclusions and institutional intervention planning.

## Page 25: System Architecture
The project consists of three major layers:
1. Data and preprocessing layer
Loads raw CSV data, cleans columns, encodes categories, and produces the training matrix.
2. Model training and evaluation layer
Trains candidate models, performs cross-validation, evaluates performance, generates plots, and serializes the final model bundle.
3. Deployment and interface layer
Loads the saved model, accepts student profile inputs, generates predictions, and displays metrics and feature analysis in the Streamlit interface.

This modular architecture makes the system easier to maintain, retrain, and demonstrate.

## Page 26: Implementation Details
The implementation is written in Python. The main libraries used are:
pandas and numpy for data handling
scikit-learn for model training and evaluation
matplotlib for plots
streamlit for the interactive interface
pickle for model serialization

The project directory contains:
`src/` for preprocessing, training, prediction, and utility logic
`app/` for the Streamlit interface
`data/` for the dataset
`models/` for the serialized final model
`plots/` for confusion matrix, learning curve, comparison charts, and feature plots

## Page 27: Streamlit Interface
The final user-facing application is built with Streamlit and provides a modern input-and-result workflow. Users can enter student details such as age, gender, CGPA, academic pressure, study satisfaction, work-study hours, financial stress, sleep duration, dietary habits, suicidal thoughts, and family history. After submitting the form, the application returns:
Prediction label
Confidence score
Probability breakdown
Feature contribution chart
Model performance metrics
Training plots

The interface is designed to be cleaner and more accessible than a code-only workflow, making the system suitable for demonstration to academic stakeholders.

## Page 28: Applications
The system can support several practical applications:
Counseling support triage
Student wellness awareness and intervention campaigns
Academic advisory and early-warning workflows
Institutional mental-health research
Demonstration of machine learning in socially relevant domains

The system is especially useful where institutions want to identify patterns at scale and direct human support more efficiently.

## Page 29: Limitations and Future Scope
Limitations:
The model depends on the quality and representativeness of the dataset.
Predictions are based on self-reported or static profile variables rather than clinical interviews.
The system provides binary classification only and does not estimate severity.
The model is intended for screening support and not for diagnosis.

Future scope:
Severity-level classification
Institution-specific retraining
Web or mobile deployment
Temporal tracking across repeated student assessments
Counselor dashboards and aggregated analytics
Incorporation of additional contextual or behavioral data sources

## Page 30: Conclusion
This project demonstrates that machine learning can be applied meaningfully to student mental-health screening when accuracy, interpretability, and usability are developed together. Using a large student dataset and a structured training pipeline, the project compared three machine learning models and selected Gradient Boosting as the final deployed classifier. The final system achieved 84.87% cross-validation accuracy and 84.70% test accuracy, with strong precision and recall for the depressed class.

Beyond predictive performance, the project also provides a usable application interface and clear insight into which factors drive the predictions. The results show that student depression can be screened using academic, behavioral, and psychological indicators with strong effectiveness. With proper ethical framing and human oversight, such systems can help institutions move from reactive support toward earlier and more scalable intervention.
