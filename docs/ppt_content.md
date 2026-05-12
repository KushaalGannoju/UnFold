# Student Depression Detection System

## Slide 1: Title Slide
- Title: Student Depression Detection System Using Machine Learning
- Subtitle: Early Risk Screening for Student Mental Health Using Behavioral, Academic, and Lifestyle Indicators
- Team: Himanshu Verma, Kushaal Sai Gannoju, Ritika Rani Ujjainia, Rayyan Zameer Baikadi
- Department: Computer Science and Engineering
- Institute: Maulana Azad National Institute of Technology
- Guide: Dr. R.K. Pateriya and Dr. Archana Balmik

Speaker notes:
This project addresses the need for early identification of depression risk among students using machine learning. The system combines academic, lifestyle, and psychological features to support preliminary screening through an interactive application.

## Slide 2: Problem Background and Motivation
- Student depression is increasing due to academic stress, financial burden, uncertainty about careers, social isolation, and unhealthy routines.
- Many students do not seek help early because of stigma, lack of awareness, and limited access to counseling services.
- Traditional screening depends on manual interviews and self-reporting after symptoms become severe.
- Institutions need a scalable, data-driven support system that can identify at-risk students earlier.
- The proposed system acts as a screening aid, helping counselors prioritize follow-up interventions.

Speaker notes:
The motivation of this project is not to replace professionals, but to support them with a system that can process large student populations quickly and consistently.

## Slide 3: Problem Statement and Objectives
- Problem statement:
Current student mental health support is mostly reactive rather than proactive. There is no simple, reliable, and scalable way to screen depression risk using routinely available student information.
- Primary objectives:
Develop an accurate machine learning model for depression-risk prediction.
Compare multiple models and select the best final classifier.
Build an easy-to-use interface for real-time prediction.
- Secondary objectives:
Design a robust preprocessing pipeline.
Identify the most influential predictors.
Present interpretable results for academic stakeholders.

Speaker notes:
The project combines predictive accuracy, usability, and interpretability so that the system is not only technically correct but also practically deployable.

## Slide 4: Dataset Description
- Dataset source: Kaggle Student Depression Dataset
- Total records: 27,901 student records
- Target classes:
Depressed: 16,336
Not Depressed: 11,565
- Candidate input dimensions covered:
Demographic information
Academic performance and pressure
Study satisfaction and work-study hours
Sleep and diet habits
Financial stress
Suicidal thoughts
Family history of mental illness
- Removed from the final workflow:
ID, city, degree, profession as irrelevant or low-value fields
Job satisfaction and work pressure because the dataset contains non-informative values for them

Speaker notes:
The dataset is large enough to support stable cross-validation and meaningful model comparison. It captures multiple dimensions that influence student mental health.

## Slide 5: Data Preprocessing Pipeline
- Step 1: Load CSV data and standardize column names
- Step 2: Detect and clean the depression target variable
- Step 3: Remove irrelevant or weak fields
- Step 4: Encode categorical values into numeric form
Sleep duration mapped to ordinal hour equivalents
Dietary habits mapped to ordinal quality scores
Yes/No and binary fields converted to 0/1
- Step 5: Keep numeric features only and fill missing values with median values
- Step 6: Apply train-test split with stratification
Training set: 22,320 samples
Testing set: 5,581 samples
- Step 7: Apply 5-fold stratified cross-validation during training

Speaker notes:
The preprocessing pipeline ensures consistency, avoids leakage, and converts mixed data into a clean machine-learning-ready matrix.

## Slide 6: Models Considered and Final Selection
- Models trained:
Gradient Boosting
Random Forest
Logistic Regression
- Final deployed model: Gradient Boosting
- Why Gradient Boosting was selected:
Highest overall generalization performance
Strong handling of non-linear decision boundaries
Stable cross-validation results
Best balance between precision, recall, and F1-score
- Gradient Boosting final tuned configuration produced the strongest production-ready performance.

Speaker notes:
Tree-based boosting was best suited to the dataset because depression risk is influenced by interacting thresholds rather than purely linear relationships.

## Slide 7: Model Performance Comparison
- Gradient Boosting:
Training Accuracy: 85.57%
CV Accuracy: 84.87%
CV Std Dev: 0.19%
Test Accuracy: 84.70%
Precision: 85.84%
Recall: 88.46%
F1 Score: 87.13%
- Random Forest:
Training Accuracy: 79.41%
CV Accuracy: 79.46%
CV Std Dev: 1.66%
Test Accuracy: 78.77%
Precision: 79.78%
Recall: 85.37%
F1 Score: 82.48%
- Logistic Regression:
Training Accuracy: 78.53%
CV Accuracy: 75.82%
CV Std Dev: 0.25%
Test Accuracy: 78.01%
Precision: 73.51%
Recall: 97.64%
F1 Score: 83.87%

Speaker notes:
The comparison clearly shows that Gradient Boosting performs best overall. Random Forest acts as a solid baseline, while Logistic Regression remains useful as a simpler linear benchmark.

## Slide 8: Confusion Matrix and Classification Analysis
- Confusion matrix interpretation for the final Gradient Boosting model:
TN: correctly identified non-depressed students
FP: non-depressed students predicted as depressed
FN: depressed students missed by the model
TP: correctly identified depressed students
- Key interpretation:
High recall on the depressed class means the model successfully captures most at-risk students.
False negatives are especially important because missed cases can delay support.
The final model is more suitable for screening because it prioritizes identifying vulnerable students.

Speaker notes:
In mental-health screening, recall for the depressed class is highly important because missing a genuine case is more harmful than flagging an extra student for follow-up.

## Slide 9: Feature Importance and Insights
- Most influential predictors in the final model:
Have You Ever Had Suicidal Thoughts
Academic Pressure
Financial Stress
Age
Work/Study Hours
Dietary Habits
Study Satisfaction
Sleep Duration
Family History of Mental Illness
CGPA
- Key insight 1:
Suicidal thoughts are the strongest direct risk indicator.
- Key insight 2:
Academic pressure and financial stress strongly shape mental-health outcomes in students.
- Key insight 3:
Lifestyle variables such as sleep, diet, and study hours meaningfully affect prediction quality.

Speaker notes:
The importance ranking shows that the final model captures both psychological warning signs and contextual stress factors instead of relying on only one category of input.

## Slide 10: System Interface, Applications, and Future Scope
- System interface:
Built using Streamlit for a clean and interactive user experience
Allows direct profile entry and instant prediction
Displays prediction outcome, confidence, model metrics, and analysis plots
- Applications:
Early screening support for counselors
Student wellness awareness programs
Institutional risk monitoring and referral workflows
- Future scope:
Severity-level prediction instead of binary classification
Integration with mobile and web platforms
Longitudinal tracking of repeated student assessments
Institution-specific retraining with local data
Integration with counseling support dashboards

Speaker notes:
The project demonstrates that machine learning can be used responsibly as a scalable student-support aid when paired with clear disclaimers and human oversight.
