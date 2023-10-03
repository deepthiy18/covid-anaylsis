# covid anaylsis
 covid-19 detection anaylsis
Covid-19 Prediction Analysis

Data Source:
•	The data was obtained from the official government website 'ABC' and comprises information about individuals who were subjected to RT-PCR testing for COVID-19. The dataset covers the period from March 11, 2020, to April 30, 2020.

Data Description:
•	Total records: 2,78,848 individuals
•	Columns: 11 columns including 8 features suspected to influence COVID-19  
•	outcomes
•	Target Variable: COVID-19 Test result (Positive or negative)

Problem Statement:
•	A speedy and accurate diagnosis of COVID-19 is made possible by effective SARS-CoV-2 screening, which can also lessen the burden on healthcare systems. There have been built prediction models that assess the likelihood of infection by combining a number of parameters. These are meant to help medical professionals all over the world treat patients, especially in light of the scarcity of healthcare resources. The current dataset has been downloaded from ‘ABC’ government website and contains around 2,78,848 individuals who have gone through the RT-PCR test. Data set contains 11 columns, including 8 features suspected to play an important role in the prediction of COVID19 outcome. Outcome variable is covid result test positive or negative. We have data from 11th March 2020 till 30th April 2020. Please consider 11th March till 15th April as a training and validation set. From 16th April till 30th April as a test set. Please further divide training and validation set at a ratio of 4:1.  

Section 1: Questions to Answer:


1.	Why is your proposal important in today’s world? How can accurately predicting a disease improve medical treatment?

In the current global scenario dominated by the COVID-19 pandemic, the ability to   swiftly and accurately predict the disease status of individuals is crucial. It allows for:

	Timely isolation and treatment of positive cases, preventing further spread.

	Efficient allocation of healthcare resources, ensuring that critical cases receive prompt attention.

	Reduction in unnecessary testing, thereby conserving resources and minimizing inconvenience to individuals.

	Early intervention, which can lead to milder symptoms and better overall outcomes.


2.	How is it going to impact the medical field when it comes to effective screening and reducing health care burden.

 Implementing our proposed predictive Modeling approach will revolutionize disease screening in the medical field. By leveraging machine learning, we can:
•	Expedite the diagnosis process, especially in regions with limited testing capabilities.

•	Alleviate strain on healthcare systems by efficiently identifying COVID-19 positive individuals.

•	Ensure that urgent cases receive immediate care, optimizing resource allocation.

•	Streamline patient influx, enhancing the quality of healthcare services and potentially saving lives.


3.	If applicable, what is the knowledge gap or how could your proposed method be beneficial for future applications in predicting other diseases?

•	The methodology developed for predicting COVID-19 based on symptoms can serve as a versatile platform for predicting various diseases. The expertise gained in constructing this model, including feature selection strategies and computational methodologies, can be applied to future disease prediction efforts. This has the potential to address knowledge gaps in numerous healthcare sectors, enabling quicker and more accurate identification of disorders beyond COVID-19. The ability to predict diseases accurately holds far-reaching implications for personalized medicine, public health planning, and the development of tailored treatment approaches, ultimately leading to improved healthcare outcomes for individuals and communities alike.
Section 2: Initial Hypothesis (or hypotheses):
	Hypothesis 1:
•	Patients who came in direct contact with Confirmed (Covid Positive) Patients are more likely to be Corona positive.
•	This hypothesis suggests that individuals with known contact with confirmed COVID-19 positive patients are at a higher risk of being infected themselves. The belief is that the virus primarily spreads through close proximity and direct contact. This hypothesis can be tested by analysing the variable "Known_contact" in the dataset to see if there is a correlation between known contact with COVID-19 positive patients and the likelihood of testing positive for the virus.
	Hypothesis 2:
•	Shortness_of_breath, Fever, and Cough_symptoms are essential variables in determining COVID-19 positive or negative cases.
•	This hypothesis proposes that symptoms like shortness of breath, fever, and cough are crucial indicators in diagnosing COVID-19 positive cases. These symptoms are commonly associated with respiratory infections and have been recognized as major indicators of COVID-19. This hypothesis can be tested by evaluating the variables "Shortness_of_breath," "Fever," and "Cough_symptoms" in the dataset to study the association between these symptoms and the likelihood of a person being COVID-19 positive or negative.

•	It's important to note that these are initial hypotheses and further analysis, including the application of machine learning models, will be necessary to validate and gain deeper insights from the data.

Data Analysis Approach:
	Data Understanding & Exploration steps we performed-
1.	We imported all necessary libraries require like pandas, numpy, matplotlib, seaborn, plotly and sklearn.
2.	We imported our dataset that is in .csv file format and we made copy of it so that any errors will not damage original dataset.
3.	In our observation- Except 'Ind_ID' column, every column is 'object' datatype.
4.	For our understanding we renamed to columns as 'Corona' to 'Test_result' and  'Ind_ID' to'ID'.
5.	In our observation- We have more number of female patients records (1,30,158) in this dataset.
6.	After checking unique values in each column we found alphabetical discrepancies in many columns. So, with the help of ‘Replace’ function we removed those alphabetical discrepancies.
7.	 After evaluating Test_result column we found that Most number of covid-19 tests are examined on '20-04-2020' = 10921.
	Handling Missing/Incorrect values-
1.	In all columns True and False are written in two different alphabetical types. We need to update this typing errors. We also have null(None) values in few columns, we need to remove them.
2.	After evaluating we observed that we don’t have any ‘Missing Values’ in our dataset. Instead we have "None" as values in many columns. Here we need to replace them with 'Mode' of particular column.
3.	After replacing ‘None’ values and all anomalies we need to save this file at this stage for MySQL analysis.
	Data Visualisation-
1.	We visualised our columns with help of heatmap and we observed that-
	Cough symptoms and Fever are highly correlated.
	After that Soar throat and Headache have next higher values.
	Values which are close to 0 are having less correlation and values which are more towards 1 are highly correlated.
	Correlation shows strength of relationship between two variables.
2.	After comparing all Symptoms with Test result we observed that –
	Among all 'positive test results' most common symptom is 'Cough'.
	Shortness of breath' is least common symptom in 'positive test results'.
3.	 After Analysing “Known Contact” column we found that ‘Other’ category in that columns is giving highest count.
	Feature Encoding-
1.	In this step we converted categorical values to continuous values.
2.	We used ‘astype’ method to change wrong datatype of our column in appropriate type.
3.	We used ‘map’ method to encode categorical values to continuous values in our columns.
4.	Since there are no categorical columns, so feature scaling and feature transformation is not required here.
	Feature Selection-
1.	Here we performed ‘Chi-square’ test because in our dataset we have more than two independent categorical variables.
2.	After performing Chi-square test we observed that- we have all categorical columns. Here we got p_value [Probability value] for columns less than 0.05 so all independent columns have relationship with dependent column i.e., 'Test_result'. We are getting p-value for 'ID' column = 0.499. We if p-value is greater than 0.05 then we can drop particular column.

Machine Learning Approach:
We need to predict whether Covid-19 test will be positive or negative, basically categorcial output. So we can use algorithsm which are best for categorical output here.
Here we are going to use following 4 algorithms:-
1.	Logistic Regression,
2.	Decision Tree,
3.	Random Forest,
4.	K Nearest Neighbors
and after comparing all algorithms we will decide which is best fit algorithm for our dataset.
	Decision Tree Algorithm- we performed this with help of DecisionTreeClassifier.
	Random Forest- we performed this with help of RandomForestClassifier.
	Logistic Regression Algorithm- we performed this with help of LogisticRegression.
	K Nearest Neighbors (KNN) Algorithm- we performed this with help of KNeighborsClassifier.
	After performing confusion matrix and classification report on each algorithm we got following results- 
Accuracy of our 4 Algorithms:-
1.	Decision Tree = 98.37%
2.	Random Forest = 98.37%
3.	Logistic Regression = 97.81%
4.	K Nearest Neighbors Algorithms = 97.42%
From above results we can observe that accuracy of our 4 Algorithms are very close to each other. If we compare all then we can conclude that 'Decision Tree' or 'Random Forest' Algorithm are best for our Covid-19 dataset.
	After visualisation of our algorithms we performed Model evaluation and Optimization for our Algorithms like- 
•	Performing any one of the three methods is usually sufficient for assessing model performance.
1.	KFold cross-validation,
2.	Cross-validation_score and
3.	GridSearchCV(hyperparameter tuning)

	A standard deviation of zero (0.00) in cross-validation scores usually indicates that the model's performance is consistent across different folds of the cross-validation process. In other words, the model is consistently making predictions with very similar accuracy across all subsets of the data.
	After our Model evaluation we observed that- 
Through cross-validation scoring, we obtained the following mean accuracy scores:

Logistic Regression = 94.98%

Decision Tree = 95.66%

Random Forest = 95.68%

K Nearest Neighbors Algorithm = 94.59%

Upon close examination, it's evident that the accuracies of all four algorithms are quite similar. Upon comparison, it is apparent that the 'Random Forest' algorithm outperforms the others, making it the most suitable choice for our Covid-19 dataset.



Preventative Measures:

Vaccination: Getting vaccinated with authorized COVID-19 vaccines is one of the most effective ways to prevent severe illness and reduce the spread of the virus. Follow your country's vaccination guidelines and schedules.

Mask-Wearing: Wear masks, particularly in indoor settings and crowded areas, where physical distancing is challenging. Use masks that meet local guidelines and cover both your nose and mouth.

Hand Hygiene: Wash your hands frequently with soap and water for at least 20 seconds. If soap and water are unavailable, use hand sanitizer with at least 60% alcohol.

Physical Distancing: Maintain physical distance (e.g., at least 6 feet) from individuals who do not live in your household, especially in crowded places.

Cough and Sneezing Etiquette: Cover your mouth and nose with a tissue or your elbow when coughing or sneezing. Dispose of used tissues properly and wash your hands immediately.

Regular Cleaning: Clean and disinfect frequently-touched surfaces in your home, workplace, and public areas.

Treatment and Government Guidelines:
Isolation: In case of a positive COVID-19 test result or the presence of symptoms, adhere to the local directives for self-isolation to prevent transmission to others.

Seeking Medical Attention: If you encounter severe symptoms like breathing difficulties, chest pain, confusion, or bluish lips or face, promptly seek professional medical assistance.

Medication: Certain treatments such as antiviral medications and monoclonal antibodies may be prescribed by healthcare experts in specific situations. It is important to follow your healthcare provider's recommendations regarding treatment choices.

Quarantine: If you have come into contact with an individual who has tested positive for COVID-19, follow the applicable local guidelines for quarantine to reduce the potential for transmission. Quarantine requirements may vary depending on your location.

Testing: Under the guidance of your healthcare provider or local health authorities, undergo COVID-19 testing, especially if you exhibit symptoms or have been in close proximity to someone with the virus.
