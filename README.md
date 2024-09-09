# Optimizing Bank Marketing Campaign
*This is the final project from the Digital Talent Incubator Program offered by Purwadhika Digital Technology School*  

## **Overview**

This project focuses on developing a machine learning model to predict whether a customer will subscribe to a term deposit after being contacted through a telemarketing campaign. The goal is to improve the effectiveness of Electric Bank's telemarketing strategy by increasing the conversion rate and enhancing the return on marketing investment (ROMI) to match or exceed industry benchmarks.

## **Context**

### Telemarketing Campaign Overview
- **Total Customers Contacted**: 41,176
- **Total Subscriptions Achieved**: 4,640
- **Cost Call Per Customers**: [25 dollars](https://www.cloudtalk.io/blog/how-much-does-call-center-software-cost/?_gl=1*q3ml5d*_up*MQ..*_ga*OTM4ODM3ODg4LjE3MjM0NDExNjY.*_ga_SMHV632SNF*MTcyMzQ1MTA5NS4yLjAuMTcyMzQ1MTA5NS4wLjAuMA..) -> 23 euros
- **Minimum Deposit**: [500 euro](https://www.activobank.pt/simple-account)
- **Total Cost of Calls**: 41,176 customers * 23 euros = 947,048 euros
- **Total Revenue from Subscriptions**: 4,640 subscriptions * 500 euros = 2,320,000 euros

### Conversion Rate Calculation
$$
\text{Conversion Rate} = \left( \frac{\text{Total Subscriptions Achieved}}{\text{Total Customers Contacted}} \right) \times 100 = \left( \frac{4,640}{41,176} \right) \times 100 \approx 11.27%\%
$$

An 11.27% conversion rate, while a solid performance, still lags behind the top performers in the financial industry, who are converting at a rate of 23% ([Ruler Analytics](https://www.ruleranalytics.com/blog/reporting/financial-services-marketing-statistics/#:~:text=Marketers%20in%20the%20financial%20services,at%20a%20rate%20of%2023%25)). This gap highlights the need for Electric Bank to improve its telemarketing strategy to reach the success levels seen by industry leaders. 

### ROMI Calculation
$$
\text{ROMI} = \left( \frac{\text{Total Revenue} - \text{Total Cost of Calls}}{\text{Total Cost of Calls}} \right) \times 100 = \left( \frac{2,320,000 - 947,048}{947,048} \right) \times 100 \approx 144.92%\%
$$

Electric Bank’s ROMI of 144.92% indicates that for every euro spent on telemarketing, the bank generates an additional 1.45 euros in profit. However, this figure is below the industry benchmark of 5:1 or 500%, which is considered a good ROMI ([Improvado](https://improvado.io/blog/return-on-marketing-investment)). This suggests that there is substantial room for improvement in the profitability of the bank's marketing efforts, as achieving a higher ROMI is essential for ensuring that marketing investments yield substantial returns.

## **Problem Statement**
The main challenge is to refine Electric Bank's telemarketing approach to increase the conversion rate and ROMI. The current conversion rate of **11.27%** and a ROMI of **144.92%** indicate potential, but there is significant room for improvement. The objective is to develop a machine learning model that accurately predicts which customers are likely to subscribe to a term deposit, allowing the bank to focus its efforts on high-potential leads, with the ultimate goal of achieving conversion rates similar to top performers and maximizing ROMI.

## **Goal**

The primary objectives of this project are:
- **Achieve Top Performer Conversion Rates**: Improve the precision of targeting potential subscribers to increase the conversion rate to match the **top performers in the industry at 23%**.
- **Maximize ROMI**: Enhance the return on marketing investment by ensuring that the profit generated from successful term deposit subscriptions significantly exceeds the costs of the telemarketing campaigns, aiming for a ROMI closer to the **industry benchmark of 500%**.

## Evaluation Metrics
- **Precision**: Chosen because the cost of false positives (predicting that a client will subscribe when they actually won’t) is high. With precision, we can focus on ensuring that the clients we predict as likely to subscribe are indeed the ones who will do so, which directly supports the goal of reducing unnecessary telemarketing efforts.

## Cost Analysis
Below are the net gains associated with each possible prediction outcome:

- **True Positive (TP):**
  - Net Gain: 500 euros (deposit revenue) − 23 euros (call cost) = 477 euros

- **False Positive (FP):**
  - Net Gain:  0 euros (no revenue) − 23 euros (call cost) = − 23 euros

- **True Negative (TN):**
  - Net Gain: 0 euros (no revenue, no cost)

- **False Negative (FN):**
  - Net Gain: 0 euros (no revenue, no cost)

## Stakeholder

The primary stakeholder for this project is the **Marketing Team Electric Bank**.

## Data Understanding

* For this project, we use a dataset that describes Portugal bank marketing campaigns conducted using telemarketing, offering customers to place a term deposit. If after all marking efforts customer has agreed to place a deposit - the target variable is marked 'yes', otherwise 'no'.  

* Each row represents information from a customer and the socio-economic circumstances of the previous marketing campaign.

(source: https://www.kaggle.com/datasets/volodymyrgavrysh/bank-marketing-campaigns-dataset)

### **Attribute Information**


**Customer Demographic**
<br>

| Attribute | Data Type | Description |
| --- | --- | --- |
|Age |Integer | age of customer |
|Job |Text | type of job (categorical: "admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown") |
|Marital |Text | marital status (categorical: "divorced","married","single","unknown"; note: "divorced" means divorced or widowed) |
|Education |Text | level of education (categorical: "basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown") |
|Default |Text | has credit in default? (categorical: "no","yes","unknown") |
|Housing |Text | has housing loan? (categorical: "no","yes","unknown") |
|Loan |Text | has personal loan? (categorical: "no","yes","unknown") |
<br>

**Information During This Campaign**

| Attribute | Data Type | Description |
| --- | --- | --- |
|Contact |Text | contact communication type (categorical: "cellular","telephone") |
|Month |Text | last contact month of year (categorical: "jan", "feb", "mar", …, "nov", "dec") |
|Day_of_week |Text | last contact day of the week (categorical: "mon","tue","wed","thu","fri") |
|Duration |Integer | last contact duration, in seconds. Important note: this attribute highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known |
|Campaign |Integer | number of contacts performed during this campaign and for this customer (numeric, includes the last contact) |
<br>

**Information From Previous Campaign**

| Attribute | Data Type | Description |
| --- | --- | --- |
|Pdays |Integer | number of days that passed by after the customer was last contacted from a previous campaign (numeric; 999 means the customer was not previously contacted) |
|Previous |Integer | number of contacts performed before this campaign and for this customer |
|Poutcome |Text | outcome of the previous marketing campaign (categorical: "failure","nonexistent","success") |
<br>

**Customer Socio-Economic**

| Attribute | Data Type | Description |
| --- | --- | --- |
|Emp.var.rate |Float | employment variation rate - quarterly indicator |
|Cons.price.idx |Float | consumer price index - monthly indicator |
|Cons.conf.idx |Float | consumer confidence index - monthly indicator |
|Euribor3m |Float | euribor 3 month rate - daily indicator |
|Nr.employed |Float | number of employees - quarterly indicator |
<br>

**Target**

| Attribute | Data Type | Description |
| --- | --- | --- |
|Y |Text | Has the customer subscribed to a term deposit? (binary: "yes", "no") |

## Data Preparation

**Data Cleaning**
* Removed duplicate records.
* Deleted features with mostly missing values (e.g., the 999 value in the pdays column).
* Handled outliers.
* Checking invalid values
* Checking label ambiguities.
* Checking cardinality
* Regrouped certain features for better analysis.
* Applied binning with KBinsDiscretizer using the quintile strategy.

**Feature Engineering**
* Extracted features such as Loan Status from Housing_Loan and Personal_loan
* Conducted feature selection using Power Predictive Score (PPS).
* Applied feature transformation, including custom encoding, binary encoding, one-hot encoding, label encoding, and scaling using Robust Scaler.

## Model Choose: XGBoost Classifier

After evaluating the models' performance and tuning the hyperparameter, the XGBoost Classifier was chosen over Gradient Boosting and any other model.

**Before and After Hyperparameter Tuning on XGBoost**
________________

|  |  Precision |Convension Rate|ROMI|
| --- | --- | --- | --- |
| Before | 0.65 | 63.59 | 1282 |
| After | 0.84 | 83.57 | 1716 |

We can see that hyperparameter tuning give a better result from the baseline from 0.6 to 0.8. Precision, Conversion Rate, and ROMI are increased from the previous model performance. From 2 models, Gradient Boosting and XGBoost Classifier, the best model after tuning is XGBoost which could reach 0.84. **So, we choose Tuned XGBoost Classifier for our final model**.

## Model Evaluation
**Classification Report**

From the confusion matrix, based on our business purpose where we want to have an efficient marketing campaign, means a high True Positive among all Positive Predicted Label, we got 117 of 140, indicating our model can predict 84% of predictions. 

<img width="430" alt="Screenshot 2024-08-25 at 21 56 23" src="https://github.com/user-attachments/assets/496d9aff-f1c6-46f5-ad01-6fe08a209c3f">

![output](https://github.com/user-attachments/assets/ea39c54a-6ef5-4564-8980-375d19e7e906)

**Learning Curve**

From the learning curve, the model shows stable and similar precision on both training and validation data as the dataset size increases, suggesting that the model is well-generalized and not overfitting.

![Learning Curve XGB](https://github.com/user-attachments/assets/a3fc65b4-ea86-4f3d-8a35-5771b063c1d2)


**Reliability Curve**

A lower Brier score indicates better calibration and better prediction accuracy of probabilities, meaning the predicted probabilities are closer to the actual outcomes.
In our model, the Brier score is approximately 0.07, which indicates how well-calibrated the predicted probabilities are. This may be due to the preprocessing steps being done correctly for the train and test data.

![Reliability Curve](https://github.com/user-attachments/assets/1fc3cf02-dcfc-42fe-a839-02e62e1fffc6)


## Business Simulation  

**1. Scenario without Modeling**

In this scenario our bank will give a campaign to all customers, that is 8,236 unseen data with a conversion rate of 11.27%. Here is the cost revenue calculation :

Total deposit subscription = 928 customers  
Conversion Rate = 11.27%
<br>Total telemarketing cost = 8,236 x EUR 23 = EUR 189,428
<br>ROMI (Return on Marketing Investment) = 144.92 %

**2. Scenario with Modeling**

After we do modeling, we can calculate the possible revenue and ROMI from 8.236 unseen data based on the confusion matrix.

TP (Predict Deposit, Actual Deposit): 117
<br>FP (Predict Deposit, Actual No Deposit): 23
<br>FN (Predict No Deposit, Actual Deposit): 811
<br>TN (Predict No Deposit, Actual No Deposit):  7,285

We will give the campaign only to customers who are predicted to Deposit (TP and FP) :

Total deposit subscription = 117 of 140 customers   
Conversion Rate = 117/(117+23) * 100 = 83.57 %
<br>Total revenue =  117 x EUR 500 = EUR 58,500
<br>Total telemarketing cost = (117 + 23) x EUR 23 = EUR 3,220
<br>ROMI (Return on Marketing Investment) = (58,500 - 3,220)/3,220 = 1,716.7%

**3. Comparison:**

* **Without modeling:**  
  CVR: 11.27%  
  ROMI: 144.92%.

* **With modeling**:  
  CVR: 83.57 %  
  ROMI: 1,716.7%

## Conclusion
We developed the best model using data cleaning, feature extraction, preprocessing techniques, and model benchmarking which is the XGBoost Classifier. The model achieves a high accuracy of 90% and a precision of 84%, meaning it is effective at correctly predicting deposits, and from the model evaluation using the Learning Curve and Brier Score, the model shows the best performance and is expected to give accurate predictions. In summary, **the model can help Electric Bank improve its Conversion Rate 7.4 times and Return On Marketing Investment (ROMI) 11.84 times from original state by effective targeted telemarketing.**

## Recommendations   

1. **Targeted Campaign**  
      - Use the predictive model to identify **customers who are most likely to subscribe** to a term deposit and design targeted marketing campaigns to engage these potential customers.  
      - Based on Global Explanation using SHAP, we found that the longer the duration of the call, the more likely it is that the client will agree to the deposit. We also found specific months and contact types have a significant impact. This global explanation was also supported by the findings in predictive modeling in Local explanation: Counterfactual, where the desired outcomes are highly influenced by those features. 

2. **Specific Recommendation Samples**  
      Based on the Counterfactual, found examples of how we can change the non-deposit customers to subscribe to the deposit.  
      - Call duration 3.35 mins (No Deposit) => Deposit
            a. ± 12 mins
            b. ± 29 mins
            c. ± 36 mins
      - Call duration 4 mins via telephone (No Deposit) => Same duration via cellular (No Deposit)

  3. **Customer Database**  
      To ensure the accuracy and relevance of predictive models, it's crucial to maintain a robust customer database and collect high-quality data. This involves several key practices based on our EDA findings:   
      - **Maintain Data Integrity**: Establish a reliable system to store and manage customer data. Regularly update the database to prevent data from becoming outdated, and continuously monitor for data drift—a situation where changes in data patterns over time can reduce the accuracy of our model.  
      - **Ensure Complete and Accurate Data Collection**: Encourage customers to provide complete information, avoiding placeholders like 'Unknown.' Collect comprehensive economic details, such as income, which are vital for accurate analysis and modeling.   
      - **Capture Important Behavioral Data**: Include additional relevant data points, such as call timestamps, to understand customer behavior over time. This temporal information can improve the model’s predictive power by highlighting trends and patterns.  

  4. **Campaign Regulation**  
      Define guidelines based on the model's insights to improve the probability of customer subscriptions. Use these regulations to refine campaign strategies and improve conversion rates. The recommendations are **what's the maximum call we should perform for a customer(in this case 56 max calls - based on our EDA findings), how many days should we call for a month, etc**.   

## Dashboard and Story
To explore more about how this dataset looks like, we provide you [Tableau Dashboard]()  

