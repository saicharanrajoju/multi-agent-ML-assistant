# Adult Income Dataset (Census Income)

**Source Citation**: Becker, B. and Kohavi, R. (1996). Adult dataset. UCI Machine Learning Repository. [https://archive.ics.uci.edu/ml/datasets/Adult]

## Why This Dataset Is Interesting for ML Agents

This dataset represents a significant step up from standard "tutorial" datasets (like exactly-clean Titanic or Telco Churn), effectively stress-testing our Multi-Agent ML pipeline due to several unique challenges:

1. **Real-World Null Encoding (" ?") Trap** 
   Unlike standard CSVs where missing values are explicitly formatted as `NaN` or `None`, missing categorical features (like `workclass` and `occupation`) are disguised as the string `" ?"`. A naive profiling pipeline will entirely miss these nulls, misinterpreting them as an actual category. Our system's Profiler agent's ability to detect this proves its intelligence against "999 = null"-style problems.

2. **Severe Class Imbalance**
   The target variable `income` is significantly skewed with ~75.9% representing `" <=50K"` and ~24.0% representing `" >50K"`. It evaluates the pipeline's capability to natively detect imbalance and implement techniques like SMOTE, class weighting, or optimized thresholds without human intervention.

3. **Size & Complexity**
   While 48,842 total recorded entries (32,561 in `adult.data`) across 15 attributes shouldn't exceed local memory limits, it perfectly balances numeric constraints, high-cardinality nominal variations (e.g. `native-country` with 40+ variations), and sparse representation challenges unseen in tiny toy datasets.

4. **Continuous + Categorical Mix**
   Contains heavily skewed continuous features like `capital-gain`, zero-inflated distributions (`capital-loss`), alongside purely categorical descriptions perfectly built for testing how the automation pipeline builds categorical encoders or transforms skewed data automatically.

## Feature Descriptions

* **age**: continuous.
* **workclass**: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked. (Contains " ?" values).
* **fnlwgt**: continuous. This is the demographic weight given by the census bureau.
* **education**: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
* **education-num**: continuous. Ordinal counterpart of education.
* **marital-status**: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
* **occupation**: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces. (Contains " ?" values).
* **relationship**: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
* **race**: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
* **sex**: Female, Male.
* **capital-gain**: continuous. Skewed heavily to 0.
* **capital-loss**: continuous. Skewed heavily to 0.
* **hours-per-week**: continuous.
* **native-country**: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands. (Contains " ?" values).
* **income** (Target Variable): >50K, <=50K.
