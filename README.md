# Revenue Prediction in a Successful Mobile Puzzle Game (Live Operations)
The user data used in this project is from a successful puzzle game of a gaming studio based in Istanbul, Turkey.

The aim of this project is to train a Machine Learning model using the available user data and their behavior while also considering their generated 'revenue', which is the target output of the model. This data is of a successful crossword puzzle from the Google Play Store and AppStore, and it involves both their first 24 hours of gameplay behavior and all time behavior. There are 23 predictors in the available dataset of the users and the response variable is revenue. After the model is trained, it is used to predict the approximate revenue generation of future players and their demographics, so that the marketing, ad strategy and in game monetization can aim the best possible scenario in terms of revenue generation.


### Table 1: List of all variables in the dataset, with similar variables grouped in the same category
| Category  | Variable | Variable Description |
| ------------- | ------------- | ------------ |
| `General User Info`  | Os  | Operating system of the user's mobile phone |
| `General User Info`  | Country  | Country of the user |
| `General User Info`  | Device Brand  | Mobile device brand of the user |
| `General User Info`  | Device Model  | Mobile device model of the user |
| `General User Info`  | Lang  | Language the user plays the game |
| `General User Info`  | Multi_lang  | If the user plays the game in more than one language |
| `General User Info`  | Reinstall  | If the user deletes the game and re-downloads it |
| `Session Info`  | Max_ses_length  | Maximum session length of the user in the first 24 hours |
| `Session Info`  | Median_ses_length  | Median session length of the user in the first 24 hours |
| `Session Info`  | Avg_ses_length  | Average session length of the user in the first 24 hours |
| `Session Info`  | Session_cnt  | Session count in the first 24 hours |
| `Session Info`  | Session_length  | Sum of the session length of the user in the first 24 hours |
| `Session Info`  | Gameplay_duration | Sum of the duration which the user plays the game in  the first 24 hours |
| `Game Success` | Max_lvl_no | Maximum number of levels the user reached in the first 24 hours |
| `Game Success` | Gold_cnt | The user's gold count at the end of the first 24 hours |
| `Game Success` | Claim_gold_cnt | The number of gold claims the user made in the first 24 hours |
| `Help` | Bonus_cnt | Sum of the bonus which the user used while playing the game |
| `Help` | Hint1_cnt | Sum of the type 1 hint count which the user used while playing the game |
| `Help` | Hint2_cnt | Sum of the type 2 hint count which the user used while playing the game |
| `Help` | Hint3_cnt | Sum of the type 3 hint count which the user used while playing the game |
| `Help` | Repeat_cnt | Sum of the type 3 hint count which the user used while playing the game |
| `Ads` | Banner_cnt | Number of banner-type ads the user watched in the first 24 hours |
| `Ads` | Is_cnt | Number of interstitial-type ads the user watched in the first 24 hours |
| `Ads` | Rv_cnt | Number of banner-type ads the user watched in the first 24 hours |
| `Revenue` | Rev | Target column. Indicates the users first 24 hours revenue |

## Data Visualization
To understand the user data and how they affect each other and revenue, first data is visualized through different methods like scatterplots, boxplots, heatmaps etc.

Below are some of the important figures that I have came across throughout this process.
### Figure 1: Average Revenue vs Country in Bar Graph
![Figure 1: Average Revenue vs Country in Bar Graph.](https://github.com/demirelozan/liveOpsGamingUserDataPrediction/blob/main/gamingUserDataFigures/Average%20Revenue%20by%20country.png)

### Figure 2: Numerical Variables Relations with Each Other for Outlier Detection
![Figure 2: Numerical Variables Relations with Each Other for Outlier Detection.](https://github.com/demirelozan/liveOpsGamingUserDataPrediction/blob/main/gamingUserDataFigures/Figure_1%20Updated%20v3.png?raw=true)

### Figure 3: Correlation of Features with Revenue in Heatmap
![Figure 3: Correlation of Features with Revenue in Heatmap.](https://github.com/demirelozan/liveOpsGamingUserDataPrediction/blob/main/gamingUserDataFigures/Correlation%20of%20Features%20with%20Revenue.png?raw=true)

### Figure 4: Correlation Matrix for All Variables
![Figure 4: Correlation Matrix for All Variables.](https://github.com/demirelozan/liveOpsGamingUserDataPrediction/blob/main/gamingUserDataFigures/Correlation%20Matrix%20for%20All%20Variables.png?raw=true)

## Feature Engineering Using the Importances
The Exploratory Data Analysis (EDA) is used to help create new features, which is used to create better relationships to revenue and perform better accuracy in the training and test datasets. 

Below are the features that are added by me with Feature Engineering:
### Table 2: List of New Features created by category, along with how it is calculated
| Features Category  | Feature (Variable) | Feature Formulas |
| ------------- | ------------- | ------------ |
| `Session Engagement`  | LevelPerSession  | max_lvl_no / session_cnt |
| `Session Engagement`  | InteractivityPerSession  | hint1_cnt + hint2_cnt + hint3_cnt + bonus_cnt + (repeat_cnt / session_cnt) |
| `Session Engagement`  | AvgGameplayDurationPerSession  | gameplay_duration / session_cnt |
| `Player Efficiency and Success`  | PositiveGameplay  | max_lvl_no + gameplay_duration + claim_gold_cnt |
| `Player Efficiency and Success`  | Penalty  | hint1_cnt + hint2_cnt + hint3_cnt + bonus_cnt + repeat_cnt + (bonus_cnt / session_cnt) |
| `Player Efficiency and Success`  | PenaltyInteractivity  |hint1_cnt + hint2_cnt + hint3_cnt + bonus_cnt + repeat_cnt + bonus_cnt |
| `Player Efficiency and Success`  | GameEfficiencyRate  | PositiveGameplay - Penalty |
| `Ad Interaction Features`  | WeightedAdInteraction (Used the weights from Figure 3) | Banner_cnt * banner_weight + is_cnt * is_weight + rv_cnt * rv_weight |
| `Ad Interaction Features`  | AdInteractionPerSession | (banner_cnt + is_cnt + rv_cnt) /session_cnt |

While Correlations do not directly showcase the importance of the variables will be doing inside the model training, just to see the relations, another Correlation with Revenue has been performed with these new features specifically.

### Figure 5: Correlation of New Features with Revenue
![Figure 5: Correlation of New Features with Revenue.](https://github.com/demirelozan/liveOpsGamingUserDataPrediction/blob/main/gamingUserDataFigures/Correlation%20of%20New%20Features%20with%20Revenue%20v2.png?raw=true)

## Application of the Models
The Machine Learning models that are used in this project are:
• **Linear Regression:** a simple but effective method. Many of the correlations found between features and revenue are relatively linear like the graphs shown in figure 2. For example, in the graph of "Revenue vs banner_cnt", watching more banner ads increases revenue generated fairly linearly. With multiple features, this is called multiple linear regression, and the model has an equation as shown 
below:
**• Decision Tree:** the method produces a tree with branches where the data is partitioned and the response value for each datapoint is determined based on the range of values the data fits in, while minimizing RSS. However, with many features it becomes too complicated to comprehend decision tree graphs. 
**•	Random Forest:** this method builds multiple decision trees and combines their predictions and is more suitable for datasets with many features. This enhances accuracy and mitigates overfitting, and can help show relationships among various features.
**• Gradient Boosting:** this is one of the best methods on tabular datasets for classification and regression predictive models. It uses trees but minimizes prediction error but iteratively building a model. This makes sense for the provided problem since many players will have similar gaming habits, so classification combined with regression is effective.
**•	XGBoost:** this last method is very similar to gradient boosting with many optimization methods. It is expected that this model will do the best since it is similar to Gradient Boosting, but is better optimized.

In order to further enhance the models, feature importance was tracked for each method application. This identifies which features are most useful for each method, might hint at which features work better for which model and might explain why a model would be successful.

### Validation Techniques
In order to be able to assess the models, using the training dataset which includes revenue, the data was split for the training: with 80% of the data being used to train each model, and 20% used to test.
In order to assess models, cross validation which calculates the average Root Mean Squared Error (RMSE), and average R-squared, are used, where:

![image](https://github.com/demirelozan/liveOpsGamingUserDataPrediction/assets/57192515/33756886-a74e-4325-b484-2ea1c5a4e578)

![image](https://github.com/demirelozan/liveOpsGamingUserDataPrediction/assets/57192515/1be740a5-beda-41c2-a5b2-46d3523d69cf)

Cross validation was performed using 5 folds, meaning the data is split into 5 and the model is trained and evaluated 5 times, which gives a more robust estimate of how successful each model is. 


