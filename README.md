# Revenue Prediction in Gaming Industry
The user data used in this project is from a successful puzzle game of a gaming studio based in Istanbul, Turkey.

The aim of this project is to train a Machine Learning model using the available user data and their behavior while also considering their generated 'revenue', which is the target output of the model. This data is of a successful crossword puzzle from the Google Play Store and AppStore, and it involves both their first 24 hours of gameplay behavior and all time behavior. There are 23 predictors in the available dataset of the users and the response variable is revenue. After the model is trained, it is used to predict the approximate revenue generation of future players and their demographics, so that the marketing, ad strategy and in game monetization can aim the best possible scenario in terms of revenue generation.


Table 1: List of all variables in the dataset, with similar variables grouped in the same category
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
![Figure 1: Average Revenue vs Country in Bar Graph.](https://github.com/demirelozan/liveOpsGamingUserDataPrediction/blob/main/gamingUserDataFigures/Average%20Revenue%20by%20country.png)

