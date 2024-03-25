# Offline Metrics 

These offline metrics are commonly used in search, information retrieval, and recommendation systems to evaluate the quality of results or recommendations:

### Recall@k:
  - Definition: Recall@k evaluates the fraction of relevant items retrieved among the top k recommendations over total relevant items. It measures the system's ability to find all relevant items in a fixed-sized list.
  - Use Case: In information retrieval and recommendation systems, Recall@k is crucial when it's essential to ensure that no relevant items are missed in the top k recommendations.

### Precision@k:

  - Definition: Precision@k assesses the fraction of retrieved items that are relevant among the top k recommendations. It measures the system's ability to provide relevant content at the top of the list.
  - Use Case: Precision@k is vital when there's a need to present users with highly relevant content in the initial recommendations. It helps in reducing user frustration caused by irrelevant suggestions.

### Mean Reciprocal Rank (MRR):

  - Definition: MRR measures the effectiveness of a system in ranking the most relevant items at the top of a list. It calculates the average of reciprocal ranks of the first correct item found in each ranked list of results: 
  MRR = 1/m \Sum(1/rank_i)
  - Use Case: MRR is often used in search and recommendation systems to assess how quickly users find relevant content. It's particularly useful when there is only one correct answer or when the order of results matters.
  - Reciprocal Rank: 검색엔진의 예에서, 검색 결과들이 순서대로 보여진다고 하자. 이 중에, 실제로 관련있는 링크가 몇번째로 나열되었는지를 측정하는것. 예를들어, 리스트 중에 실제 관련된 링크가 첫번째로 나왔다면, reciprocal rank는 1/1에서 1이다. N번째로 나왔다면, 1/N이 될것.
  - Mean Reciprocal Rank (MRR): 여러번의 검색 결과에서 얻은 reciprocal rank들을 그냥 평균한것. 

### Average Precision for single query
  - 현재 추천된 리스트가 있다고 할 때, 실제 관련있는 아이템들의 위치에서 precision을 계산하여 평균한것. 주어진 리스트가 [1, 1, 0, 0, 1, 0, 0] 이라고 하자. 1은 실제 관련이 있는 아이템, 0은 실제로는 관련이 없는 아이템임. 이때, 첫 아이템부터 끝 아이템까지, 1인 값에서만 precision을 계산하여 평균함.
  - At index 0 (value 1), precision = 1/1. At index 1 (value 1), precision = 2/2. At index 3 (value 0), precision = 2/3 ...
  - 이것은 전체 추천 리스트 중에 몇개가 실제 관련있는지를 측정하는 동시에, 추천의 순서도 측정하게 됨. 관련된 아이템이 높은 순서에 배치되어 있다면, 더 높은 average precision을 가지게 됨. l1=[1, 1, 1, 0, 0, 0], l2=[0, 0, 0, 1, 1, 1]의 예를 생각 해 보면, 뒷쪽의 index에서는 precision이 비슷할것이다 (둘 다 3/6). 그런데, 앞쪽의 아이템에서는 l1의 경우엔 1, 1, 1, 3/4, 3/5, 3/6이고, l2의 경우엔 0, 0, 0, 1/4, 2/5, 3/6이 되어서, 이들의 평균을 비교하면 l1이 더 큰 값을 가지게 됨. 


### Mean Average Precision (mAP):
  - Definition: mAP computes the average precision across multiple queries or users. Precision is calculated for each query, and the mean of these precisions is taken to provide a single performance score.
  - Use Case: mAP is valuable in scenarios where there are multiple users or queries, and you want to assess the overall quality of recommendations or search results across a diverse set of queries. mAP works well for binary relevances. For continues scores, we use nDCG.
  - mAP는 binary label에 대해서 사용. 아래의 DCG는 varying score에 대해서 사용.

### Discounted Cumulative Gain (DCG):
  - Definition: Discounted Cumulative Gain (DCG) is a widely used evaluation metric primarily applied in the fields of information retrieval, search engines, and recommendation systems.
    - DCG quantifies the quality of a ranked list of items or search results by considering two key aspects:
      1. Relevance: Each item in the list is associated with a relevance score, which indicates how relevant it is to the user's query or preferences. Relevance scores are typically on a scale, with higher values indicating greater relevance.
      2. Position: DCG takes into account the position of each item in the ranked list. Items appearing higher in the list are considered more important because users are more likely to interact with or click on items at the top of the list.
    - DCG calculates the cumulative gain by summing the relevance scores of items in the ranked list up to a specified position.
    - To reflect the decreasing importance of items further down the list, DCG applies a discount factor, often logarithmic in nature.
  - Use case: 
    - DCG is employed to evaluate how effectively a system ranks and presents relevant items to users.
    - It is instrumental in optimizing search and recommendation algorithms, ensuring that highly relevant items are positioned at the top of the list for user engagement and satisfaction.
    - 예를들어, relavance score가 다음과 같이 주어졌다고 하면, [8, 5, 3, 2, 1], DGC는 8f(0)+5f(1)+3f(2)+2f(3)+1f(4)로 계산됨. 여기서 f()는 discount function.
    - nDCG는 normalized DCG임. 이것은 nDCG = DCG/iDCG로 계산됨. iDCG는 ideal DCG로, 주어진 score list를 sorting해서 계산함. 만약, score list가 이미 decresing order로 되어있다면, DCG와 iDCG는 동일할것임. 

### Normalized Discounted Cumulative Gain (nDCG):

  - Definition: nDCG measures the quality of a ranked list by considering the graded relevance of items. It discounts the relevance of items as they appear further down the list and normalizes the score. It is calculated as the fraction of DCG over the Ideal DCG(IDCG) for an ideal ranking. 
  - Use Case: nDCG is beneficial when relevance is not binary (i.e., there are degrees of relevance), and you want to account for the diminishing importance of items lower in the ranking.

# Cross Entropy and Normalized Cross Entropy 
- The CE (also a loss function), measures how well the predicted probabilities align with the true class labels. It's defined as:

    - For binary classification:
    CE = - [y * log(p) + (1 - y) * log(1 - p)]
    
    - For multi-class classification:
    CE = - Σ(y_i * log(p_i))
    
    Where:
    - y is the true class label (0 or 1 for binary, one-hot encoded vector for multi-class).
    - p is the predicted probability assigned to the true class label.
    - The negative sign ensures that the loss is minimized when the predicted probabilities match the true labels. (the lower the better)
- NCE: 1 - CE(ML model) / CE(simple baseline such as random guessing). CE(model)은 작을수록 좋고 CE(baseline)은 model보다 크다. 그래서 NCE가 1에 가까울수록 모델이 잘 되는것, 0에 가까울수록 잘 안되는것으로 해석 가능. (CE 값은 항상 양수임. log(p)는 항상 0보다 작은데, 앞에 -가 붙어있으니. )

### Ranking:
* Precision @k and Recall @k not a good fit (not consider ranking quality of out) 
* MRR, mAP, and nDCG good: 
  * MRR: focus on rank of 1st relevant item 
  * nDCG: relevance b/w user and item is non-binary 
  * mAP: relevance is binary 
* Ads ranking: NCE. Model prediction on whether the ads will be clicked or not is compared to the actual click-through rate (CTR). 




  
# Online metrics 
* CTR 


- Definition:

    - Click-Through Rate (CTR) is a metric that quantifies user engagement with a specific item or element, such as an advertisement, a search result, a recommended product, or a link.
    - It is calculated by dividing the number of clicks on the item by the total number of impressions (or views) it received.
    - Formula for CTR:
      CTR= Number of Clicks/Number of Impressions ×100%

    - Impressions: Impressions refer to the total number of times the item was displayed or viewed by users. For ads, it's the number of times the ad was shown to users. For recommendations, it's the number of times an item was recommended to users.

- Use Cases:
  - Online Advertising campaigns: widely used to assess how well ads are performing. A high CTR indicates that the ad is compelling and relevant to the target audience.
  - Recommendation Systems: CTR is used to measure how effectively recommended items attract user clicks.
- Search Engines: CTR is used to evaluate the quality of search results. High CTR for a search result indicates that it was relevant to the user's query.

* Conversion Rate: Conversion Rate measures the percentage of users who take a specific desired action after interacting with an item, such as making a purchase, signing up for a newsletter, or filling out a form. It helps assess the effectiveness of a call to action. (engagement rate와 비슷한데, conversion rate는 "얼마나 많은 유저가"를 측정하고, engagement rate는 "얼마나 많은 활동을" 측정함. )

* Bounce Rate: Bounce Rate calculates the percentage of users who visit a webpage or view an item but leave without taking any further action, such as navigating to another page or interacting with additional content. A high bounce rate may indicate that users are not finding the content engaging.

* Engagement Rate: Engagement Rate evaluates the level of user interaction and participation with content or ads. It can include metrics like comments, shares, likes, or time spent on a webpage. A high engagement rate suggests that users are actively involved with the content.

* Time on Page: Time on Page measures how long users spend on a webpage or interacting with a specific piece of content. It helps evaluate user engagement and the effectiveness of content in holding user attention.

* Return on Investment (ROI): ROI assesses the financial performance of an advertising or marketing campaign by comparing the costs of the campaign to the revenue generated from it. It's crucial for measuring the profitability of marketing efforts.
