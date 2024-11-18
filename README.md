# PCDE
This is the code of a personalized conversational debiasing framework (PCDe) for next POI recommendation with uncertain check-ins. By leveraging the advantages of conversational techniques, PCDe captures personalized dynamic user preferences, enabling it to effectively mitigate both scale and popularity biases at a personalized level.


## Pre-requisits



* ### Datasets
Three datasets which are generated from Foursquare in three cities, i.e., Calgary (CAL), Charlotte (CHA) and Phoenix (PHO).
```bash
https://developer.foursquare.com/docs/build-with-foursquare/categories/
```
```
https://sites.google.com/site/yangdingqi/home/foursquare-dataset
```

* ###  Environment Requirement
* Python = 3.9.13
* Numpy >= 1.23.1
* PyTorch = 1.12.0
* pandas = 1.5.1


## Example to Run the Code

**Train Agent & Evaluate**
```
python RL_model.py --data_name <data_name> 
```

## Data Description
* `popular_POI_dict_v2.pkl`
* The data of categorized popular POIs.

* `user_category_probability.json`
* The historical preference distribution of user u across different attributes.

* `2-layer taxonomy.json`
* The mapping relationship between the first-level categories and their subcategories in Foursquare.

* `UI_interaction_data.zip`
* The training set, validation set, and test set used in this study.

    
