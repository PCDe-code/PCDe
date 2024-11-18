# PCDE

Paper: **[PCDe: A Personalized Conversational Debiasing Framework for Next POI Recommendation with Uncertain Check-Ins]**.

We propose a personalized conversational debiasing framework for next POI recommendation with uncertain check-ins, equipped with two delicately designed components to mitigate scale bias and popularity bias.



## Pre-requisits
* ### Running environment
  - Python 3.9.13
  - Pytorch 1.12.0
  - pandas 1.5.1


* ### Datasets
Three datasets which are generated from Foursquare in three cities, i.e., Calgary (CAL), Charlotte (CHA) and Phoenix (PHO).
```bash
https://developer.foursquare.com/docs/build-with-foursquare/categories/
```
```
https://sites.google.com/site/yangdingqi/home/foursquare-dataset
```

**This is our torch implementation for the paper:**
## Environment Requirement
* Python = 3.9.13
* Numpy >= 1.23.1
* PyTorch = 1.12.0
* pandas = 1.5.1

## Example to Run the Code

**Train Agent & Evaluate**
```
python RL_model.py --data_name <data_name> 
```

**More Details:**

Use `python RL_model.py -h` to get more argument setting details.

```
  -h, --help            show this help message and exit
  --seed <seed>         random seed.
  --epochs <epochs>     the number of RL train epoch.
  --fm_epoch <fm_epoch> the epoch of FM embedding
  --batch_size <batch_size>
                        batch size.
  --gamma <gamma>       reward discount factor.
  --lr <lr>             learning rate.
  --hidden <hidden>     hidden size
  --memory_size <memory_size>
                        the size of memory
  --data_name <data_name>
                        One of {LAST_FM*, LAST_FM, YELP*, YELP}.
  --entropy_method <entropy_method>
                        entropy_method is one of {entropy, weight entropy}
  --max_turn <max_turn>
                        max conversation turn
  --ask_num <attr_num>   the number of attributes for <data_set>
  --observe_num <observe_num>
                        the number of epochs to save RL model and metric
  --target_update <target_update>
                        the number of epochs to update policy parameters

```

## Data Description
**1. Graph Generate Data**

* `popular_POI_dict_v2.pkl`
* The data of categorized popular POIs.

* `user_category_probability.json`
* The historical preference distribution of user u across different attributes.

* `2-layer taxonomy.json`
* The mapping relationship between the first-level categories and their subcategories in Foursquare.

* `UI_interaction_data.zip`
* The training set, validation set, and test set used in this study.

    
