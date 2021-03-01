# Tadpole Challenge:

- This code was used to participate in international Alzheimer’s disease
  prediction challenge - TADPOLE challange
  https://tadpole.grand-challenge.org/


- The submission (IBM-RES-OZ) achieved MAUC of 0.905 for the clinical
  prediction task on the cross sectional data (D3). 
  
- The results are described in the paper
  https://arxiv.org/abs/2002.03419


##  Steps:
 
- To extract features, run feature extractor.py. This will save features
  as numpy file in data directory
- Train model: run train.py. This will train the models and save them in
  models directory
- To generate the leaderboard submission file run 
  generate_lb_submission.py
  
## Data files required
- data/d1_data.csv
- data/d2_data.csv
- data/TADPOLE_D1_D2_Dict.csv
- Note: d3 data is generated from last visit of d2

 