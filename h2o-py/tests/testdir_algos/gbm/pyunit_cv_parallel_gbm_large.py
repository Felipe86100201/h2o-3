import sys
sys.path.insert(1,"../../../")
import h2o
from tests import pyunit_utils
from h2o.estimators.gbm import H2OGradientBoostingEstimator


def set_parallel(value):
  h2o.rapids("(setproperty \"{}\" \"{}\")".format("sys.ai.h2o.sharedtree.crossvalidation.parallelMainModelBuilding", value))


def cv_nfolds_gbm():
  loan_data = h2o.import_file(path=pyunit_utils.locate("bigdata/laptop/lending-club/loan.csv"))
  loan_data["bad_loan"] = loan_data["bad_loan"].asfactor()

  model_default = H2OGradientBoostingEstimator(nfolds=5, distribution="bernoulli", ntrees=500, 
                                               score_tree_interval=3, stopping_rounds=2, seed=42)
  try:
    set_parallel("true")
    model_default.train(y="bad_loan", training_frame=loan_data)
  finally:
    set_parallel("false")
  preds_default = model_default.predict(loan_data)

  model_sequential = H2OGradientBoostingEstimator(nfolds=5, distribution="bernoulli", ntrees=500,
                                                  score_tree_interval=3, stopping_rounds=2, seed=42)
  model_sequential.train(y="bad_loan", training_frame=loan_data)
  preds_sequential = model_sequential.predict(loan_data)

  assert model_default.actual_params["ntrees"] == model_sequential.actual_params["ntrees"]
  pyunit_utils.compare_frames_local(preds_default, preds_sequential, prob=1.0)


if __name__ == "__main__":
  pyunit_utils.standalone_test(cv_nfolds_gbm)
else:
  cv_nfolds_gbm()
