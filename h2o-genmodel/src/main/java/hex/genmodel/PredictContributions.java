package hex.genmodel;

import hex.genmodel.attributes.parameters.KeyValue;

import java.io.Serializable;

public interface PredictContributions extends Serializable {

  /**
   * Calculate contributions (SHAP values) for a given input row.
   * @param input input data
   * @return per-feature contributions, last value is the model bias
   */
  float[] calculateContributions(double[] input);
  
  KeyValue[] calculateContributions(double[] input, int topN, int topBottomN, boolean abs);

  String[] getContributionNames();
}
