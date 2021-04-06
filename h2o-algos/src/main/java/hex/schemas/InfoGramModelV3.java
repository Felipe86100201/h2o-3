package hex.schemas;

import hex.infogram.InfoGramModel;
import water.api.API;
import water.api.schemas3.ModelOutputSchemaV3;
import water.api.schemas3.ModelSchemaV3;

public class InfoGramModelV3 extends ModelSchemaV3<InfoGramModel, InfoGramModelV3, InfoGramModel.InfoGramParameter, 
        InfoGramV3.INFOGRAMParametersV3, InfoGramModel.InfoGramOutput, InfoGramModelV3.InfoGramOutputV3> {
  public static final class InfoGramOutputV3 extends ModelOutputSchemaV3<InfoGramModel.InfoGramOutput, InfoGramOutputV3> {
    @API(help="Array of conditional information for admissible features")
    public double[] conditional_info;  // conditional info for admissible features in _admissible_features
    
    @API(help="Array of variable importance for admissible features")
    public double[] varimp;  // varimp values for admissible features in _admissible_features
    
    @API(help="Array containing names of admissible features for the user")
    public String[] _admissible_features; // predictors chosen that exceeds both conditional_info and varimp thresholds
  }
  
  public InfoGramV3.INFOGRAMParametersV3 createparametersSchema() { return new InfoGramV3.INFOGRAMParametersV3(); }
  
  public InfoGramOutputV3 createOutputSchema() { return new InfoGramOutputV3(); }
  
  @Override
  public InfoGramModel createImpl() {
    InfoGramModel.InfoGramParameter parms = parameters.createImpl();
    return new InfoGramModel(model_id.key(), parms, null);
  }
}
